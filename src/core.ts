/**
 * Pure transformation logic for pi-squeeze. No pi runtime imports here so it can be unit-tested.
 */
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";

export interface SqueezeConfig {
	/** Master switch. */
	enabled: boolean;
	/** Compression model as "provider/modelId". Empty = not configured (extension stays idle). */
	compressorModel: string;
	/**
	 * Only squeeze when the active (expensive) model matches one of these "provider/modelId"
	 * patterns. `*` wildcards allowed. Empty list = squeeze for every model except the compressor.
	 */
	targetModels: string[];
	/** Tool results belonging to the last N tool-calling assistant messages stay verbatim. */
	keepRecentToolTurns: number;
	/** Tool results of the first N tool-calling assistant messages stay verbatim (initial exploration). */
	keepFirstToolTurns: number;
	/** Only tool results with at least this many text characters are squeezed. */
	minChars: number;
	/** Soft length target for summaries, in words. */
	summaryWords: number;
	/** Keep the original if summary is not shorter than this fraction of the original. */
	maxRatio: number;
	/** Characters of tool output sent to the compressor at most (head + tail kept). */
	maxCompressorInputChars: number;
	/** Parallel compressor calls. */
	concurrency: number;
	/** Max output tokens for one summary call. */
	maxSummaryTokens: number;
	/** Compressor prompt: "squeeze" (terse, per tool output) or "pi" (pi's built-in compaction prompt). */
	promptStyle: PromptStyle;
	/** Compact when the (squeezed) prompt reaches this % of the main model's context window. 0 = off. */
	compactAtPercent: number;
	/** Run pi compactions (auto, /compact, /squeeze-compact) with the compressor model instead of the chat model. */
	compactWithCompressor: boolean;
}

export const DEFAULT_CONFIG: SqueezeConfig = {
	enabled: true,
	compressorModel: "",
	targetModels: [],
	keepRecentToolTurns: 2,
	keepFirstToolTurns: 1,
	minChars: 1500,
	summaryWords: 200,
	maxRatio: 0.6,
	maxCompressorInputChars: 400_000,
	concurrency: 4,
	maxSummaryTokens: 2048,
	promptStyle: "squeeze",
	compactAtPercent: 70,
	compactWithCompressor: true,
};

/** Rough token estimate (chars / 4) of what a message list costs as LLM input. */
export function estimateTokens(messages: MsgLike[]): number {
	let chars = 0;
	for (const m of messages) {
		if (typeof m.content === "string") {
			chars += m.content.length;
			continue;
		}
		if (!Array.isArray(m.content)) continue;
		for (const p of m.content as AnyPart[]) {
			if (typeof p.text === "string") chars += p.text.length;
			else if (typeof p.thinking === "string") chars += p.thinking.length;
			else if (p.type === "toolCall") chars += JSON.stringify(p.arguments ?? {}).length;
			else if (p.type === "image") chars += 4800; // ~1.2k tokens
		}
	}
	return Math.ceil(chars / 4);
}

/** 1234 -> "1.2k", 2_500_000 -> "2.5M". */
export function formatTokens(n: number): string {
	if (n < 1000) return String(Math.round(n));
	if (n < 1_000_000) return `${(n / 1000).toFixed(n < 10_000 ? 1 : 0)}k`;
	return `${(n / 1_000_000).toFixed(n < 10_000_000 ? 2 : 1)}M`;
}

export function loadConfig(path: string): SqueezeConfig {
	if (!existsSync(path)) return { ...DEFAULT_CONFIG };
	try {
		const raw = JSON.parse(readFileSync(path, "utf8"));
		return { ...DEFAULT_CONFIG, ...raw };
	} catch {
		return { ...DEFAULT_CONFIG };
	}
}

export function saveConfig(path: string, config: SqueezeConfig): void {
	writeFileSync(path, `${JSON.stringify(config, null, 2)}\n`, "utf8");
}

export function globMatch(pattern: string, value: string): boolean {
	const re = new RegExp(`^${pattern.split("*").map(escapeRe).join(".*")}$`, "i");
	return re.test(value);
}
function escapeRe(s: string): string {
	return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function shouldSqueezeFor(config: SqueezeConfig, activeModel: string | undefined): boolean {
	if (!config.enabled || !config.compressorModel || !activeModel) return false;
	if (activeModel.toLowerCase() === config.compressorModel.toLowerCase()) return false;
	if (config.targetModels.length === 0) return true;
	return config.targetModels.some((p) => globMatch(p, activeModel));
}

// ---- minimal structural message types (subset of pi-ai Message) ----
type TextPart = { type: "text"; text: string };
type AnyPart = { type: string; [k: string]: unknown };
export interface MsgLike {
	role: string;
	content?: unknown;
	toolCallId?: string;
	toolName?: string;
	[k: string]: unknown;
}

export interface SummaryRecord {
	summary: string;
	file: string;
	originalChars: number;
}

export interface SummarizeRequest {
	toolName: string;
	toolArgs: string;
	output: string;
	userGoal: string;
}

export type Summarizer = (req: SummarizeRequest) => Promise<string>;

/** Persistent per-session store: summaries cache + raw output files. */
export class SqueezeStore {
	readonly dir: string;
	private cache: Record<string, SummaryRecord> = {};
	private cachePath: string;

	constructor(dir: string) {
		this.dir = dir;
		mkdirSync(dir, { recursive: true });
		this.cachePath = join(dir, "summaries.json");
		if (existsSync(this.cachePath)) {
			try {
				this.cache = JSON.parse(readFileSync(this.cachePath, "utf8"));
			} catch {
				this.cache = {};
			}
		}
	}

	get(key: string): SummaryRecord | undefined {
		const rec = this.cache[key];
		// Raw file may have been wiped by tmp cleaning; then re-squeeze.
		if (rec && !existsSync(rec.file)) return undefined;
		return rec;
	}

	writeRaw(key: string, toolName: string, args: string, output: string): string {
		const safeTool = toolName.replace(/[^a-zA-Z0-9_-]/g, "_");
		const file = join(this.dir, `${safeTool}-${key}.txt`);
		// The tmp dir can vanish mid-session (tmp cleaners, manual rm); recreate it.
		mkdirSync(this.dir, { recursive: true });
		writeFileSync(file, `# tool: ${toolName}\n# args: ${args}\n\n${output}`, "utf8");
		return file;
	}

	set(key: string, rec: SummaryRecord): void {
		this.cache[key] = rec;
		mkdirSync(this.dir, { recursive: true });
		writeFileSync(this.cachePath, JSON.stringify(this.cache), "utf8");
	}

	stats(): { count: number; originalChars: number; summaryChars: number } {
		let originalChars = 0;
		let summaryChars = 0;
		const recs = Object.values(this.cache);
		for (const r of recs) {
			originalChars += r.originalChars;
			summaryChars += r.summary.length;
		}
		return { count: recs.length, originalChars, summaryChars };
	}
}

function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return (content as AnyPart[])
		.filter((p) => p.type === "text")
		.map((p) => (p as TextPart).text)
		.join("\n");
}

function hashKey(toolCallId: string, text: string): string {
	return createHash("sha256").update(toolCallId).update("\0").update(text).digest("hex").slice(0, 16);
}

export function clip(text: string, max: number): string {
	if (text.length <= max) return text;
	const half = Math.floor(max / 2);
	return `${text.slice(0, half)}\n\n[... ${text.length - max} chars omitted ...]\n\n${text.slice(-half)}`;
}

export function placeholder(rec: SummaryRecord, toolName: string): string {
	return [
		`[pi-squeeze: this ${toolName} output (${rec.originalChars} chars) was summarized by a smaller model to save context.`,
		`Full original output: ${rec.file}`,
		`Use rg / read on that file if you need exact details.]`,
		"",
		rec.summary,
	].join("\n");
}

async function mapLimit<T>(items: T[], limit: number, fn: (t: T) => Promise<void>): Promise<void> {
	let i = 0;
	const workers = Array.from({ length: Math.max(1, Math.min(limit, items.length)) }, async () => {
		while (i < items.length) {
			const item = items[i++];
			await fn(item);
		}
	});
	await Promise.all(workers);
}

export interface SqueezeResult {
	messages: MsgLike[];
	squeezed: number;
	newlySummarized: number;
	charsSaved: number;
	errors: string[];
}

/**
 * Replace older, large tool results with summaries. Returns a new array; input is not mutated.
 */
export async function squeezeMessages(
	messages: MsgLike[],
	config: SqueezeConfig,
	store: SqueezeStore,
	summarize: Summarizer,
	signal?: AbortSignal,
): Promise<SqueezeResult> {
	const result: SqueezeResult = { messages, squeezed: 0, newlySummarized: 0, charsSaved: 0, errors: [] };

	// Map toolCallId -> args, and find which assistant-turns are "recent".
	const toolArgs = new Map<string, string>();
	const toolCallOwner = new Map<string, number>(); // toolCallId -> assistant msg index
	const toolAssistantIdx: number[] = [];
	messages.forEach((m, idx) => {
		if (m.role !== "assistant" || !Array.isArray(m.content)) return;
		let hasCall = false;
		for (const p of m.content as AnyPart[]) {
			if (p.type === "toolCall" && typeof p.id === "string") {
				hasCall = true;
				toolArgs.set(p.id, JSON.stringify(p.arguments ?? {}));
				toolCallOwner.set(p.id, idx);
			}
		}
		if (hasCall) toolAssistantIdx.push(idx);
	});
	const keep = Math.max(0, config.keepRecentToolTurns);
	const cutoffAssistantIdx =
		toolAssistantIdx.length > keep ? toolAssistantIdx[toolAssistantIdx.length - keep - 1] : -1;
	const firstKeep = Math.max(0, config.keepFirstToolTurns);
	const protectedFirst = new Set(toolAssistantIdx.slice(0, firstKeep));

	// Last user message text, used as relevance hint for the compressor.
	let userGoal = "";
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role === "user") {
			userGoal = clip(textOf(messages[i].content), 3000);
			break;
		}
	}

	type Job = { idx: number; key: string; text: string; toolName: string; args: string };
	const jobs: Job[] = [];
	const ready = new Map<number, SummaryRecord>();

	messages.forEach((m, idx) => {
		if (m.role !== "toolResult" || !m.toolCallId) return;
		const owner = toolCallOwner.get(m.toolCallId);
		// Only tool results are ever rewritten: system prompt, user and assistant messages pass through untouched.
		if (owner === undefined || owner > cutoffAssistantIdx) return; // recent: keep verbatim
		if (protectedFirst.has(owner)) return; // first turn(s): keep verbatim
		const text = textOf(m.content);
		if (text.length < config.minChars) return;
		if (text.startsWith("[pi-squeeze:")) return;
		const key = hashKey(m.toolCallId, text);
		const cached = store.get(key);
		if (cached) ready.set(idx, cached);
		else jobs.push({ idx, key, text, toolName: m.toolName ?? "tool", args: toolArgs.get(m.toolCallId) ?? "{}" });
	});

	await mapLimit(jobs, config.concurrency, async (job) => {
		if (signal?.aborted) return;
		try {
			const summary = (
				await summarize({
					toolName: job.toolName,
					toolArgs: clip(job.args, 4000),
					output: clip(job.text, config.maxCompressorInputChars),
					userGoal,
				})
			).trim();
			const file = store.writeRaw(job.key, job.toolName, job.args, job.text);
			// Summary not worth it: cache a marker so we don't retry every call.
			const worth = summary.length > 0 && summary.length <= job.text.length * config.maxRatio;
			const rec: SummaryRecord = { summary: worth ? summary : "", file, originalChars: job.text.length };
			store.set(job.key, rec);
			ready.set(job.idx, rec);
			result.newlySummarized++;
		} catch (e) {
			result.errors.push(`${job.toolName}: ${e instanceof Error ? e.message : String(e)}`);
		}
	});

	if (ready.size === 0) return result;

	result.messages = messages.map((m, idx) => {
		const rec = ready.get(idx);
		if (!rec || !rec.summary) return m;
		const original = textOf(m.content);
		const text = placeholder(rec, m.toolName ?? "tool");
		// Preserve non-text parts (e.g. images) as-is.
		const nonText = Array.isArray(m.content) ? (m.content as AnyPart[]).filter((p) => p.type !== "text") : [];
		result.squeezed++;
		result.charsSaved += original.length - text.length;
		return { ...m, content: [{ type: "text", text }, ...nonText] };
	});
	return result;
}

// ---- compressor prompts ----

export type PromptStyle = "squeeze" | "pi";
export const PROMPT_STYLES: PromptStyle[] = ["squeeze", "pi"];

/** "squeeze" style: terse, tool-output-specific prompt (pi-squeeze's own). */
export const SQUEEZE_SYSTEM = `You compress tool outputs for a coding agent. The agent's context is expensive, so you replace a tool output with a dense summary it will read instead of the original.
Rules:
- Keep everything the agent is likely to need later: file paths, line numbers, function/class names, exact error messages, failing test names, versions, key values, counts, and conclusions.
- Prefer exact snippets (short) over paraphrase for code and errors.
- Drop boilerplate, repetition, progress bars, and irrelevant noise.
- Say what was NOT found if the output shows absence (e.g. "no matches for X").
- Do not add advice or commentary. Output only the summary, plain text or terse markdown bullets.`;

/**
 * "pi" style: pi's built-in compaction prompts, copied verbatim from pi-coding-agent
 * dist/core/compaction/utils.js (SUMMARIZATION_SYSTEM_PROMPT) and compaction.js (SUMMARIZATION_PROMPT).
 * The input is serialized like pi's serializeConversation().
 */
export const PI_SYSTEM = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary.`;

export const PI_SUMMARIZATION_PROMPT = `The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages.`;

function piArgs(argsJson: string): string {
	try {
		const args = JSON.parse(argsJson) as Record<string, unknown>;
		return Object.entries(args)
			.map(([k, v]) => `${k}=${JSON.stringify(v)}`)
			.join(", ");
	} catch {
		return argsJson;
	}
}

export function compressorSystem(style: PromptStyle): string {
	return style === "pi" ? PI_SYSTEM : SQUEEZE_SYSTEM;
}

export function compressorPrompt(req: SummarizeRequest, words: number, style: PromptStyle = "squeeze"): string {
	if (style === "pi") {
		const conversation = [
			`[User]: ${req.userGoal || "(unknown)"}`,
			`[Assistant tool calls]: ${req.toolName}(${piArgs(req.toolArgs)})`,
			`[Tool result]: ${req.output}`,
		].join("\n\n");
		return `<conversation>\n${conversation}\n</conversation>\n\n${PI_SUMMARIZATION_PROMPT}`;
	}
	return [
		`Agent's latest user request (for relevance):\n<goal>\n${req.userGoal || "(unknown)"}\n</goal>`,
		`Tool: ${req.toolName}\nArguments: ${req.toolArgs}`,
		`<tool_output>\n${req.output}\n</tool_output>`,
		`Summarize the tool output in at most ~${words} words (fewer if the output is simple).`,
	].join("\n\n");
}
