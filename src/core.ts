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
	/**
	 * Soft target for estimated input tokens sent to the main model (messages + system prompt).
	 * Compression is applied oldest-first only until the estimate is under this target.
	 * 0 = compress everything eligible (smallest possible context).
	 */
	contextTarget: number;
	/** Once over `contextTarget`, compress down to this fraction of it (headroom for following turns). */
	targetLowRatio: number;
	/** Max consecutive tool-calling steps collapsed into one block summary. */
	maxBlockSteps: number;
	/** Soft length target for block summaries, in words. */
	blockSummaryWords: number;
	/**
	 * Characters of surrounding (uncompressed) session transcript given to the compressor as context.
	 * The compressor is cheap, so it sees the whole session; this transcript is a shared, cacheable prefix.
	 */
	compressorContextChars: number;
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
	contextTarget: 0,
	targetLowRatio: 0.7,
	maxBlockSteps: 10,
	blockSummaryWords: 350,
	compressorContextChars: 300_000,
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

/** One compressor call. `system` carries the shared session transcript (identical across calls in one pass, so it caches). */
export interface SummarizeRequest {
	kind: "output" | "block";
	/** Human-readable label, e.g. "bash" or "steps S3–S9". */
	label: string;
	system: string;
	prompt: string;
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

	/** Raw transcript of a collapsed block of steps. */
	writeBlock(key: string, label: string, transcript: string): string {
		const file = join(this.dir, `block-${key}.txt`);
		mkdirSync(this.dir, { recursive: true });
		writeFileSync(file, `# pi-squeeze collapsed block: ${label}\n\n${transcript}`, "utf8");
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
	/** Tool outputs replaced by per-output summaries. */
	squeezed: number;
	/** Blocks of steps collapsed into a single summary. */
	blocks: number;
	/** Messages removed by block collapsing (net). */
	messagesRemoved: number;
	newlySummarized: number;
	charsSaved: number;
	/** Estimated tokens of the returned messages (excl. system prompt). */
	tokensAfter: number;
	errors: string[];
}

// ---------------------------------------------------------------------------
// Algorithm overview
//
// 1. Segment the transcript into "steps": an assistant message that makes tool calls plus the
//    tool results answering it. Everything else (user, system, custom, compaction summaries,
//    assistant messages without tool calls) is a protected barrier and is never touched.
// 2. Protect the first `keepFirstToolTurns` and last `keepRecentToolTurns` steps, plus steps
//    that are still unresolved (missing results).
// 3. Group remaining consecutive steps (no barrier in between) into runs, then chunk each run
//    into blocks of at most `maxBlockSteps`. Chunk boundaries are fixed per run (anchored at the
//    run start), so block identity, and therefore the summary cache, is stable as the session grows.
// 4. Compression levels, cheapest first:
//      L1: replace a large tool output with a summary (old behaviour)
//      L2: replace a whole block of steps (assistant text, tool calls, results) with one summary
//    Apply oldest-first until the estimated context is under `contextTarget` (0 = all the way).
//    A block of a single step is only squeezed at L1; L2 needs >= 2 steps.
// 5. Compressor calls share one system prompt containing the full uncompressed session transcript
//    (cheap model, cacheable prefix), so summaries are written with whole-session awareness.
// ---------------------------------------------------------------------------

interface ToolCallInfo {
	id: string;
	name: string;
	args: string;
}

interface Step {
	/** Index of the assistant message in `messages`. */
	aIdx: number;
	/** Indices of the tool results belonging to this step. */
	resultIdx: number[];
	calls: ToolCallInfo[];
	complete: boolean;
}

interface Block {
	steps: Step[];
	/** First/last message index covered (assistant .. last result). */
	from: number;
	to: number;
	/** Step ordinal (1-based) for labels. */
	firstOrdinal: number;
}

function toolCallsOf(m: MsgLike): ToolCallInfo[] {
	if (m.role !== "assistant" || !Array.isArray(m.content)) return [];
	const out: ToolCallInfo[] = [];
	for (const p of m.content as AnyPart[]) {
		if (p.type === "toolCall" && typeof p.id === "string") {
			out.push({ id: p.id, name: String(p.name ?? "tool"), args: JSON.stringify(p.arguments ?? {}) });
		}
	}
	return out;
}

/** Split messages into tool steps. Messages between steps are barriers. */
function segmentSteps(messages: MsgLike[]): { steps: Step[]; barrierBefore: boolean[] } {
	const steps: Step[] = [];
	const barrierBefore: boolean[] = []; // barrierBefore[k]: a protected message sits between step k-1 and k
	let sawBarrier = false;
	let i = 0;
	while (i < messages.length) {
		const m = messages[i];
		const calls = toolCallsOf(m);
		if (calls.length === 0) {
			sawBarrier = true;
			i++;
			continue;
		}
		const ids = new Set(calls.map((c) => c.id));
		const resultIdx: number[] = [];
		let j = i + 1;
		while (j < messages.length && messages[j].role === "toolResult" && ids.has(String(messages[j].toolCallId))) {
			resultIdx.push(j);
			j++;
		}
		barrierBefore.push(sawBarrier || steps.length === 0);
		steps.push({ aIdx: i, resultIdx, calls, complete: resultIdx.length === calls.length });
		sawBarrier = false;
		i = j;
	}
	return { steps, barrierBefore };
}

function buildBlocks(steps: Step[], barrierBefore: boolean[], config: SqueezeConfig): Block[] {
	const n = steps.length;
	const lo = Math.max(0, config.keepFirstToolTurns);
	const hi = n - Math.max(0, config.keepRecentToolTurns);
	const blocks: Block[] = [];
	const size = Math.max(1, config.maxBlockSteps);
	let run: number[] = [];
	const flush = () => {
		for (let s = 0; s < run.length; s += size) {
			const ks = run.slice(s, s + size);
			const st = ks.map((k) => steps[k]);
			const last = st[st.length - 1];
			blocks.push({
				steps: st,
				from: st[0].aIdx,
				to: last.resultIdx.length ? last.resultIdx[last.resultIdx.length - 1] : last.aIdx,
				firstOrdinal: ks[0] + 1,
			});
		}
		run = [];
	};
	for (let k = 0; k < n; k++) {
		const eligible = k >= lo && k < hi && steps[k].complete;
		if (!eligible || barrierBefore[k]) flush();
		if (eligible) run.push(k);
	}
	flush();
	return blocks;
}

/** Plain-text rendering of messages for the compressor (and for raw block files). */
export function renderTranscript(messages: MsgLike[], from = 0, to = messages.length - 1, outputCap = Infinity): string {
	const lines: string[] = [];
	for (let i = from; i <= to && i < messages.length; i++) {
		const m = messages[i];
		if (m.role === "toolResult") {
			const t = textOf(m.content);
			lines.push(`[Tool result ${m.toolName ?? "tool"}${m.isError ? " (error)" : ""}]:\n${clip(t, outputCap)}`);
			continue;
		}
		if (m.role === "assistant" && Array.isArray(m.content)) {
			const parts: string[] = [];
			for (const p of m.content as AnyPart[]) {
				if (p.type === "text" && typeof p.text === "string" && p.text.trim()) parts.push(p.text);
				else if (p.type === "toolCall") parts.push(`<tool call> ${String(p.name)}(${piArgs(JSON.stringify(p.arguments ?? {}))})`);
			}
			if (parts.length) lines.push(`[Assistant]:\n${parts.join("\n")}`);
			continue;
		}
		const body = typeof m.summary === "string" ? m.summary : textOf(m.content);
		if (body) lines.push(`[${m.role === "user" ? "User" : m.role}]:\n${body}`);
	}
	return lines.join("\n\n");
}

/**
 * Session context for the compressor: the full uncompressed transcript, capped at `maxChars`.
 * When over budget, each tool output is clipped to an equal share (head + tail); user and
 * assistant text are always included in full. As a last resort the middle is dropped.
 */
export function sessionContext(messages: MsgLike[], maxChars: number): string {
	if (maxChars <= 0) return "";
	let full = renderTranscript(messages);
	if (full.length <= maxChars) return full;
	const outputs = messages.filter((m) => m.role === "toolResult").length || 1;
	let cap = Math.floor(maxChars / outputs);
	for (let tries = 0; tries < 6 && cap > 200; tries++) {
		full = renderTranscript(messages, 0, messages.length - 1, cap);
		if (full.length <= maxChars) return full;
		cap = Math.floor(cap / 2);
	}
	return clip(full, maxChars);
}

function blockKey(messages: MsgLike[], b: Block): string {
	const h = createHash("sha256");
	for (let i = b.from; i <= b.to; i++) {
		const m = messages[i];
		h.update(m.role + "\0" + String(m.toolCallId ?? "") + "\0");
		h.update(m.role === "assistant" ? JSON.stringify(m.content) : textOf(m.content));
		h.update("\u0001");
	}
	return "b" + h.digest("hex").slice(0, 16);
}

function blockLabel(b: Block): string {
	const a = b.firstOrdinal;
	const z = a + b.steps.length - 1;
	return a === z ? `step S${a}` : `steps S${a}-S${z}`;
}

function blockMessage(rec: SummaryRecord, b: Block, original: MsgLike): MsgLike {
	const tools = new Map<string, number>();
	for (const s of b.steps) for (const c of s.calls) tools.set(c.name, (tools.get(c.name) ?? 0) + 1);
	const toolList = [...tools].map(([n, c]) => (c > 1 ? `${n}x${c}` : n)).join(", ");
	const text = [
		`[pi-squeeze: ${blockLabel(b)} (${b.steps.length} tool-calling steps: ${toolList}; ${rec.originalChars} chars) were collapsed into this summary by a smaller model.`,
		`Full original transcript: ${rec.file}`,
		`Use rg / read on that file if you need exact details.]`,
		"",
		rec.summary,
	].join("\n");
	// Keep provider/model metadata of the original assistant message; drop tool calls.
	const { content: _c, ...meta } = original;
	return { ...meta, stopReason: "stop", content: [{ type: "text", text }] };
}

function tokensOf(m: MsgLike): number {
	return estimateTokens([m]);
}

/**
 * Compress older steps until the context fits `contextTarget`. Returns a new array; input is not mutated.
 * `systemTokens` is added to the message estimate when comparing against the target.
 */
export async function squeezeMessages(
	messages: MsgLike[],
	config: SqueezeConfig,
	store: SqueezeStore,
	summarize: Summarizer,
	signal?: AbortSignal,
	systemTokens = 0,
): Promise<SqueezeResult> {
	const result: SqueezeResult = {
		messages,
		squeezed: 0,
		blocks: 0,
		messagesRemoved: 0,
		newlySummarized: 0,
		charsSaved: 0,
		tokensAfter: 0,
		errors: [],
	};
	const msgTokens = messages.map(tokensOf);
	let total = msgTokens.reduce((a, b) => a + b, 0) + systemTokens;
	// Hysteresis: new compression only starts when over `contextTarget`, then continues down to
	// `contextTarget * targetLowRatio`. Undershooting leaves headroom so the next turns reuse cached
	// summaries (no compressor calls, stable prompt prefix for the main model's cache).
	const target = Math.max(0, config.contextTarget);
	const low = target * Math.min(1, Math.max(0, config.targetLowRatio));
	const triggered = target === 0 || total > target;
	const overTarget = () => target === 0 || (triggered && total > low);

	const { steps, barrierBefore } = segmentSteps(messages);
	const blocks = buildBlocks(steps, barrierBefore, config);
	if (blocks.length === 0) {
		result.tokensAfter = total - systemTokens;
		return result;
	}

	// Shared compressor context: whole uncompressed session, built lazily (only if a call is needed).
	let system: string | undefined;
	const systemFor = (style: PromptStyle) => {
		system ??= compressorSystem(style) + sessionSection(sessionContext(messages, config.compressorContextChars));
		return system;
	};
	const userGoal = clip(latestUserText(messages), 3000);

	const replaced = new Map<number, MsgLike>(); // message index -> replacement
	const removed = new Set<number>();

	// ---- Level 1: per-output summaries, oldest first ----
	interface OutJob {
		idx: number;
		key: string;
		text: string;
		toolName: string;
		args: string;
	}
	const argsById = new Map<string, string>();
	for (const s of steps) for (const c of s.calls) argsById.set(c.id, c.args);
	const l1Candidates = (b: Block): OutJob[] => {
		const out: OutJob[] = [];
		for (const s of b.steps) {
			for (const idx of s.resultIdx) {
				const m = messages[idx];
				const text = textOf(m.content);
				if (text.length < config.minChars || text.startsWith("[pi-squeeze:")) continue;
				out.push({
					idx,
					key: hashKey(String(m.toolCallId), text),
					text,
					toolName: String(m.toolName ?? "tool"),
					args: argsById.get(String(m.toolCallId)) ?? "{}",
				});
			}
		}
		return out;
	};
	const applyL1 = (job: OutJob, rec: SummaryRecord) => {
		if (!rec.summary) return;
		const m = messages[job.idx];
		const parts = Array.isArray(m.content) ? (m.content as AnyPart[]) : [{ type: "text", text: String(m.content ?? "") }];
		const nonText = parts.filter((p) => p.type !== "text");
		const next: MsgLike = { ...m, content: [{ type: "text", text: placeholder(rec, job.toolName) } as TextPart, ...nonText] };
		const before = msgTokens[job.idx];
		const after = tokensOf(next);
		replaced.set(job.idx, next);
		msgTokens[job.idx] = after;
		total += after - before;
		result.charsSaved += Math.max(0, rec.originalChars - rec.summary.length);
		result.squeezed++;
	};
	const runL1 = async (jobs: OutJob[]) => {
		const todo: OutJob[] = [];
		for (const j of jobs) {
			const rec = store.get(j.key);
			if (rec) applyL1(j, rec);
			else todo.push(j);
		}
		await mapLimit(todo, config.concurrency, async (job) => {
			if (signal?.aborted) return;
			try {
				const summary = (
					await summarize({
						kind: "output",
						label: job.toolName,
						system: systemFor(config.promptStyle),
						prompt: outputPrompt(job, userGoal, config),
					})
				).trim();
				const file = store.writeRaw(job.key, job.toolName, job.args, job.text);
				const worth = summary.length > 0 && summary.length <= job.text.length * config.maxRatio;
				const rec: SummaryRecord = { summary: worth ? summary : "", file, originalChars: job.text.length };
				store.set(job.key, rec);
				result.newlySummarized++;
				applyL1(job, rec);
			} catch (e) {
				result.errors.push(`${job.toolName}: ${e instanceof Error ? e.message : String(e)}`);
			}
		});
	};

	// Level 1, oldest first. Cached summaries are always applied: they're free, and they keep the
	// main model's prompt prefix stable. New ones run in windows of `concurrency` until under target.
	const allJobs = blocks.flatMap(l1Candidates);
	const uncached: OutJob[] = [];
	for (const j of allJobs) {
		const rec = store.get(j.key);
		if (rec) applyL1(j, rec);
		else uncached.push(j);
	}
	const win = Math.max(1, config.concurrency);
	for (let s = 0; s < uncached.length && overTarget() && !signal?.aborted; s += win) {
		await runL1(uncached.slice(s, s + win));
	}

	// ---- Level 2: collapse whole blocks, oldest first ----
	const applyL2 = (b: Block, rec: SummaryRecord) => {
		if (!rec.summary) return;
		let before = 0;
		for (let i = b.from; i <= b.to; i++) before += msgTokens[i];
		const msg = blockMessage(rec, b, messages[b.from]);
		const after = tokensOf(msg);
		if (after >= before) return;
		for (let i = b.from; i <= b.to; i++) {
			if (replaced.has(i)) {
				const r = replaced.get(i)!;
				if (r.role === "toolResult") result.squeezed--; // L1 superseded by L2
				replaced.delete(i);
			}
			if (i === b.from) continue;
			removed.add(i);
			msgTokens[i] = 0;
		}
		replaced.set(b.from, msg);
		msgTokens[b.from] = after;
		total += after - before;
		result.charsSaved += Math.max(0, rec.originalChars - rec.summary.length);
		result.messagesRemoved += b.to - b.from;
		result.blocks++;
	};

	const l2 = blocks.filter((b) => b.steps.length >= 2);
	for (const b of l2) {
		const key = blockKey(messages, b);
		const cached = store.get(key);
		if (cached) {
			// Always reuse cached block summaries (stable prefix for the main model).
			applyL2(b, cached);
			continue;
		}
		if (!overTarget()) continue;
		if (signal?.aborted) break;
		const transcript = renderTranscript(messages, b.from, b.to);
		try {
			const summary = (
				await summarize({
					kind: "block",
					label: blockLabel(b),
					system: systemFor(config.promptStyle),
					prompt: blockPrompt(messages, b, userGoal, config),
				})
			).trim();
			const file = store.writeBlock(key, blockLabel(b), transcript);
			const worth = summary.length > 0 && summary.length <= transcript.length * config.maxRatio;
			const rec: SummaryRecord = { summary: worth ? summary : "", file, originalChars: transcript.length };
			store.set(key, rec);
			result.newlySummarized++;
			applyL2(b, rec);
		} catch (e) {
			result.errors.push(`${blockLabel(b)}: ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	result.tokensAfter = total - systemTokens;
	if (replaced.size === 0 && removed.size === 0) return result;
	const out: MsgLike[] = [];
	for (let i = 0; i < messages.length; i++) {
		if (removed.has(i)) continue;
		out.push(replaced.get(i) ?? messages[i]);
	}
	result.messages = out;
	return result;
}

function latestUserText(messages: MsgLike[]): string {
	for (let i = messages.length - 1; i >= 0; i--) {
		if (messages[i].role === "user") {
			const t = textOf(messages[i].content);
			if (t) return t;
		}
	}
	return "";
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

/** Appended to the compressor system prompt. Identical for every call in a pass, so it is a cacheable prefix. */
export function sessionSection(transcript: string): string {
	if (!transcript) return "";
	return [
		"",
		"",
		"Below is the agent's session so far (uncompressed, for background only). Use it to judge what matters:",
		"what the user asked for, what the agent is trying to do, and which facts later steps relied on.",
		"Only summarize the material the user message asks you to summarize; do not summarize this session.",
		"<session>",
		transcript,
		"</session>",
	].join("\n");
}

interface OutputJobLike {
	toolName: string;
	args: string;
	text: string;
}

export function outputPrompt(job: OutputJobLike, userGoal: string, config: SqueezeConfig): string {
	const output = clip(job.text, config.maxCompressorInputChars);
	const args = clip(job.args, 4000);
	if (config.promptStyle === "pi") {
		const conversation = [
			`[User]: ${userGoal || "(unknown)"}`,
			`[Assistant tool calls]: ${job.toolName}(${piArgs(args)})`,
			`[Tool result]: ${output}`,
		].join("\n\n");
		return `<conversation>\n${conversation}\n</conversation>\n\n${PI_SUMMARIZATION_PROMPT}`;
	}
	return [
		`Agent's latest user request (for relevance):\n<goal>\n${userGoal || "(unknown)"}\n</goal>`,
		`Tool: ${job.toolName}\nArguments: ${args}`,
		`<tool_output>\n${output}\n</tool_output>`,
		`Summarize this tool output in at most ~${config.summaryWords} words (fewer if the output is simple).`,
	].join("\n\n");
}

export function blockPrompt(
	messages: MsgLike[],
	b: { from: number; to: number; steps: unknown[] },
	userGoal: string,
	config: SqueezeConfig,
): string {
	const transcript = clip(renderTranscript(messages, b.from, b.to), config.maxCompressorInputChars);
	if (config.promptStyle === "pi") {
		return `<conversation>\n[User]: ${userGoal || "(unknown)"}\n\n${transcript}\n</conversation>\n\n${PI_SUMMARIZATION_PROMPT}`;
	}
	return [
		`Agent's latest user request (for relevance):\n<goal>\n${userGoal || "(unknown)"}\n</goal>`,
		`The agent took the following ${b.steps.length} consecutive tool-calling steps. They will be removed from its context and replaced by your summary.`,
		`<steps>\n${transcript}\n</steps>`,
		[
			`Summarize these steps in at most ~${config.blockSummaryWords} words as a record of what the agent did and learned:`,
			"actions taken (commands, files read/edited), results and findings, exact errors, decisions and their reasons,",
			"and anything still open. Preserve paths, line numbers, symbols, and values the agent may need later.",
		].join(" "),
	].join("\n\n");
}
