/**
 * pi-squeeze: before each LLM call, replace older large tool outputs with summaries written by a
 * cheap "compressor" model. Originals are kept in tmp files so the agent can rg/read them.
 *
 * Also: pi compactions (auto, /compact, /squeeze-compact) are summarized by the compressor model,
 * and a compaction is triggered when the squeezed prompt reaches compactAtPercent of the context window.
 *
 * Commands: /squeeze [status|on|off|model|targets|style|set <key> <value>], /squeeze-compact [instructions]
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
// Host-provided at runtime (peer dependency); pi's own compaction routine, reused with another model.
import { compact as piCompact } from "@earendil-works/pi-coding-agent";
import {
	formatTokens,
	PROMPT_STYLES,
	type PromptStyle,
	DEFAULT_CONFIG,
	loadConfig,
	type MsgLike,
	type SqueezeConfig,
	SqueezeStore,
	saveConfig,
	shouldSqueezeFor,
	squeezeMessages,
} from "./core.ts";

const agentDir = () => process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
const configPath = () => process.env.PI_SQUEEZE_CONFIG || join(agentDir(), "pi-squeeze.json");

export default function (pi: ExtensionAPI) {
	let config: SqueezeConfig = loadConfig(configPath());
	let store: SqueezeStore | undefined;
	let storeSession = "";
	let lastReport = "";
	const totals = { calls: 0, charsSaved: 0, summarized: 0, errors: 0 };
	// Cumulative main-model input tokens avoided this session (squeezing every call + compactions run on the compressor).
	let saved = { squeeze: 0, compaction: 0 };
	let savedPath = "";
	// Estimated main-model prompt size after the latest squeeze; null until measured after a compaction.
	let lastPromptTokens: number | null = null;
	let compacting = false;

	const persist = () => saveConfig(configPath(), config);
	const modelKey = (m: { provider: string; id: string } | undefined) => (m ? `${m.provider}/${m.id}` : undefined);

	function getStore(ctx: ExtensionContext): SqueezeStore {
		const sid = ctx.sessionManager.getSessionId() || "ephemeral";
		if (!store || storeSession !== sid) {
			store = new SqueezeStore(join(tmpdir(), "pi-squeeze", sid));
			storeSession = sid;
			savedPath = join(store.dir, "saved.json");
			saved = { squeeze: 0, compaction: 0 };
			try {
				if (existsSync(savedPath)) saved = { ...saved, ...JSON.parse(readFileSync(savedPath, "utf8")) };
			} catch {
				/* start from zero */
			}
		}
		return store;
	}

	function persistSaved() {
		if (!store || !savedPath) return;
		try {
			mkdirSync(store.dir, { recursive: true });
			writeFileSync(savedPath, JSON.stringify(saved), "utf8");
		} catch {
			/* stats are best-effort */
		}
	}

	function updateStatus(ctx: ExtensionContext) {
		if (!ctx.hasUI) return;
		if (!config.enabled) return ctx.ui.setStatus("pi-squeeze", undefined);
		const label = config.compressorModel ? `squeeze:${config.compressorModel.split("/").pop()}` : "squeeze:no-model";
		const total = saved.squeeze + saved.compaction;
		const savedText = total > 0 ? ` ▼${formatTokens(total)} tok saved` : "";
		const ctxText = lastPromptTokens !== null && ctx.model?.contextWindow
			? ` ${Math.round((lastPromptTokens / ctx.model.contextWindow) * 100)}%`
			: "";
		ctx.ui.setStatus("pi-squeeze", label + savedText + ctxText);
	}

	function compressorUsable(ctx: ExtensionContext) {
		if (!config.enabled || !config.compactWithCompressor || !config.compressorModel) return undefined;
		const m = resolveCompressor(ctx);
		if (!m || modelKey(m)?.toLowerCase() === modelKey(ctx.model)?.toLowerCase()) return undefined;
		return m;
	}

	function triggerCompact(ctx: ExtensionContext, why: string, customInstructions?: string) {
		if (compacting) return;
		compacting = true;
		if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: compacting (${why})`, "info");
		ctx.compact({
			customInstructions,
			onComplete: () => {
				compacting = false;
				updateStatus(ctx);
			},
			onError: (err) => {
				compacting = false;
				if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: compaction failed: ${err.message}`, "error");
			},
		});
	}

	function resolveCompressor(ctx: ExtensionContext) {
		const [provider, ...rest] = config.compressorModel.split("/");
		return ctx.modelRegistry.find(provider, rest.join("/"));
	}

	pi.on("session_start", async (_e, ctx) => {
		config = loadConfig(configPath());
		lastPromptTokens = null;
		getStore(ctx);
		updateStatus(ctx);
	});

	// Run pi's compaction (same prompts, split-turn and file-list handling) but on the compressor model.
	pi.on("session_before_compact", async (event, ctx) => {
		const model = compressorUsable(ctx);
		if (!model) return;
		const { preparation, customInstructions, signal } = event;
		try {
			const auth = await ctx.modelRegistry.getApiKeyAndHeaders(model);
			if (!auth.ok) throw new Error(auth.error);
			const result = await piCompact(
				preparation,
				model,
				auth.apiKey,
				auth.headers
					? Object.fromEntries(
							Object.entries(auth.headers).filter((e): e is [string, string] => typeof e[1] === "string"),
						)
					: undefined,
				customInstructions,
				signal,
				undefined,
				// Route through the registry so request-time auth (OAuth, custom providers) applies.
				(m, context, options) => ctx.modelRegistry.streamSimple(m, context, options),
			);
			if (!result.summary.trim()) throw new Error("empty summary");
			// The chat model would have read the whole history to write this summary.
			getStore(ctx);
			saved.compaction += preparation.tokensBefore;
			persistSaved();
			if (ctx.hasUI) {
				ctx.ui.notify(
					`pi-squeeze: compacted ${formatTokens(preparation.tokensBefore)} tokens with ${modelKey(model)} (${event.reason})`,
					"info",
				);
			}
			return { compaction: result };
		} catch (error) {
			if (signal.aborted) return;
			const msg = error instanceof Error ? error.message : String(error);
			// Fall back to pi's default compaction with the chat model.
			if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: compressor compaction failed (${msg}); using default`, "warning");
			return;
		}
	});

	pi.on("session_compact", async (_e, ctx) => {
		lastPromptTokens = null; // re-measure on the next call
		updateStatus(ctx);
	});

	// Threshold check at idle: ctx.compact() aborts a running turn, so never fire mid-run.
	// (Mid-run, pi's own threshold/overflow compaction still runs, on the compressor via the hook above.)
	pi.on("agent_settled", async (_e, ctx) => {
		if (!config.enabled || config.compactAtPercent <= 0 || lastPromptTokens === null) return;
		const window = ctx.model?.contextWindow;
		if (!window) return;
		const pct = (lastPromptTokens / window) * 100;
		if (pct < config.compactAtPercent) return;
		lastPromptTokens = null; // loop guard: needs a fresh measurement before triggering again
		triggerCompact(ctx, `prompt ~${Math.round(pct)}% of ${formatTokens(window)} >= ${config.compactAtPercent}%`);
	});

	pi.on("context", async (event, ctx) => {
		if (!shouldSqueezeFor(config, modelKey(ctx.model))) return;
		const compressor = resolveCompressor(ctx);
		if (!compressor) {
			lastReport = `compressor model "${config.compressorModel}" not found`;
			return;
		}

		const systemTokens = Math.ceil(ctx.getSystemPrompt().length / 4);
		const result = await squeezeMessages(
			event.messages as unknown as MsgLike[],
			config,
			getStore(ctx),
			async (req) => {
				const res = await ctx.modelRegistry.complete(
					compressor,
					{
						// Shared per pass (whole-session transcript): cache it so parallel/following calls are cheap.
						systemPrompt: req.system,
						messages: [
							{
								role: "user",
								content: [{ type: "text", text: req.prompt }],
								timestamp: Date.now(),
							},
						],
					},
					{ maxTokens: config.maxSummaryTokens, signal: ctx.signal, cacheRetention: "short" },
				);
				if (res.stopReason === "error" || res.stopReason === "aborted") {
					throw new Error(res.errorMessage || res.stopReason);
				}
				return res.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");
			},
			ctx.signal,
			systemTokens,
		);

		totals.calls++;
		totals.charsSaved = result.charsSaved; // savings on the latest call
		saved.squeeze += Math.round(result.charsSaved / 4);
		if (result.charsSaved > 0) persistSaved();
		lastPromptTokens = result.tokensAfter + systemTokens;
		totals.summarized += result.newlySummarized;
		totals.errors += result.errors.length;
		const target = config.contextTarget > 0 ? ` (target ${formatTokens(config.contextTarget)})` : "";
		lastReport = `last call: ${result.squeezed} outputs squeezed, ${result.blocks} blocks collapsed, ~${formatTokens(lastPromptTokens)} tokens sent${target}, ~${Math.round(result.charsSaved / 4)} saved, ${result.newlySummarized} new summaries`;
		if (result.errors.length) {
			lastReport += `, ${result.errors.length} errors (${result.errors[0]})`;
			if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: ${result.errors.length} summary call(s) failed: ${result.errors[0]}`, "warning");
		}
		updateStatus(ctx);
		if (result.squeezed === 0 && result.blocks === 0) return;
		return { messages: result.messages as unknown as typeof event.messages };
	});

	async function pickModel(ctx: ExtensionContext) {
		const models = ctx.modelRegistry.getAvailable().map((m) => `${m.provider}/${m.id}`);
		if (models.length === 0) {
			ctx.ui.notify("No models with configured auth found", "error");
			return;
		}
		const choice = await ctx.ui.select("pi-squeeze: compressor (cheap) model", models);
		if (!choice) return;
		config.compressorModel = choice;
		persist();
		ctx.ui.notify(`pi-squeeze compressor model: ${choice}`, "info");
	}

	async function pickStyle(ctx: ExtensionContext) {
		const labels: Record<PromptStyle, string> = {
			squeeze: "squeeze - terse summary of each tool output (default)",
			pi: "pi - pi's built-in compaction prompt (structured checkpoint)",
		};
		const choice = await ctx.ui.select(
			`pi-squeeze: compressor prompt style (current: ${config.promptStyle})`,
			PROMPT_STYLES.map((s) => labels[s]),
		);
		if (!choice) return;
		config.promptStyle = choice.split(" ")[0] as PromptStyle;
		persist();
		ctx.ui.notify(`pi-squeeze prompt style: ${config.promptStyle}`, "info");
	}

	async function pickTargets(ctx: ExtensionContext) {
		const current = config.targetModels.join(", ");
		const value = await ctx.ui.input(
			"pi-squeeze: only squeeze for these active models (comma-separated provider/id, * wildcards; empty = all)",
			current || (modelKey(ctx.model) ?? ""),
		);
		if (value === undefined) return;
		config.targetModels = value
			.split(",")
			.map((s) => s.trim())
			.filter(Boolean);
		persist();
		ctx.ui.notify(`pi-squeeze targets: ${config.targetModels.join(", ") || "(all models)"}`, "info");
	}

	function statusText(ctx: ExtensionContext): string {
		const s = store?.stats();
		return [
			`pi-squeeze ${config.enabled ? "ON" : "OFF"}`,
			`compressor: ${config.compressorModel || "(not set - run /squeeze model)"} (style: ${config.promptStyle})`,
			`targets: ${config.targetModels.join(", ") || "(all models)"}`,
			`active model: ${modelKey(ctx.model) ?? "?"} -> ${shouldSqueezeFor(config, modelKey(ctx.model)) ? "squeezing" : "not squeezing"}`,
			`keepFirst=${config.keepFirstToolTurns} keepRecent=${config.keepRecentToolTurns} minChars=${config.minChars} summaryWords=${config.summaryWords}`,
			`contextTarget=${config.contextTarget > 0 ? formatTokens(config.contextTarget) : "0 (compress all eligible)"} low=${config.targetLowRatio} maxBlockSteps=${config.maxBlockSteps} blockSummaryWords=${config.blockSummaryWords} compressorContextChars=${config.compressorContextChars}`,
			lastReport || "no calls yet",
			`compaction: ${config.compactWithCompressor ? "via compressor" : "via chat model"}, auto at ${config.compactAtPercent > 0 ? `${config.compactAtPercent}%` : "off"}${lastPromptTokens !== null && ctx.model?.contextWindow ? ` (now ~${formatTokens(lastPromptTokens)} / ${formatTokens(ctx.model.contextWindow)})` : ""}`,
			`saved this session: ~${formatTokens(saved.squeeze + saved.compaction)} tokens (squeeze ${formatTokens(saved.squeeze)}, compaction ${formatTokens(saved.compaction)})`,
			s ? `session cache: ${s.count} summaries, ${s.originalChars} -> ${s.summaryChars} chars, dir ${store?.dir}` : "",
			`config: ${configPath()}`,
		]
			.filter(Boolean)
			.join("\n");
	}

	pi.registerCommand("squeeze-compact", {
		description: "Compact the context now, summarized by the pi-squeeze compressor model: /squeeze-compact [instructions]",
		handler: async (args, ctx) => {
			if (!compressorUsable(ctx)) {
				ctx.ui.notify("pi-squeeze: compressor unavailable (off, not set, or same as chat model); using default compaction", "warning");
			}
			triggerCompact(ctx, "manual", (args ?? "").trim() || undefined);
		},
	});

	pi.registerCommand("squeeze", {
		description: "Summarize old tool outputs with a cheap model: /squeeze [status|on|off|model|targets|style|set <key> <value>]",
		handler: async (args, ctx) => {
			const [sub = "", key, ...rest] = (args ?? "").trim().split(/\s+/);
			switch (sub) {
				case "on":
				case "off":
					config.enabled = sub === "on";
					persist();
					if (config.enabled && !config.compressorModel) await pickModel(ctx);
					break;
				case "toggle":
					config.enabled = !config.enabled;
					persist();
					break;
				case "model":
					if (key) {
						config.compressorModel = key;
						persist();
					} else await pickModel(ctx);
					break;
				case "targets":
					await pickTargets(ctx);
					break;
				case "style":
					if (key && (PROMPT_STYLES as string[]).includes(key)) {
						config.promptStyle = key as PromptStyle;
						persist();
					} else if (key) {
						ctx.ui.notify(`Unknown style. Styles: ${PROMPT_STYLES.join(", ")}`, "error");
						return;
					} else await pickStyle(ctx);
					break;
				case "set": {
					if (!key || !(key in DEFAULT_CONFIG)) {
						ctx.ui.notify(`Unknown key. Keys: ${Object.keys(DEFAULT_CONFIG).join(", ")}`, "error");
						return;
					}
					const raw = rest.join(" ");
					const def = (DEFAULT_CONFIG as unknown as Record<string, unknown>)[key];
					let value: unknown = raw;
					if (typeof def === "number") {
						// Allow 40k / 1.5M shorthands.
						const mm = /^([\d.]+)\s*([kKmM])?$/.exec(raw.replace(/_/g, ""));
						value = mm ? Number(mm[1]) * (mm[2] ? (mm[2].toLowerCase() === "k" ? 1e3 : 1e6) : 1) : Number.NaN;
					} else if (typeof def === "boolean") value = raw === "true" || raw === "on";
					else if (Array.isArray(def)) value = raw ? raw.split(",").map((s) => s.trim()) : [];
					if (typeof value === "number" && Number.isNaN(value)) {
						ctx.ui.notify(`${key} needs a number`, "error");
						return;
					}
					(config as unknown as Record<string, unknown>)[key] = value;
					persist();
					break;
				}
				case "":
				case "status":
					break;
				default: {
					// No/unknown arg: interactive menu (works in TUI and pi-web).
					break;
				}
			}
			if (sub === "" && ctx.hasUI) {
				const action = await ctx.ui.select(statusText(ctx), [
					config.enabled ? "Turn off" : "Turn on",
					"Choose compressor model",
					"Set target models",
					"Choose prompt style",
					"Close",
				]);
				if (action === "Turn off" || action === "Turn on") {
					config.enabled = action === "Turn on";
					persist();
					if (config.enabled && !config.compressorModel) await pickModel(ctx);
				} else if (action === "Choose compressor model") await pickModel(ctx);
				else if (action === "Set target models") await pickTargets(ctx);
				else if (action === "Choose prompt style") await pickStyle(ctx);
			} else {
				ctx.ui.notify(statusText(ctx), "info");
			}
			updateStatus(ctx);
		},
	});
}
