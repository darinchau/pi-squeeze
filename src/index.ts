/**
 * pi-squeeze: before each LLM call, replace older large tool outputs with summaries written by a
 * cheap "compressor" model. Originals are kept in tmp files so the agent can rg/read them.
 *
 * Also: pi compactions (auto, /compact, /squeeze-compact) are summarized by the compressor model,
 * and once the prompt reaches compactAtPercent of the context window a compaction is
 * deferred to the next prompt (never fired at idle, so nothing gets interrupted).
 * Compaction is global: it runs even when squeezing is off (enabled=false or the model
 * isn't a target) — only compactAtPercent and compactWithCompressor gate it.
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
	compactThresholdCrossed,
	estimateTokens,
	formatTokens,
	compressorPrompt,
	compressorSystem,
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
	statusBarText,
	withRetry,
	CHEAP_MODEL_RETRIES,
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
	// Threshold crossed at idle: compact just before the next prompt is processed, not now.
	let compactPending = false;

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
		const text = statusBarText(config, modelKey(ctx.model), saved.squeeze + saved.compaction, ctx.model?.contextWindow, compactPending);
		// Empty text = squeezing inactive for the active model: clear the footer entry entirely.
		ctx.ui.setStatus("pi-squeeze", text || undefined);
	}

	// Compaction on the compressor is global: gated only by compactWithCompressor and a
	// configured compressor, never by the squeezing master switch or targetModels.
	function compressorUsable(ctx: ExtensionContext) {
		if (!config.compactWithCompressor || !config.compressorModel) return undefined;
		const m = resolveCompressor(ctx);
		if (!m || modelKey(m)?.toLowerCase() === modelKey(ctx.model)?.toLowerCase()) return undefined;
		return m;
	}

	/**
	 * Run a compaction via pi (summarized by the compressor when possible; see the
	 * session_before_compact hook). Always resolves, never rejects, so callers can
	 * await it without hanging the surrounding pipeline.
	 */
	function runCompact(ctx: ExtensionContext, why: string, customInstructions?: string): Promise<void> {
		return new Promise<void>((resolve) => {
			if (compacting) return resolve();
			compacting = true;
			if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: compacting (${why})`, "info");
			let settled = false;
			const done = () => {
				if (settled) return;
				settled = true;
				compacting = false;
				resolve();
			};
			try {
				ctx.compact({
					customInstructions,
					// The session may have been reloaded mid-compaction; never leave callers hanging.
					onComplete: () => {
						try {
							updateStatus(ctx);
						} catch {
							/* runtime gone */
						}
						done();
					},
					onError: (err) => {
						try {
							if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: compaction failed: ${err.message}`, "error");
						} catch {
							/* runtime gone */
						}
						done();
					},
				});
			} catch {
				done();
			}
		});
	}

	function resolveCompressor(ctx: ExtensionContext) {
		const [provider, ...rest] = config.compressorModel.split("/");
		return ctx.modelRegistry.find(provider, rest.join("/"));
	}

	pi.on("session_start", async (_e, ctx) => {
		config = loadConfig(configPath());
		lastPromptTokens = null;
		compactPending = false;
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
			// Retry failed compressor calls (including empty summaries) before falling back.
			const result = await withRetry(async () => {
				const r = await piCompact(
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
				if (!r.summary.trim()) throw new Error("empty summary");
				return r;
			}, CHEAP_MODEL_RETRIES, signal);
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
		compactPending = false; // the context is fresh now
		updateStatus(ctx);
	});

	// Threshold crossed at idle: don't compact now. ctx.compact() would abort whatever
	// runs next and pi rejects prompt submission while a compaction is in flight, which is
	// what killed the session before. Instead arm a deferred compaction and run it from the
	// input handler below, just before the next prompt is processed.
	// Not gated on config.enabled: threshold compaction is global (squeeze off still compacts,
	// via the chat model when no compressor is usable).
	pi.on("agent_settled", async (_e, ctx) => {
		if (config.compactAtPercent <= 0 || lastPromptTokens === null || compactPending) return;
		const window = ctx.model?.contextWindow;
		if (!compactThresholdCrossed(config, lastPromptTokens, window)) return;
		compactPending = true;
		const pct = (lastPromptTokens / (window as number)) * 100;
		if (ctx.hasUI) {
			ctx.ui.notify(
				`pi-squeeze: prompt ~${Math.round(pct)}% of ${formatTokens(window as number)} — will compact on your next prompt`,
				"info",
			);
		}
		updateStatus(ctx);
	});

	// "Compact at next prompt": the input event fires before pi queues the message, so awaiting
	// the compaction here means the next turn starts on the compacted context. Only fresh,
	// idle prompts qualify (streamingBehavior is set while steering a live run, and compacting
	// would abort it). Steering inputs pass through; the pending compaction stays armed.
	// Not gated on config.enabled: an armed compaction runs even if squeezing was turned
	// off after arming (global compaction).
	pi.on("input", async (event, ctx) => {
		if (!compactPending || event.streamingBehavior) return;
		compactPending = false;
		const window = ctx.model?.contextWindow;
		if (config.compactAtPercent <= 0 || lastPromptTokens === null || !window) {
			updateStatus(ctx);
			return;
		}
		const pct = (lastPromptTokens / window) * 100;
		updateStatus(ctx);
		if (!compactThresholdCrossed(config, lastPromptTokens, window)) return; // e.g. switched to a model with a bigger window
		await runCompact(ctx, `prompt ~${Math.round(pct)}% of ${formatTokens(window)} >= ${config.compactAtPercent}%`);
		updateStatus(ctx);
		// Fall through: the submitted prompt is processed on the compacted context.
	});

	// Squeezing activity depends on the active model; refresh the footer when it changes.
	pi.on("model_select", async (_e, ctx) => {
		updateStatus(ctx);
	});

	pi.on("context", async (event, ctx) => {
		// Always measure the outgoing prompt so the global threshold compaction stays armed
		// even when squeezing is inactive (off, not a target, or no compressor): without a
		// fresh measurement nothing would ever cross compactAtPercent.
		const measure = (msgs: MsgLike[]) => {
			lastPromptTokens = estimateTokens(msgs) + Math.ceil(ctx.getSystemPrompt().length / 4);
		};
		if (!shouldSqueezeFor(config, modelKey(ctx.model))) {
			measure(event.messages as unknown as MsgLike[]);
			return;
		}
		const compressor = resolveCompressor(ctx);
		if (!compressor) {
			lastReport = `compressor model "${config.compressorModel}" not found`;
			measure(event.messages as unknown as MsgLike[]);
			return;
		}

		const result = await squeezeMessages(
			event.messages as unknown as MsgLike[],
			config,
			getStore(ctx),
			async (req) => {
				const res = await ctx.modelRegistry.complete(
					compressor,
					{
						systemPrompt: compressorSystem(config.promptStyle),
						messages: [
							{
								role: "user",
								content: [{ type: "text", text: compressorPrompt(req, config.summaryWords, config.promptStyle) }],
								timestamp: Date.now(),
							},
						],
					},
					{ maxTokens: config.maxSummaryTokens, signal: ctx.signal, cacheRetention: "none" },
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
		);

		totals.calls++;
		totals.charsSaved = result.charsSaved; // savings on the latest call
		saved.squeeze += Math.round(result.charsSaved / 4);
		if (result.charsSaved > 0) persistSaved();
		measure(result.messages);
		totals.summarized += result.newlySummarized;
		totals.errors += result.errors.length;
		lastReport = `last call: ${result.squeezed} outputs squeezed, ~${Math.round(result.charsSaved / 4)} tokens saved, ${result.newlySummarized} new summaries`;
		if (result.errors.length) {
			lastReport += `, ${result.errors.length} errors (${result.errors[0]})`;
			if (ctx.hasUI) ctx.ui.notify(`pi-squeeze: ${result.errors.length} summary call(s) failed: ${result.errors[0]}`, "warning");
		}
		updateStatus(ctx);
		if (result.squeezed === 0) return;
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
			lastReport || "no calls yet",
			`compaction: ${config.compactWithCompressor ? "via compressor" : "via chat model"}, auto at ${config.compactAtPercent > 0 ? `${config.compactAtPercent}% on next prompt${compactPending ? " (pending)" : ""}` : "off"} — runs regardless of squeeze on/off${lastPromptTokens !== null && ctx.model?.contextWindow ? ` (now ~${formatTokens(lastPromptTokens)} / ${formatTokens(ctx.model.contextWindow)})` : ""}`,
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
				ctx.ui.notify("pi-squeeze: compressor unavailable (not set, disabled via compactWithCompressor=false, or same as chat model); using default compaction", "warning");
			}
			await runCompact(ctx, "manual", (args ?? "").trim() || undefined);
			updateStatus(ctx);
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
					if (typeof def === "number") value = Number(raw);
					else if (typeof def === "boolean") value = raw === "true" || raw === "on";
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
