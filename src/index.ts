/**
 * pi-squeeze: before each LLM call, replace older large tool outputs with summaries written by a
 * cheap "compressor" model. Originals are kept in tmp files so the agent can rg/read them.
 *
 * Commands: /squeeze [status|on|off|model|targets|style|set <key> <value>]
 */
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import {
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
} from "./core.ts";

const agentDir = () => process.env.PI_CODING_AGENT_DIR || join(homedir(), ".pi", "agent");
const configPath = () => process.env.PI_SQUEEZE_CONFIG || join(agentDir(), "pi-squeeze.json");

export default function (pi: ExtensionAPI) {
	let config: SqueezeConfig = loadConfig(configPath());
	let store: SqueezeStore | undefined;
	let storeSession = "";
	let lastReport = "";
	const totals = { calls: 0, charsSaved: 0, summarized: 0, errors: 0 };

	const persist = () => saveConfig(configPath(), config);
	const modelKey = (m: { provider: string; id: string } | undefined) => (m ? `${m.provider}/${m.id}` : undefined);

	function getStore(ctx: ExtensionContext): SqueezeStore {
		const sid = ctx.sessionManager.getSessionId() || "ephemeral";
		if (!store || storeSession !== sid) {
			store = new SqueezeStore(join(tmpdir(), "pi-squeeze", sid));
			storeSession = sid;
		}
		return store;
	}

	function updateStatus(ctx: ExtensionContext) {
		if (!ctx.hasUI) return;
		if (!config.enabled) return ctx.ui.setStatus("pi-squeeze", undefined);
		const label = config.compressorModel ? `squeeze:${config.compressorModel.split("/").pop()}` : "squeeze:no-model";
		const saved = totals.charsSaved > 0 ? ` -${Math.round(totals.charsSaved / 4 / 1000)}k tok/call` : "";
		ctx.ui.setStatus("pi-squeeze", label + saved);
	}

	function resolveCompressor(ctx: ExtensionContext) {
		const [provider, ...rest] = config.compressorModel.split("/");
		return ctx.modelRegistry.find(provider, rest.join("/"));
	}

	pi.on("session_start", async (_e, ctx) => {
		config = loadConfig(configPath());
		updateStatus(ctx);
	});

	pi.on("context", async (event, ctx) => {
		if (!shouldSqueezeFor(config, modelKey(ctx.model))) return;
		const compressor = resolveCompressor(ctx);
		if (!compressor) {
			lastReport = `compressor model "${config.compressorModel}" not found`;
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
			s ? `session cache: ${s.count} summaries, ${s.originalChars} -> ${s.summaryChars} chars, dir ${store?.dir}` : "",
			`config: ${configPath()}`,
		]
			.filter(Boolean)
			.join("\n");
	}

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
