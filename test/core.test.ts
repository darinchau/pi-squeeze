import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { compactThresholdCrossed, compressorPrompt, compressorSystem, DEFAULT_CONFIG, estimateTokens, formatTokens, type MsgLike, SqueezeStore, shouldSqueezeFor, squeezeMessages, statusBarText, withRetry } from "../src/core.ts";

const big = (tag: string) => `${tag} `.repeat(1000); // ~4000+ chars

function convo(nTurns: number): MsgLike[] {
	const msgs: MsgLike[] = [{ role: "user", content: "find the bug", timestamp: 0 }];
	for (let i = 0; i < nTurns; i++) {
		msgs.push({ role: "assistant", content: [{ type: "toolCall", id: `c${i}`, name: "bash", arguments: { cmd: `ls ${i}` } }] });
		msgs.push({ role: "toolResult", toolCallId: `c${i}`, toolName: "bash", content: [{ type: "text", text: big(`out${i}`) }] });
	}
	return msgs;
}

const cfg = { ...DEFAULT_CONFIG, compressorModel: "cheap/m", keepRecentToolTurns: 2, keepFirstToolTurns: 0 };

test("squeezes old outputs, keeps recent ones, writes raw files, caches", async () => {
	const store = new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-")));
	let calls = 0;
	const summarize = async () => {
		calls++;
		return "short summary";
	};
	const msgs = convo(4);
	const r = await squeezeMessages(msgs, cfg, store, summarize);
	assert.equal(calls, 2);
	assert.equal(r.squeezed, 2);
	const t0 = (r.messages[2].content as { text: string }[])[0].text;
	assert.match(t0, /pi-squeeze/);
	assert.match(t0, /short summary/);
	const file = t0.match(/Full original output: (.+)/)![1];
	assert.ok(existsSync(file));
	assert.match(readFileSync(file, "utf8"), /out0 out0/);
	// recent kept verbatim
	assert.equal(r.messages[8], msgs[8]);
	assert.equal(r.messages[6], msgs[6]);
	// input not mutated
	assert.match((msgs[2].content as { text: string }[])[0].text, /^out0/);
	// second call hits cache
	const r2 = await squeezeMessages(msgs, cfg, store, summarize);
	assert.equal(calls, 2);
	assert.equal(r2.squeezed, 2);
});

test("skips small outputs and useless summaries; records errors", async () => {
	const store = new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-")));
	const msgs = convo(3);
	(msgs[2].content as { text: string }[])[0].text = "tiny";
	const r = await squeezeMessages(msgs, cfg, store, async () => "x".repeat(10_000));
	assert.equal(r.squeezed, 0);
	let calls = 0;
	const r2 = await squeezeMessages(
		convo(3),
		cfg,
		new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-"))),
		async () => {
			calls++;
			throw new Error("boom");
		},
		undefined,
		3,
		0,
	);
	assert.equal(r2.squeezed, 0);
	assert.equal(r2.errors.length, 1);
	assert.equal(calls, 4); // initial attempt + 3 retries
});

test("retries failed summary calls and succeeds", async () => {
	let calls = 0;
	const r = await squeezeMessages(
		convo(3),
		cfg,
		new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-"))),
		async () => {
			if (++calls < 3) throw new Error("flaky");
			return "short summary";
		},
		undefined,
		3,
		0,
	);
	assert.equal(calls, 3);
	assert.equal(r.squeezed, 1);
	assert.equal(r.errors.length, 0);
});

test("withRetry stops after 3 retries and when aborted", async () => {
	let calls = 0;
	await assert.rejects(
		withRetry(async () => {
			calls++;
			throw new Error("nope");
		}, 3, undefined, 0),
		/nope/,
	);
	assert.equal(calls, 4);
	const ac = new AbortController();
	calls = 0;
	await assert.rejects(
		withRetry(async () => {
			calls++;
			ac.abort();
			throw new Error("aborted");
		}, 3, ac.signal, 0),
	);
	assert.equal(calls, 1);
});

test("shouldSqueezeFor", () => {
	assert.equal(shouldSqueezeFor(cfg, "big/opus"), true);
	assert.equal(shouldSqueezeFor(cfg, "cheap/m"), false);
	assert.equal(shouldSqueezeFor({ ...cfg, enabled: false }, "big/opus"), false);
	assert.equal(shouldSqueezeFor({ ...cfg, compressorModel: "" }, "big/opus"), false);
	assert.equal(shouldSqueezeFor({ ...cfg, targetModels: ["big/*"] }, "big/opus"), true);
	assert.equal(shouldSqueezeFor({ ...cfg, targetModels: ["big/*"] }, "other/x"), false);
});

test("statusBarText: empty when inactive, otherwise squeeze:<model> ▼<saved> (<current>/<window>)", () => {
	// Inactive (off, no compressor, active model is the compressor, or not a target) with
	// nothing armed: no text at all.
	assert.equal(statusBarText({ ...cfg, enabled: false }, "big/opus", 27_000, 200_000), "");
	assert.equal(statusBarText({ ...cfg, compressorModel: "" }, "big/opus", 27_000, 200_000), "");
	assert.equal(statusBarText(cfg, "cheap/m", 27_000, 200_000), "");
	assert.equal(statusBarText({ ...cfg, targetModels: ["big/*"] }, "other/x", 27_000, 200_000), "");
	assert.equal(statusBarText(cfg, undefined, 27_000, 200_000), "");
	// Compaction is global: even while squeezing is inactive, an armed deferred
	// compaction still shows the marker (no squeeze:<model> prefix — nothing is squeezed).
	assert.equal(statusBarText({ ...cfg, enabled: false }, "big/opus", 27_000, 200_000, true), "compact-next");
	assert.equal(statusBarText({ ...cfg, compressorModel: "" }, "big/opus", 27_000, 200_000, true), "compact-next");
	assert.equal(statusBarText({ ...cfg, targetModels: ["big/*"] }, "other/x", 0, undefined, true), "compact-next");
	// Active: model, cumulative tokens saved (absolute), current/window size.
	assert.equal(statusBarText(cfg, "big/opus", 13_000 + 14_000, 200_000), "squeeze:m ▼27k (200k)");
	// Multi-segment model ids: the label is the id's last segment.
	const deepCfg = { ...cfg, compressorModel: "openrouter/deep/deepseek-chat" };
	assert.equal(statusBarText(deepCfg, "big/opus", 1_250_000, 1_000_000), "squeeze:deepseek-chat ▼1.25M (1.00M)");
	// Nothing saved yet: window only.
	assert.equal(statusBarText(cfg, "big/opus", 0, 200_000), "squeeze:m (200k)");
	assert.equal(statusBarText(cfg, "big/opus", 0, undefined), "squeeze:m");
	// Current prompt measured: (<current>/<window>). 0 counts as measured, not unknown.
	assert.equal(statusBarText(cfg, "big/opus", 27_000, 200_000, false, 46_000), "squeeze:m ▼27k (46k/200k)");
	assert.equal(statusBarText(cfg, "big/opus", 0, 200_000, false, 0), "squeeze:m (0/200k)");
	// Window unknown: no parenthetical, even with a measurement (it would read as a bare window).
	assert.equal(statusBarText(cfg, "big/opus", 27_000, undefined, false, 46_000), "squeeze:m ▼27k");
	// Deferred compaction marker.
	assert.equal(statusBarText(cfg, "big/opus", 27_000, 200_000, true), "squeeze:m ▼27k (200k) compact-next");
	assert.equal(statusBarText(cfg, "big/opus", 27_000, 200_000, true, 46_000), "squeeze:m ▼27k (46k/200k) compact-next");
});

test("compactThresholdCrossed: global, only compactAtPercent gates it", () => {
	const window = 100_000;
	// The squeeze master switch never gates the threshold compaction.
	assert.equal(compactThresholdCrossed({ ...cfg, enabled: false, compactAtPercent: 70 }, 75_000, window), true);
	assert.equal(compactThresholdCrossed({ ...cfg, enabled: false, compactAtPercent: 70 }, 69_999, window), false);
	// compactAtPercent = 0 disables it.
	assert.equal(compactThresholdCrossed({ ...cfg, compactAtPercent: 0 }, 999_999, window), false);
	// No context window known: never arm.
	assert.equal(compactThresholdCrossed(cfg, 75_000, undefined), false);
	// Boundary: >= is crossed.
	assert.equal(compactThresholdCrossed(cfg, 70_000, window), true);
	assert.equal(compactThresholdCrossed(cfg, 69_999, window), false);
	// Squeezing-inactive model (not a target) still crosses — compaction is global.
	assert.equal(compactThresholdCrossed({ ...cfg, targetModels: ["big/*"], compactAtPercent: 70 }, 90_000, window), true);
});

test("never touches user/assistant/system messages; protects first tool turn", async () => {
	const store = new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-")));
	const msgs = convo(4);
	msgs.splice(1, 0, { role: "assistant", content: [{ type: "text", text: big("reply") }] });
	msgs[0] = { role: "user", content: big("question") };
	const r = await squeezeMessages(msgs, { ...cfg, keepFirstToolTurns: 1 }, store, async () => "s");
	assert.equal(r.squeezed, 1); // only c1: c0 is first turn, c2/c3 recent
	for (const i of [0, 1, 2, 3]) assert.equal(r.messages[i], msgs[i]);
	assert.match((r.messages[5].content as { text: string }[])[0].text, /pi-squeeze/);
});

test("prompt styles", () => {
	const req = { toolName: "bash", toolArgs: JSON.stringify({ command: "ls" }), output: "OUT", userGoal: "GOAL" };
	const pi = compressorPrompt(req, 100, "pi");
	assert.match(pi, /<conversation>\n\[User\]: GOAL/);
	assert.match(pi, /\[Assistant tool calls\]: bash\(command="ls"\)/);
	assert.match(pi, /## Goal/);
	assert.match(compressorSystem("pi"), /context summarization assistant/);
	assert.match(compressorPrompt(req, 100, "squeeze"), /<tool_output>\nOUT/);
	assert.match(compressorSystem("squeeze"), /compress tool outputs/);
});

test("store survives its tmp dir being deleted mid-session", () => {
	const dir = join(mkdtempSync(join(tmpdir(), "sq-")), "session");
	const store = new SqueezeStore(dir);
	rmSync(dir, { recursive: true, force: true });
	const file = store.writeRaw("k1", "read", JSON.stringify({ path: "a" }), "hello");
	store.set("k1", { summary: "s", file, originalChars: 5 } as never);
	assert.ok(existsSync(file));
	assert.ok(existsSync(join(dir, "summaries.json")));
});

test("estimateTokens / formatTokens", () => {
	const msgs: MsgLike[] = [
		{ role: "user", content: "x".repeat(400) },
		{ role: "assistant", content: [{ type: "text", text: "y".repeat(400) }, { type: "toolCall", arguments: { a: 1 } }] },
	];
	assert.equal(estimateTokens(msgs), Math.ceil((800 + 7) / 4));
	assert.equal(formatTokens(950), "950");
	assert.equal(formatTokens(1234), "1.2k");
	assert.equal(formatTokens(123_456), "123k");
	assert.equal(formatTokens(2_500_000), "2.50M");
});
