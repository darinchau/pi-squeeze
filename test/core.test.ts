import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { outputPrompt, compressorSystem, DEFAULT_CONFIG, estimateTokens, formatTokens, type MsgLike, SqueezeStore, shouldSqueezeFor, squeezeMessages } from "../src/core.ts";

const big = (tag: string) => `${tag} `.repeat(1000); // ~4000+ chars

function convo(nTurns: number): MsgLike[] {
	const msgs: MsgLike[] = [{ role: "user", content: "find the bug", timestamp: 0 }];
	for (let i = 0; i < nTurns; i++) {
		msgs.push({ role: "assistant", content: [{ type: "toolCall", id: `c${i}`, name: "bash", arguments: { cmd: `ls ${i}` } }] });
		msgs.push({ role: "toolResult", toolCallId: `c${i}`, toolName: "bash", content: [{ type: "text", text: big(`out${i}`) }] });
	}
	return msgs;
}

// maxBlockSteps: 1 = per-output summaries only (legacy behaviour); block tests override it.
const cfg = { ...DEFAULT_CONFIG, compressorModel: "cheap/m", keepRecentToolTurns: 2, keepFirstToolTurns: 0, maxBlockSteps: 1 };

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
	const r2 = await squeezeMessages(convo(3), cfg, new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-"))), async () => {
		throw new Error("boom");
	});
	assert.equal(r2.squeezed, 0);
	assert.equal(r2.errors.length, 1);
});

test("shouldSqueezeFor", () => {
	assert.equal(shouldSqueezeFor(cfg, "big/opus"), true);
	assert.equal(shouldSqueezeFor(cfg, "cheap/m"), false);
	assert.equal(shouldSqueezeFor({ ...cfg, enabled: false }, "big/opus"), false);
	assert.equal(shouldSqueezeFor({ ...cfg, compressorModel: "" }, "big/opus"), false);
	assert.equal(shouldSqueezeFor({ ...cfg, targetModels: ["big/*"] }, "big/opus"), true);
	assert.equal(shouldSqueezeFor({ ...cfg, targetModels: ["big/*"] }, "other/x"), false);
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
	const job = { toolName: "bash", args: JSON.stringify({ command: "ls" }), text: "OUT" };
	const pi = outputPrompt(job, "GOAL", { ...cfg, promptStyle: "pi" });
	assert.match(pi, /<conversation>\n\[User\]: GOAL/);
	assert.match(pi, /\[Assistant tool calls\]: bash\(command="ls"\)/);
	assert.match(pi, /## Goal/);
	assert.match(compressorSystem("pi"), /context summarization assistant/);
	assert.match(outputPrompt(job, "GOAL", cfg), /<tool_output>\nOUT/);
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

// ---- block summarization / contextTarget ----

const bcfg = { ...DEFAULT_CONFIG, compressorModel: "cheap/m", keepRecentToolTurns: 2, keepFirstToolTurns: 0, maxBlockSteps: 10 };

function mkStore() {
	return new SqueezeStore(mkdtempSync(join(tmpdir(), "sq-blk-")));
}

test("collapses consecutive old steps into one block; keeps user + recent steps verbatim", async () => {
	const msgs = convo(6); // user, then 6 x (assistant call, result)
	const reqs: { kind: string; system: string; prompt: string }[] = [];
	const r = await squeezeMessages(msgs, bcfg, mkStore(), async (req) => {
		reqs.push(req);
		return req.kind === "block" ? "BLOCK SUMMARY" : "out summary";
	});
	assert.equal(r.blocks, 1);
	assert.equal(r.messagesRemoved, 7); // steps 1-4 = 8 messages -> 1
	assert.equal(r.messages.length, msgs.length - 7);
	assert.deepEqual(r.messages[0], msgs[0]); // user message verbatim
	const blk = r.messages[1];
	assert.equal(blk.role, "assistant");
	assert.equal(blk.stopReason, "stop");
	const text = (blk.content as { text: string }[])[0].text;
	assert.match(text, /steps S1-S4/);
	assert.match(text, /BLOCK SUMMARY/);
	const file = /Full original transcript: (.+)/.exec(text)![1];
	const raw = readFileSync(file, "utf8");
	assert.match(raw, /out0 out0/);
	assert.match(raw, /out3 out3/);
	// recent 2 steps untouched, tool calls intact
	assert.deepEqual(r.messages.slice(2), msgs.slice(9));
	// no orphaned tool results
	for (const m of r.messages) if (m.role === "toolResult") assert.ok(["c4", "c5"].includes(String(m.toolCallId)));
	// compressor sees whole uncompressed session, identical system prompt across calls (cacheable prefix)
	const blockReq = reqs.find((q) => q.kind === "block")!;
	assert.match(blockReq.system, /<session>/);
	assert.match(blockReq.system, /out5 out5/);
	assert.match(blockReq.system, /find the bug/);
	assert.equal(new Set(reqs.map((q) => q.system)).size, 1);
});

test("user messages are barriers: blocks never span them", async () => {
	const msgs = convo(6);
	msgs.splice(5, 0, { role: "user", content: [{ type: "text", text: "also check X" }] }); // between step 2 and 3
	const r = await squeezeMessages(msgs, bcfg, mkStore(), async (req) => (req.kind === "block" ? "BS" : "os"));
	assert.equal(r.blocks, 2);
	assert.ok(r.messages.some((m) => m.role === "user" && JSON.stringify(m.content).includes("also check X")));
	const labels = r.messages.map((m) => JSON.stringify(m.content).match(/steps S\d-S\d/)?.[0]).filter(Boolean);
	assert.deepEqual(labels, ["steps S1-S2", "steps S3-S4"]);
});

test("contextTarget: stops compressing once under target, oldest first", async () => {
	const msgs = convo(8);
	const full = estimateTokens(msgs);
	let calls = 0;
	const summarize = async () => {
		calls++;
		return "s";
	};
	// Target already met: nothing happens, no compressor calls.
	const none = await squeezeMessages(msgs, { ...bcfg, contextTarget: full + 1000 }, mkStore(), summarize);
	assert.equal(calls, 0);
	assert.equal(none.messages, msgs);
	// Target slightly below full: only the oldest output(s) get squeezed, no blocks needed.
	const r = await squeezeMessages(msgs, { ...bcfg, contextTarget: full - 500 }, mkStore(), summarize);
	assert.ok(r.squeezed >= 1 && r.squeezed < 6);
	assert.equal(r.blocks, 0);
	assert.ok(r.tokensAfter <= full - 500);
	// Hysteresis: once triggered it compresses down to targetLowRatio * target, not just under it.
	assert.ok(r.tokensAfter <= (full - 500) * DEFAULT_CONFIG.targetLowRatio);
	const tight = await squeezeMessages(msgs, { ...bcfg, contextTarget: full - 500, targetLowRatio: 1, concurrency: 1 }, mkStore(), summarize);
	assert.equal(tight.squeezed, 1);
	assert.match(JSON.stringify(r.messages[2].content), /pi-squeeze/); // oldest result first
	assert.doesNotMatch(JSON.stringify(r.messages[12].content), /pi-squeeze/);
});

test("block summaries are cached and reused; growth keeps old block stable", async () => {
	const store = mkStore();
	let blockCalls = 0;
	const summarize = async (req: { kind: string }) => {
		if (req.kind === "block") blockCalls++;
		return req.kind === "block" ? "BS" : "os";
	};
	const msgs = convo(6);
	const r1 = await squeezeMessages(msgs, bcfg, store, summarize);
	const r2 = await squeezeMessages(msgs, bcfg, store, summarize);
	assert.equal(blockCalls, 1);
	assert.deepEqual(r2.messages, r1.messages);
	// L1 summaries of outputs inside the block were made first; cached too.
	assert.equal(r2.newlySummarized, 0);
});

test("single-step runs only get per-output summaries; compressor errors don't break", async () => {
	const msgs = convo(3); // 1 eligible step
	const r = await squeezeMessages(msgs, bcfg, mkStore(), async () => {
		throw new Error("boom");
	});
	assert.equal(r.blocks, 0);
	assert.equal(r.messages, msgs);
	assert.match(r.errors[0], /boom/);
});
