import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { compressorPrompt, compressorSystem, DEFAULT_CONFIG, estimateTokens, formatTokens, type MsgLike, SqueezeStore, shouldSqueezeFor, squeezeMessages } from "../src/core.ts";

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
