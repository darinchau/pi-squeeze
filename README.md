# pi-squeeze

A pi extension that cuts the input tokens sent to an expensive model. Before every LLM call it replaces older, large tool outputs with summaries written by a cheap "compressor" model.

- The originals are saved to `<tmp>/pi-squeeze/<session-id>/<tool>-<hash>.txt`. Each summary includes that file's path, so the agent can `rg` or `read` it for exact details.
- Summaries are cached per session in `summaries.json`, so each tool output is summarized once.
- The session file is never modified. Only the context sent to the model changes, so turning the extension off restores the full outputs.
- Only tool results are rewritten. The system prompt, every user message, and every assistant message (text, thinking, tool calls) are always sent unchanged.
- Tool results from the first `keepFirstToolTurns` and last `keepRecentToolTurns` tool-calling turns stay verbatim.
- There are two compressor prompt styles, switched with `/squeeze style`. `squeeze` (default) writes a terse summary of each output. `pi` reuses pi's built-in compaction prompt verbatim (a structured Goal/Progress/Next Steps checkpoint).
- If a summary isn't meaningfully shorter than the original (`maxRatio`), the original is kept.
- Failed compressor calls (summaries and compaction) are retried up to 3 times with backoff. If all attempts fail, the turn isn't blocked; the affected output is just sent unsqueezed.
- Compaction runs on the compressor too. Every pi compaction (auto, `/compact`, `/squeeze-compact`) uses pi's own `compact()` routine with the compressor model instead of the chat model. That covers the same prompts, split-turn and file-list handling. If the compressor fails, pi's default compaction runs.
- Once the squeezed prompt reaches `compactAtPercent` of the chat model's context window, a compaction is triggered when the agent goes idle. It never fires mid-run, because `ctx.compact()` aborts the turn.

## Install

You need pi with `@earendil-works/pi-coding-agent` 1.x and node 22.19 or newer.

```bash
# from a git remote (after pushing this repo somewhere)
pi install git:github.com/<you>/pi-squeeze

# or from a local clone
git clone <repo-url> pi-squeeze
pi install /absolute/path/to/pi-squeeze
```

Restart pi (or pi web), then pick the compressor model:

```
/squeeze model
```

Test without installing: `pi -e /path/to/pi-squeeze/src/index.ts`. To uninstall: `pi remove <same source>`.

## Usage

The commands work in the TUI and in pi web, which forwards the select/input dialogs.

| Command | Effect |
| --- | --- |
| `/squeeze` | Status plus an interactive menu (on/off, model, targets, style) |
| `/squeeze on` / `off` / `toggle` | Enable or disable |
| `/squeeze model [provider/id]` | Pick the compressor model (no arg opens a list) |
| `/squeeze style [squeeze/pi]` | Switch the compressor prompt style |
| `/squeeze targets` | Only squeeze when the active model matches these patterns, e.g. `Yunqiao/*` |
| `/squeeze set <key> <value>` | Change any config key |
| `/squeeze status` | Print status |
| `/squeeze-compact [instructions]` | Compact now with the compressor model |

The footer shows `squeeze:<model> ▼<N> tok saved <P>%`. N is the main-model input tokens avoided this session, counting squeezing on every call plus the history each compaction would have fed the chat model. P is the squeezed prompt size as a percentage of the context window.

## Config

Config lives in `~/.pi/agent/pi-squeeze.json`. You can override the path with `PI_SQUEEZE_CONFIG`.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch |
| `compressorModel` | `""` | `provider/modelId`. Nothing happens while this is empty. |
| `targetModels` | `[]` | Active-model patterns to squeeze for. Empty means all models except the compressor. |
| `keepRecentToolTurns` | `2` | Recent tool-calling turns left verbatim |
| `keepFirstToolTurns` | `1` | First tool-calling turns left verbatim (initial exploration) |
| `minChars` | `1500` | Smaller outputs are left alone |
| `summaryWords` | `200` | Soft summary length |
| `maxRatio` | `0.6` | Discard a summary longer than this fraction of the original |
| `maxCompressorInputChars` | `400000` | Head and tail clip for huge outputs sent to the compressor |
| `concurrency` | `4` | Parallel compressor calls |
| `maxSummaryTokens` | `2048` | Output cap per summary |
| `promptStyle` | `squeeze` | `squeeze` or `pi` |
| `compactAtPercent` | `70` | Auto-compact when the squeezed prompt reaches this % of the context window. `0` turns it off. |
| `compactWithCompressor` | `true` | Run compactions on the compressor model |

## Trade-offs

- Prompt style comparison (main `Poe/qwen3.8-27b-el`, compressor `deepseek/deepseek-flash`, on this repo's 14K- and 9.5K-char source files): both styles let the main model answer 5 detail questions correctly from the summaries alone. `squeeze` summaries were ~1.1K chars (8-12x smaller). `pi` summaries were ~3.4K chars (3-4x) and kept slightly more exact code lines. `squeeze` is the default because it saves about 3x more. Use `pi` if the agent keeps reopening raw files.

- Summaries can drop details. The agent is told where the raw file is, but it has to decide to look.
- It can hurt models that do have prompt caching, because rewriting old messages changes the prefix. Use `targetModels` to restrict it to the uncached model.
- Raw files live in the OS temp dir. If they are deleted, the affected outputs are re-summarized on the next call.

## Development

```bash
npm install
npm test          # node --test (pure logic in src/core.ts)
npx tsc -p .      # type-check against pi types
```
