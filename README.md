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
- **Compaction is global**: it does not depend on the squeezing master switch. `/squeeze off` (or a model outside `targetModels`) only stops tool-output squeezing — threshold compaction still fires, and compactions still run on the compressor. Only `compactAtPercent` gates the automatic compaction and `compactWithCompressor` gates the compressor model.
- Once the prompt (squeezed or not) reaches `compactAtPercent` of the chat model's context window, the compaction is **deferred to the next prompt**: pi-squeeze arms it when the agent goes idle (footer shows `compact-next`) and runs it right after you submit your next message, before that message is passed to the agent. Nothing is interrupted, nothing is compacted while the context is still in use, and a session you walk away from never spends a compaction call. (Mid-run, pi's own threshold/overflow compaction still applies, on the compressor via the hook.)

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
| `/squeeze on` / `off` / `toggle` | Enable or disable **tool-output squeezing** (compaction stays active; see `compactAtPercent` / `compactWithCompressor`) |
| `/squeeze model [provider/id]` | Pick the compressor model (no arg opens a list) |
| `/squeeze style [squeeze/pi]` | Switch the compressor prompt style |
| `/squeeze targets` | Only squeeze when the active model matches these patterns, e.g. `Yunqiao/*` |
| `/squeeze set <key> <value>` | Change any config key |
| `/squeeze status` | Print status |
| `/squeeze-compact [instructions]` | Compact now with the compressor model |

The footer shows `squeeze:<model> ▼<N> (<context window>)` while squeezing is active for the current model, e.g. `squeeze:qwen3.8-27b ▼27k (1.00M)`. N is the main-model input tokens avoided this session, summed per call: if call 1 sends 13k fewer tokens and call 2 sends 14k fewer, N is 27k, even though most of call 2's savings come from the same old outputs. Each compressor compaction also adds the history the chat model would otherwise have summarized. N is hidden until something has been saved. When a compaction is armed for your next prompt, ` compact-next` is appended. When squeezing is inactive for the current model (extension off, no compressor set, active model is the compressor, or not matching `targetModels`) the footer shows nothing — unless a compaction is armed, in which case it shows `compact-next` (compaction is global, so the marker appears even when nothing is being squeezed).

## Config

Config lives in `~/.pi/agent/pi-squeeze.json`. You can override the path with `PI_SQUEEZE_CONFIG`.

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Master switch for **tool-output squeezing**. Compaction is not affected: threshold compaction (`compactAtPercent`) and compressor-driven compaction (`compactWithCompressor`) keep working when this is off. |
| `compressorModel` | `""` | `provider/modelId`. While empty, tool-output squeezing and compressor-driven compaction are idle, but threshold compaction (`compactAtPercent`) still runs — on the chat model via pi's default compaction. |
| `targetModels` | `[]` | Active-model patterns to squeeze for. Empty means all models except the compressor. This only gates tool-output squeezing; compactions still run on the compressor for every model (see `compactWithCompressor`). |
| `keepRecentToolTurns` | `2` | Recent tool-calling turns left verbatim |
| `keepFirstToolTurns` | `1` | First tool-calling turns left verbatim (initial exploration) |
| `minChars` | `1500` | Smaller outputs are left alone |
| `summaryWords` | `200` | Soft summary length |
| `maxRatio` | `0.6` | Discard a summary longer than this fraction of the original |
| `maxCompressorInputChars` | `400000` | Head and tail clip for huge outputs sent to the compressor |
| `concurrency` | `4` | Parallel compressor calls |
| `maxSummaryTokens` | `2048` | Output cap per summary |
| `promptStyle` | `squeeze` | `squeeze` or `pi` |
| `compactAtPercent` | `70` | Arm a compaction when the prompt (squeezed or not) reaches this % of the context window; it runs right before your next prompt is processed. Works whether squeezing is on or off. `0` turns it off. |
| `compactWithCompressor` | `true` | Run compactions on the compressor model |

## Trade-offs

- Prompt style comparison (main `Poe/qwen3.8-27b-el`, compressor `deepseek/deepseek-flash`, on this repo's 14K- and 9.5K-char source files): both styles let the main model answer 5 detail questions correctly from the summaries alone. `squeeze` summaries were ~1.1K chars (8-12x smaller). `pi` summaries were ~3.4K chars (3-4x) and kept slightly more exact code lines. `squeeze` is the default because it saves about 3x more. Use `pi` if the agent keeps reopening raw files.

- Summaries can drop details. The agent is told where the raw file is, but it has to decide to look.
- It can hurt models that do have prompt caching, because rewriting old messages changes the prefix. Use `targetModels` to restrict it to the uncached model.
- Raw files live in the OS temp dir. If they are deleted, the affected outputs are re-summarized on the next call.
- The deferred compaction needs a next prompt in the same process. In one-shot modes (`pi -p`, JSON) it may never run; pi's own threshold compaction still protects those sessions (on the compressor via the hook).

## Development

```bash
npm install
npm test          # node --test (pure logic in src/core.ts)
npx tsc -p .      # type-check against pi types
```
