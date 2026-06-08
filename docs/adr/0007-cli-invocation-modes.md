# ADR 0007: CLI invocation modes

## Status

**Accepted**

## Context

The tool started as a single batch pipeline: read URLs from a file, run every
step, write everything to disk. Two additional user situations turned up later:

1. **A human wants to look at one video quickly.** "Give me the notes for this
   URL." They don't want it added to the batch corpus or polluting the FAISS
   index.
2. **A coding agent wants to run its own downstream reasoning.** When Claude,
   Cursor, or another agent orchestrates the tool, it does not need the
   tool's LLM to summarise — the agent's own LLM handles that. What the
   orchestrating agent needs is a clean, machine-consumable transcript.

Adding separate CLI commands for each situation (e.g. `yt-ingest fetch-one`,
`yt-ingest transcript-text`) would bloat the command surface. Folding all
three faces into `fetch` and `run` via flags and argument shape keeps the CLI
surface small and keeps related behavior colocated.

## Decision

The CLI exposes **three invocation modes**, distinguished by the shape of the
arguments and a single flag. The same subcommands (`fetch`, `run`) participate
in each mode, with mode-specific behavior:

### (a) Batch mode — `--file <path>`

- Triggered by providing `--file urls.txt` rather than a positional URL.
- Runs the full pipeline: fetch → extract → index → synthesize (under `run`)
  or populates the transcript cache (under `fetch`).
- Always writes to disk: transcript caches to `./transcripts/`, notes to
  `./notes/`, FAISS index to `./.faiss_index/`.
- **Idempotent via file-existence plus input checksums**, not timestamps.
  Re-running skips work whose inputs haven't changed; a changed transcript
  cache re-triggers downstream extract/synthesize for that video.
- `--agent` is a no-op in batch mode (the batch pipeline is human-facing).

### (b) Single-URL mode — positional URL argument

- Triggered by providing a YouTube URL as a positional argument to `fetch` or
  `run`, without `--file`.
- Prints to **stdout** by default (formatted transcript text for `fetch`,
  rendered Markdown note for `run`).
- `--output <path>` redirects output to a file.
- **No disk cache side-effects.** Single-URL `fetch` does not write to
  `./transcripts/` unless `--output` is given, and `run` in single-URL mode
  bypasses the disk cache for the transcript step.
- Mutually exclusive with `--file`; the CLI errors if both are supplied.

### (c) Agent mode — `--agent` flag

- Triggered by `--agent` on `run` (and no-op on `fetch`, where the output is
  already raw transcript text).
- In single-URL mode: outputs the raw formatted transcript text only and
  returns. **No extract, no synthesize, no DeepSeek call.** The calling
  agent's own LLM is expected to do its own summarisation.
- In batch mode: explicitly ignored (batch is human-operated and expects the
  full pipeline to run).
- Output goes to stdout unless `--output` is given.

The combination rule: `fetch` and `run` each accept both a URL argument and
`--file`, but not both at once. `--agent` is meaningful only on `run`. The
CLI rejects ambiguous combinations with a clear error message rather than
making a silent choice.

## Consequences

- **One command, three faces.** A reader of `yt-ingest --help` sees `fetch`
  and `run`, not four parallel commands.
- **Agents can skip the tool's LLM entirely.** The `--agent` flag provides a
  stable machine-consumable path that does not drift with prompt changes.
- **No silent cache pollution.** Single-URL invocations do not surprise the
  user with files they did not ask for.
- **Idempotency in batch mode is load-bearing.** Re-runs after a network
  failure, or after editing `urls.txt`, skip already-done work cheaply. The
  checksum is computed on the model_dump of the transcript cache, so changing
  the fetched content re-triggers downstream steps.
- **The `--agent` / `--output` combination is supported.** An orchestrating
  agent can write the raw transcript to a chosen file via `run <url> --agent
  --output transcript.txt` without touching the local cache.
