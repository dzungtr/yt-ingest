# ADR 0001: Transcript fetching chain-of-responsibility

## Status

**Accepted**

## Context

YouTube exposes several paths to a transcript for a given video, and none of
them is durable. The public transcript endpoint used by `youtube-transcript-api`
returns structured, timestamped text when it works — but YouTube periodically
rate-limits IPs, disables transcripts on individual videos, and reshapes the
underlying endpoint without notice. Auto-generated subtitles (the VTT files
`yt-dlp` can grab) cover a wider set of videos but require parsing. For the
remaining tail — videos with no captions of any kind — a local Whisper model on
a workstation GPU can transcribe raw audio, but only if the user opts into the
slower path.

A single hard-coded fetcher would break the moment YouTube touches that endpoint
or the user's IP gets throttled. The tool needs graceful degradation that is
obvious in the log output, so a failed run shows the user which tier gave up and
why.

## Decision

Adopt a **chain-of-responsibility** pattern for transcript fetching, with three
tiers in a fixed order:

1. **`YouTubeAPIFetcher`** — `youtube-transcript-api`; returns timestamped dicts
   directly, no auth required.
2. **`YtDlpFetcher`** — `yt-dlp --write-auto-sub --sub-lang en --sub-format vtt`;
   parses the VTT with a local regex (not `webvtt-py`, which is listed as a
   dependency but the current implementation uses an inline parser).
3. **`WhisperFetcher`** — `yt-dlp --extract-audio` followed by `faster-whisper`;
   **gated by `--allow-whisper`** because it is slow and requires the extra
   package to be installed.

The tiers share a single contract — the `TranscriptFetcher` interface, defined
as a `typing.Protocol` with one method:

```python
class TranscriptFetcher(Protocol):
    def fetch(self, video_id: str, url: str) -> TranscriptCache: ...
```

A chain-runner (`run_fetcher_chain`) tries each fetcher in order, catches
`FetchError`, and raises a combined error that lists every attempt when all of
them fail. Adding a fourth tier (e.g. a third-party caption aggregator) is a
one-list-append change, with no control-flow churn.

**No YouTube Data API key is used.** The tool never authenticates with Google
and never uses authenticated endpoints. This is a deliberate hard constraint:
Google API keys require OAuth or service accounts, and quota exhaustion would
turn a "fetch 20 videos" workflow into a credential-management burden. The
fetcher chain assumes unauthenticated access everywhere.

For video metadata (title, channel, duration), `yt-dlp --dump-json` is called
independently of the transcript source, so the same metadata path is used
regardless of which tier won.

## Consequences

- **YouTube breaks one path, the next picks up.** The common failure mode is
  "IP blocked on `youtube-transcript-api` today"; the ytdlp tier often still
  works because the VTT endpoint is more permissive. The user sees a clear log
  line: `FAIL youtube-transcript-api: ...` followed by `OK via ytdlp_subs`.
- **Whisper is an opt-in, not a silent fallback.** A user running `--allow-whisper`
  has made a deliberate choice to pay the latency and install the extra
  dependency. Without the flag, the chain stops at tier 2.
- **Adding fetchers is cheap and contained.** A new tier slots into the chain
  without touching existing ones or the chain runner.
- **The `webvtt-py` package is declared as a dependency** (see `pyproject.toml`)
  but the current `YtDlpFetcher` parser is inline; this is a cleanup candidate,
  not a functional issue.
- **The chain runner is sequential, not parallel.** YouTube IP throttling is
  triggered by concurrency, so fetching one video at a time is load-bearing
  (see Operational notes in `docs/architecture.md` and `ADR-0004`).
