# ADR 0002: DeepSeek as LLM provider

## Status

**Accepted**

## Context

The tool needs an LLM-backed extraction step (per-video structured notes) and a
cross-video synthesis step. Two things matter more than the headline "use an
LLM" choice: **which model plays which role**, and **how token costs are
accounted for** when the same system prompt is used for every video.

DeepSeek offers an OpenAI-compatible HTTP API. This means the `openai` Python
SDK can be pointed at DeepSeek by overriding `base_url` and supplying
`DEEPSEEK_API_KEY`, with no separate SDK, no separate auth flow, and no
provider-specific client code. It also means the same wrapper can be retargeted
at any other OpenAI-compatible provider (local Ollama, another hosted vendor)
by changing one constant.

Prompt caching on DeepSeek is automatic — repeated prefixes across requests
are cached server-side, and cache hit tokens are billed at roughly 10% of the
regular prompt rate. Because the extraction system prompt is identical across
every video in a batch, cache hits dominate the cost profile from the second
call onwards. A naive "input tokens × price" cost estimate overstimates by a
large margin; the wrapper has to surface the split.

## Decision

Use **DeepSeek as the LLM provider**, accessed through the `openai` SDK with
`base_url=https://api.deepseek.com`. Model selection is treated as a **policy**,
not a hard-pinned version string — the current choices are:

| Stage | Model | Why |
|-------|-------|-----|
| Per-chunk draft | `deepseek-v4-pro` | High fidelity, 128k context window, the chunk-level pass where hallucinations cost the most. Default for `chat_json`. |
| Cross-chunk merge | `deepseek-v4-pro` | Same model as the draft stage; merge is consolidation, not synthesis. |
| Cross-video synthesis | `deepseek-v4-flash` | Cheaper, fast, adequate for thematic extraction over concatenated note previews. Passed as an explicit `model=` override. |
| `ask` answer | `deepseek-v4-pro` (default) | Answer generation is a one-shot call; the default applies. |

All DeepSeek interaction is funnelled through a single wrapper —
`yt_ingest.llm.chat_json()` — that:

- Takes `system`, `user`, `model` (default `deepseek-v4-pro`), `temperature`.
- Uses `response_format={"type": "json_object"}` so the response is parseable
  JSON.
- Reads `usage.prompt_tokens_details.cached_tokens` (falling back to 0 when
  absent) and computes a clean split into `hit_tokens` and `miss_tokens`.
- Returns a `CacheStats` dataclass alongside the parsed JSON, so callers can
  log or aggregate token accounting across multiple calls.

The previous-spec plan named `deepseek-chat` (V3) for extraction and
`deepseek-reasoner` (R1) for synthesis. The current implementation uses the
`v4-pro` / `v4-flash` pair instead; the decision (two models, one expensive for
high-fidelity work, one cheap for aggregation) is the same — the concrete
model names track DeepSeek's catalog as it evolves.

## Consequences

- **No separate DeepSeek SDK** to maintain. The `openai` package is the only
  LLM dependency. Retargeting at another OpenAI-compatible provider is a
  one-line change.
- **Accurate cost accounting.** The wrapper's hit/miss split is logged on every
  extraction and synthesis step and is visible in the CLI success line.
  Estimating session cost using only `prompt_tokens` would be off by roughly
  an order of magnitude on cache-heavy runs.
- **Model selection is not pinned to a specific version forever.** The constants
  in `extract.py` and `synthesize.py` name the currently preferred models;
  when DeepSeek ships a successor, the policy changes, not the architecture.
- **JSON-only responses.** Every LLM call in the pipeline returns JSON. The
  `"blog_post"` key in the extraction contract is a naming artifact
  (ADR-0006), not a contract.
- **One environment variable, one credential.** `DEEPSEEK_API_KEY`, loaded via
  `python-dotenv`. No OAuth, no service accounts.
