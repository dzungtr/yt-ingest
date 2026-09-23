# ADR-0008: provider-abstract LLM layer (OpenAI-compatible vs Anthropic-compatible)

## Status

Accepted (supersedes the DeepSeek-only wiring of ADR-0002 for the transport
layer; DeepSeek remains the default OpenAI-compatible endpoint).

## Context

The LLM layer (`yt_ingest.llm`) was hard-wired to DeepSeek: an `openai.OpenAI`
client pinned to `https://api.deepseek.com` keyed on `DEEPSEEK_API_KEY`, with
DeepSeek-specific model names (`deepseek-v4-pro`, `deepseek-v4-flash`) baked
into call sites (`extract.py`, `synthesize.py`, `cli.ask`).

Requirement: support either an OpenAI-compatible endpoint (any base URL, API
key) or an Anthropic-compatible endpoint, selected purely by which
credentials are present in the environment, with OpenAI-compatible taking
precedence.

## Decision

1. **Provider protocol.** `llm.py` defines an `LLMProvider` protocol with a
   single method `chat_json(system, user, model_tier, temperature) ->
   (parsed_json, CacheStats)`. Two implementations exist:
   - `OpenAICompatProvider` — Chat Completions against any OpenAI-compatible
     base URL (`OPENAI_BASE_URL`, default `https://api.deepseek.com`).
     Uses `response_format={"type": "json_object"}` and reads cache-hit
     tokens from `usage.prompt_tokens_details.cached_tokens`.
   - `AnthropicProvider` — Messages API via the `anthropic` SDK (lazy import;
     it is an optional dependency `[project.optional-dependencies.anthropic]`).
     No structured-output flag exists, so JSON is extracted from the text
     response with code-fence stripping. Cache stats map from
     `usage.cache_read_input_tokens` / `usage.input_tokens`. Note:
     `anthropic>=1.8` removed the `temperature` parameter, so the Anthropic
     path ignores it.
2. **Selection by env-var presence, OpenAI precedence**
   (`llm._select_provider`):
   - `OPENAI_API_KEY` set → OpenAI-compatible.
   - else `ANTHROPIC_API_KEY` set → Anthropic.
   - else `DEEPSEEK_API_KEY` set → OpenAI-compatible with the DeepSeek base
     URL (legacy fallback, preserves existing `.env` files).
   - else raise `LLMConfigurationError` listing the options.
3. **Model tiers instead of provider model names.** Call sites no longer
   hard-code model names. `chat_json(model=...)` accepts a tier — `strong`
   (default) or `fast` — resolved to a per-provider default
   (OpenAI-compat: `deepseek-v4-pro` / `deepseek-v4-flash`; Anthropic:
   `claude-sonnet-4-5` / `claude-haiku-4-5`) and overridable globally via
   `YT_INGEST_MODEL` / `YT_INGEST_FAST_MODEL`. Explicit model names pass
   through unchanged. `synthesize` now requests the `fast` tier.
4. **Public API unchanged.** The module-level `chat_json` keeps the same
   signature and return shape, so `extract.py`, `synthesize.py`, and
   `cli.ask` are untouched apart from the tier rename in `synthesize`.

## Consequences

- Adding a provider means implementing the protocol and one arm of
  `_select_provider`; no call-site changes.
- Tests patch the underlying SDK clients (`openai.OpenAI`,
  `anthropic.Anthropic`) plus `get_config`; the suite still makes no network
  calls.
- Cost-relevant caveat from ADR-0002 (prompt caching billed at ~10% on
  OpenAI-compatible DeepSeek) still applies to that provider; Anthropic
  cache reads surface as `hit_tokens` in `CacheStats` the same way.
- New env vars: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`,
  `YT_INGEST_MODEL`, `YT_INGEST_FAST_MODEL`. `DEEPSEEK_API_KEY` is legacy.