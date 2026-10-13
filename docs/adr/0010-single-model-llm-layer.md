# ADR 0010: Single-model LLM layer (remove the strong/fast tiers)

## Status

**Proposed** — implementation tracked in issue #16.

Amends ADR-0008: the model-tier half of that decision is removed; the
provider-abstraction half (OpenAI-compatible vs Anthropic, env-var
selection, precedence) survives unchanged.

## Context

ADR-0008 introduced two model tiers (`strong` / `fast`) so call sites
could request a cheap model without hard-coding provider names. The
gem-preserving pipeline (ADR-0009) initially considered per-stage tier
assignment — e.g. the verifier on `fast` to save cost on its many small
calls, the miner on `strong` for judgment.

Working through it: the dual-process benefit is **structural** (stage
decomposition — different prompts, different jobs, verification anchored
to the transcript), not model-tiered. A per-stage tiered design would
tune cost against verdict quality — a metric that cannot be measured
until the acceptance re-audit (issue #21) — and would add per-stage knobs
to a personal tool (ADR-0005: prompt policy is not config surface).
CacheStats already reports the real cost after one batch run, so a later
downgrade of any stage is a one-line change via explicit model
pass-through.

## Decision

The LLM layer uses exactly one model. Providers take a single model
instead of a strong/fast pair; the fast-tier default constants and the
`YT_INGEST_FAST_MODEL` env var are removed; `YT_INGEST_MODEL` remains the
sole override; explicit model names still pass through unchanged.
Synthesis and ask move to the single model. All new pipeline stages use
it by construction.

## Consequences

- One model to configure; `.env.example` loses a variable.
- Synthesis cost rises from the fast default to the single model — one
  call per batch, cents on a personal corpus.
- No tier-resolution logic left in the provider layer to maintain or
  test; provider selection, precedence, and cache-stats behavior are
  unchanged.
- A later per-stage downgrade remains possible via explicit model names
  (pass-through) without reinstating tier machinery.
