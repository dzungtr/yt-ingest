# ADR 0009: Multi-stage gem-preserving extraction (mine → write → verify)

## Status

**Proposed** — implementation tracked in issues #16–#21 (spec #15).

Amends ADR-0005 (the strip/preserve rules gain structural enforcement
rather than prompt-text-only enforcement). Realizes the two-pass escalation
clause of ADR-0006: its rule-driven prompt strategy survives, its
single-pass compression assumption does not.

## Context

A claim-by-claim audit of two videos (`yt-ingest-notes-audit` session,
2026-10-13) confirmed that the single-pass rule-driven extraction
systematically loses five classes of "hidden gems": general causal laws
(the savings-buildup → risk-chasing law), conditional/bet structures (the
drain metaphor), definitional asides that set up later claims (what rare
earths are), embedded quotes carrying factual anchors (the 67% tariff
clip), and thesis-restating closing predictions. Concrete numbers, names,
and procedures survived almost perfectly — the losses concentrated in
connective tissue. Loss is also partly luck-shaped: the same video
extracted twice lost the drain metaphor on the second run (temperature 0,
provider non-determinism).

Root cause: one model call doing two fundamentally different jobs.
Judging what is load-bearing is deliberate, step-by-step judgment
(System 2 work); compressing and writing is pattern-following execution
(System 1 work). To a pattern-matcher, hedging and conditionals *read*
like filler. A single rule-driven prompt cannot distinguish filler from
the argument without understanding the video's narrative arc — which is
exactly what the audit predicted. Research grounding: dual-process
separation (`arXiv:2502.12470`), CoVe independent verification
(`arXiv:2309.11495`), and the circular-verification failure mode to
avoid (`arXiv:2509.05741`, a negative result).

## Decision

Split extraction into stages, each doing one kind of thinking, with an
external verification loop that makes gem preservation deterministic
rather than luck-dependent. Accepted cost envelope: ~2–2.5× token-weighted
per video.

1. **Mine** (S2 judgment): per chunk, an over-inclusive, recall-first LLM
   pass extracts candidate gems — typed (the five audit loss classes plus
   `other`), near-verbatim quoted (the verification anchor), timestamped,
   with a one-line why and a rank, capped per chunk (~12, rank-ordered).
   Recall-first posture: an unneeded gem costs one checklist line; a
   missed gem is lost forever. The miner never pre-drops its own gems.
2. **Write** (S1 execution): the existing rule-driven draft prompt,
   extended with the chunk's gems as a must-include checklist. Writing
   stays cheap pattern-following; judgment has already happened.
3. **Merge**: drafts via the existing merge stage (guards intact); gem
   lists via deterministic dedupe (no LLM — dedupe by quote similarity
   and timestamp adjacency).
4. **Verify** (per gem, independent): one LLM call per gem — never a
   batched gem list, so verdicts cannot bias each other (CoVe
   independence). The judge receives the note, the gem (quote, why,
   timestamp), and a deterministic ±60s transcript excerpt around the
   gem's timestamp as the external anchor — never the note's own
   self-assessment (the circular-verification trap). Content-match, not
   string-match: paraphrase counts as covered. Verdicts: `covered` (with
   note-span citation) / `missing` / `inapplicable`, each with a one-line
   reason.
5. **Patch** (bounded): iff ≥1 gem is missing, exactly one insert-only
   patch folds the missing gems in at their most useful location without
   altering existing note text or lengthening the note; then all gems
   re-verify once. Residual missing/inapplicable verdicts are recorded in
   a persisted persistence report — visible data, never silent drops.
   `inapplicable` never triggers retry.

Verification runs against the **merged** note — the point where text can
no longer change — covering both writer-loss and merge-loss with a single
retry budget per video.

Gem lists (`notes/{video_id}.gems.json`) and persistence reports
(`notes/{video_id}.audit.md`) persist as first-class, gitignored
artifacts. The gem `type` field is metadata only — no stage may branch on
it; it exists for per-loss-class retention metrics and taxonomy growth.

**Prompt-fingerprint checksums.** The extraction checksum input becomes
transcript JSON plus an auto-hash (fingerprint) of all extraction prompt
constants, so any prompt text edit invalidates every checksum and forces
re-extraction of stale notes. Chosen over a manual version constant
because forgetting to bump it reproduces the silent-staleness bug class
this ADR exists to kill.

### Rejected alternatives

- **Extending the keep-list** (the audit's "cheapest fix"): moves the
  problem; prompt rules cannot create narrative-arc judgment. The audit
  itself concluded a single prompt can't distinguish filler from argument.
- **Whole-transcript mining**: breaks the chunk architecture; the arc is
  needed for placement and emphasis — the writer's job — not for spotting
  — the miner's job (the five loss classes are locally type-recognizable).
- **Embedding-similarity verification**: deterministic and cheap, but
  mid-range similarity scores are exactly where silent misses hide
  (same-idea paraphrases score mid-range).
- **Batched gem-list verification**: violates verdict independence;
  verdict N gets colored by verdicts 1..N−1 and each gem's attention
  slice shrinks as the list grows.
- **Per-stage model tiers**: rejected with the single-model refactor
  (ADR-0010); the dual-process benefit is structural, not model-tiered.

## Consequences

- Gem preservation becomes deterministic and measurable: every drop is a
  recorded verdict with a reason; every re-extraction is auditable.
- Cost per video rises to ~2–2.5× token-weighted (approved envelope).
  Provider prefix caching keeps per-gem verification incremental cost
  near excerpt + verdict; CacheStats surfaces the real split on the CLI.
- Stage failures (API errors, invalid JSON) stay fail-loud — no
  error-driven retries or backoff machinery. The only retry in the system
  is the quality-driven patch.
- ADR-0005's strip/preserve rules gain structural enforcement.
- Prompt edits auto-invalidate extraction checksums; no prompt change can
  leave notes silently stale.
- Acceptance is defined by re-audit: the two audited videos must preserve
  all 8 documented lost gems with zero residual missing, density within
  the 20–30% band (issue #21).
