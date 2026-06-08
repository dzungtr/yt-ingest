# ADR 0006: Rule-driven prompts over few-shot

## Status

**Accepted**

## Context

The per-video extraction is a chunk-and-merge pipeline: the transcript is
split into token-bounded chunks, each chunk is turned into a draft study note
by an LLM call, and — when multiple chunks exist — the drafts are merged into
a single cohesive note by a second LLM call. Two prompt styles were on the
table for each stage:

- **Few-shot**: prepend 2-3 exemplar (transcript, note) pairs so the model
  imitates their style. Adds 500-1500 tokens per call, and the examples carry
  the voice and topic of the exemplar video into every unrelated video.
- **Rule-driven**: state the role, the output contract (TL;DR + free body,
  third-person, prose default), the explicit strip and keep rules, and a soft
  density target as instructions. No examples.
- **Two-pass draft-then-compress**: draft the note first, then run a second
  LLM pass that shortens it. More expensive, architecturally different.

Also on the table: whether the existing JSON contract envelope
(`{"blog_post": "..."}`) introduced by an earlier version of the extraction
prompt should be renamed to match the new role framing ("study note" rather
than "blog post").

## Decision

Use a **single rule-driven prompt per stage**:

- `_DRAFT_PROMPT` for the per-chunk drafting pass.
- `_MERGE_PROMPT` for the cross-chunk merge pass (only fires when a video
  produces 2+ chunks).

Both prompts live as module-level constants in `src/yt_ingest/extract.py`.
They contain:

- A role descriptor ("study-note writer" or "study-note editor").
- The output structure (TL;DR + free body; third-person; prose default).
- The soft density target (20-30%) paired with the anchor "when in doubt, cut".
- The strip and preserve rules from ADR-0005.
- An explicit **JSON envelope**: `Return valid JSON: {"blog_post": "..."}`.

**Few-shot was rejected** because: each exemplar adds 500-1500 tokens per
call (cost and latency), exemplars leak topic and voice into unrelated videos,
and there is no single good exemplar that works across the tool's topic range.
Maintenance cost is high relative to benefit.

**Two-pass draft-then-compress is held as a named fallback**, not implemented.
If real outputs consistently come back longer than the 20-30% target despite
the explicit strip rules, the next iteration escalates to a separate
two-pass design rather than attempting to patch the single prompt further.
The decision to escalate rather than tweak is made per batch of ~5 real
outputs, not per video.

**The `"blog_post"` JSON key is preserved** despite its name no longer
matching the "study note" framing. The key is a label, not a contract:
renaming it would churn prompt constants and test fixtures for zero
functional benefit. The substantive shift is the role descriptor inside the
prompt itself.

### Merge-stage guards

The `_MERGE_PROMPT` includes explicit guards that are load-bearing and not
obvious:

- **"Replace the per-chunk TL;DRs — do not stack them."** Without this,
  models commonly concatenate per-chunk summaries into the merged note.
- **"Do not add new information that is not in the drafts."** This constrains
  the merge to consolidation, not synthesis; prevents the LLM from filling
  in "plausible" claims that weren't actually in the transcript.

## Consequences

- **Prompt edits are local to `extract.py`.** No example files, no fixture
  updates, no configuration toggles.
- **Cost per call is predictable.** No few-shot tokens means the prompt
  contribution to input tokens is bounded by the rule text (a few hundred
  tokens), not by example size.
- **Density drift is possible.** The 20-30% target is a soft instruction and
  LLMs often miss numeric length targets. Mitigation is the explicit strip
  list plus the "when in doubt, cut" anchor. If drift is observed in
  practice, escalation is to the two-pass approach, not more prompt tweaking.
- **The JSON response envelope is a permanent artifact.** Any future consumer
  of `extract_from_cache`'s return value reads it off `raw["blog_post"]`,
  not a more intuitively named key.
