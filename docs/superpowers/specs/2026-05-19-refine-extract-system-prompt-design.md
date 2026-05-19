# Refine extract system prompts — design spec

**Date:** 2026-05-19
**Topic slug:** `refine-extract-system-prompt`
**File touched:** `src/yt_ingest/extract.py` (lines 19-31, prompt constants only)

## Goal

Refine the per-chunk and merge system prompts used by `extract_from_cache()` so
the generated note is a **dense study brief** — substance over style — rather
than a "blog post". Strip filler, personal storytelling, dramatic/hype language,
and word-count inflation. Preserve claims, numbers, caveats, procedures, and the
speaker's opinions.

## Audience and use-case (decided)

- **Use-case (Q1):** Personal study notes. The reader (the user) consumes these
  to absorb a video's content faster than watching. Skimmable, dense, structured
  for later review. Not optimised for publication polish, and not primarily for
  RAG retrieval.
- **Structure (Q2):** Loose template. Mandatory short TL;DR at the top; the rest
  is the model's call (sections, sub-headings, bullets where useful).
- **Density target (Q3):** Medium — roughly 20-30% of the input transcript's word
  count. Claims and key reasoning preserved; most anecdotes/examples dropped
  unless load-bearing. Reads like a study brief.
- **Voice (Q5a):** Third-person, neutral. Refer to the speaker as "the speaker"
  or by name. No first-person narration.
- **Formatting (Q5b):** Prose default. Bullets/numbered lists permitted only when
  they genuinely aid scanning (enumerations, steps, comparisons).

## Strip / preserve matrix (Q4)

| # | Item | Treatment |
|---|------|-----------|
| 1 | Concrete numbers, dates, names of tools/people/companies/products | **Keep** |
| 2 | Specific examples / case studies | **Keep if load-bearing** |
| 3 | Personal anecdotes from the speaker | **Drop** |
| 4 | Analogies and metaphors | **Keep** |
| 5 | Caveats, counterarguments, stated limitations | **Keep** |
| 6 | Step-by-step procedures / how-tos | **Keep** |
| 7 | Speaker's opinions / recommendations | **Keep** |
| 8 | Verbatim quotes | **Keep if load-bearing** |
| 9 | Speculation about the future / predictions | **Keep** |
| 10 | Sponsor reads, channel plugs, intros, outros | **Drop** (interpreted from "C") |

> Item 10 is treated as drop-by-default. Sponsor reads / channel plugs are
> effectively never load-bearing for study purposes. If the user later wants
> them kept in some cases, revisit the prompt.

## Approach (decided)

**Approach 1 — rule-driven prompt.** A single tight system prompt per stage,
naming the role ("study-note writer/editor"), stating the output contract
(TL;DR + free body, third-person, prose default), giving explicit strip and
keep lists, and stating a soft density target. No few-shot examples. No
second compression pass.

Approaches considered and rejected:

- **Few-shot prompt** — adds 500-1500 tokens per call, examples leak topic/voice
  into unrelated videos, no good example exists. High maintenance, low upside.
- **Two-pass (rewrite + compress)** — extra LLM call per video, architectural
  change beyond prompt text. Held as the fallback if Approach 1 outputs come
  back too long in practice.

## Scope

In scope (prompt text only):

- Rewrite `_SYSTEM_PROMPT` → renamed to `_DRAFT_PROMPT`.
- Rewrite `_MERGE_PROMPT` (keep the name).
- Update the call site references in `extract_from_cache()` so they reference
  `_DRAFT_PROMPT` instead of `_SYSTEM_PROMPT`.

Explicitly out of scope (do **not** change as part of this work):

- The JSON contract `{"blog_post": "..."}` — the key stays, even though the
  content is no longer a blog post. The key is a label, not a contract;
  renaming it would churn `extract.py` and any fixtures with no functional
  benefit. The role descriptor inside the prompt is where the substantive
  shift happens.
- The chunk-and-merge control flow in `extract_from_cache()`.
- `_CHUNK_TOKENS` / `_OVERLAP_TOKENS` and `_chunk_transcript()`.
- The Jinja note template (`templates/note.md.j2`) and `render_note()`.
- The FAISS index, `synthesize`, `ask`, or any other CLI command.
- Adding CLI flags, config knobs, or environment toggles for prompt content.
- Adding automated tests for prompt output (brittle for a personal tool).

## Proposed prompt text

### `_DRAFT_PROMPT` (per-chunk drafting)

```text
You are a study-note writer. Transform the provided YouTube video transcript
chunk into dense study notes for a reader who wants to absorb the content
faster than watching the video.

Output format:
- Start with a short TL;DR (3-5 lines) summarising what this chunk covers.
- Then write the body using clear prose paragraphs with section headings. Use
bullet or numbered lists only when they genuinely aid scanning (enumerations,
steps, comparisons). Default to prose.
- Third-person, neutral voice. Refer to the speaker as "the speaker" or by
name. No first-person narration.

Density target: roughly 20-30% of the input chunk's word count. When in doubt,
cut.

Always keep:
- Concrete numbers, dates, names of tools/people/companies/products.
- Analogies and metaphors used to explain concepts.
- Caveats, counterarguments, and stated limitations.
- Step-by-step procedures and how-tos.
- The speaker's opinions, recommendations, and predictions about the future.

Keep only if load-bearing (i.e. it carries the argument, not just illustrates
it):
- Specific examples and case studies.
- Verbatim quotes.

Always drop:
- Personal anecdotes from the speaker ("when I was at X, I...").
- Sponsor reads, channel plugs, intros, outros, calls to subscribe.
- Filler words, verbal tics, hedging phrases, hype language, dramatic build-up.
- Repetition and restating of the same point.

Return valid JSON: {"blog_post": "study notes in markdown format"}
```

Notes:

- The TL;DR is **per chunk**, not per video. The merge stage replaces stacked
  per-chunk TL;DRs with a single video-level TL;DR.
- The JSON contract envelope (`{"blog_post": "..."}`) is preserved.

### `_MERGE_PROMPT` (cross-chunk merge)

Only runs when a video produced 2+ chunks
(`src/yt_ingest/extract.py:81-84`).

```text
You are a study-note editor. The following are draft study-note sections
written from sequential chunks of a single YouTube video. Merge them into one
cohesive study note.

Required output structure:
- A single TL;DR at the top (3-7 lines) summarising the entire video. Replace
the per-chunk TL;DRs — do not stack them.
- Then a unified body with section headings drawn from the actual content, not
"Part 1" / "Chunk 2". Reorder material if it improves flow.
- Use prose paragraphs by default. Bullet or numbered lists only where they
genuinely aid scanning (enumerations, steps, comparisons).
- Third-person, neutral voice throughout.

Editing rules:
- Eliminate redundancy across chunks: if the same claim, number, or caveat
appears twice, keep it once at its most useful location.
- Preserve every distinct claim, number, name, caveat, procedure, opinion,
analogy, and prediction from the drafts. Do not add new information that is
not in the drafts.
- Do not re-introduce filler, hype, personal anecdotes, or sponsor content
even if it survived into a draft.
- The merged note should be no longer than the sum of the drafts, and ideally
shorter once redundancy is removed.

Return valid JSON: {"blog_post": "merged study notes in markdown format"}
```

Notes:

- "Replace the per-chunk TL;DRs — do not stack them." is an explicit guard;
  without it, models often concatenate them.
- "Do not add new information that is not in the drafts." constrains the merge
  to consolidation, not synthesis (i.e. no hallucinated facts).
- No density ratio at this stage — drafts are already compressed; the merge
  de-duplicates rather than re-compresses.

## Risks and mitigations

**Density drift.** The 20-30% target is a soft instruction; LLMs often miss
numeric length targets. Mitigation in the prompt: pair the ratio with "When in
doubt, cut" and explicit strip lists, so content rules still bind even if the
ratio drifts. If the next ~5 real outputs come back consistently flabby,
escalate to Approach 3 (two-pass: draft then compress) — a separate spec at
that point, not a quick patch on top of these prompts.

**Cross-chunk redundancy not collapsed.** Mitigated by the explicit
deduplication rule in `_MERGE_PROMPT`. The "no longer than the sum of the
drafts, ideally shorter" anchor reinforces it.

**Hallucination during merge.** Mitigated by the explicit "Do not add new
information" rule. This is consolidation, not synthesis.

## Validation plan (post-rollout)

Manual, qualitative. No automated tests — brittle and out of scope.

1. Run `extract` on 2-3 transcripts already present in `transcripts/` (or
   listed in `urls.txt`). Pick a mix of lengths so at least one video produces
   2+ chunks and exercises the merge path.
2. For each output note, eyeball:
   - TL;DR present at the top? Length feels right (≤7 lines for the merge
     case, 3-5 for a single-chunk case)?
   - Third-person voice throughout, no first-person?
   - No anecdotes, sponsor reads, channel plugs, or hype language?
   - Numbers, dates, named tools/people, caveats, and procedures preserved?
   - Output length roughly 20-30% of transcript word count (rough eyeball,
     not strict)?
   - Reads as something to **skim and study**, not as a "blog post"?
3. For chunked videos, confirm the merge produced **one** TL;DR, not
   stacked ones, and that section headings reflect content rather than
   "Part 1"/"Chunk 2".

If any of these fail consistently across 3+ videos, capture the failure pattern
and decide between: (a) patching specific prompt lines, or (b) escalating to
the two-pass approach.

## Decisions log

| Decision | Choice | Reason |
|----------|--------|--------|
| Use-case | Personal study notes | User answer to Q1 |
| Structure | Loose template (TL;DR + free body) | User answer to Q2 |
| Density target | Medium (~20-30% of input) | User answer to Q3 |
| Voice | Third-person, neutral | User answer to Q5a |
| Formatting | Prose default, lists where useful | User answer to Q5b |
| Approach | Rule-driven single prompt per stage | User selected over few-shot and two-pass |
| JSON key | Keep `"blog_post"` | Label, not contract; avoids churn |
| Constant naming | `_SYSTEM_PROMPT` → `_DRAFT_PROMPT`; `_MERGE_PROMPT` unchanged | More accurately names the stage |
| Tests | No automated prompt-output tests | Brittle; out of scope for a personal tool |
| Worktree branch | `spec/refine-extract-system-prompt` | Per user-scope rule: spec lands via PR, not committed to main in root |

## Next step

Implementation is **not** executed as part of this design session. The next
step is to invoke `superpowers:writing-plans` to produce a concrete
implementation plan (file edits, verification commands). The main session
will dispatch execution separately.
