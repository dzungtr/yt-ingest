# Refine Extract System Prompts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the two LLM system prompts in `src/yt_ingest/extract.py` so the extracted note is a dense personal study brief (TL;DR + free body, ~20-30% of transcript word count, third-person, explicit strip/keep rules) rather than a "blog post".

**Architecture:** Prompt-text-only change. The chunk-and-merge control flow in `extract_from_cache()` is unchanged. The JSON contract `{"blog_post": "..."}` is unchanged (label, not contract). One module-level constant is renamed (`_SYSTEM_PROMPT` → `_DRAFT_PROMPT`) so it reflects the per-chunk drafting role; `_MERGE_PROMPT` keeps its name. The single call site that references `_SYSTEM_PROMPT` is updated to the new name. The existing tests in `tests/test_extract.py` mock `chat_json` and assert on the `blog_post` JSON key only, so they continue to pass without modification.

**Tech Stack:** Python 3, `pytest`, `tiktoken` (already used for chunking). No new dependencies.

**Spec:** [`docs/superpowers/specs/2026-05-19-refine-extract-system-prompt-design.md`](../specs/2026-05-19-refine-extract-system-prompt-design.md)

---

## Pre-work assumptions

- Working in a fresh worktree on a new feature branch (e.g. `feat/refine-extract-system-prompt`), created via `superpowers:using-git-worktrees`. **This is a different branch from `spec/refine-extract-system-prompt`** (the spec PR), so the implementation can be reviewed independently.
- All commands are relative to the worktree root unless otherwise noted.
- The implementer has read the spec linked above and will not re-litigate any decision recorded there.

---

## Task 1: Rename `_SYSTEM_PROMPT` to `_DRAFT_PROMPT` and replace its body

**Files:**
- Modify: `src/yt_ingest/extract.py:19-24` (the `_SYSTEM_PROMPT` constant)
- Modify: `src/yt_ingest/extract.py:74` (the only call site that references `_SYSTEM_PROMPT`)

- [ ] **Step 1: Confirm the constant is referenced only once**

Run:

```bash
grep -n '_SYSTEM_PROMPT' src/yt_ingest/extract.py
```

Expected output (exact line numbers may vary, two matches total):

```
19:_SYSTEM_PROMPT = """\
74:        raw, stats = chat_json(system=_SYSTEM_PROMPT, user=user_msg)
```

Also confirm no other file imports it:

```bash
grep -rn '_SYSTEM_PROMPT' src tests
```

Expected: only the two lines above. If anything else appears, stop and report — the rename needs broader treatment than this plan covers.

- [ ] **Step 2: Replace the `_SYSTEM_PROMPT` constant**

Replace lines 19-24 of `src/yt_ingest/extract.py` (the entire `_SYSTEM_PROMPT = """..."""` block) with:

```python
_DRAFT_PROMPT = """\
You are a study-note writer. Transform the provided YouTube video transcript \
chunk into dense study notes for a reader who wants to absorb the content \
faster than watching the video.

Output format:
- Start with a short TL;DR (3-5 lines) summarising what this chunk covers.
- Then write the body using clear prose paragraphs with section headings. Use \
bullet or numbered lists only when they genuinely aid scanning (enumerations, \
steps, comparisons). Default to prose.
- Third-person, neutral voice. Refer to the speaker as "the speaker" or by \
name. No first-person narration.

Density target: roughly 20-30% of the input chunk's word count. When in doubt, \
cut.

Always keep:
- Concrete numbers, dates, names of tools/people/companies/products.
- Analogies and metaphors used to explain concepts.
- Caveats, counterarguments, and stated limitations.
- Step-by-step procedures and how-tos.
- The speaker's opinions, recommendations, and predictions about the future.

Keep only if load-bearing (i.e. it carries the argument, not just illustrates \
it):
- Specific examples and case studies.
- Verbatim quotes.

Always drop:
- Personal anecdotes from the speaker ("when I was at X, I...").
- Sponsor reads, channel plugs, intros, outros, calls to subscribe.
- Filler words, verbal tics, hedging phrases, hype language, dramatic build-up.
- Repetition and restating of the same point.

Return valid JSON: {"blog_post": "study notes in markdown format"}"""
```

Note: this preserves the existing Python line-continuation style (trailing `\` joined inside one `"""..."""` literal), matches the indentation already in the file (the constant is at module top level, not nested), and keeps the JSON envelope (`{"blog_post": "..."}`) so downstream code does not break.

- [ ] **Step 3: Update the single call site**

In `src/yt_ingest/extract.py`, find the line that currently reads:

```python
        raw, stats = chat_json(system=_SYSTEM_PROMPT, user=user_msg)
```

(formerly line 74, inside `extract_from_cache`'s per-chunk loop).

Replace it with:

```python
        raw, stats = chat_json(system=_DRAFT_PROMPT, user=user_msg)
```

- [ ] **Step 4: Verify the rename is complete**

Run:

```bash
grep -n '_SYSTEM_PROMPT' src tests
```

Expected: **no output** (the symbol no longer exists anywhere).

Run:

```bash
grep -n '_DRAFT_PROMPT' src/yt_ingest/extract.py
```

Expected: exactly two matches — the constant definition line and the `chat_json(system=_DRAFT_PROMPT, ...)` call site.

- [ ] **Step 5: Run the existing extract tests to confirm nothing broke**

Run:

```bash
pytest tests/test_extract.py -v
```

Expected: all tests pass. The existing tests mock `chat_json` and assert on JSON envelope only — they do not reference `_SYSTEM_PROMPT` or `_DRAFT_PROMPT` by name, so the rename is invisible to them.

If any test fails, stop and read the failure. Do not skip or modify a test unless it is genuinely asserting on the old prompt text (none should — checked at plan-writing time).

- [ ] **Step 6: Commit**

```bash
git add src/yt_ingest/extract.py
git commit -m "refactor: rename _SYSTEM_PROMPT to _DRAFT_PROMPT and rewrite as study-note prompt

Replaces the per-chunk 'blog writer' prompt with a 'study-note writer' prompt
that targets dense personal study notes: short TL;DR plus free body, ~20-30%
density target, third-person voice, explicit strip/keep lists. The JSON
envelope ({\"blog_post\": ...}) is unchanged."
```

---

## Task 2: Replace `_MERGE_PROMPT` body

**Files:**
- Modify: `src/yt_ingest/extract.py:26-31` (the `_MERGE_PROMPT` constant)

No call sites change in this task — `_MERGE_PROMPT` keeps its name; only the prompt text changes.

- [ ] **Step 1: Replace the `_MERGE_PROMPT` constant**

Replace the entire `_MERGE_PROMPT = """..."""` block (formerly lines 26-31, may have shifted after Task 1) with:

```python
_MERGE_PROMPT = """\
You are a study-note editor. The following are draft study-note sections \
written from sequential chunks of a single YouTube video. Merge them into one \
cohesive study note.

Required output structure:
- A single TL;DR at the top (3-7 lines) summarising the entire video. Replace \
the per-chunk TL;DRs — do not stack them.
- Then a unified body with section headings drawn from the actual content, \
not "Part 1" / "Chunk 2". Reorder material if it improves flow.
- Use prose paragraphs by default. Bullet or numbered lists only where they \
genuinely aid scanning (enumerations, steps, comparisons).
- Third-person, neutral voice throughout.

Editing rules:
- Eliminate redundancy across chunks: if the same claim, number, or caveat \
appears twice, keep it once at its most useful location.
- Preserve every distinct claim, number, name, caveat, procedure, opinion, \
analogy, and prediction from the drafts. Do not add new information that is \
not in the drafts.
- Do not re-introduce filler, hype, personal anecdotes, or sponsor content \
even if it survived into a draft.
- The merged note should be no longer than the sum of the drafts, and ideally \
shorter once redundancy is removed.

Return valid JSON: {"blog_post": "merged study notes in markdown format"}"""
```

Note: indentation is module top level (matching the existing constant), the line-continuation `\` style is preserved, and the JSON envelope key `blog_post` is unchanged.

- [ ] **Step 2: Verify the constant still parses**

Run:

```bash
python -c "from yt_ingest.extract import _MERGE_PROMPT, _DRAFT_PROMPT; print(len(_DRAFT_PROMPT), len(_MERGE_PROMPT))"
```

Expected: two integers printed, both > 500 (rough lower bound — the new prompts are noticeably longer than the originals, which were ~300-400 chars each). No `SyntaxError`, no `ImportError`.

- [ ] **Step 3: Sanity-check the prompt content with a grep**

Run:

```bash
grep -c 'study-note' src/yt_ingest/extract.py
```

Expected: `4` (or more). The string "study-note" should appear in both the writer prompt and the editor prompt, multiple times.

Run:

```bash
grep -c 'blog' src/yt_ingest/extract.py
```

Expected: `4` — exactly the four surviving uses of `blog_post` as the JSON key and the `raw.get("blog_post", ...)` fallback. **If you see "blog writer" or "blog editor", you missed editing the role descriptor — go back to Task 1 Step 2 or this task's Step 1.**

- [ ] **Step 4: Run the full extract test suite**

Run:

```bash
pytest tests/test_extract.py -v
```

Expected: all tests pass. The merge test (`test_extract_from_cache_merges_multiple_chunks`) exercises the merge path with mocked `chat_json` and should pass unchanged.

- [ ] **Step 5: Run the full project test suite**

Run:

```bash
pytest -v
```

Expected: all tests pass. This is a prompt-text-only change with no signature or contract changes; nothing outside `tests/test_extract.py` should be affected, but running the full suite is cheap insurance.

- [ ] **Step 6: Commit**

```bash
git add src/yt_ingest/extract.py
git commit -m "refactor: rewrite _MERGE_PROMPT as study-note editor prompt

Replaces the cross-chunk 'blog editor' prompt with a 'study-note editor'
prompt that consolidates per-chunk TL;DRs into a single top-level TL;DR,
enforces redundancy removal, forbids hallucination ('do not add new
information'), and re-asserts the strip rules so filler does not leak back
in during merge. JSON envelope is unchanged."
```

---

## Task 3: Manual validation on a real transcript

This task is **not optional** — the spec's validation plan is qualitative and lives here. Two of the three checks (TL;DR present, third-person voice, no anecdotes) can only be confirmed by reading actual output.

**Files:**
- No code changes. This task produces a short validation log in the commit message and confirms the implementation meets the spec's success criteria.

- [ ] **Step 1: Identify a short test transcript**

Run:

```bash
ls transcripts/ | head -20
```

If `transcripts/` contains at least one cached transcript, pick one that is **short enough to fit in a single chunk** (under ~6000 tokens, which roughly means a video under ~30-40 minutes). If you cannot tell length from the filename alone, run:

```bash
for f in transcripts/*.json; do
  python -c "
import json, sys
d = json.load(open(sys.argv[1]))
n = sum(len(s.get('text', '').split()) for s in d.get('segments', []))
print(f'{sys.argv[1]}: ~{n} words')
" "$f"
done | sort -t'~' -k2 -n | head -5
```

Pick the shortest (smallest word count).

If `transcripts/` is empty or missing, run an `extract` invocation from a URL — but **do not let this become a yak shave**. If no transcript is available within ~5 minutes of looking, stop and ask the user to provide one or skip to Step 4 with the empty-state caveat noted.

- [ ] **Step 2: Run extraction on the chosen transcript**

The exact command depends on the CLI shape. Try:

```bash
python -m yt_ingest extract --help
```

and pick the invocation that runs extraction against an already-cached transcript without re-fetching. Typical shapes (one of these will work — try in order):

```bash
python -m yt_ingest extract <video_id_or_url>
# or
python -m yt_ingest run <video_id_or_url>   # may include fetch + extract
```

The output should land in the project's `notes/` directory as `<video_id>.md`. If you cannot determine the right invocation in under 2 minutes, ask the user.

- [ ] **Step 3: Read the resulting note and check it against the spec's criteria**

Open the generated note file (e.g. `notes/<video_id>.md`) and verify, in order:

1. **TL;DR at the top?** First non-template content should be a short TL;DR section (3-5 lines for a single-chunk video, 3-7 for a merged one). Section heading "TL;DR" is expected but not strictly mandatory — a clearly-labelled summary at the top counts.
2. **Third-person voice?** No "I", "my", "we" (other than inside a quoted phrase). The speaker is referred to as "the speaker", "the author", or by name.
3. **No anecdotes or sponsor content?** No "when I was at...", no "this video is sponsored by...", no "smash that subscribe button".
4. **Numbers, names, and caveats preserved?** If the transcript mentioned specific tools, dates, or limitations, they should appear in the note.
5. **Length feels right?** Roughly eyeball — the note should be substantially shorter than the transcript. Counting words is overkill; "feels like a study brief, not a blog post" is the bar.
6. **Reads as something to skim and study?** Subjective, but the headline test the spec optimises for.

If a chunked video is available, also test that case (run on a transcript large enough to produce 2+ chunks) and confirm:

7. **One TL;DR**, not stacked per-chunk TL;DRs.
8. **Section headings reflect content**, not "Part 1" / "Chunk 2".

- [ ] **Step 4: Record the validation result**

Create a short note in the commit message for this step. If validation passed cleanly, the commit body should list which transcript was tested, whether it was single-chunk or chunked, and a one-line "all criteria met" assertion.

If validation **failed** on any criterion, stop and report back to the spec author before proceeding. Do not patch the prompt inline as part of this task — failed validation is a signal to discuss the escalation path (likely Approach 3, two-pass compress) recorded in the spec's Risks section.

- [ ] **Step 5: Commit (empty commit if no code changed)**

```bash
git commit --allow-empty -m "chore: validate refined extract prompts on real transcript

Tested transcript: <video_id_or_filename>
Chunked: <yes/no>
TL;DR at top: <yes/no>
Third-person voice: <yes/no>
No anecdotes/sponsor content: <yes/no>
Numbers and caveats preserved: <yes/no>
Output ~20-30% of transcript length: <yes/no — rough eyeball>
Reads as a study brief: <yes/no>

[If chunked]
Single top-level TL;DR (not stacked): <yes/no>
Section headings reflect content (not 'Part N'): <yes/no>

Overall: <all criteria met / see notes above>"
```

If validation failed and the user instructs you to skip the empty commit, omit Step 5 — the validation result is already captured in the conversation.

---

## Task 4: Open the implementation PR

**Files:**
- No code changes.

- [ ] **Step 1: Verify the worktree is clean and on the implementation branch**

```bash
git status --short
git branch --show-current
```

Expected: clean working tree, branch name is the implementation branch (not `main` and not the spec branch `spec/refine-extract-system-prompt`).

- [ ] **Step 2: Push the branch**

```bash
git push -u origin "$(git branch --show-current)"
```

Expected: branch pushed; remote tracking set up. If the push fails because the branch already exists on remote, stop and check with the user — do not force-push.

- [ ] **Step 3: Open the PR**

Use `gh pr create`. The PR body should reference the spec PR (#5 at the time of plan-writing — confirm the actual number with `gh pr list --search "spec/refine-extract-system-prompt"` if unsure):

```bash
gh pr create --title "feat: refine extract system prompts to dense study-note style" --body "$(cat <<'EOF'
## Summary

- Implements the prompts approved in the spec PR (see "Related" below).
- Renames `_SYSTEM_PROMPT` → `_DRAFT_PROMPT` and rewrites it as a study-note writer prompt (TL;DR + free body, ~20-30% density target, third-person voice, explicit strip/keep lists).
- Rewrites `_MERGE_PROMPT` as a study-note editor prompt (single top-level TL;DR, redundancy removal, no hallucinated content).
- JSON envelope (`{"blog_post": "..."}`) is unchanged — only the role and content rules change.
- No control-flow, chunking, or downstream changes.

## Related

- Spec PR: #5 (replace with actual number if different)
- Spec document: `docs/superpowers/specs/2026-05-19-refine-extract-system-prompt-design.md`
- Plan document: `docs/superpowers/plans/2026-05-19-refine-extract-system-prompt.md`

## Test plan

- [x] `pytest tests/test_extract.py -v` passes
- [x] `pytest -v` (full suite) passes
- [x] Manual validation on at least one real transcript (see commit `<sha>` for details)
- [ ] Reviewer eyeballs one generated note and confirms it reads as a study brief, not a blog post
EOF
)"
```

Expected: PR URL printed. Report the URL back.

- [ ] **Step 4: Hand off**

Report the PR URL and confirm the implementation is done. No further steps in this plan.

---

## Self-review (run after writing the plan)

**Spec coverage:**

- Goal (dense study notes, strip/preserve matrix, ~20-30% density, third-person, prose default) → Task 1 + Task 2 (the prompt bodies contain every rule from the spec's matrix).
- Use-case = personal study notes → reflected in role descriptor "study-note writer/editor" in Task 1 Step 2 and Task 2 Step 1.
- Structure = loose template (TL;DR + free body) → "Start with a short TL;DR... Then write the body" in `_DRAFT_PROMPT`; "A single TL;DR at the top... Then a unified body" in `_MERGE_PROMPT`.
- Density target = 20-30% → present verbatim in `_DRAFT_PROMPT` Task 1 Step 2.
- Voice = third-person neutral → present in both prompts.
- Formatting = prose default, lists where useful → present in both prompts.
- Approach = rule-driven single prompt per stage → no few-shot, no two-pass added.
- Strip/preserve matrix (items 1-10) → mapped into the "Always keep" / "Keep only if load-bearing" / "Always drop" sections in Task 1 Step 2.
- Out-of-scope items (JSON key rename, chunking, FAISS, synthesize, ask, CLI flags, automated prompt tests) → none of these are touched by any task. ✓
- Validation plan (eyeball TL;DR, third-person, no anecdotes/sponsor, numbers/caveats preserved, length, skim test, single TL;DR on merge) → Task 3.
- Risks section (density drift, hallucination, cross-chunk redundancy) → addressed inside the prompts; no separate task needed.

**Placeholder scan:** no `TBD`, no `TODO`, no "implement later", no "add error handling" hand-waves. Each step has the actual content needed.

**Type consistency:** the only identifier introduced is `_DRAFT_PROMPT`. It appears in Task 1 Step 2 (definition), Task 1 Step 3 (single call site), Task 1 Step 4 (grep verification), and Task 2 Step 2 (import smoke test). Same casing and same leading underscore throughout. The retained identifier `_MERGE_PROMPT` is also consistent.

**Gaps fixed inline:** none found on this pass.
