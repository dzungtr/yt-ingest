from __future__ import annotations
import difflib
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tiktoken
from jinja2 import Environment, FileSystemLoader, select_autoescape

from yt_ingest.llm import CacheStats, chat_json
from yt_ingest.models import Gem, TranscriptCache, Verdict, VerdictStatus
from yt_ingest.utils import seconds_to_mmss

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_CHUNK_TOKENS = 6_000
_OVERLAP_TOKENS = 200
_GEMS_PER_CHUNK_CAP = 12
_EXCERPT_WINDOW_SECONDS = 60.0

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

_GEM_GUIDANCE = """

Gem checklist handling:
- The user message includes a MUST-INCLUDE gem checklist mined from this \
chunk by a separate judgment pass. Every gem on the checklist must appear \
in the note. Convey each gem's idea faithfully — paraphrase is fine; the \
near-verbatim quote need not appear word-for-word.
- Weave gems in at their most useful location in the flow. Do not append \
them as a separate list and do not mention checklist mechanics.
- If a gem seems to conflict with the strip rules below, the gem wins: \
mined gems are must-include by definition, and judgment has already \
happened upstream.
"""

_MINING_PROMPT = """\
You are a gem miner. Your job is deliberate, over-inclusive judgment: extract \
candidate "gems" from the provided YouTube video transcript chunk — content \
that must survive into the study note. Recall-first: an unneeded gem costs \
one checklist line; a missed gem is lost forever. When in doubt, include it.

Mine for these classes:
- causal_law: general causal laws and transferable rules ("X leads to Y").
- conditional_bet: conditional or bet structures ("if A, then B"), and the \
stated conditions under which the video's claims hold.
- definitional_aside: definitional asides that set up later arguments (what \
a term means, why it matters downstream).
- embedded_quote: verbatim or near-verbatim quotes carrying factual anchors \
(numbers, named triggers, quoted claims).
- closing_prediction: thesis restatements and forward-looking predictions, \
especially near the end of the video.
- other: anything else load-bearing that fits none of the above.

Rules:
- Quote near-verbatim from the transcript — the quote is the verification \
anchor checked against the transcript later. Never paraphrase the quote.
- timestamp_start: seconds into the video where the gem begins, read from \
the [mm:ss] markers in the transcript.
- why: one line on why this gem must survive into the study note.
- rank: importance order, 1 = most important. Return gems already ordered \
by rank.
- Cap: at most 12 gems per chunk. If the chunk is data-dense, keep the 12 \
most important. Never pad the list with filler; never pre-drop a gem you \
have judged load-bearing.

Return valid JSON:
{"gems": [{"id": "g1", "type": "causal_law", "quote": "...", \
"timestamp_start": 0.0, "why": "...", "rank": 1}]}"""

_VERIFY_PROMPT = """\
You are a gem verifier. You judge exactly one gem: did this gem's idea survive \
into the finished study note? Judge content, not strings — a paraphrase that \
conveys the same idea counts as covered; the near-verbatim quote need not \
appear word-for-word. The provided transcript excerpt around the gem's \
timestamp is the external anchor: judge the gem against the source material, \
never against the note's own self-description.

Verdicts:
- covered: the gem's idea is present in the note. Cite the note span — a \
short quote or section heading taken from the note itself — where it survives.
- missing: the gem's idea is absent from the note.
- inapplicable: the gem cannot apply to this note (for example the excerpt \
shows the quote is channel chatter or outside the video's substance).

Always give a one-line reason. Return valid JSON:
{"verdict": "covered", "reason": "...", "note_span": "..."}

note_span is required for covered; use "" for missing and inapplicable."""

_PATCH_PROMPT = """\
You are a study-note editor performing an insert-only patch. The user message \
lists gems that were judged missing from the finished study note. Fold each \
missing gem's idea into the note at its most useful location.

Insert-only rules:
- Do not alter, reword, or reorder any existing note text.
- Do not lengthen the note: keep the total length at or below the original.
- Convey each missing gem's idea faithfully — paraphrase is fine; the \
near-verbatim quote need not appear word-for-word.
- Do not mention gem, verification, or patch mechanics.

Return valid JSON: {"blog_post": "patched study notes in markdown format"}"""


def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _chunk_transcript(cache: TranscriptCache) -> list[str]:
    """Split transcript into overlapping chunks by token count."""
    enc = _encoding()
    chunks: list[str] = []
    current_tokens: list[int] = []
    current_count = 0

    for seg in cache.segments:
        line = f"[{seconds_to_mmss(seg.start)}] {seg.text}"
        toks = enc.encode(line)
        if current_count + len(toks) > _CHUNK_TOKENS and current_tokens:
            chunks.append(enc.decode(current_tokens))
            overlap = current_tokens[-_OVERLAP_TOKENS:]
            current_tokens = list(overlap) + list(toks)
            current_count = len(current_tokens)
        else:
            current_tokens.extend(toks)
            current_count += len(toks)

    if current_tokens:
        chunks.append(enc.decode(current_tokens))

    return chunks or [""]


def _parse_gems(raw: dict[str, Any]) -> list[Gem]:
    """Validate and normalize the miner's output into the gem data contract.

    Rank-ordered, capped at _GEMS_PER_CHUNK_CAP, ranks renumbered 1..n.
    Invalid items raise (pydantic) — stage failures stay fail-loud.
    """
    items = raw.get("gems", [])
    if not isinstance(items, list):
        raise ValueError("Mining stage returned no gem list.")
    gems = [Gem.model_validate(item) for item in items]
    gems.sort(key=lambda g: g.rank)
    gems = gems[:_GEMS_PER_CHUNK_CAP]
    return [
        g.model_copy(update={"rank": i + 1}) for i, g in enumerate(gems)
    ]


def _transcript_excerpt(cache: TranscriptCache, timestamp_start: float) -> str:
    """Deterministic ±60s transcript excerpt around the gem's timestamp.

    Code-sliced (never LLM-selected): every segment whose start falls inside
    the closed window [ts-60, ts+60], in transcript order, as [mm:ss] lines.
    """
    lo = timestamp_start - _EXCERPT_WINDOW_SECONDS
    hi = timestamp_start + _EXCERPT_WINDOW_SECONDS
    lines = [
        f"[{seconds_to_mmss(seg.start)}] {seg.text}"
        for seg in cache.segments
        if lo <= seg.start <= hi
    ]
    if not lines:
        return "(no transcript content within ±60s of the gem timestamp)"
    return "\n".join(lines)


def _parse_verdict(gem_id: str, raw: dict[str, Any]) -> Verdict:
    """Validate the verifier's output into a Verdict; failures stay fail-loud."""
    try:
        return Verdict.model_validate(
            {
                "gem_id": gem_id,
                "verdict": raw.get("verdict"),
                "reason": raw.get("reason", ""),
                "note_span": raw.get("note_span", ""),
            }
        )
    except ValueError as exc:
        raise ValueError(
            f"Verifier returned an invalid verdict for gem {gem_id}: {exc}"
        ) from exc


def _verify_user_message(note: str, gem: Gem, excerpt: str) -> str:
    """Verify prompt body: stable note prefix first, per-gem tail last, so the
    shared prefix (system + note block) hits provider prefix caching."""
    return (
        "Finished study note:\n\n"
        f"{note}\n\n"
        "---\n\n"
        f"Gem to verify (id {gem.id}):\n"
        f'- quote: "{gem.quote}"\n'
        f"- why: {gem.why}\n"
        f"- timestamp: {seconds_to_mmss(gem.timestamp_start)}\n\n"
        "Transcript excerpt (±60s around the gem's timestamp):\n\n"
        f"{excerpt}"
    )


def _verify_round(
    cache: TranscriptCache, note: str, gems: list[Gem]
) -> tuple[list[Verdict], CacheStats]:
    """One independent LLM call per gem — never a batched gem list, so verdicts
    cannot bias each other (CoVe-style)."""
    verdicts: list[Verdict] = []
    total_stats = CacheStats()
    for gem in gems:
        user = _verify_user_message(
            note, gem, _transcript_excerpt(cache, gem.timestamp_start)
        )
        raw, stats = chat_json(system=_VERIFY_PROMPT, user=user)
        total_stats = total_stats.merge(stats)
        verdicts.append(_parse_verdict(gem.id, raw))
    return verdicts, total_stats


@dataclass
class VerifyReport:
    """Full verification audit trail for one video (issue #18)."""

    initial: list[Verdict]
    final: list[Verdict]
    recheck: list[Verdict] = field(default_factory=list)
    patch_fired: bool = False


def _md_cell(text: str) -> str:
    """Flatten a reason/note span into a safe single-line markdown table cell.

    Pipes are escaped (they would split the cell) and newlines are collapsed
    so a multi-line LLM reason cannot break the table structure.
    """
    return text.replace("|", "\\|").replace("\r", " ").replace("\n", " ").strip()


def verify_and_patch_gems(
    cache: TranscriptCache, note: str, gems: list[Gem]
) -> tuple[str, VerifyReport, CacheStats]:
    """Verify every gem against the finished note; if any is missing, fire
    exactly one insert-only patch and re-verify all gems once.

    Inapplicable never triggers retry. Whatever remains missing or
    inapplicable is recorded in the report, never silently dropped.

    Returns (possibly patched note, VerifyReport, CacheStats).
    """
    initial, stats = _verify_round(cache, note, gems)
    report = VerifyReport(initial=initial, final=initial)
    missing_ids = {v.gem_id for v in initial if v.verdict is VerdictStatus.MISSING}
    if not missing_ids:
        return note, report, stats

    missing = [g for g in gems if g.id in missing_ids]
    user = (
        "Finished study note:\n\n"
        f"{note}\n\n"
        "---\n\n"
        "Missing gems to fold in (insert-only):\n\n"
        + _gem_checklist(missing)
    )
    raw, patch_stats = chat_json(system=_PATCH_PROMPT, user=user)
    stats = stats.merge(patch_stats)
    patched_note = raw.get("blog_post", note)

    recheck, recheck_stats = _verify_round(cache, patched_note, gems)
    stats = stats.merge(recheck_stats)
    report.patch_fired = True
    report.recheck = recheck
    report.final = recheck
    return patched_note, report, stats


def render_audit_md(
    video_id: str, gems: list[Gem], report: VerifyReport
) -> str:
    """Render the audit artifact: verdict per gem, reasons, note spans, retry
    history, and residuals — the honest, re-auditable loss trail."""
    gem_by_id = {g.id: g for g in gems}
    lines = [
        f"# Verification audit — {video_id}",
        "",
        "## Verdicts (final)",
        "",
        "| gem | type | verdict | reason | note span |",
        "|---|---|---|---|---|",
    ]
    for v in report.final:
        gem = gem_by_id[v.gem_id]
        span = v.note_span or "—"
        lines.append(
            f"| {v.gem_id} | {gem.type.value} | {v.verdict.value} "
            f"| {_md_cell(v.reason)} | {_md_cell(span)} |"
        )

    lines += ["", "## Retry history", ""]
    if report.patch_fired:
        lines.append(
            "- Patch fired (at least one gem missing); all gems re-verified once."
        )
        for v in report.initial:
            lines.append(
                f"- Initial verdict {v.gem_id}: {v.verdict.value} — {v.reason}"
            )
    else:
        lines.append("- No patch fired (no gem was missing).")

    lines += ["", "## Residuals", ""]
    residuals = [v for v in report.final if v.verdict is not VerdictStatus.COVERED]
    if residuals:
        for v in residuals:
            lines.append(f"- {v.gem_id}: {v.verdict.value} — {v.reason}")
    else:
        lines.append("- None. Every gem ended covered or reasoned-inapplicable.")
    return "\n".join(lines) + "\n"


def _gem_checklist(gems: list[Gem]) -> str:
    """Render the chunk's gems as the writer's must-include checklist."""
    lines = [
        f"{g.rank}. [{g.type.value}] \"{g.quote}\" "
        f"(at {seconds_to_mmss(g.timestamp_start)}) — {g.why}"
        for g in gems
    ]
    return "MUST-INCLUDE gem checklist (mined from this chunk):\n" + "\n".join(lines)


def _normalized_quote(quote: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", quote.lower()))


def _same_gem(a: Gem, b: Gem) -> bool:
    """Near-duplicate test: quote similarity AND timestamp adjacency (ADR-0009)."""
    if abs(a.timestamp_start - b.timestamp_start) > 60.0:
        return False
    na, nb = _normalized_quote(a.quote), _normalized_quote(b.quote)
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb).ratio() >= 0.85


def merge_gems(gem_lists: list[list[Gem]]) -> list[Gem]:
    """Deterministic gem merge: dedupe near-duplicates across chunk lists
    (no LLM), then reassign ids/ranks 1..n in stable rank order."""
    merged: list[Gem] = []
    for gems in gem_lists:
        for gem in gems:
            if not any(_same_gem(gem, kept) for kept in merged):
                merged.append(gem)
    merged.sort(key=lambda g: g.rank)
    return [
        m.model_copy(update={"id": f"g{i + 1:03d}", "rank": i + 1})
        for i, m in enumerate(merged)
    ]


def extract_from_cache(
    cache: TranscriptCache,
) -> tuple[str, list[Gem], VerifyReport | None, CacheStats]:
    """Transform a transcript into a study note via the gem pipeline.

    Per chunk: mine gems (S2 judgment), then write the draft with that
    chunk's gems as a must-include checklist (S1 execution). Gem lists are
    merged deterministically. On the single-chunk path the finished note is
    then verified per gem and patched at most once (issue #18).

    Returns (blog_post_markdown, merged_gems, VerifyReport, CacheStats).
    On the multi-chunk path verification is not wired yet (issue #19): the
    report is None so no audit artifact is written for unverified notes.
    """
    chunks = _chunk_transcript(cache)
    drafts: list[str] = []
    chunk_gems: list[list[Gem]] = []
    total_stats = CacheStats()

    for i, chunk in enumerate(chunks):
        user_msg = (
            f"Video: {cache.title!r} by {cache.channel}\n\n"
            f"Transcript chunk {i + 1}/{len(chunks)}:\n\n{chunk}"
        )
        raw, stats = chat_json(system=_MINING_PROMPT, user=user_msg)
        total_stats = total_stats.merge(stats)
        gems = _parse_gems(raw)
        chunk_gems.append(gems)

        system = _DRAFT_PROMPT + _GEM_GUIDANCE if gems else _DRAFT_PROMPT
        write_msg = user_msg + "\n\n" + _gem_checklist(gems) if gems else user_msg
        raw, stats = chat_json(system=system, user=write_msg)
        drafts.append(raw.get("blog_post", ""))
        total_stats = total_stats.merge(stats)

    merged_gems = merge_gems(chunk_gems)

    if len(drafts) == 1:
        # Single-chunk path: verify every gem against the finished note and
        # apply the single bounded patch if needed (issue #18). Multi-chunk
        # post-merge verification is wired by issue #19.
        note, report, verify_stats = verify_and_patch_gems(
            cache, drafts[0], merged_gems
        )
        total_stats = total_stats.merge(verify_stats)
        return note, merged_gems, report, total_stats

    combined = "\n\n---\n\n".join(
        f"## Draft section {i + 1}\n\n{d}" for i, d in enumerate(drafts)
    )
    raw, stats = chat_json(system=_MERGE_PROMPT, user=combined)
    total_stats = total_stats.merge(stats)
    # Multi-chunk: no per-gem verification yet (issue #19) — report stays None
    # so the CLI skips the audit artifact instead of writing a false
    # "everything covered" trail for an unverified note.
    return raw.get("blog_post", "\n\n".join(drafts)), merged_gems, None, total_stats


def _checksum(cache: TranscriptCache) -> str:
    raw = cache.model_dump_json()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def render_note(
    cache: TranscriptCache,
    blog_post: str,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape([]),
    )
    template = env.get_template("note.md.j2")
    return template.render(
        title=cache.title,
        channel=cache.channel,
        duration=seconds_to_mmss(cache.duration_seconds),
        video_id=cache.video_id,
        url=cache.url,
        source=cache.source.value,
        blog_post=blog_post,
    )


def note_path(notes_dir: Path, video_id: str) -> Path:
    return notes_dir / f"{video_id}.md"


def checksum_path(notes_dir: Path, video_id: str) -> Path:
    return notes_dir / f".{video_id}.sha"


def gems_path(notes_dir: Path, video_id: str) -> Path:
    return notes_dir / f"{video_id}.gems.json"


def audit_path(notes_dir: Path, video_id: str) -> Path:
    return notes_dir / f"{video_id}.audit.md"
