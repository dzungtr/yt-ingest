from __future__ import annotations
import difflib
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import tiktoken
from jinja2 import Environment, FileSystemLoader, select_autoescape

from yt_ingest.llm import CacheStats, chat_json
from yt_ingest.models import Gem, TranscriptCache
from yt_ingest.utils import seconds_to_mmss

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_CHUNK_TOKENS = 6_000
_OVERLAP_TOKENS = 200
_GEMS_PER_CHUNK_CAP = 12

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


def extract_from_cache(cache: TranscriptCache) -> tuple[str, list[Gem], CacheStats]:
    """Transform a transcript into a study note via the gem pipeline.

    Per chunk: mine gems (S2 judgment), then write the draft with that
    chunk's gems as a must-include checklist (S1 execution). Gem lists are
    merged deterministically.

    Returns (blog_post_markdown, merged_gems, CacheStats).
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
        return drafts[0], merged_gems, total_stats

    combined = "\n\n---\n\n".join(
        f"## Draft section {i + 1}\n\n{d}" for i, d in enumerate(drafts)
    )
    raw, stats = chat_json(system=_MERGE_PROMPT, user=combined)
    total_stats = total_stats.merge(stats)
    return raw.get("blog_post", "\n\n".join(drafts)), merged_gems, total_stats


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
