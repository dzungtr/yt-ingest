from __future__ import annotations
import hashlib
import logging
from pathlib import Path

import tiktoken
from jinja2 import Environment, FileSystemLoader, select_autoescape

from yt_ingest.llm import CacheStats, chat_json
from yt_ingest.models import TranscriptCache
from yt_ingest.utils import seconds_to_mmss

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_CHUNK_TOKENS = 6_000
_OVERLAP_TOKENS = 200

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


def extract_from_cache(cache: TranscriptCache) -> tuple[str, CacheStats]:
    """Transform a transcript into a blog post. Returns (blog_post_markdown, CacheStats)."""
    chunks = _chunk_transcript(cache)
    drafts: list[str] = []
    total_stats = CacheStats()

    for i, chunk in enumerate(chunks):
        user_msg = (
            f"Video: {cache.title!r} by {cache.channel}\n\n"
            f"Transcript chunk {i + 1}/{len(chunks)}:\n\n{chunk}"
        )
        raw, stats = chat_json(system=_DRAFT_PROMPT, user=user_msg)
        drafts.append(raw.get("blog_post", ""))
        total_stats = total_stats.merge(stats)

    if len(drafts) == 1:
        return drafts[0], total_stats

    combined = "\n\n---\n\n".join(
        f"## Draft section {i + 1}\n\n{d}" for i, d in enumerate(drafts)
    )
    raw, stats = chat_json(system=_MERGE_PROMPT, user=combined)
    total_stats = total_stats.merge(stats)
    return raw.get("blog_post", "\n\n".join(drafts)), total_stats


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
