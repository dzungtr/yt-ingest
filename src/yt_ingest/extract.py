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

_SYSTEM_PROMPT = """\
You are a professional blog writer. Transform the provided YouTube video transcript \
into an engaging, well-structured blog post. Write in clear prose paragraphs with \
natural section headings. Capture the core ideas, insights, and narrative of the video. \
Do not use bullet lists — write flowing sentences and paragraphs instead. \
Return valid JSON: {"blog_post": "full blog post in markdown format"}"""

_MERGE_PROMPT = """\
You are a professional blog editor. The following are draft blog sections written \
from sequential parts of a single YouTube video. Merge them into one cohesive, \
well-structured blog post. Eliminate redundancy, ensure smooth transitions, and \
preserve all key insights. Write in clear prose — no bullet lists. \
Return valid JSON: {"blog_post": "full merged blog post in markdown format"}"""


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
        raw, stats = chat_json(system=_SYSTEM_PROMPT, user=user_msg)
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
