from __future__ import annotations
import hashlib
import logging
from pathlib import Path

import tiktoken
from jinja2 import Environment, FileSystemLoader, select_autoescape

from yt_ingest.llm import CacheStats, chat_json
from yt_ingest.models import ExtractedContent, TranscriptCache
from yt_ingest.utils import seconds_to_mmss

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_CHUNK_TOKENS = 6_000
_OVERLAP_TOKENS = 200

_SYSTEM_PROMPT = """\
You are a precise research assistant. Extract structured knowledge from the \
provided YouTube video transcript. Return valid JSON matching this schema exactly:

{
  "summary": "string — 3-5 sentences",
  "key_claims": [{"text": "string", "timestamp_seconds": number}],
  "frameworks_and_mental_models": ["string"],
  "definitions": {"term": "definition"},
  "worth_rewatching": [{"timestamp_seconds": number, "reason": "string"}],
  "counterpoints_and_caveats": ["string"],
  "open_questions": ["string"]
}

Use only information present in the transcript. For timestamps, use the \
start time of the relevant segment."""


def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str) -> int:
    return len(_encoding().encode(text))


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
            # keep overlap
            overlap = current_tokens[-_OVERLAP_TOKENS:]
            current_tokens = list(overlap) + list(toks)
            current_count = len(current_tokens)
        else:
            current_tokens.extend(toks)
            current_count += len(toks)

    if current_tokens:
        chunks.append(enc.decode(current_tokens))

    return chunks or [""]


def _merge_extractions(parts: list[ExtractedContent]) -> ExtractedContent:
    if len(parts) == 1:
        return parts[0]
    summaries = " ".join(p.summary for p in parts)
    key_claims = [c for p in parts for c in p.key_claims]
    frameworks: list[str] = []
    seen_fw: set[str] = set()
    for p in parts:
        for fm in p.frameworks_and_mental_models:
            if fm not in seen_fw:
                frameworks.append(fm)
                seen_fw.add(fm)
    definitions: dict[str, str] = {}
    for p in parts:
        definitions.update(p.definitions)
    worth = [w for p in parts for w in p.worth_rewatching]
    caveats: list[str] = []
    seen_c: set[str] = set()
    for p in parts:
        for c in p.counterpoints_and_caveats:
            if c not in seen_c:
                caveats.append(c)
                seen_c.add(c)
    questions: list[str] = []
    seen_q: set[str] = set()
    for p in parts:
        for q in p.open_questions:
            if q not in seen_q:
                questions.append(q)
                seen_q.add(q)
    return ExtractedContent(
        summary=summaries,
        key_claims=key_claims,
        frameworks_and_mental_models=frameworks,
        definitions=definitions,
        worth_rewatching=worth,
        counterpoints_and_caveats=caveats,
        open_questions=questions,
    )


def extract_from_cache(cache: TranscriptCache) -> tuple[ExtractedContent, CacheStats]:
    """Extract structured notes from a transcript cache entry."""
    chunks = _chunk_transcript(cache)
    parts: list[ExtractedContent] = []
    total_stats = CacheStats()

    for i, chunk in enumerate(chunks):
        user_msg = (
            f"Video: {cache.title!r} by {cache.channel}\n\n"
            f"Transcript chunk {i + 1}/{len(chunks)}:\n\n{chunk}"
        )
        raw, stats = chat_json(system=_SYSTEM_PROMPT, user=user_msg)
        parts.append(ExtractedContent.model_validate(raw))
        total_stats = total_stats.merge(stats)

    content = _merge_extractions(parts)
    return content, total_stats


def _checksum(cache: TranscriptCache) -> str:
    raw = cache.model_dump_json()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def render_note(
    cache: TranscriptCache,
    content: ExtractedContent,
) -> str:
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape([]),
    )
    env.filters["mmss"] = seconds_to_mmss
    template = env.get_template("note.md.j2")
    return template.render(
        title=cache.title,
        channel=cache.channel,
        duration=seconds_to_mmss(cache.duration_seconds),
        video_id=cache.video_id,
        url=cache.url,
        source=cache.source.value,
        content=content,
    )


def note_path(notes_dir: Path, video_id: str) -> Path:
    return notes_dir / f"{video_id}.md"


def checksum_path(notes_dir: Path, video_id: str) -> Path:
    return notes_dir / f".{video_id}.sha"
