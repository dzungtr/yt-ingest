from __future__ import annotations
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from yt_ingest.extract import (
    _checksum,
    _chunk_transcript,
    _merge_extractions,
    extract_from_cache,
    render_note,
)
from yt_ingest.llm import CacheStats
from yt_ingest.models import (
    ExtractedContent,
    KeyClaim,
    TranscriptCache,
    TranscriptSegment,
    TranscriptSource,
    WorthRewatching,
)


def _make_cache(n_segments: int = 3) -> TranscriptCache:
    segments = [
        TranscriptSegment(text=f"segment {i}", start=float(i * 5), duration=5.0)
        for i in range(n_segments)
    ]
    return TranscriptCache(
        video_id="dQw4w9WgXcQ",
        url="https://youtu.be/dQw4w9WgXcQ",
        source=TranscriptSource.YOUTUBE_TRANSCRIPT_API,
        fetched_at=datetime(2026, 1, 1),
        title="Test Video",
        channel="Test Channel",
        duration_seconds=n_segments * 5,
        segments=segments,
    )


def _make_content() -> ExtractedContent:
    return ExtractedContent(
        summary="A test summary.",
        key_claims=[KeyClaim(text="Claim one", timestamp_seconds=5.0)],
        frameworks_and_mental_models=["Framework A"],
        definitions={"term": "definition"},
        worth_rewatching=[WorthRewatching(timestamp_seconds=10.0, reason="Great example")],
        counterpoints_and_caveats=["Caveat one"],
        open_questions=["Question one?"],
    )


# --- _chunk_transcript ---

def test_chunk_transcript_small_fits_in_one() -> None:
    cache = _make_cache(3)
    chunks = _chunk_transcript(cache)
    assert len(chunks) == 1
    assert "segment 0" in chunks[0]


def test_chunk_transcript_preserves_all_text() -> None:
    cache = _make_cache(3)
    chunks = _chunk_transcript(cache)
    full = " ".join(chunks)
    for i in range(3):
        assert f"segment {i}" in full


# --- _merge_extractions ---

def test_merge_single_returns_same() -> None:
    content = _make_content()
    merged = _merge_extractions([content])
    assert merged.summary == content.summary


def test_merge_deduplicates_frameworks() -> None:
    c1 = _make_content()
    c2 = _make_content()
    merged = _merge_extractions([c1, c2])
    assert merged.frameworks_and_mental_models.count("Framework A") == 1


def test_merge_combines_key_claims() -> None:
    c1 = _make_content()
    c2 = ExtractedContent(
        summary="Second",
        key_claims=[KeyClaim(text="Claim two", timestamp_seconds=20.0)],
        frameworks_and_mental_models=[],
        definitions={},
        worth_rewatching=[],
        counterpoints_and_caveats=[],
        open_questions=[],
    )
    merged = _merge_extractions([c1, c2])
    assert len(merged.key_claims) == 2


# --- render_note ---

def test_render_note_contains_title() -> None:
    cache = _make_cache()
    content = _make_content()
    note = render_note(cache, content)
    assert "Test Video" in note
    assert "Test Channel" in note
    assert "dQw4w9WgXcQ" in note


def test_render_note_formats_timestamps() -> None:
    cache = _make_cache()
    content = _make_content()
    note = render_note(cache, content)
    assert "0:05" in note or "0:10" in note


# --- extract_from_cache ---

def test_extract_from_cache_calls_llm() -> None:
    cache = _make_cache()
    content = _make_content()
    raw = content.model_dump()
    stats = CacheStats(hit_tokens=50, miss_tokens=50, total_calls=1)

    with patch("yt_ingest.extract.chat_json", return_value=(raw, stats)) as mock_llm:
        result, result_stats = extract_from_cache(cache)

    assert mock_llm.called
    assert result.summary == content.summary
    assert result_stats.total_calls == 1


# --- _checksum ---

def test_checksum_same_for_same_cache() -> None:
    cache = _make_cache()
    assert _checksum(cache) == _checksum(cache)


def test_checksum_differs_for_different_caches() -> None:
    c1 = _make_cache(3)
    c2 = _make_cache(5)
    assert _checksum(c1) != _checksum(c2)
