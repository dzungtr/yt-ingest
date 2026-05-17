from __future__ import annotations
from datetime import datetime
from unittest.mock import patch

from yt_ingest.extract import (
    _checksum,
    _chunk_transcript,
    extract_from_cache,
    render_note,
)
from yt_ingest.llm import CacheStats
from yt_ingest.models import (
    TranscriptCache,
    TranscriptSegment,
    TranscriptSource,
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


# --- render_note ---

def test_render_note_contains_title() -> None:
    cache = _make_cache()
    note = render_note(cache, "This is the blog post content.")
    assert "Test Video" in note
    assert "Test Channel" in note
    assert "dQw4w9WgXcQ" in note


def test_render_note_includes_blog_post() -> None:
    cache = _make_cache()
    blog = "## Introduction\n\nThis is the blog post body."
    note = render_note(cache, blog)
    assert "## Introduction" in note
    assert "This is the blog post body." in note


# --- extract_from_cache ---

def test_extract_from_cache_returns_blog_string() -> None:
    cache = _make_cache()
    stats = CacheStats(hit_tokens=50, miss_tokens=50, total_calls=1)
    blog_text = "## My Blog Post\n\nGreat content here."

    with patch("yt_ingest.extract.chat_json", return_value=({"blog_post": blog_text}, stats)) as mock_llm:
        result, result_stats = extract_from_cache(cache)

    assert mock_llm.called
    assert result == blog_text
    assert result_stats.total_calls == 1


def test_extract_from_cache_merges_multiple_chunks() -> None:
    cache = _make_cache()
    stats = CacheStats(hit_tokens=10, miss_tokens=10, total_calls=1)
    draft = "Draft content."
    merged = "## Merged Blog\n\nUnified content."

    call_count = 0

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, str], CacheStats]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {"blog_post": draft}, stats
        return {"blog_post": merged}, stats

    with patch("yt_ingest.extract._chunk_transcript", return_value=["chunk1", "chunk2"]):
        with patch("yt_ingest.extract.chat_json", side_effect=fake_chat_json):
            result, _ = extract_from_cache(cache)

    assert call_count == 3  # 2 chunk calls + 1 merge call
    assert result == merged


# --- _checksum ---

def test_checksum_same_for_same_cache() -> None:
    cache = _make_cache()
    assert _checksum(cache) == _checksum(cache)


def test_checksum_differs_for_different_caches() -> None:
    c1 = _make_cache(3)
    c2 = _make_cache(5)
    assert _checksum(c1) != _checksum(c2)
