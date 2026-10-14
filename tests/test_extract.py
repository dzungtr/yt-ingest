from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from yt_ingest.extract import (
    _checksum,
    _chunk_transcript,
    extract_from_cache,
    gems_path,
    render_note,
)
from yt_ingest.llm import CacheStats
from yt_ingest.models import (
    Gem,
    GemType,
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

_GEMS_PAYLOAD = {
    "gems": [
        {
            "id": "g1",
            "type": "causal_law",
            "quote": "Savings buildup leads to risk chasing.",
            "timestamp_start": 61.0,
            "why": "Core causal law.",
            "rank": 1,
        },
        {
            "id": "g2",
            "type": "conditional_bet",
            "quote": "If the giants do not deliver, the pipeline becomes a drain.",
            "timestamp_start": 120.0,
            "why": "Bet condition.",
            "rank": 2,
        },
    ]
}


def _dispatch_mine_write(
    mine_payload: dict[str, object], blog_text: str, stats: CacheStats
):
    """chat_json test double dispatching on the system prompt (mine vs write)."""

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        system = str(kwargs.get("system", ""))
        if "gem miner" in system:
            return mine_payload, stats
        if "gem verifier" in system:
            return {"verdict": "covered", "reason": "present", "note_span": "b"}, stats
        return {"blog_post": blog_text}, stats

    return fake_chat_json


def test_extract_from_cache_single_chunk_mines_then_writes() -> None:
    cache = _make_cache()
    stats = CacheStats(hit_tokens=50, miss_tokens=50, total_calls=1)
    blog_text = "## My Blog Post\n\nGreat content here."
    calls: list[dict[str, object]] = []

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        calls.append(dict(kwargs))
        return _dispatch_mine_write(_GEMS_PAYLOAD, blog_text, stats)(**kwargs)

    with patch("yt_ingest.extract.chat_json", side_effect=fake_chat_json):
        result, gems, _report, result_stats = extract_from_cache(cache)

    assert result == blog_text
    assert result_stats.total_calls == 4
    assert len(calls) == 4
    # Stage order: mine first, then gem-guided write, then per-gem verification.
    assert "gem miner" in str(calls[0]["system"])
    assert "study-note writer" in str(calls[1]["system"])
    assert all("gem verifier" in str(c["system"]) for c in calls[2:])
    # Gem-guided writing: the writer sees the chunk's gems as a must-include checklist.
    assert "MUST-INCLUDE" in str(calls[1]["user"])
    assert "Savings buildup leads to risk chasing." in str(calls[1]["user"])
    assert (
        "If the giants do not deliver, the pipeline becomes a drain."
        in str(calls[1]["user"])
    )
    # Merged gem list lands with ranks normalized.
    assert [g.id for g in gems] == ["g001", "g002"]
    assert [g.rank for g in gems] == [1, 2]
    assert gems[0].type is GemType.CAUSAL_LAW


def test_extract_from_cache_caps_gems_at_per_chunk_cap() -> None:
    cache = _make_cache()
    stats = CacheStats(total_calls=1)
    blog_text = "blog"
    big_payload = {
        "gems": [
            {
                "id": f"g{i}",
                "type": "other",
                "quote": f"Gem number {i} quote.",
                "timestamp_start": float(i) * 300.0,
                "why": "why",
                "rank": i,
            }
            for i in range(1, 16)  # 15 gems, above the cap
        ]
    }

    with patch(
        "yt_ingest.extract.chat_json",
        side_effect=_dispatch_mine_write(big_payload, blog_text, stats),
    ):
        _, gems, _report, _ = extract_from_cache(cache)

    assert len(gems) == 12
    assert [g.rank for g in gems] == list(range(1, 13))


def test_extract_from_cache_merges_multiple_chunks_and_gems() -> None:
    cache = _make_cache()
    stats = CacheStats(hit_tokens=10, miss_tokens=10, total_calls=1)
    draft = "Draft content."
    merged = "## Merged Blog\n\nUnified content."
    chunk2_payload = {
        "gems": [
            {
                "id": "g1",
                "type": "causal_law",
                "quote": "Savings buildup leads to risk chasing.",
                "timestamp_start": 65.0,
                "why": "Core causal law.",
                "rank": 1,
            }
        ]
    }

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        system = str(kwargs.get("system", ""))
        if "gem miner" in system:
            user = str(kwargs.get("user", ""))
            if "chunk 1/2" in user:
                return _GEMS_PAYLOAD, stats
            return chunk2_payload, stats
        if "study-note writer" in system:
            return {"blog_post": draft}, stats
        return {"blog_post": merged}, stats

    with (
        patch(
            "yt_ingest.extract._chunk_transcript", return_value=["chunk1", "chunk2"]
        ),
        patch("yt_ingest.extract.chat_json", side_effect=fake_chat_json),
    ):
        result, gems, _report, _ = extract_from_cache(cache)

    assert result == merged
    # 2 chunks x (mine + write) + 1 merge call
    # Overlapping-chunk duplicate gem deduped deterministically (no LLM).
    assert [g.quote for g in gems] == [
        "Savings buildup leads to risk chasing.",
        "If the giants do not deliver, the pipeline becomes a drain.",
    ]
    assert [g.rank for g in gems] == [1, 2]


# --- gems artifact paths ---

def test_gems_path_matches_persistence_contract() -> None:
    assert gems_path(Path("notes"), "dQw4w9WgXcQ") == (
        Path("notes") / "dQw4w9WgXcQ.gems.json"
    )


def test_gems_file_serializes_schema(tmp_path: Path) -> None:
    gems = [
        Gem(
            id="g001",
            type=GemType.DEFINITIONAL_ASIDE,
            quote="Rare earths are a group of metals.",
            timestamp_start=45.0,
            why="Sets up the refining argument.",
            rank=1,
        )
    ]
    out = tmp_path / "dQw4w9WgXcQ.gems.json"
    out.write_text(json.dumps([g.model_dump(mode="json") for g in gems], indent=2))

    loaded = [Gem.model_validate(item) for item in json.loads(out.read_text())]
    assert loaded == gems


# --- _checksum ---

def test_checksum_same_for_same_cache() -> None:
    cache = _make_cache()
    assert _checksum(cache) == _checksum(cache)


def test_checksum_differs_for_different_caches() -> None:
    c1 = _make_cache(3)
    c2 = _make_cache(5)
    assert _checksum(c1) != _checksum(c2)
