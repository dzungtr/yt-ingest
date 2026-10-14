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
    VerdictStatus,
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
        if "gem verifier" in system:
            return {
                "verdict": "covered",
                "reason": "present",
                "note_span": "Unified content.",
            }, stats
        return {"blog_post": merged}, stats

    with (
        patch(
            "yt_ingest.extract._chunk_transcript", return_value=["chunk1", "chunk2"]
        ),
        patch("yt_ingest.extract.chat_json", side_effect=fake_chat_json),
    ):
        result, gems, report, _ = extract_from_cache(cache)

    assert result == merged
    # Overlapping-chunk duplicate gem deduped deterministically (no LLM).
    assert [g.quote for g in gems] == [
        "Savings buildup leads to risk chasing.",
        "If the giants do not deliver, the pipeline becomes a drain.",
    ]
    assert [g.rank for g in gems] == [1, 2]
    # Post-merge verification runs against the merged note (issue #19).
    assert report is not None
    assert not report.patch_fired
    assert all(v.verdict is VerdictStatus.COVERED for v in report.final)


def _dispatch_multi_chunk(
    gem_payloads: dict[int, dict[str, object]],
    draft: str,
    merged: str,
    verdicts: dict[str, tuple[dict[str, object], ...]],
    stats: CacheStats,
    patched: str = "",
):
    """chat_json test double for the multi-chunk flow, dispatching on the
    system prompt (mine / write / merge / verify / patch).

    verdicts maps gem id -> tuple of (payload, stats) verdicts returned in
    verify-round order, so the same double serves initial and re-verify rounds.
    """
    verify_rounds: dict[str, list[tuple[dict[str, object], CacheStats]]] = {
        gem_id: [(payload, stats) for payload in payloads]
        for gem_id, payloads in verdicts.items()
    }

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        system = str(kwargs.get("system", ""))
        if "gem miner" in system:
            user = str(kwargs.get("user", ""))
            if "chunk 1/2" in user:
                return gem_payloads[1], stats
            return gem_payloads[2], stats
        if "study-note writer" in system:
            return {"blog_post": draft}, stats
        if "gem verifier" in system:
            user = str(kwargs.get("user", ""))
            for gem_id, round_calls in verify_rounds.items():
                if f"id {gem_id}" in user and round_calls:
                    return round_calls.pop(0)
            # Default: covered (used when no verdict rounds are scripted).
            return {
                "verdict": "covered",
                "reason": "present",
                "note_span": "Unified content.",
            }, stats
        if "insert-only patch" in system:
            return {"blog_post": patched or merged + " patched"}, stats
        return {"blog_post": merged}, stats

    return fake_chat_json


def test_extract_from_cache_multi_chunk_all_covered_no_patch() -> None:
    cache = _make_cache()
    stats = CacheStats(hit_tokens=10, miss_tokens=10, total_calls=1)
    calls: list[dict[str, object]] = []
    fake = _dispatch_multi_chunk(
        {1: _GEMS_PAYLOAD, 2: {"gems": []}},
        "Draft content.",
        "## Merged Blog\n\nUnified content.",
        {},
        stats,
    )

    def wrapper(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        calls.append(dict(kwargs))
        return fake(**kwargs)

    with (
        patch(
            "yt_ingest.extract._chunk_transcript", return_value=["chunk1", "chunk2"]
        ),
        patch("yt_ingest.extract.chat_json", side_effect=wrapper),
    ):
        result, gems, report, _ = extract_from_cache(cache)

    # Stage order: mine, write, mine, write, merge, verify x gems — no patch.
    assert "gem miner" in str(calls[0]["system"])
    assert "study-note writer" in str(calls[1]["system"])
    assert "gem miner" in str(calls[2]["system"])
    assert "study-note writer" in str(calls[3]["system"])
    assert "editor" in str(calls[4]["system"])
    assert all("gem verifier" in str(c["system"]) for c in calls[5:])
    assert len(calls) == 7  # 2 mine + 2 write + 1 merge + 2 verify
    assert result == "## Merged Blog\n\nUnified content."
    assert report is not None
    assert not report.patch_fired
    assert [v.gem_id for v in report.final] == ["g001", "g002"]
    assert [g.id for g in gems] == ["g001", "g002"]


def test_extract_from_cache_multi_chunk_writer_stays_locally_grounded() -> None:
    """Each writer call sees only its own chunk and that chunk's gem checklist."""
    cache = _make_cache()
    stats = CacheStats(total_calls=1)
    write_users: list[str] = []
    fake = _dispatch_multi_chunk(
        {1: _GEMS_PAYLOAD, 2: {"gems": []}},
        "Draft content.",
        "Merged.",
        {},
        stats,
    )

    def wrapper(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        if "study-note writer" in str(kwargs.get("system", "")):
            write_users.append(str(kwargs.get("user", "")))
        return fake(**kwargs)

    with (
        patch(
            "yt_ingest.extract._chunk_transcript",
            return_value=["chunk one text", "chunk two text"],
        ),
        patch("yt_ingest.extract.chat_json", side_effect=wrapper),
    ):
        extract_from_cache(cache)

    assert len(write_users) == 2
    assert "chunk 1/2" in write_users[0] and "chunk one text" in write_users[0]
    assert "chunk two text" not in write_users[0]
    assert "MUST-INCLUDE" in write_users[0]
    assert "Savings buildup leads to risk chasing." in write_users[0]
    assert "chunk 2/2" in write_users[1] and "chunk two text" in write_users[1]
    assert "chunk one text" not in write_users[1]
    assert "Savings buildup leads to risk chasing." not in write_users[1]


def test_extract_from_cache_multi_chunk_missing_fires_single_patch() -> None:
    """Verify the merged note -> one gem missing -> one patch -> re-verify all
    gems once. One retry budget for the whole video (issue #19)."""
    cache = _make_cache()
    stats = CacheStats(total_calls=1)
    patched = "## Merged Blog\n\nUnified content. Plus the missing idea."
    calls: list[dict[str, object]] = []
    fake = _dispatch_multi_chunk(
        {1: _GEMS_PAYLOAD, 2: {"gems": []}},
        "Draft content.",
        "## Merged Blog\n\nUnified content.",
        {
            "g001": (
                {"verdict": "missing", "reason": "not in note", "note_span": ""},
                {
                    "verdict": "covered",
                    "reason": "folded in",
                    "note_span": "Plus the missing idea.",
                },
            ),
            "g002": (
                {
                    "verdict": "covered",
                    "reason": "present",
                    "note_span": "Unified content.",
                },
                {
                    "verdict": "covered",
                    "reason": "still present",
                    "note_span": "Unified content.",
                },
            ),
        },
        stats,
        patched=patched,
    )

    def wrapper(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        calls.append(dict(kwargs))
        return fake(**kwargs)

    with (
        patch(
            "yt_ingest.extract._chunk_transcript", return_value=["chunk1", "chunk2"]
        ),
        patch("yt_ingest.extract.chat_json", side_effect=wrapper),
    ):
        result, _gems, report, _ = extract_from_cache(cache)

    assert result == patched
    assert report is not None
    assert report.patch_fired
    assert [v.verdict for v in report.final] == [
        VerdictStatus.COVERED,
        VerdictStatus.COVERED,
    ]
    # 2 mine + 2 write + 1 merge + 2 verify + 1 patch + 2 re-verify.
    assert len(calls) == 10
    patch_calls = [c for c in calls if "insert-only patch" in str(c["system"])]
    assert len(patch_calls) == 1
    # The patch prompt receives the merged note and the missing gem only.
    assert "## Merged Blog" in str(patch_calls[0]["user"])
    assert "g002" not in str(patch_calls[0]["user"])


def test_extract_from_cache_multi_chunk_end_to_end_over_6k_tokens() -> None:
    """Real chunking on a >6k-token transcript: mine x2 -> write x2 ->
    gem-merge -> draft-merge -> verify x2 -> no patch (all covered)."""
    cache = _make_cache(n_segments=50)
    # ~140 tokens per line, so 50 segments (~7k tokens) split into 2 chunks.
    filler = "the speaker explains the mechanism in careful detail step by step " * 12
    cache = cache.model_copy(
        update={
            "segments": [
                seg.model_copy(update={"text": filler}) for seg in cache.segments
            ]
        }
    )
    stats = CacheStats(total_calls=1)
    systems: list[str] = []
    fake = _dispatch_multi_chunk(
        {1: _GEMS_PAYLOAD, 2: {"gems": []}},
        "Draft content.",
        "## Merged Blog\n\nUnified content.",
        {},
        stats,
    )

    def wrapper(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        systems.append(str(kwargs.get("system", "")))
        return fake(**kwargs)

    with patch("yt_ingest.extract.chat_json", side_effect=wrapper):
        result, gems, report, _ = extract_from_cache(cache)

    assert result == "## Merged Blog\n\nUnified content."
    assert [g.id for g in gems] == ["g001", "g002"]
    assert report is not None and not report.patch_fired
    mine = [s for s in systems if "gem miner" in s]
    write = [s for s in systems if "study-note writer" in s]
    verify = [s for s in systems if "gem verifier" in s]
    merge = [s for s in systems if "study-note editor" in s]
    assert len(mine) == 2 and len(write) == 2 and len(verify) == 2 and len(merge) == 1


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
