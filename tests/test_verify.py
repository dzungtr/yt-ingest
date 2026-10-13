"""Tests for per-gem verification + bounded patch + audit artifact (issue #18).

Primary seam: `yt_ingest.extract.chat_json` patched with side_effect dispatch
on the system prompt (mine / write / verify / patch), as in test_extract.py.
Pure-function seam: direct unit tests for the deterministic ±60s excerpt
slicing and the verdict parser.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from yt_ingest.extract import (
    VerifyReport,
    _parse_verdict,
    _transcript_excerpt,
    audit_path,
    extract_from_cache,
    render_audit_md,
)
from yt_ingest.llm import CacheStats
from yt_ingest.models import (
    Gem,
    GemType,
    TranscriptCache,
    TranscriptSegment,
    TranscriptSource,
    Verdict,
    VerdictStatus,
)


def _make_cache() -> TranscriptCache:
    segments = [
        TranscriptSegment(text=f"segment {i}", start=float(i * 30), duration=5.0)
        for i in range(11)  # starts at 0, 30, 60, ... 300
    ]
    return TranscriptCache(
        video_id="dQw4w9WgXcQ",
        url="https://youtu.be/dQw4w9WgXcQ",
        source=TranscriptSource.YOUTUBE_TRANSCRIPT_API,
        fetched_at=datetime(2026, 1, 1),
        title="Test Video",
        channel="Test Channel",
        duration_seconds=310,
        segments=segments,
    )


# --- deterministic transcript excerpt (pure-function seam) ---


def test_transcript_excerpt_slices_pm60s_window_deterministically() -> None:
    cache = _make_cache()
    # Gem at 65s -> window [5s, 125s] covers segments starting 30, 60, 90, 120.
    excerpt = _transcript_excerpt(cache, 65.0)
    assert "[0:30] segment 1" in excerpt
    assert "[1:00] segment 2" in excerpt
    assert "[1:30] segment 3" in excerpt
    assert "[2:00] segment 4" in excerpt
    assert "[0:00] segment 0" not in excerpt
    assert "[2:30] segment 5" not in excerpt
    # Deterministic: identical output on repeated calls (code, not LLM-selected).
    assert excerpt == _transcript_excerpt(cache, 65.0)


def test_transcript_excerpt_empty_window_reports_no_content() -> None:
    cache = _make_cache()
    excerpt = _transcript_excerpt(cache, 10_000.0)
    assert excerpt == "(no transcript content within ±60s of the gem timestamp)"


# --- chat_json dispatch seam ---


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

_BLOG = (
    "## Note\n\nSavings buildup leads to risk chasing. "
    "If the giants do not deliver, the pipeline becomes a drain."
)


def _covered(gem_id: str) -> dict[str, str]:
    return {
        "verdict": "covered",
        "reason": "idea present in note",
        "note_span": "risk chasing",
    }


def _run_extract(rounds: list[dict[str, dict[str, str]]], patched: str):
    """Run extract_from_cache with the given verify rounds; returns
    (result, gems, report, all_calls, verify_calls). verify_rounds[i] maps gem
    id to verdict payload for verify round i (0-based)."""
    all_calls: list[dict[str, object]] = []
    verify_calls: list[dict[str, object]] = []

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        all_calls.append(dict(kwargs))
        system = str(kwargs.get("system", ""))
        user = str(kwargs.get("user", ""))
        stats = CacheStats(hit_tokens=1, miss_tokens=1, total_calls=1)
        if "gem miner" in system:
            return _GEMS_PAYLOAD, stats
        if "study-note writer" in system:
            return {"blog_post": _BLOG}, stats
        if "gem verifier" in system:
            verify_calls.append(dict(kwargs))
            gem_id = "g1" if "g001" in user else "g2"
            round = min((len(verify_calls) - 1) // 2, len(rounds) - 1)
            return rounds[round][gem_id], stats
        assert "insert-only" in system
        return {"blog_post": patched}, stats

    with patch("yt_ingest.extract.chat_json", side_effect=fake_chat_json):
        result, gems, report, stats = extract_from_cache(_make_cache())
    return result, gems, report, all_calls, verify_calls


# --- all-covered no-patch path ---


def test_all_covered_fires_no_patch() -> None:
    covered = {"g1": _covered("g1"), "g2": _covered("g2")}
    result, gems, report, all_calls, verify_calls = _run_extract(
        [covered], "SHOULD NOT BE USED"
    )

    assert result == _BLOG  # note unchanged — no patch
    assert not report.patch_fired
    assert [v.verdict for v in report.final] == [VerdictStatus.COVERED] * 2
    assert all(v.note_span for v in report.final)
    # 1 mine + 1 write + 2 per-gem verify calls; no patch call.
    assert len(all_calls) == 4
    assert not any("insert-only" in str(c["system"]) for c in all_calls)


def test_verification_is_one_call_per_gem_never_batched() -> None:
    covered = {"g1": _covered("g1"), "g2": _covered("g2")}
    _, _, _, _, verify_calls = _run_extract([covered], "")
    assert len(verify_calls) == 2
    # Each call carries exactly one gem in its tail.
    for call in verify_calls:
        user = str(call["user"])
        assert ("g001" in user) != ("g002" in user)


# --- prefix-caching prompt structure ---


def test_verify_prompts_share_note_prefix_with_per_gem_tail() -> None:
    covered = {"g1": _covered("g1"), "g2": _covered("g2")}
    _, _, _, _, verify_calls = _run_extract([covered], "")
    marker = "Gem to verify"
    prefixes = [str(c["user"]).split(marker)[0] for c in verify_calls]
    assert prefixes[0] == prefixes[1]  # stable note prefix for provider caching
    tails = [str(c["user"]).split(marker)[1] for c in verify_calls]
    assert tails[0] != tails[1]  # per-gem tail last
    # The note block comes first, before any gem details.
    assert "## Note" in prefixes[0]


def test_verify_user_includes_deterministic_transcript_excerpt() -> None:
    covered = {"g1": _covered("g1"), "g2": _covered("g2")}
    _, _, _, _, verify_calls = _run_extract([covered], "")
    users = [str(c["user"]) for c in verify_calls]
    # g001 sits at 61s -> excerpt window [1s, 121s]
    assert "[1:00] segment 2" in users[0]
    assert "[2:30] segment 5" not in users[0]
    # g002 sits at 120s -> window [60s, 180s]
    assert "[1:30] segment 3" in users[1]
    assert "[0:30] segment 1" not in users[1]


# --- missing -> patch -> re-verify path ---


def test_missing_gem_fires_single_patch_then_full_reverify() -> None:
    rounds = [
        {
            "g1": {"verdict": "missing", "reason": "law absent from note"},
            "g2": _covered("g2"),
        },
        {"g1": _covered("g1"), "g2": _covered("g2")},
    ]
    patched = _BLOG + "\n\nPatched addition."
    result, gems, report, all_calls, verify_calls = _run_extract(rounds, patched)

    assert report.patch_fired
    assert result == patched  # patched note becomes the final note
    # 1 mine + 1 write + 2 verify + 1 patch + 2 re-verify = 7 calls.
    assert len(all_calls) == 7
    assert len(verify_calls) == 4  # all gems re-verified once after the patch
    assert [v.verdict for v in report.final] == [VerdictStatus.COVERED] * 2
    # No second patch: exactly one insert-only call.
    assert sum(1 for c in all_calls if "insert-only" in str(c["system"])) == 1


def test_patch_receives_only_missing_gems_insert_only_rules() -> None:
    rounds = [
        {"g1": {"verdict": "missing", "reason": "law absent"}, "g2": _covered("g2")},
        {"g1": _covered("g1"), "g2": _covered("g2")},
    ]
    _, _, _, all_calls, _ = _run_extract(rounds, "Patched.")

    patch_calls = [c for c in all_calls if "insert-only" in str(c["system"])]
    assert len(patch_calls) == 1
    tail = str(patch_calls[0]["user"]).split("Missing gems to fold in")[1]
    assert "Savings buildup leads to risk chasing." in tail  # missing gem
    assert "If the giants do not deliver" not in tail  # covered gem stays out
    system = str(patch_calls[0]["system"])
    assert "insert-only" in system
    assert "do not alter" in system.lower() or "without altering" in system.lower()


# --- residual-missing persistence path ---


def test_residual_missing_recorded_after_reverify() -> None:
    rounds = [
        {"g1": {"verdict": "missing", "reason": "law absent"}, "g2": _covered("g2")},
        {
            "g1": {"verdict": "missing", "reason": "still absent after patch"},
            "g2": _covered("g2"),
        },
    ]
    _, gems, report, _, _ = _run_extract(rounds, "Patched.")

    assert report.patch_fired
    final_by_id = {v.gem_id: v for v in report.final}
    assert final_by_id["g001"].verdict is VerdictStatus.MISSING
    assert final_by_id["g001"].reason == "still absent after patch"
    audit = render_audit_md("dQw4w9WgXcQ", gems, report)
    assert "still absent after patch" in audit
    assert "g001" in audit


# --- inapplicable never triggers retry ---


def test_inapplicable_never_triggers_patch() -> None:
    rounds = [
        {
            "g1": {"verdict": "inapplicable", "reason": "gem is channel promo chatter"},
            "g2": _covered("g2"),
        },
    ]
    result, gems, report, all_calls, _ = _run_extract(rounds, "SHOULD NOT BE USED")

    assert not report.patch_fired
    assert result == _BLOG
    final_by_id = {v.gem_id: v for v in report.final}
    assert final_by_id["g001"].verdict is VerdictStatus.INAPPLICABLE
    assert not any("insert-only" in str(c["system"]) for c in all_calls)


# --- verdict parsing (pure-function seam) ---


def test_verdict_parser_rejects_unknown_verdict() -> None:
    with pytest.raises(ValueError, match="invalid verdict"):
        _parse_verdict("g1", {"verdict": "sort_of", "reason": "x", "note_span": ""})


def test_verdict_parser_requires_note_span_for_covered() -> None:
    with pytest.raises(ValueError, match="note span"):
        _parse_verdict("g1", {"verdict": "covered", "reason": "x", "note_span": ""})


# --- audit artifact ---


def test_audit_path_matches_persistence_contract() -> None:
    assert audit_path(Path("notes"), "dQw4w9WgXcQ") == (
        Path("notes") / "dQw4w9WgXcQ.audit.md"
    )


def test_audit_markdown_contains_verdicts_reasons_spans_history_residuals() -> None:
    rounds = [
        {"g1": {"verdict": "missing", "reason": "law absent"}, "g2": _covered("g2")},
        {
            "g1": {"verdict": "missing", "reason": "still absent after patch"},
            "g2": _covered("g2"),
        },
    ]
    _, gems, report, _, _ = _run_extract(rounds, "Patched.")
    audit = render_audit_md("dQw4w9WgXcQ", gems, report)

    assert "dQw4w9WgXcQ" in audit
    assert "g001" in audit and "g002" in audit
    assert "covered" in audit and "missing" in audit
    assert "risk chasing" in audit  # note span citation
    assert "law absent" in audit  # initial reason in retry history
    assert "patch" in audit.lower()  # retry history mentions the patch
    assert "still absent after patch" in audit  # residual recorded, not dropped


def test_audit_markdown_no_patch_history() -> None:
    covered = {"g1": _covered("g1"), "g2": _covered("g2")}
    _, gems, report, _, _ = _run_extract([covered], "")
    audit = render_audit_md("dQw4w9WgXcQ", gems, report)
    assert "no patch" in audit.lower()


# --- audit table-cell escaping + verdict reason contract ---


def test_audit_markdown_escapes_pipes_and_newlines_in_cells() -> None:
    gem = Gem(
        id="g001",
        type=GemType.CAUSAL_LAW,
        quote="Quote.",
        timestamp_start=10.0,
        why="why",
        rank=1,
    )
    verdict = Verdict(
        gem_id="g001",
        verdict=VerdictStatus.INAPPLICABLE,
        reason="channel | promo\nchatter",
        note_span="",
    )
    report = VerifyReport(initial=[verdict], final=[verdict])

    audit = render_audit_md("dQw4w9WgXcQ", [gem], report)

    row = [ln for ln in audit.splitlines() if ln.startswith("| g001")][0]
    # Raw pipe escaped so the cell does not split the table row.
    assert "channel \\| promo chatter" in row
    assert row.count("\\|") == 1
    # Row structure stays a 5-column table (unescaped pipes only).
    assert row.replace("\\|", "").count("|") == 6


def test_verdict_parser_requires_reason_for_missing_and_inapplicable() -> None:
    with pytest.raises(ValueError, match="one-line reason"):
        _parse_verdict("g1", {"verdict": "missing", "reason": "", "note_span": ""})
    with pytest.raises(ValueError, match="one-line reason"):
        _parse_verdict(
            "g1", {"verdict": "inapplicable", "reason": "   ", "note_span": ""}
        )


# --- multi-chunk path: no verification yet, no report ---


def test_multi_chunk_returns_no_report_and_never_verifies() -> None:
    def fake_chat_json(**kwargs: object) -> tuple[dict[str, Any], CacheStats]:
        system = str(kwargs.get("system", ""))
        stats = CacheStats(hit_tokens=1, miss_tokens=1, total_calls=1)
        if "gem miner" in system:
            return _GEMS_PAYLOAD, stats
        if "gem verifier" in system:
            raise AssertionError("multi-chunk path must not verify gems")
        if "study-note writer" in system:
            return {"blog_post": "draft"}, stats
        assert "study-note editor" in system  # merge stage
        return {"blog_post": "merged"}, stats

    with (
        patch(
            "yt_ingest.extract._chunk_transcript", return_value=["a", "b"]
        ),
        patch("yt_ingest.extract.chat_json", side_effect=fake_chat_json),
    ):
        result, gems, report, _ = extract_from_cache(_make_cache())

    assert result == "merged"
    # No per-gem verification on multi-chunk yet (issue #19): report is None
    # so the CLI writes no audit artifact for this unverified note.
    assert report is None
    assert [g.id for g in gems] == ["g001", "g002"]
