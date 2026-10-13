"""Pure-function tests for the gem data contract and deterministic merge-dedupe.

Issue #17: gem schema (id, type, quote, timestamp_start, why, rank) and the
deterministic gem merge helper consumed by the extraction pipeline.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from yt_ingest.extract import merge_gems
from yt_ingest.models import Gem, GemType


def _gem(**overrides: object) -> Gem:
    defaults: dict[str, object] = {
        "id": "g1",
        "type": GemType.CAUSAL_LAW,
        "quote": "Savings buildup leads to risk chasing.",
        "timestamp_start": 61.0,
        "why": "Core causal law of the video.",
        "rank": 1,
    }
    defaults.update(overrides)
    return Gem.model_validate(defaults)


# --- Gem schema ---


def test_gem_type_has_five_loss_classes_plus_other() -> None:
    assert {t.value for t in GemType} == {
        "causal_law",
        "conditional_bet",
        "definitional_aside",
        "embedded_quote",
        "closing_prediction",
        "other",
    }


def test_gem_has_full_data_contract() -> None:
    gem = _gem()
    assert gem.id == "g1"
    assert gem.type is GemType.CAUSAL_LAW
    assert gem.quote == "Savings buildup leads to risk chasing."
    assert gem.timestamp_start == 61.0
    assert gem.why == "Core causal law of the video."
    assert gem.rank == 1


def test_gem_rejects_unknown_type() -> None:
    with pytest.raises(ValidationError):
        Gem.model_validate(
            {
                "id": "g1",
                "type": "highlight",
                "quote": "quote",
                "timestamp_start": 0.0,
                "why": "why",
                "rank": 1,
            }
        )


def test_gem_rejects_missing_fields() -> None:
    with pytest.raises(ValidationError):
        Gem.model_validate({"id": "g1", "quote": "quote"})


def test_gem_type_is_metadata_only_roundtrip() -> None:
    """Type must not alter shape or validation of the other fields."""
    a = _gem(type=GemType.OTHER)
    b = _gem(type=GemType.CLOSING_PREDICTION)
    a_dump = a.model_dump()
    b_dump = b.model_dump()
    a_dump["type"] = b_dump["type"] = None
    assert a_dump.keys() == b_dump.keys()


# --- merge_gems (deterministic dedupe, no LLM) ---


def test_merge_gems_dedupes_same_quote_across_chunks() -> None:
    chunk1 = [_gem(id="g1", quote="The tariff is 67 percent.", timestamp_start=200.0)]
    chunk2 = [_gem(id="g1", quote="The tariff is 67 percent.", timestamp_start=210.0)]

    merged = merge_gems([chunk1, chunk2])

    assert len(merged) == 1


def test_merge_gems_dedupes_identical_quotes_across_chunks() -> None:
    chunk1 = [
        _gem(id="g1", rank=1),
        _gem(
            id="g2",
            type=GemType.CONDITIONAL_BET,
            quote="If the giants do not deliver, the pipeline becomes a drain.",
            timestamp_start=120.0,
            why="Bet condition.",
            rank=2,
        ),
    ]
    chunk2 = [
        _gem(id="g1", rank=1),
        _gem(
            id="g2",
            type=GemType.CONDITIONAL_BET,
            quote="If the Giants Don't Deliver, the pipeline becomes a drain.",
            timestamp_start=150.0,
            why="Bet condition.",
            rank=2,
        ),
    ]

    merged = merge_gems([chunk1, chunk2])

    assert len(merged) == 2


def test_merge_gems_dedupes_near_duplicate_quotes_within_60s() -> None:
    chunk1 = [
        _gem(
            id="g1",
            quote="Savings buildup leads to risk chasing.",
            timestamp_start=100.0,
        )
    ]
    chunk2 = [
        _gem(
            id="g1",
            quote="Savings buildup leads to a risk chasing behaviour.",
            timestamp_start=130.0,
        )
    ]

    merged = merge_gems([chunk1, chunk2])

    assert len(merged) == 1


def test_merge_gems_keeps_similar_quotes_far_apart_in_time() -> None:
    chunk1 = [
        _gem(
            id="g1",
            quote="Savings buildup leads to risk chasing.",
            timestamp_start=100.0,
        )
    ]
    chunk2 = [
        _gem(
            id="g1",
            quote="Savings buildup leads to risk chasing.",
            timestamp_start=900.0,
        )
    ]

    merged = merge_gems([chunk1, chunk2])

    assert len(merged) == 2


def test_merge_gems_keeps_distinct_gems() -> None:
    chunk1 = [
        _gem(
            id="g1",
            quote="Savings buildup leads to risk chasing.",
            timestamp_start=100.0,
        ),
        _gem(
            id="g2",
            type=GemType.EMBEDDED_QUOTE,
            quote="The tariff is 67 percent.",
            timestamp_start=200.0,
            rank=2,
        ),
    ]
    chunk2 = [
        _gem(
            id="g1",
            type=GemType.CLOSING_PREDICTION,
            quote="Rates will stay higher for longer.",
            timestamp_start=600.0,
        ),
    ]

    merged = merge_gems([chunk1, chunk2])

    assert len(merged) == 3
    assert {m.quote for m in merged} == {
        "Savings buildup leads to risk chasing.",
        "The tariff is 67 percent.",
        "Rates will stay higher for longer.",
    }


def test_merge_gems_reassigns_ids_and_ranks_deterministically() -> None:
    chunk1 = [
        _gem(id="chunk2-g9", rank=1),
        _gem(
            id="chunk1-g1", quote="Second gem.", timestamp_start=90.0, why="w", rank=2
        ),
    ]
    chunk2 = [_gem(id="g1", quote="Third gem.", timestamp_start=300.0, why="w")]

    merged = merge_gems([chunk1, chunk2])

    assert [m.id for m in merged] == ["g001", "g002", "g003"]
    assert [m.rank for m in merged] == [1, 2, 3]


def test_merge_gems_deterministic_for_same_input() -> None:
    chunks = [
        [
            _gem(id="g1", rank=1),
            _gem(
                id="g2",
                type=GemType.OTHER,
                quote="Another gem.",
                timestamp_start=200.0,
                why="w",
                rank=2,
            ),
        ],
        [
            _gem(
                id="g1",
                type=GemType.OTHER,
                quote="Another gem.",
                timestamp_start=200.0,
                why="w",
            )
        ],
    ]
    assert merge_gems(chunks) == merge_gems(chunks)


def test_merge_gems_handles_empty_input() -> None:
    assert merge_gems([]) == []
    assert merge_gems([[], []]) == []
