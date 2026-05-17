from __future__ import annotations
from pathlib import Path
from unittest.mock import patch

import pytest

from yt_ingest.llm import CacheStats
from yt_ingest.synthesize import _notes_checksum, synthesize_notes


def _write_note(notes_dir: Path, video_id: str, content: str = "Some note content.") -> Path:
    p = notes_dir / f"{video_id}.md"
    p.write_text(content)
    return p


_RAW_RESPONSE = {
    "synthesis": "These videos share common insights about productivity.",
    "cross_cutting_themes": ["Focus", "Deep work"],
    "contradictions": [{"topic": "Rest", "views": ["Rest is overrated", "Rest is essential"]}],
    "complementary_ideas": ["Both emphasize eliminating distractions"],
    "recommended_watch_order": ["abc1234567a", "xyz9876543b"],
    "further_questions": ["How do these apply to creative work?"],
}


def test_synthesize_notes_produces_markdown(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir, "abc1234567a", "Note for video A")
    _write_note(notes_dir, "xyz9876543b", "Note for video B")

    stats = CacheStats(hit_tokens=100, miss_tokens=50, total_calls=1)

    with patch("yt_ingest.synthesize.chat_json", return_value=(_RAW_RESPONSE, stats)):
        markdown, result_stats = synthesize_notes(notes_dir, tmp_path / "index")

    assert "Cross-Video Synthesis" in markdown
    assert "Focus" in markdown
    assert "Deep work" in markdown
    assert result_stats.hit_tokens == 100


def test_synthesize_notes_raises_when_empty(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()

    with pytest.raises(ValueError, match="No notes found"):
        synthesize_notes(notes_dir, tmp_path / "index")


def test_notes_checksum_stable(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir, "abc1234567a")
    cs1 = _notes_checksum(notes_dir)
    cs2 = _notes_checksum(notes_dir)
    assert cs1 == cs2


def test_notes_checksum_changes_on_new_note(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    _write_note(notes_dir, "abc1234567a")
    cs1 = _notes_checksum(notes_dir)
    _write_note(notes_dir, "xyz9876543b")
    cs2 = _notes_checksum(notes_dir)
    assert cs1 != cs2
