from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from yt_ingest.retrieval import Chunk, _chunk_note, build_index, search


# --- _chunk_note ---

def test_chunk_note_single_chunk(tmp_path: Path) -> None:
    note = tmp_path / "abc1234567a.md"
    note.write_text("word " * 100)
    chunks = _chunk_note(note, "abc1234567a", chunk_size=500)
    assert len(chunks) == 1
    assert chunks[0].video_id == "abc1234567a"


def test_chunk_note_multiple_chunks(tmp_path: Path) -> None:
    note = tmp_path / "abc1234567a.md"
    note.write_text("word " * 1200)
    chunks = _chunk_note(note, "abc1234567a", chunk_size=500)
    assert len(chunks) == 3


def test_chunk_note_empty_file(tmp_path: Path) -> None:
    note = tmp_path / "abc1234567a.md"
    note.write_text("")
    chunks = _chunk_note(note, "abc1234567a")
    assert len(chunks) == 1
    assert chunks[0].text == ""


# --- build_index ---

def _mock_model(dim: int = 4) -> MagicMock:
    model = MagicMock()
    model.encode.side_effect = lambda texts, **kw: np.random.rand(len(texts), dim).astype(np.float32)
    return model


def test_build_index_returns_zero_when_no_notes(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    index_dir = tmp_path / "index"
    n = build_index(notes_dir, index_dir)
    assert n == 0


def test_build_index_creates_files(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "abc1234567a.md").write_text("hello world this is a test note")
    index_dir = tmp_path / "index"

    with patch("yt_ingest.retrieval._load_model", return_value=_mock_model()):
        n = build_index(notes_dir, index_dir)

    assert n >= 1
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "meta.json").exists()


# --- search ---

def test_search_raises_when_no_index(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Index not found"):
        search("what is this?", tmp_path)


def test_search_returns_chunks(tmp_path: Path) -> None:
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    (notes_dir / "abc1234567a.md").write_text("hello world this is a test note with some content")
    index_dir = tmp_path / "index"

    with patch("yt_ingest.retrieval._load_model", return_value=_mock_model()):
        build_index(notes_dir, index_dir)
        results = search("hello world", index_dir, top_k=1)

    assert len(results) == 1
    assert isinstance(results[0], Chunk)
    assert results[0].video_id == "abc1234567a"
