from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from yt_ingest.cli import app
from yt_ingest.models import (
    TranscriptCache,
    TranscriptSegment,
    TranscriptSource,
)
from datetime import datetime

runner = CliRunner()


def _make_cache(video_id: str = "dQw4w9WgXcQ") -> TranscriptCache:
    return TranscriptCache(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
        source=TranscriptSource.YOUTUBE_TRANSCRIPT_API,
        fetched_at=datetime(2026, 1, 1),
        title="Test",
        channel="Test Channel",
        duration_seconds=213,
        segments=[TranscriptSegment(text="hello", start=0.0, duration=1.0)],
    )


def _mock_config(tmp_path: Path) -> MagicMock:
    cfg = MagicMock()
    cfg.transcripts_dir = tmp_path / "transcripts"
    cfg.transcripts_dir.mkdir()
    return cfg


def test_fetch_new_video(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    cache = _make_cache()

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
    ):
        result = runner.invoke(app, ["fetch", str(urls_file)])

    assert result.exit_code == 0
    assert "OK" in result.output
    assert "dQw4w9WgXcQ" in result.output


def test_fetch_cache_hit(tmp_path: Path) -> None:
    cfg = _mock_config(tmp_path)
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    # Pre-write cached file
    cache = _make_cache()
    (cfg.transcripts_dir / "dQw4w9WgXcQ.json").write_text(cache.model_dump_json())

    with patch("yt_ingest.cli.get_config", return_value=cfg):
        result = runner.invoke(app, ["fetch", str(urls_file)])

    assert result.exit_code == 0
    assert "HIT" in result.output


def test_fetch_failure_exits_nonzero(tmp_path: Path) -> None:
    from yt_ingest.fetchers.base import FetchError

    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", side_effect=FetchError("all failed")),
    ):
        result = runner.invoke(app, ["fetch", str(urls_file)])

    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_fetch_empty_file_exits_nonzero(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("# just a comment\n")

    with patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)):
        result = runner.invoke(app, ["fetch", str(urls_file)])

    assert result.exit_code == 1
