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


# --- --file mode tests ---

def test_fetch_new_video_via_file(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    cache = _make_cache()

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
    ):
        result = runner.invoke(app, ["fetch", "--file", str(urls_file)])

    assert result.exit_code == 0
    assert "OK" in result.output
    assert "dQw4w9WgXcQ" in result.output


def test_fetch_cache_hit_via_file(tmp_path: Path) -> None:
    cfg = _mock_config(tmp_path)
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    # Pre-write cached file
    cache = _make_cache()
    (cfg.transcripts_dir / "dQw4w9WgXcQ.json").write_text(cache.model_dump_json())

    with patch("yt_ingest.cli.get_config", return_value=cfg):
        result = runner.invoke(app, ["fetch", "--file", str(urls_file)])

    assert result.exit_code == 0
    assert "HIT" in result.output


def test_fetch_failure_exits_nonzero_via_file(tmp_path: Path) -> None:
    from yt_ingest.fetchers.base import FetchError

    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", side_effect=FetchError("all failed")),
    ):
        result = runner.invoke(app, ["fetch", "--file", str(urls_file)])

    assert result.exit_code == 1
    assert "FAIL" in result.output


def test_fetch_empty_file_exits_nonzero(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("# just a comment\n")

    with patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)):
        result = runner.invoke(app, ["fetch", "--file", str(urls_file)])

    assert result.exit_code == 1


# --- positional URL mode tests ---

def test_fetch_new_video_via_url_arg(tmp_path: Path) -> None:
    cache = _make_cache()

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
    ):
        result = runner.invoke(app, ["fetch", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert "[0:00] hello" in result.output


def test_fetch_cache_hit_via_url_arg(tmp_path: Path) -> None:
    cfg = _mock_config(tmp_path)

    cache = _make_cache()
    (cfg.transcripts_dir / "dQw4w9WgXcQ.json").write_text(cache.model_dump_json())

    with patch("yt_ingest.cli.get_config", return_value=cfg):
        result = runner.invoke(app, ["fetch", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert "[0:00] hello" in result.output


def test_fetch_failure_exits_nonzero_via_url_arg(tmp_path: Path) -> None:
    from yt_ingest.fetchers.base import FetchError

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", side_effect=FetchError("all failed")),
    ):
        result = runner.invoke(app, ["fetch", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 1
    assert "Error" in result.stderr


def test_fetch_url_arg_invalid_url(tmp_path: Path) -> None:
    with patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)):
        result = runner.invoke(app, ["fetch", "not-a-valid-url"])

    assert result.exit_code == 1
    assert "Error" in result.stderr


# --- --agent (no-op on fetch) ---

def test_fetch_agent_flag_prints_transcript(tmp_path: Path) -> None:
    cache = _make_cache()

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
    ):
        result = runner.invoke(app, ["fetch", "https://youtu.be/dQw4w9WgXcQ", "--agent"])

    assert result.exit_code == 0
    assert "[0:00] hello" in result.output


# --- --output ---

def test_fetch_output_flag_writes_file(tmp_path: Path) -> None:
    cache = _make_cache()
    out = tmp_path / "transcript.txt"

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
    ):
        result = runner.invoke(
            app, ["fetch", "https://youtu.be/dQw4w9WgXcQ", "--output", str(out)]
        )

    assert result.exit_code == 0
    assert result.output == "" or result.output == "\n"
    assert out.read_text() == "[0:00] hello"


# --- mutual exclusion / missing arg tests ---

def test_fetch_no_args_exits_nonzero() -> None:
    result = runner.invoke(app, ["fetch"])
    assert result.exit_code == 1


def test_fetch_both_url_and_file_exits_nonzero(tmp_path: Path) -> None:
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    result = runner.invoke(
        app,
        ["fetch", "https://youtu.be/dQw4w9WgXcQ", "--file", str(urls_file)],
    )
    assert result.exit_code == 1


# --- run command single-URL tests ---

def test_run_url_prints_study_note(tmp_path: Path) -> None:
    cache = _make_cache()
    mock_stats = MagicMock()
    mock_stats.hit_tokens = 0
    mock_stats.miss_tokens = 100

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
        patch("yt_ingest.cli.extract_from_cache", return_value=("# Study Note", mock_stats)),
    ):
        result = runner.invoke(app, ["run", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 0
    assert "# Study Note" in result.output


def test_run_url_agent_prints_transcript_no_llm(tmp_path: Path) -> None:
    cache = _make_cache()

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
        patch("yt_ingest.cli.extract_from_cache") as mock_extract,
    ):
        result = runner.invoke(
            app, ["run", "https://youtu.be/dQw4w9WgXcQ", "--agent"]
        )

    assert result.exit_code == 0
    assert "[0:00] hello" in result.output
    mock_extract.assert_not_called()


def test_run_url_output_writes_study_note(tmp_path: Path) -> None:
    cache = _make_cache()
    mock_stats = MagicMock()
    mock_stats.hit_tokens = 0
    mock_stats.miss_tokens = 100
    out = tmp_path / "note.md"

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
        patch("yt_ingest.cli.extract_from_cache", return_value=("# Study Note", mock_stats)),
    ):
        result = runner.invoke(
            app, ["run", "https://youtu.be/dQw4w9WgXcQ", "--output", str(out)]
        )

    assert result.exit_code == 0
    assert result.output == "" or result.output == "\n"
    assert "# Study Note" in out.read_text()


def test_run_url_agent_output_writes_transcript(tmp_path: Path) -> None:
    cache = _make_cache()
    out = tmp_path / "transcript.txt"

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
        patch("yt_ingest.cli.extract_from_cache") as mock_extract,
    ):
        result = runner.invoke(
            app, ["run", "https://youtu.be/dQw4w9WgXcQ", "--agent", "--output", str(out)]
        )

    assert result.exit_code == 0
    assert result.output == "" or result.output == "\n"
    assert "[0:00] hello" in out.read_text()
    mock_extract.assert_not_called()


def test_run_file_mode_unchanged(tmp_path: Path) -> None:
    """Batch mode still runs full pipeline (fetch -> extract -> index -> synthesize)."""
    cache = _make_cache()
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
        patch("yt_ingest.cli.extract") as mock_extract,
        patch("yt_ingest.cli.index") as mock_index,
        patch("yt_ingest.cli.synthesize") as mock_synthesize,
    ):
        result = runner.invoke(app, ["run", "--file", str(urls_file)])

    assert result.exit_code == 0
    mock_extract.assert_called_once()
    mock_index.assert_called_once()
    mock_synthesize.assert_called_once()


def test_run_file_with_agent_ignored(tmp_path: Path) -> None:
    """--agent is ignored in batch mode."""
    cache = _make_cache()
    urls_file = tmp_path / "urls.txt"
    urls_file.write_text("https://youtu.be/dQw4w9WgXcQ\n")

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", return_value=cache),
        patch("yt_ingest.cli.extract") as mock_extract,
        patch("yt_ingest.cli.index") as mock_index,
        patch("yt_ingest.cli.synthesize") as mock_synthesize,
    ):
        result = runner.invoke(app, ["run", "--file", str(urls_file), "--agent"])

    assert result.exit_code == 0
    mock_extract.assert_called_once()
    mock_index.assert_called_once()
    mock_synthesize.assert_called_once()


def test_run_fetch_failure_exits_nonzero(tmp_path: Path) -> None:
    from yt_ingest.fetchers.base import FetchError

    with (
        patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)),
        patch("yt_ingest.cli.run_fetcher_chain", side_effect=FetchError("all failed")),
    ):
        result = runner.invoke(app, ["run", "https://youtu.be/dQw4w9WgXcQ"])

    assert result.exit_code == 1
    assert "Error" in result.stderr


def test_run_invalid_url_exits_nonzero(tmp_path: Path) -> None:
    with patch("yt_ingest.cli.get_config", return_value=_mock_config(tmp_path)):
        result = runner.invoke(app, ["run", "invalid"])

    assert result.exit_code == 1
    assert "Error" in result.stderr
