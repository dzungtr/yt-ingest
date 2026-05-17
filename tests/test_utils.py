from __future__ import annotations
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yt_ingest.models import VideoMeta
from yt_ingest.utils import (
    extract_video_id,
    fetch_video_meta,
    file_checksum,
    load_urls,
    seconds_to_mmss,
    vtt_time_to_seconds,
)


# --- extract_video_id ---

@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
])
def test_extract_video_id_valid(url: str, expected: str) -> None:
    assert extract_video_id(url) == expected


def test_extract_video_id_invalid() -> None:
    with pytest.raises(ValueError, match="Could not extract"):
        extract_video_id("https://example.com/not-a-youtube-url")


# --- load_urls ---

def test_load_urls(tmp_path: Path) -> None:
    f = tmp_path / "urls.txt"
    f.write_text(
        "# comment\n"
        "https://youtu.be/dQw4w9WgXcQ\n"
        "\n"
        "https://youtu.be/dQw4w9WgXcQ\n"  # duplicate
        "https://youtu.be/oHg5SJYRHA0\n"
    )
    result = load_urls(f)
    assert result == [
        "https://youtu.be/dQw4w9WgXcQ",
        "https://youtu.be/oHg5SJYRHA0",
    ]


# --- seconds_to_mmss ---

@pytest.mark.parametrize("seconds,expected", [
    (0.0, "0:00"),
    (59.9, "0:59"),
    (60.0, "1:00"),
    (90.5, "1:30"),
    (3661.0, "61:01"),
])
def test_seconds_to_mmss(seconds: float, expected: str) -> None:
    assert seconds_to_mmss(seconds) == expected


# --- vtt_time_to_seconds ---

@pytest.mark.parametrize("time_str,expected", [
    ("00:00:00.000", 0.0),
    ("00:00:03.500", 3.5),
    ("00:01:07.000", 67.0),
    ("01:00:00.000", 3600.0),
    ("01:30.000", 90.0),  # MM:SS form
])
def test_vtt_time_to_seconds(time_str: str, expected: float) -> None:
    assert vtt_time_to_seconds(time_str) == pytest.approx(expected)


# --- file_checksum ---

def test_file_checksum(tmp_path: Path) -> None:
    f = tmp_path / "data.txt"
    f.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert file_checksum(f) == expected


# --- fetch_video_meta ---

def test_fetch_video_meta_success(ytdlp_meta_json: str) -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ytdlp_meta_json

    with patch("yt_ingest.utils.subprocess.run", return_value=mock_result):
        meta = fetch_video_meta("https://youtu.be/dQw4w9WgXcQ")

    data = json.loads(ytdlp_meta_json)
    assert meta == VideoMeta(
        title=data["title"],
        channel=data["uploader"],
        duration_seconds=data["duration"],
    )


def test_fetch_video_meta_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "yt-dlp error"

    with patch("yt_ingest.utils.subprocess.run", return_value=mock_result):
        with pytest.raises(RuntimeError, match="yt-dlp --dump-json failed"):
            fetch_video_meta("https://youtu.be/dQw4w9WgXcQ")
