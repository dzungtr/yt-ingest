from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from yt_ingest.fetchers.base import FetchError
from yt_ingest.fetchers.whisper import WhisperFetcher
from yt_ingest.models import TranscriptSource, VideoMeta


_META = VideoMeta(title="Test Video", channel="Test Channel", duration_seconds=60)


@dataclass
class _WhisperSegment:
    text: str
    start: float
    end: float


_WHISPER_SEGMENTS = [
    _WhisperSegment(text="  Never gonna give you up  ", start=0.0, end=3.5),
    _WhisperSegment(text="Never gonna let you down", start=3.5, end=7.0),
]


def _make_dl_ok() -> MagicMock:
    r = MagicMock()
    r.returncode = 0
    return r


def test_fetch_success() -> None:
    mock_model = MagicMock()
    mock_model.transcribe.return_value = (_WHISPER_SEGMENTS, MagicMock())

    with (
        patch("yt_ingest.fetchers.whisper.subprocess.run", return_value=_make_dl_ok()),
        patch("yt_ingest.fetchers.whisper.fetch_video_meta", return_value=_META),
        patch.dict("sys.modules", {"faster_whisper": MagicMock(WhisperModel=MagicMock(return_value=mock_model))}),
    ):
        result = WhisperFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")

    assert result.source == TranscriptSource.WHISPER
    assert len(result.segments) == 2
    assert result.segments[0].text == "Never gonna give you up"
    assert result.segments[0].start == pytest.approx(0.0)
    assert result.segments[0].duration == pytest.approx(3.5)


def test_fetch_raises_when_not_installed() -> None:
    import sys

    with patch.dict("sys.modules", {"faster_whisper": None}):  # type: ignore[dict-item]
        with pytest.raises(FetchError, match="faster-whisper is not installed"):
            WhisperFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")


def test_fetch_raises_on_download_failure() -> None:
    mock_dl = MagicMock()
    mock_dl.returncode = 1
    mock_dl.stderr = "download error"

    with (
        patch("yt_ingest.fetchers.whisper.subprocess.run", return_value=mock_dl),
        patch.dict("sys.modules", {"faster_whisper": MagicMock()}),
    ):
        with pytest.raises(FetchError, match="yt-dlp audio download failed"):
            WhisperFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
