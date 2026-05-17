from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import patch

import pytest

from yt_ingest.fetchers.base import FetchError
from yt_ingest.fetchers.youtube_api import YouTubeAPIFetcher
from yt_ingest.models import TranscriptSource, VideoMeta


@dataclass
class _Snippet:
    text: str
    start: float
    duration: float


_RAW_SNIPPETS = [
    _Snippet(text="Never gonna give you up", start=0.0, duration=3.5),
    _Snippet(text="Never gonna let you down", start=3.5, duration=3.5),
]

_META = VideoMeta(
    title="Rick Astley - Never Gonna Give You Up",
    channel="Rick Astley",
    duration_seconds=213,
)


def test_fetch_success() -> None:
    with (
        patch(
            "yt_ingest.fetchers.youtube_api.YouTubeTranscriptApi.fetch",
            return_value=_RAW_SNIPPETS,
        ),
        patch("yt_ingest.fetchers.youtube_api.fetch_video_meta", return_value=_META),
    ):
        result = YouTubeAPIFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")

    assert result.video_id == "dQw4w9WgXcQ"
    assert result.source == TranscriptSource.YOUTUBE_TRANSCRIPT_API
    assert len(result.segments) == 2
    assert result.segments[0].text == "Never gonna give you up"
    assert result.title == _META.title
    assert result.channel == _META.channel


def test_fetch_raises_on_no_transcript() -> None:
    from youtube_transcript_api import NoTranscriptFound  # type: ignore[import-untyped]

    with patch(
        "yt_ingest.fetchers.youtube_api.YouTubeTranscriptApi.fetch",
        side_effect=NoTranscriptFound("dQw4w9WgXcQ", [], []),
    ):
        with pytest.raises(FetchError, match="youtube-transcript-api"):
            YouTubeAPIFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")


def test_fetch_raises_on_transcripts_disabled() -> None:
    from youtube_transcript_api import TranscriptsDisabled  # type: ignore[import-untyped]

    with patch(
        "yt_ingest.fetchers.youtube_api.YouTubeTranscriptApi.fetch",
        side_effect=TranscriptsDisabled("dQw4w9WgXcQ"),
    ):
        with pytest.raises(FetchError, match="youtube-transcript-api"):
            YouTubeAPIFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
