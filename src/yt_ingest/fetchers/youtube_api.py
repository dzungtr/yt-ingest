from __future__ import annotations
from datetime import datetime, timezone

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from yt_ingest.fetchers.base import FetchError
from yt_ingest.models import TranscriptCache, TranscriptSegment, TranscriptSource
from yt_ingest.utils import fetch_video_meta


class YouTubeAPIFetcher:
    """Fetch transcript via youtube-transcript-api (no auth required)."""

    def fetch(self, video_id: str, url: str) -> TranscriptCache:
        try:
            fetched = YouTubeTranscriptApi().fetch(video_id)
        except (NoTranscriptFound, TranscriptsDisabled) as exc:
            raise FetchError(f"youtube-transcript-api: {exc}") from exc
        except Exception as exc:
            raise FetchError(f"youtube-transcript-api unexpected error: {exc}") from exc

        segments = [
            TranscriptSegment(
                text=snippet.text,
                start=snippet.start,
                duration=snippet.duration,
            )
            for snippet in fetched
        ]
        meta = fetch_video_meta(url)
        return TranscriptCache(
            video_id=video_id,
            url=url,
            source=TranscriptSource.YOUTUBE_TRANSCRIPT_API,
            fetched_at=datetime.now(timezone.utc),
            title=meta.title,
            channel=meta.channel,
            duration_seconds=meta.duration_seconds,
            segments=segments,
        )
