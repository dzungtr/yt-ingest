from __future__ import annotations
from typing import Protocol

from yt_ingest.models import TranscriptCache


class FetchError(Exception):
    """Raised when a fetcher cannot retrieve the transcript."""


class TranscriptFetcher(Protocol):
    def fetch(self, video_id: str, url: str) -> TranscriptCache:
        """Attempt to fetch the transcript; raise FetchError on failure."""
        ...


def run_fetcher_chain(
    fetchers: list[TranscriptFetcher],
    video_id: str,
    url: str,
) -> TranscriptCache:
    """Try each fetcher in order; return first success or raise FetchError."""
    errors: list[str] = []
    for fetcher in fetchers:
        try:
            return fetcher.fetch(video_id, url)
        except FetchError as exc:
            errors.append(str(exc))
    raise FetchError(
        f"All fetchers failed for {video_id!r}: " + "; ".join(errors)
    )
