from __future__ import annotations

import pytest

from yt_ingest.fetchers.base import FetchError, run_fetcher_chain
from yt_ingest.models import TranscriptCache, TranscriptSource, TranscriptSegment
from datetime import datetime


def _make_cache(video_id: str = "abc1234567a") -> TranscriptCache:
    return TranscriptCache(
        video_id=video_id,
        url=f"https://youtu.be/{video_id}",
        source=TranscriptSource.YOUTUBE_TRANSCRIPT_API,
        fetched_at=datetime(2026, 1, 1),
        title="Test",
        channel="Test Channel",
        duration_seconds=60,
        segments=[TranscriptSegment(text="hello", start=0.0, duration=1.0)],
    )


class _SuccessFetcher:
    def fetch(self, video_id: str, url: str) -> TranscriptCache:
        return _make_cache(video_id)


class _FailingFetcher:
    def __init__(self, msg: str = "unavailable") -> None:
        self._msg = msg

    def fetch(self, video_id: str, url: str) -> TranscriptCache:
        raise FetchError(self._msg)


def test_chain_returns_first_success() -> None:
    result = run_fetcher_chain(
        [_FailingFetcher("first"), _SuccessFetcher()],
        "abc1234567a",
        "https://youtu.be/abc1234567a",
    )
    assert result.video_id == "abc1234567a"


def test_chain_uses_first_fetcher_when_it_succeeds() -> None:
    result = run_fetcher_chain(
        [_SuccessFetcher(), _FailingFetcher("should not run")],
        "abc1234567a",
        "https://youtu.be/abc1234567a",
    )
    assert result.video_id == "abc1234567a"


def test_chain_raises_when_all_fail() -> None:
    with pytest.raises(FetchError, match="All fetchers failed"):
        run_fetcher_chain(
            [_FailingFetcher("err1"), _FailingFetcher("err2")],
            "abc1234567a",
            "https://youtu.be/abc1234567a",
        )


def test_chain_error_includes_individual_messages() -> None:
    with pytest.raises(FetchError) as exc_info:
        run_fetcher_chain(
            [_FailingFetcher("reason_one"), _FailingFetcher("reason_two")],
            "abc1234567a",
            "https://youtu.be/abc1234567a",
        )
    msg = str(exc_info.value)
    assert "reason_one" in msg
    assert "reason_two" in msg
