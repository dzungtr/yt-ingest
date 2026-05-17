from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from yt_ingest.fetchers.base import FetchError
from yt_ingest.fetchers.ytdlp import YtDlpFetcher, _parse_vtt
from yt_ingest.models import TranscriptSource, VideoMeta


_META = VideoMeta(title="Test Video", channel="Test Channel", duration_seconds=60)

_VTT_CONTENT = """\
WEBVTT
Kind: captions
Language: en

00:00:00.000 --> 00:00:03.500
Never gonna give you up

00:00:03.500 --> 00:00:07.000
Never gonna let you down
"""


# --- _parse_vtt unit tests ---

def test_parse_vtt_extracts_segments(sample_vtt: Path) -> None:
    segments = _parse_vtt(sample_vtt.read_text())
    assert len(segments) == 3
    assert segments[0].text == "Never gonna give you up"
    assert segments[0].start == pytest.approx(0.0)
    assert segments[0].duration == pytest.approx(3.5)


def test_parse_vtt_inline() -> None:
    segments = _parse_vtt(_VTT_CONTENT)
    assert len(segments) == 2
    assert segments[1].text == "Never gonna let you down"
    assert segments[1].start == pytest.approx(3.5)


def test_parse_vtt_skips_empty_cues() -> None:
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\n"
    assert _parse_vtt(vtt) == []


# --- YtDlpFetcher integration tests ---

def _make_run_ok(tmpdir: str, vtt_content: str = _VTT_CONTENT) -> MagicMock:
    """Side-effect that writes a VTT file into tmpdir before returning success."""

    def _side_effect(cmd: list[str], **kwargs: object) -> MagicMock:
        # Extract output template: the value after --output
        out_idx = cmd.index("--output") + 1
        base = cmd[out_idx].replace("%(id)s", "dQw4w9WgXcQ")
        Path(base + ".en.vtt").write_text(vtt_content)
        r = MagicMock()
        r.returncode = 0
        return r

    return MagicMock(side_effect=_side_effect)


def test_fetch_success() -> None:
    with (
        patch("yt_ingest.fetchers.ytdlp.subprocess.run") as mock_run,
        patch("yt_ingest.fetchers.ytdlp.fetch_video_meta", return_value=_META),
    ):
        def _run(cmd: list[str], **kw: object) -> MagicMock:
            out_idx = cmd.index("--output") + 1
            base = cmd[out_idx].replace("%(id)s", "dQw4w9WgXcQ")
            Path(base + ".en.vtt").write_text(_VTT_CONTENT)
            r = MagicMock()
            r.returncode = 0
            return r

        mock_run.side_effect = _run
        result = YtDlpFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")

    assert result.source == TranscriptSource.YTDLP_SUBS
    assert len(result.segments) == 2
    assert result.video_id == "dQw4w9WgXcQ"


def test_fetch_raises_on_yt_dlp_failure() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "some yt-dlp error"

    with patch("yt_ingest.fetchers.ytdlp.subprocess.run", return_value=mock_result):
        with pytest.raises(FetchError, match="yt-dlp subtitles failed"):
            YtDlpFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")


def test_fetch_raises_when_no_vtt_produced() -> None:
    mock_result = MagicMock()
    mock_result.returncode = 0

    with patch("yt_ingest.fetchers.ytdlp.subprocess.run", return_value=mock_result):
        with pytest.raises(FetchError, match="no VTT subtitles"):
            YtDlpFetcher().fetch("dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ")
