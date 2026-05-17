from __future__ import annotations
import re
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from yt_ingest.fetchers.base import FetchError
from yt_ingest.models import TranscriptCache, TranscriptSegment, TranscriptSource
from yt_ingest.utils import fetch_video_meta, vtt_time_to_seconds


_CUE_PATTERN = re.compile(
    r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
    r"\s+-->\s+"
    r"(\d{2}:\d{2}:\d{2}\.\d{3}|\d{2}:\d{2}\.\d{3})"
)


def _parse_vtt(vtt_text: str) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    lines = vtt_text.splitlines()
    i = 0
    while i < len(lines):
        m = _CUE_PATTERN.match(lines[i].strip())
        if m:
            start = vtt_time_to_seconds(m.group(1))
            end = vtt_time_to_seconds(m.group(2))
            duration = end - start
            i += 1
            text_lines: list[str] = []
            while i < len(lines) and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            text = " ".join(text_lines)
            if text:
                segments.append(TranscriptSegment(text=text, start=start, duration=duration))
        else:
            i += 1
    return segments


class YtDlpFetcher:
    """Fetch transcript by downloading auto-generated subtitles via yt-dlp."""

    def fetch(self, video_id: str, url: str) -> TranscriptCache:
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "yt-dlp",
                    "--write-auto-sub",
                    "--sub-lang", "en",
                    "--sub-format", "vtt",
                    "--skip-download",
                    "--output", f"{tmpdir}/%(id)s",
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise FetchError(f"yt-dlp subtitles failed: {result.stderr[:300]}")

            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                raise FetchError(f"yt-dlp produced no VTT subtitles for {video_id!r}")

            vtt_text = vtt_files[0].read_text(encoding="utf-8")

        segments = _parse_vtt(vtt_text)
        if not segments:
            raise FetchError(f"VTT file parsed to empty transcript for {video_id!r}")

        meta = fetch_video_meta(url)
        return TranscriptCache(
            video_id=video_id,
            url=url,
            source=TranscriptSource.YTDLP_SUBS,
            fetched_at=datetime.now(timezone.utc),
            title=meta.title,
            channel=meta.channel,
            duration_seconds=meta.duration_seconds,
            segments=segments,
        )
