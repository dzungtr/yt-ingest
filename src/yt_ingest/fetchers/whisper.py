from __future__ import annotations
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from yt_ingest.fetchers.base import FetchError
from yt_ingest.models import TranscriptCache, TranscriptSegment, TranscriptSource
from yt_ingest.utils import fetch_video_meta


class WhisperFetcher:
    """Transcribe audio via faster-whisper (local, GPU/CPU). Slow fallback."""

    def __init__(self, model_size: str = "base") -> None:
        self._model_size = model_size

    def fetch(self, video_id: str, url: str) -> TranscriptCache:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-untyped]
        except ImportError as exc:
            raise FetchError("faster-whisper is not installed") from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / f"{video_id}.m4a"
            dl = subprocess.run(
                [
                    "yt-dlp",
                    "--extract-audio",
                    "--audio-format", "m4a",
                    "--output", str(audio_path),
                    url,
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if dl.returncode != 0:
                raise FetchError(f"yt-dlp audio download failed: {dl.stderr[:300]}")

            model = WhisperModel(self._model_size, compute_type="int8")
            whisper_segments, _ = model.transcribe(str(audio_path))

            segments = [
                TranscriptSegment(
                    text=seg.text.strip(),
                    start=float(seg.start),
                    duration=float(seg.end) - float(seg.start),
                )
                for seg in whisper_segments
                if seg.text.strip()
            ]

        if not segments:
            raise FetchError(f"Whisper produced empty transcript for {video_id!r}")

        meta = fetch_video_meta(url)
        return TranscriptCache(
            video_id=video_id,
            url=url,
            source=TranscriptSource.WHISPER,
            fetched_at=datetime.now(timezone.utc),
            title=meta.title,
            channel=meta.channel,
            duration_seconds=meta.duration_seconds,
            segments=segments,
        )
