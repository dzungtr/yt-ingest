from __future__ import annotations
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_ingest.models import VideoMeta


def extract_video_id(url: str) -> str:
    """Parse a YouTube video ID from any standard YouTube URL format."""
    parsed = urlparse(url)
    video_id = ""
    if parsed.netloc in ("youtu.be",):
        video_id = parsed.path.lstrip("/").split("/")[0]
    elif "youtube.com" in parsed.netloc:
        if parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/")[2]
        elif parsed.path.startswith("/embed/"):
            video_id = parsed.path.split("/")[2]
        else:
            qs = parse_qs(parsed.query)
            video_id = qs.get("v", [""])[0]
    if not re.match(r"^[A-Za-z0-9_-]{11}$", video_id):
        raise ValueError(f"Could not extract a valid video_id from URL: {url!r}")
    return video_id


def load_urls(path: Path) -> list[str]:
    """Load YouTube URLs from a file, skipping blank lines and comments, deduplicating."""
    seen: set[str] = set()
    result: list[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result


def seconds_to_mmss(seconds: float) -> str:
    """Convert a float number of seconds to mm:ss string (no zero-padding on minutes)."""
    total = int(seconds)
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def vtt_time_to_seconds(time_str: str) -> float:
    """Convert a VTT timestamp ('HH:MM:SS.mmm' or 'MM:SS.mmm') to seconds."""
    time_str = time_str.replace(",", ".")
    parts = time_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    m, s = parts
    return int(m) * 60 + float(s)


def file_checksum(path: Path) -> str:
    """SHA-256 hex digest of a file's contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_video_meta(url: str) -> VideoMeta:
    """Fetch video title, channel, and duration via yt-dlp --dump-json."""
    result = subprocess.run(
        ["yt-dlp", "--dump-json", url],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp --dump-json failed: {result.stderr[:300]}")
    data = json.loads(result.stdout)
    return VideoMeta(
        title=str(data.get("title", "Unknown")),
        channel=str(data.get("uploader", data.get("channel", "Unknown"))),
        duration_seconds=int(data.get("duration", 0)),
    )
