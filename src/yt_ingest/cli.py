from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import typer

from yt_ingest.config import get_config
from yt_ingest.extract import (
    checksum_path,
    extract_from_cache,
    note_path,
    render_note,
    _checksum,
)
from yt_ingest.fetchers.base import FetchError, TranscriptFetcher, run_fetcher_chain
from yt_ingest.fetchers.youtube_api import YouTubeAPIFetcher
from yt_ingest.fetchers.ytdlp import YtDlpFetcher
from yt_ingest.fetchers.whisper import WhisperFetcher
from yt_ingest.models import TranscriptCache
from yt_ingest.utils import extract_video_id, load_urls

app = typer.Typer(name="yt-ingest", add_completion=False)


def _cache_path(transcripts_dir: Path, video_id: str) -> Path:
    return transcripts_dir / f"{video_id}.json"


def _load_cache(path: Path) -> TranscriptCache | None:
    if not path.exists():
        return None
    return TranscriptCache.model_validate_json(path.read_text())


def _save_cache(cache: TranscriptCache, path: Path) -> None:
    path.write_text(cache.model_dump_json(indent=2))


@app.command()
def fetch(
    urls_file: str = typer.Argument(..., help="Path to file with YouTube URLs"),
    allow_whisper: bool = typer.Option(False, "--allow-whisper", help="Enable Whisper fallback (slow)"),
) -> None:
    """Fetch transcripts for all URLs and cache them to disk."""
    cfg = get_config()
    cfg.ensure_dirs()

    urls = load_urls(Path(urls_file))
    if not urls:
        typer.echo("No URLs found in file.")
        raise typer.Exit(1)

    fetchers: list[TranscriptFetcher] = [YouTubeAPIFetcher(), YtDlpFetcher()]
    if allow_whisper:
        fetchers.append(WhisperFetcher())

    ok = 0
    failed = 0
    for url in urls:
        try:
            video_id = extract_video_id(url)
        except ValueError as exc:
            typer.echo(f"SKIP  {url}: {exc}", err=True)
            failed += 1
            continue

        cache_file = _cache_path(cfg.transcripts_dir, video_id)
        if _load_cache(cache_file) is not None:
            typer.echo(f"HIT   {video_id}")
            ok += 1
            continue

        try:
            cache = run_fetcher_chain(fetchers, video_id, url)
        except FetchError as exc:
            typer.echo(f"FAIL  {video_id}: {exc}", err=True)
            failed += 1
            continue

        _save_cache(cache, cache_file)
        typer.echo(f"OK    {video_id}  [{cache.source.value}]  {len(cache.segments)} segments")
        ok += 1

    typer.echo(f"\nDone: {ok} ok, {failed} failed.")
    if failed:
        raise typer.Exit(1)


@app.command()
def extract() -> None:
    """Extract structured notes from cached transcripts."""
    cfg = get_config()
    cfg.ensure_dirs()

    transcript_files = sorted(cfg.transcripts_dir.glob("*.json"))
    if not transcript_files:
        typer.echo("No cached transcripts found. Run 'fetch' first.")
        raise typer.Exit(1)

    ok = 0
    skipped = 0
    for tf in transcript_files:
        cache = TranscriptCache.model_validate_json(tf.read_text())
        npath = note_path(cfg.notes_dir, cache.video_id)
        cpath = checksum_path(cfg.notes_dir, cache.video_id)

        current_cs = _checksum(cache)
        if cpath.exists() and npath.exists() and cpath.read_text().strip() == current_cs:
            typer.echo(f"SKIP  {cache.video_id}  (up-to-date)")
            skipped += 1
            continue

        typer.echo(f"EXTRACT {cache.video_id}  ({cache.title[:50]})")
        content, stats = extract_from_cache(cache)
        note = render_note(cache, content)
        npath.write_text(note)
        cpath.write_text(current_cs)
        typer.echo(
            f"OK    {cache.video_id}  "
            f"cache hit={stats.hit_tokens} miss={stats.miss_tokens}"
        )
        ok += 1

    typer.echo(f"\nDone: {ok} extracted, {skipped} skipped.")


@app.command()
def index() -> None:
    """Build FAISS vector index from notes."""
    typer.echo("index — not yet implemented")


@app.command()
def synthesize() -> None:
    """Generate cross-video synthesis index."""
    typer.echo("synthesize — not yet implemented")


@app.command()
def ask(question: str = typer.Argument(..., help="Question to answer")) -> None:
    """Answer a question using the knowledge base."""
    typer.echo(f"ask: {question!r} — not yet implemented")


@app.command()
def run(
    urls_file: str = typer.Argument(..., help="Path to file with YouTube URLs"),
    allow_whisper: bool = typer.Option(False, "--allow-whisper"),
) -> None:
    """Run the full pipeline: fetch → extract → index → synthesize."""
    typer.echo("run — not yet implemented")


if __name__ == "__main__":
    app()
