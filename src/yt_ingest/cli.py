from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich import print as rprint

from yt_ingest.config import get_config
from yt_ingest.retrieval import build_index
from yt_ingest.synthesize import (
    _CHECKSUM_FILE,
    _SYNTHESIS_FILE,
    _notes_checksum,
    synthesize_notes,
)
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
console = Console()
err_console = Console(stderr=True)


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
    url: Optional[str] = typer.Argument(None, help="Single YouTube URL to fetch"),
    file: Optional[str] = typer.Option(None, "--file", help="Path to file with YouTube URLs"),
    allow_whisper: bool = typer.Option(False, "--allow-whisper", help="Enable Whisper fallback (slow)"),
) -> None:
    """Fetch transcripts for all URLs and cache them to disk."""
    if url is not None and file is not None:
        err_console.print("[red]Error:[/red] provide either a URL argument or --file, not both.")
        raise typer.Exit(1)
    if url is None and file is None:
        err_console.print("[red]Error:[/red] provide either a URL argument or --file.")
        raise typer.Exit(1)

    cfg = get_config()
    cfg.ensure_dirs()

    if file is not None:
        urls = load_urls(Path(file))
        if not urls:
            console.print("[yellow]No URLs found in file.[/yellow]")
            raise typer.Exit(1)
    else:
        urls = [url]

    fetchers: list[TranscriptFetcher] = [YouTubeAPIFetcher(), YtDlpFetcher()]
    if allow_whisper:
        fetchers.append(WhisperFetcher())

    ok = 0
    failed = 0
    for u in urls:
        try:
            video_id = extract_video_id(u)
        except ValueError as exc:
            err_console.print(f"[yellow]SKIP[/yellow]  {u}: {exc}")
            failed += 1
            continue

        cache_file = _cache_path(cfg.transcripts_dir, video_id)
        if _load_cache(cache_file) is not None:
            console.print(f"[cyan]HIT [/cyan]  {video_id}")
            ok += 1
            continue

        with console.status(f"[dim]Fetching[/dim] {video_id}…", spinner="dots"):
            try:
                cache = run_fetcher_chain(fetchers, video_id, u)
            except FetchError as exc:
                err_console.print(f"[red]FAIL[/red]  {video_id}: {exc}")
                failed += 1
                continue

        _save_cache(cache, cache_file)
        console.print(
            f"[green]OK  [/green]  {video_id}"
            f"  [dim]via {cache.source.value}  {len(cache.segments)} segments[/dim]"
        )
        ok += 1

    console.rule()
    console.print(f"[green]{ok} ok[/green]  [red]{failed} failed[/red]")
    if failed:
        raise typer.Exit(1)


@app.command()
def extract() -> None:
    """Extract structured notes from cached transcripts."""
    cfg = get_config()
    cfg.ensure_dirs()

    transcript_files = sorted(cfg.transcripts_dir.glob("*.json"))
    if not transcript_files:
        err_console.print("[yellow]No cached transcripts found. Run 'fetch' first.[/yellow]")
        raise typer.Exit(1)

    ok = 0
    skipped = 0
    for tf in transcript_files:
        cache = TranscriptCache.model_validate_json(tf.read_text())
        npath = note_path(cfg.notes_dir, cache.video_id)
        cpath = checksum_path(cfg.notes_dir, cache.video_id)

        current_cs = _checksum(cache)
        if cpath.exists() and npath.exists() and cpath.read_text().strip() == current_cs:
            console.print(f"[cyan]SKIP[/cyan]  {cache.video_id}  [dim](up-to-date)[/dim]")
            skipped += 1
            continue

        with console.status(
            f"[dim]Extracting[/dim] {cache.video_id}  [dim]{cache.title[:50]}[/dim]…",
            spinner="dots",
        ):
            content, stats = extract_from_cache(cache)

        note = render_note(cache, content)
        npath.write_text(note)
        cpath.write_text(current_cs)
        console.print(
            f"[green]OK  [/green]  {cache.video_id}  "
            f"[dim]cache hit={stats.hit_tokens} miss={stats.miss_tokens}[/dim]"
        )
        ok += 1

    console.rule()
    console.print(f"[green]{ok} extracted[/green]  [cyan]{skipped} skipped[/cyan]")


@app.command()
def index() -> None:
    """Build FAISS vector index from notes."""
    cfg = get_config()
    with console.status("[dim]Building vector index…[/dim]", spinner="dots"):
        n = build_index(cfg.notes_dir, cfg.faiss_index_dir)
    if n == 0:
        err_console.print("[yellow]No notes found. Run 'extract' first.[/yellow]")
        raise typer.Exit(1)
    console.print(f"[green]Indexed[/green] {n} chunks.")


@app.command()
def synthesize() -> None:
    """Generate cross-video synthesis from all notes."""
    cfg = get_config()
    cs = _notes_checksum(cfg.notes_dir)
    cs_path = cfg.notes_dir / _CHECKSUM_FILE
    out_path = cfg.notes_dir / _SYNTHESIS_FILE

    if cs_path.exists() and out_path.exists() and cs_path.read_text().strip() == cs:
        console.print("[cyan]Synthesis up-to-date.[/cyan]")
        return

    with console.status("[dim]Synthesizing notes…[/dim]", spinner="dots"):
        try:
            markdown, stats = synthesize_notes(cfg.notes_dir, cfg.faiss_index_dir)
        except ValueError as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1) from exc

    out_path.write_text(markdown)
    cs_path.write_text(cs)
    console.print(
        f"[green]Synthesis written to[/green] {out_path}  "
        f"[dim](cache hit={stats.hit_tokens} miss={stats.miss_tokens})[/dim]"
    )


_ASK_SYSTEM = """\
You are a research assistant with access to notes from multiple YouTube videos. \
Answer the user's question using only the provided context. Be concise and cite \
which video the information comes from where relevant. \
Return valid JSON: {"answer": "your answer here"}"""


@app.command()
def ask(question: str = typer.Argument(..., help="Question to answer")) -> None:
    """Answer a question using the knowledge base."""
    from yt_ingest.llm import chat_json
    from yt_ingest.retrieval import search

    cfg = get_config()
    try:
        chunks = search(question, cfg.faiss_index_dir, top_k=5)
    except FileNotFoundError as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    context = "\n\n---\n\n".join(
        f"[{c.video_id}]\n{c.text}" for c in chunks
    )
    user_msg = f"Context:\n\n{context}\n\nQuestion: {question}"
    raw, stats = chat_json(system=_ASK_SYSTEM, user=user_msg)
    answer = raw.get("answer", "") if isinstance(raw, dict) else str(raw)

    console.rule()
    console.print(answer)
    console.rule()
    err_console.print(
        f"[dim]cache hit={stats.hit_tokens} miss={stats.miss_tokens}[/dim]"
    )


@app.command()
def run(
    url: Optional[str] = typer.Argument(None, help="Single YouTube URL to process"),
    file: Optional[str] = typer.Option(None, "--file", help="Path to file with YouTube URLs"),
    allow_whisper: bool = typer.Option(False, "--allow-whisper"),
) -> None:
    """Run the full pipeline: fetch → extract → index → synthesize."""
    console.rule("[bold]fetch[/bold]")
    fetch(url=url, file=file, allow_whisper=allow_whisper)
    console.rule("[bold]extract[/bold]")
    extract()
    console.rule("[bold]index[/bold]")
    index()
    console.rule("[bold]synthesize[/bold]")
    synthesize()


if __name__ == "__main__":
    app()
