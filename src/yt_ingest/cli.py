from __future__ import annotations
import typer

app = typer.Typer(name="yt-ingest", add_completion=False)


@app.command()
def fetch(
    urls_file: str = typer.Argument(..., help="Path to file with YouTube URLs"),
    allow_whisper: bool = typer.Option(False, "--allow-whisper", help="Enable Whisper fallback (slow)"),
) -> None:
    """Fetch transcripts for all URLs."""
    typer.echo(f"fetch {urls_file} allow_whisper={allow_whisper} — not yet implemented")


@app.command()
def extract() -> None:
    """Extract structured notes from cached transcripts."""
    typer.echo("extract — not yet implemented")


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
