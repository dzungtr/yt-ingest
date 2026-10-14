"""MCP server face (issue #28, spec #26).

Single-tool FastMCP server: ``write_study_note(url) -> note text``. Wraps the
existing in-memory pipeline one function deep — fetcher chain →
``extract_from_cache`` → ``render_note`` — no new pipeline logic. Progress
notifications are emitted at the pipeline call site by wrapping the
``yt_ingest.extract.chat_json`` seam (the module-level LLM boundary the test
suite already patches); ``extract.py`` itself is untouched, so gem-epic
refactors (#19–#20) that shift its internals resolve here.

Transport: streamable-HTTP with stateless sessions; the SSE half is retained
so progress notifications reach clients (pi resets its 60s per-request timeout
on each notification). No disk access: the server is local-state by design —
``transcripts/`` and ``notes/`` are never touched.
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, Iterator

from yt_ingest.extract import _chunk_transcript, extract_from_cache, render_note
from yt_ingest.fetchers.base import FetchError, TranscriptFetcher, run_fetcher_chain
from yt_ingest.fetchers.youtube_api import YouTubeAPIFetcher
from yt_ingest.fetchers.ytdlp import YtDlpFetcher
from yt_ingest.models import TranscriptCache
from yt_ingest.utils import extract_video_id

_CAPTIONLESS_ERROR = (
    "No captions found for {url!r} (transcript-api and yt-dlp tiers both "
    "failed). This server does not run Whisper. Run locally instead: "
    "`yt-ingest fetch {url} --allow-whisper`"
)


async def _report_progress(progress: float, message: str, total: float) -> None:
    """Emit a progress notification from the tool coroutine (awaited)."""
    from fastmcp.server.context import _current_context

    ctx = _current_context.get()
    if ctx is None:
        return
    await ctx.report_progress(progress=progress, total=total, message=message)


def _report_progress_from_thread(progress: float, message: str, total: float) -> None:
    """Emit a progress notification from the sync pipeline thread.

    ``extract_from_cache`` is synchronous, so it runs via ``asyncio.to_thread``
    (contextvars propagate into the worker thread). FastMCP's request
    ``Context`` is contextvar-bound; the coroutine runs on a fresh loop inside
    the worker thread because the server loop is not running there.
    """
    from fastmcp.server.context import _current_context

    ctx = _current_context.get()
    if ctx is None:
        return

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            ctx.report_progress(progress=progress, total=total, message=message)
        )
    finally:
        loop.close()


class _StageReporter:
    """Wraps the ``extract.chat_json`` seam to emit stage progress.

    Classifies each LLM call by its system prompt (the same dispatch the test
    doubles use) and maps it onto the notification stages from spec #26:
    gems mined → chunk drafted i/n → merged. Verification/patch calls emit no
    separate stage (they are part of the merge/verify budget).
    """

    def __init__(self, total_chunks: int) -> None:
        self._total_chunks = total_chunks
        self._stage = 1.0

    def _advance(self, message: str) -> None:
        self._stage += 1.0
        _report_progress_from_thread(
            self._stage, message, 2.0 + self._total_chunks + 2.0
        )

    @contextmanager
    def wrap_chat_json(self) -> Iterator[None]:
        import yt_ingest.extract as extract_module

        original: Callable[..., Any] = extract_module.chat_json
        miner_calls = 0
        writer_calls = 0
        merge_seen = False

        def wrapped(**kwargs: Any) -> Any:
            nonlocal miner_calls, writer_calls, merge_seen
            result = original(**kwargs)
            system = str(kwargs.get("system", ""))
            if "gem miner" in system:
                miner_calls += 1
                self._advance(f"gems mined (chunk {miner_calls}/{self._total_chunks})")
            elif "study-note writer" in system:
                writer_calls += 1
                self._advance(
                    f"chunk drafted {writer_calls}/{self._total_chunks}"
                )
            elif not merge_seen and "merger" in system:
                merge_seen = True
                self._advance("drafts merged")
            return result

        extract_module.chat_json = wrapped  # type: ignore[assignment]
        try:
            yield
        finally:
            extract_module.chat_json = original  # type: ignore[assignment]

    def __enter__(self) -> _StageReporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


def _fetch_transcript(url: str) -> TranscriptCache:
    """Fetch via the existing chain — same tiers as the CLI's default path.

    The whisper tier is deliberately absent: the ``[server]`` install omits
    the ``whisper`` extra, so both HTTP tiers failing is a clean tool error
    pointing at the CLI's ``--allow-whisper`` escape hatch.
    """
    video_id = extract_video_id(url)
    fetchers: list[TranscriptFetcher] = [YouTubeAPIFetcher(), YtDlpFetcher()]
    return run_fetcher_chain(fetchers, video_id, url)


async def write_study_note_impl(url: str) -> str:
    """Pipeline body: fetch → extract → render, with stage notifications."""
    video_id = extract_video_id(url)
    try:
        cache = await asyncio.to_thread(
            run_fetcher_chain, [YouTubeAPIFetcher(), YtDlpFetcher()], video_id, url
        )
    except FetchError as exc:
        raise ToolCaptionlessError(_CAPTIONLESS_ERROR.format(url=url)) from exc
    await _report_progress(1.0, "transcript fetched", 5.0)

    total_chunks = len(_chunk_transcript(cache))
    reporter = _StageReporter(total_chunks)

    def pipeline() -> str:
        with reporter.wrap_chat_json():
            blog_post, _gems, _report, _stats = extract_from_cache(cache)
        return render_note(cache, blog_post)

    note = await asyncio.to_thread(pipeline)
    await _report_progress(
        float(2 + total_chunks + 2), "done", 5.0 + total_chunks
    )
    return note


class ToolCaptionlessError(Exception):
    """Raised when the fetch chain finds no captions (whisper excluded)."""


def build_mcp_server() -> Any:
    """Build the FastMCP server exposing ``write_study_note`` and ``/health``."""
    from fastmcp import FastMCP

    mcp = FastMCP(
        name="yt-ingest",
        instructions=(
            "YouTube URL in, finished study note out. One tool: "
            "write_study_note(url)."
        ),
    )

    @mcp.custom_route("/health", methods=["GET"])
    async def health(request: Any) -> Any:
        from starlette.responses import JSONResponse

        return JSONResponse({"status": "ok"})

    @mcp.tool
    async def write_study_note(url: str) -> str:
        """Turn a YouTube URL into a finished study note (markdown).

        Runs the full yt-ingest pipeline server-side: fetch captions, mine
        gems, draft per-chunk notes, merge, verify. Emits MCP progress
        notifications at each stage (fetched → mined → chunk drafted i/n →
        merged → done); pi's MCP client resets its 60s per-request timeout on
        every progress notification, so no client config is needed. Residual
        risk: a single stage between notifications exceeding 60s (rare —
        chunk drafts are ~10-30s) can still time out the request.
        """
        try:
            return await write_study_note_impl(url)
        except ToolCaptionlessError as exc:
            raise _tool_error(str(exc)) from exc

    return mcp


def _tool_error(message: str) -> Exception:
    from fastmcp.exceptions import ToolError

    return ToolError(message)


def serve() -> None:
    """Entry point: ``yt-ingest-serve`` — streamable-HTTP on 0.0.0.0:8000/mcp."""
    mcp = build_mcp_server()
    mcp.run_http_async(
        host="0.0.0.0",
        port=8000,
        transport="http",
        path="/mcp",
        stateless_http=True,
        json_response=False,
        show_banner=False,
    )
