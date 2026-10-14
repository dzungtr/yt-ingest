"""Server-face tests (issue #28).

Drives the ``write_study_note`` MCP tool through FastMCP's in-memory client
transport — no sockets, no real network calls. The fetcher chain is patched to
return an in-memory ``TranscriptCache`` (success) or raise ``FetchError``
(caption-less video); the LLM boundary is mocked via the existing
``yt_ingest.extract.chat_json`` patch seam.

Smoke-test command shape for the deploy slice (issue #30):

    async with Client(mcp_server) as client:          # in-memory transport
        result = await client.call_tool("write_study_note", {"url": URL})
        # result.data is the rendered note text; progress notifications are
        # captured by passing progress_handler=... to Client(...).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable
from unittest.mock import patch

import pytest

from yt_ingest.fetchers.base import FetchError
from yt_ingest.models import (
    TranscriptCache,
    TranscriptSegment,
    TranscriptSource,
)

mimetypes_guarded = False


def _ensure_mimetypes() -> None:
    """FastMCP import reads /etc/mime.types, which is denied in this sandbox."""
    global mimetypes_guarded
    if not mimetypes_guarded:
        import mimetypes

        mimetypes.knownfiles = []
        mimetypes_guarded = True


@pytest.fixture()
def fastmcp_modules() -> None:
    """Import fastmcp with the sandbox-blocked mimetypes DB disabled."""
    pytest.importorskip("fastmcp")
    _ensure_mimetypes()
    import fastmcp  # noqa: F401

    fastmcp  # keep linters honest


def _make_cache() -> TranscriptCache:
    segments = [
        TranscriptSegment(text=f"segment {i}", start=float(i * 5), duration=5.0)
        for i in range(3)
    ]
    return TranscriptCache(
        video_id="dQw4w9WgXcQ",
        url="https://youtu.be/dQw4w9WgXcQ",
        source=TranscriptSource.YOUTUBE_TRANSCRIPT_API,
        fetched_at=datetime(2026, 1, 1),
        title="Test Video",
        channel="Test Channel",
        duration_seconds=15.0,
        segments=segments,
    )


def _fake_chat_json(blog_text: str):
    from yt_ingest.llm import CacheStats

    def fake_chat_json(**kwargs: object) -> tuple[dict[str, object], CacheStats]:
        system = str(kwargs.get("system", ""))
        if "gem miner" in system:
            return {
                "gems": [
                    {
                        "type": "causal_law",
                        "quote": "Savings buildup leads to risk chasing.",
                        "timestamp_start": 5.0,
                        "why": "Core causal law.",
                        "rank": 1,
                    }
                ]
            }, CacheStats(total_calls=1)
        if "gem verifier" in system:
            return {
                "verdict": "covered",
                "reason": "present",
                "note_span": "b",
            }, CacheStats(total_calls=1)
        return {"blog_post": blog_text}, CacheStats(total_calls=1)

    return fake_chat_json


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


def test_write_study_note_returns_rendered_note(fastmcp_modules: None) -> None:
    import anyio

    async def scenario() -> str:
        from fastmcp import Client

        from yt_ingest.server import build_mcp_server

        mcp = build_mcp_server()
        blog_text = "## My Blog Post\n\nGreat content."
        async with Client(mcp) as client:
            with (
                patch(
                    "yt_ingest.server.run_fetcher_chain",
                    return_value=_make_cache(),
                ),
                patch(
                    "yt_ingest.server.extract_from_cache",
                    return_value=(
                        blog_text,
                        [],
                        None,
                        None,
                    ),
                ),
                patch(
                    "yt_ingest.server.render_note",
                    return_value=f"# Test Video\n\n{blog_text}",
                ),
            ):
                result = await client.call_tool(
                    "write_study_note", {"url": "https://youtu.be/dQw4w9WgXcQ"}
                )
        return str(result.data)

    note = _run(scenario())
    assert "Test Video" in note
    assert "My Blog Post" in note


def test_write_study_note_captionless_clean_error(fastmcp_modules: None) -> None:
    async def scenario() -> object:
        from fastmcp import Client

        from yt_ingest.server import build_mcp_server

        mcp = build_mcp_server()
        async with Client(mcp) as client:
            with patch(
                "yt_ingest.server.run_fetcher_chain",
                side_effect=FetchError("all fetchers failed"),
            ):
                try:
                    await client.call_tool(
                        "write_study_note", {"url": "https://youtu.be/dQw4w9WgXcQ"}
                    )
                except Exception as exc:  # ToolError surfaces to the caller
                    return exc
        raise AssertionError("tool call should have raised")

    exc = _run(scenario())
    text = str(exc)
    assert "--allow-whisper" in text
    assert "yt-ingest fetch" in text


def test_progress_notifications_emitted(fastmcp_modules: None) -> None:
    """Progress notifications reach the in-memory client's progress handler."""
    from fastmcp import Client

    from yt_ingest.server import build_mcp_server

    messages: list[tuple[float, float | None, str | None]] = []

    async def handler(
        progress: float, total: float | None, message: str | None
    ) -> None:
        messages.append((progress, total, message))

    async def scenario() -> None:
        mcp = build_mcp_server()
        blog_text = "## My Blog Post"

        import yt_ingest.extract as extract_module

        def fake_extract(cache: TranscriptCache) -> tuple[str, object, object, object]:
            real_chat_json = extract_module.chat_json
            with patch(
                "yt_ingest.extract.chat_json",
                side_effect=lambda **kw: ({"blog_post": "x"}, None),
            ):
                real_chat_json(system="gem miner", user="x")
                real_chat_json(system="study-note writer", user="x")
            return blog_text, [], None, None

        async with Client(mcp, progress_handler=handler) as client:
            with (
                patch(
                    "yt_ingest.server.run_fetcher_chain",
                    return_value=_make_cache(),
                ),
                patch(
                    "yt_ingest.server._chunk_transcript",
                    return_value=["chunk a"],
                ),
                patch(
                    "yt_ingest.server.extract_from_cache",
                    side_effect=fake_extract,
                ),
                patch(
                    "yt_ingest.server.render_note",
                    return_value=blog_text,
                ),
            ):
                await client.call_tool(
                    "write_study_note", {"url": "https://youtu.be/dQw4w9WgXcQ"}
                )

    _run(scenario())
    texts = [str(m[2]) for m in messages]
    assert any("fetched" in t for t in texts)
    assert any("mined" in t for t in texts)
    assert any("drafted 1/1" in t for t in texts)
    assert any("done" in t for t in texts)


def test_progress_notifications_multi_chunk_merges(fastmcp_modules: None) -> None:
    """Two chunks → merged stage appears between drafting and done."""
    from fastmcp import Client

    from yt_ingest.server import build_mcp_server

    messages: list[tuple[float, float | None, str | None]] = []

    async def handler(
        progress: float, total: float | None, message: str | None
    ) -> None:
        messages.append((progress, total, message))

    async def scenario() -> None:
        mcp = build_mcp_server()
        blog_text = "## My Blog Post"
        merged_text = "## Merged"

        import yt_ingest.extract as extract_module

        def fake_extract(cache: TranscriptCache) -> tuple[str, object, object, object]:
            # Route each call through the real chat_json seam — but patched at
            # yt_ingest.extract, so no network. The _StageReporter wrapper in
            # server.py wraps this same module attribute.
            real_chat_json = extract_module.chat_json
            with patch(
                "yt_ingest.extract.chat_json",
                side_effect=lambda **kw: ({"blog_post": "x"}, None),
            ):
                for system in ("gem miner", "study-note writer", "gem miner",
                               "study-note writer", "merger", "gem verifier"):
                    real_chat_json(system=system, user="x")
            return merged_text, [], None, None

        async with Client(mcp, progress_handler=handler) as client:
            with (
                patch(
                    "yt_ingest.server.run_fetcher_chain",
                    return_value=_make_cache(),
                ),
                patch(
                    "yt_ingest.server._chunk_transcript",
                    return_value=["chunk a", "chunk b"],
                ),
                patch(
                    "yt_ingest.server.extract_from_cache",
                    side_effect=fake_extract,
                ),
                patch(
                    "yt_ingest.server.render_note",
                    return_value=blog_text,
                ),
            ):
                await client.call_tool(
                    "write_study_note", {"url": "https://youtu.be/dQw4w9WgXcQ"}
                )

    _run(scenario())
    texts = [str(m[2]) for m in messages]
    assert any("drafted 1/2" in t for t in texts)
    assert any("drafted 2/2" in t for t in texts)
    assert any("merged" in t for t in texts)
    assert any("done" in t for t in texts)


def test_health_route_returns_200(fastmcp_modules: None) -> None:
    from fastmcp import Client
    from starlette.testclient import TestClient

    from yt_ingest.server import build_mcp_server

    mcp = build_mcp_server()
    # http_app with stateless_http matches the production run configuration.
    asgi = mcp.http_app(stateless_http=True, json_response=False, transport="http")
    http_client = TestClient(asgi)
    response = http_client.get("/health")
    assert response.status_code == 200


def test_server_module_does_not_touch_disk_dirs() -> None:
    """The server face is stateless: no writes to transcripts/ or notes/."""
    import inspect

    from yt_ingest import server

    src = inspect.getsource(server)
    assert "ensure_dirs" not in src
