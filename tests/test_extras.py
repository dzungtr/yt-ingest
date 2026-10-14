"""Extras-restructure import guards (issue #27).

Base install (no extras) must import the fetch+extract path without the
whisper/retrieval/CLI packages present; the CLI must fail cleanly
(ModuleNotFoundError) when the ``[cli]`` extra is absent.

These tests run helper subprocesses with a sys.meta_path blocker so the
sandboxed import graph is exercised even though the dev environment has
the full stack installed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_BASE_MODULES = [
    "yt_ingest",
    "yt_ingest.config",
    "yt_ingest.extract",
    "yt_ingest.fetchers.base",
    "yt_ingest.fetchers.youtube_api",
    "yt_ingest.fetchers.ytdlp",
    "yt_ingest.fetchers.whisper",
    "yt_ingest.models",
    "yt_ingest.llm",
    "yt_ingest.synthesize",
    "yt_ingest.utils",
]

_EXTRA_ONLY_MODULES = {
    "whisper": ["faster_whisper"],
    "retrieval": ["faiss", "sentence_transformers", "langchain_text_splitters"],
    "cli": ["typer", "rich"],
    "server": ["fastmcp", "uvicorn"],
}

_BLOCKER = textwrap.dedent(
    """
    import sys
    from importlib.abc import MetaPathFinder

    class _Blocker(MetaPathFinder):
        def __init__(self, names: list[str]) -> None:
            self._names = set(names)

        def find_spec(self, fullname, path=None, target=None):
            if fullname in self._names or fullname.split(".")[0] in self._names:
                raise ModuleNotFoundError(
                    f"import of {fullname!r} blocked by test harness",
                    name=fullname,
                )
            return None

    def install(names: list[str]) -> None:
        sys.meta_path.insert(0, _Blocker(names))
    """
)


def _run_with_blocked(blocks: list[str], code: str) -> subprocess.CompletedProcess[str]:
    script = f"{_BLOCKER}\ninstall({blocks!r})\n{code}"
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )


def test_base_imports_without_extras() -> None:
    blocked = [name for names in _EXTRA_ONLY_MODULES.values() for name in names]
    imports = "\n".join(f"import {mod}" for mod in _BASE_MODULES)
    result = _run_with_blocked(blocked, imports + "\nprint('ok')")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


@pytest.mark.parametrize("extra,packages", sorted(_EXTRA_ONLY_MODULES.items()))
def test_extra_only_module_import_fails_cleanly_without_extra(
    extra: str, packages: list[str]
) -> None:
    if extra == "cli":
        code = "import yt_ingest.cli"
    else:
        # Server module does not exist yet (issue #28); assert the raw packages fail.
        code = f"import {packages[0]}"
    result = _run_with_blocked(packages, code)
    assert result.returncode != 0
    assert "ModuleNotFoundError" in result.stderr
    # Not silent breakage: the failure names the missing package.
    assert packages[0] in result.stderr


def test_pyproject_dependency_layout() -> None:
    import tomllib

    data: dict[str, Any] = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    base = set(data["project"]["dependencies"])
    assert base == {
        "youtube-transcript-api>=0.6",
        "yt-dlp>=2024.1",
        "webvtt-py>=0.4",
        "openai>=1.30",
        "tiktoken>=0.7",
        "pydantic>=2.0",
        "jinja2>=3.1",
        "python-dotenv>=1.0",
        "pyyaml>=6.0",
    }
    extras = data["project"]["optional-dependencies"]
    assert extras["whisper"] == ["faster-whisper>=1.0"]
    assert extras["retrieval"] == [
        "sentence-transformers>=3.0",
        "faiss-cpu>=1.8",
        "langchain-text-splitters>=0.2",
    ]
    assert extras["cli"] == ["typer>=0.12", "rich>=13.0"]
    assert extras["server"] == ["fastmcp>=2.0", "uvicorn>=0.30"]
    assert set(extras["full"]) == {
        "yt-ingest[cli]",
        "yt-ingest[whisper]",
        "yt-ingest[retrieval]",
    }
    # `[full]` reproduces today's install: old base deps minus dev-only = base + extras.
    resolved = base | {
        dep
        for key in ("cli", "whisper", "retrieval")
        for dep in extras[key]
    }
    assert resolved == {
        "youtube-transcript-api>=0.6",
        "yt-dlp>=2024.1",
        "webvtt-py>=0.4",
        "faster-whisper>=1.0",
        "openai>=1.30",
        "tiktoken>=0.7",
        "pydantic>=2.0",
        "jinja2>=3.1",
        "typer>=0.12",
        "sentence-transformers>=3.0",
        "faiss-cpu>=1.8",
        "langchain-text-splitters>=0.2",
        "python-dotenv>=1.0",
        "pyyaml>=6.0",
        "rich>=13.0",
    }
