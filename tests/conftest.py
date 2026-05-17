from __future__ import annotations
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture()
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture()
def sample_vtt(fixtures_dir: Path) -> Path:
    return fixtures_dir / "dQw4w9WgXcQ.en.vtt"


@pytest.fixture()
def ytdlp_meta_json(fixtures_dir: Path) -> str:
    return (fixtures_dir / "ytdlp_meta.json").read_text()
