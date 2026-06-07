from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv

_config: "Config | None" = None


class Config:
    def __init__(self) -> None:
        load_dotenv()
        self.deepseek_api_key: str = os.environ.get("DEEPSEEK_API_KEY", "")
        self.transcripts_dir: Path = Path("transcripts")
        self.notes_dir: Path = Path("notes")
        self.faiss_index_dir: Path = Path(".faiss_index")

    def ensure_dirs(self) -> None:
        self.transcripts_dir.mkdir(exist_ok=True)
        self.notes_dir.mkdir(exist_ok=True)


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
