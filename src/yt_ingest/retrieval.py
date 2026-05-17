from __future__ import annotations
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_NAME = "BAAI/bge-large-en-v1.5"
_INDEX_FILE = "index.faiss"
_META_FILE = "meta.json"


@dataclass
class Chunk:
    video_id: str
    note_path: str
    text: str


def _load_model() -> Any:
    return SentenceTransformer(_MODEL_NAME)


def _chunk_note(note_path: Path, video_id: str, chunk_size: int = 500) -> list[Chunk]:
    """Split a markdown note into word-level chunks."""
    text = note_path.read_text()
    words = text.split()
    chunks: list[Chunk] = []
    for i in range(0, max(1, len(words)), chunk_size):
        chunk_text = " ".join(words[i : i + chunk_size])
        chunks.append(Chunk(video_id=video_id, note_path=str(note_path), text=chunk_text))
    return chunks


def build_index(notes_dir: Path, index_dir: Path) -> int:
    """Build FAISS index from all note files. Returns number of chunks indexed."""
    note_files = sorted(notes_dir.glob("*.md"))
    if not note_files:
        return 0

    all_chunks: list[Chunk] = []
    for nf in note_files:
        video_id = nf.stem
        all_chunks.extend(_chunk_note(nf, video_id))

    model = _load_model()
    texts = [c.text for c in all_chunks]
    logger.info("Encoding %d chunks with %s", len(texts), _MODEL_NAME)
    embeddings: Any = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))

    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / _INDEX_FILE))

    meta = [{"video_id": c.video_id, "note_path": c.note_path, "text": c.text} for c in all_chunks]
    (index_dir / _META_FILE).write_text(json.dumps(meta, indent=2))

    return len(all_chunks)


def search(query: str, index_dir: Path, top_k: int = 5) -> list[Chunk]:
    """Search the FAISS index and return top-k matching chunks."""
    idx_path = index_dir / _INDEX_FILE
    meta_path = index_dir / _META_FILE
    if not idx_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Index not found in {index_dir}. Run 'index' first.")

    index = faiss.read_index(str(idx_path))
    meta = json.loads(meta_path.read_text())

    model = _load_model()
    query_emb: Any = model.encode([query], normalize_embeddings=True)
    _, indices = index.search(query_emb.astype(np.float32), top_k)

    results: list[Chunk] = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(meta):
            continue
        m = meta[idx]
        results.append(Chunk(video_id=m["video_id"], note_path=m["note_path"], text=m["text"]))
    return results
