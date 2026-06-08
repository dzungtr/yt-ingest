# ADR 0003: Local-first vector retrieval

## Status

**Accepted**

## Context

The tool builds a semantic index over the notes it produces, so the user can
ask natural-language questions (`yt-ingest ask "..."`) against the knowledge
base. The expected corpus size is small — fewer than a few hundred notes, a
few thousand chunks in total — and the tool runs on a single-user workstation
that already has an NVIDIA GPU resident for Whisper.

A cloud vector database (Pinecone, Weaviate, Qdrant-as-a-Service) or a
self-hosted Milvus would give multi-user access, backups, and a managed API
surface that the tool does not need and would have to be operated. The same
applies to embedding providers (OpenAI, Voyage): they add a per-query cost,
a network dependency, and an extra credential to manage, for a corpus that
fits comfortably in memory.

## Decision

Index locally with **`faiss-cpu`** and embed locally with
**`sentence-transformers`** using **`BAAI/bge-large-en-v1.5`**.

- Embeddings run via the same GPU that is already loaded for Whisper, so no
  extra hardware is required.
- The indexing step is word-level: each note is split into ~500-word chunks,
  embedded, and written to `index.faiss` with a paired `meta.json` holding the
  `(video_id, note_path, text)` tuple for every chunk.
- Retrieval is a single `index.search(...)` call returning the top-k neighbours.

The chosen embedder (`BAAI/bge-large-en-v1.5`) is open-weight and runs offline
after the first download; HuggingFace caches it under `HF_HOME` (overridable
via env), so re-runs do not re-download.

## Consequences

- **Single-user scope.** The FAISS index is one disk file; there is no
  concurrency, no cloud backup, no access control. This is acceptable because
  the whole tool is single-user (ADR-0004).
- **No per-query cost.** Embedding runs locally; queries are free after the
  model is downloaded.
- **Index rebuild is cheap.** `yt-ingest index` is idempotent and can be
  re-run any time notes change; the rebuild is bounded by the number of notes,
  which is small.
- **No multi-user or remote access.** Exposing the index to another machine
  would require a separate design; the current choice explicitly does not
  provide it.
- **The `langchain-text-splitters` package is declared as a dependency**
  (see `pyproject.toml`) but the current `retrieval.py` implements chunking
  with a simple word-split; this is a cleanup candidate, not a functional
  issue.
