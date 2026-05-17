# yt-ingest

A CLI tool that fetches YouTube transcripts, extracts structured notes with DeepSeek, builds a FAISS vector index, and lets you query your knowledge base.

## Setup

```bash
pip install -e .
cp .env.example .env        # then add DEEPSEEK_API_KEY
```

## Usage

### Full pipeline (one command)

```bash
yt-ingest run urls.txt
```

### Step by step

```bash
# 1. Fetch transcripts for all URLs in the file
yt-ingest fetch urls.txt

# 2. Extract structured notes via DeepSeek
yt-ingest extract

# 3. Build FAISS vector index
yt-ingest index

# 4. Generate cross-video synthesis
yt-ingest synthesize

# 5. Ask questions
yt-ingest ask "What are the main arguments for deep work?"
```

### URL file format

One URL per line. Blank lines and `#` comments are ignored. Duplicates are skipped.

```
# My learning list
https://www.youtube.com/watch?v=...
https://youtu.be/...
```

### Options

- `--allow-whisper` (fetch/run): Enable faster-whisper fallback for videos without captions. Requires `faster-whisper` installed and can be slow.

## Transcript fetcher chain

1. **youtube-transcript-api** — fastest, no download required
2. **yt-dlp subtitles** — downloads auto-generated VTT subtitles
3. **faster-whisper** — local transcription (opt-in via `--allow-whisper`)

Each fetcher is tried in order; the first success wins.

## Output

- `transcripts/<video_id>.json` — cached transcript
- `notes/<video_id>.md` — structured notes
- `notes/synthesis.md` — cross-video synthesis
- `.faiss_index/` — FAISS vector index

## Development

```bash
pip install -e ".[dev]"
pytest
mypy --strict src
```
