from __future__ import annotations
import hashlib
import logging
from pathlib import Path
from typing import Any

from yt_ingest.llm import CacheStats, chat_json

logger = logging.getLogger(__name__)

_SYNTHESIS_FILE = "synthesis.md"
_CHECKSUM_FILE = ".synthesis.sha"

_SYSTEM_PROMPT = """\
You are a research synthesizer. Given excerpts from multiple YouTube video notes, \
identify cross-cutting themes, contradictions, complementary ideas, and a unified \
synthesis of the knowledge. Return valid JSON:

{
  "cross_cutting_themes": ["string"],
  "contradictions": [{"topic": "string", "views": ["string"]}],
  "complementary_ideas": ["string"],
  "synthesis": "string — comprehensive 5-8 sentence synthesis",
  "recommended_watch_order": ["video_id"],
  "further_questions": ["string"]
}"""


def _notes_checksum(notes_dir: Path) -> str:
    content = ""
    for nf in sorted(notes_dir.glob("*.md")):
        content += nf.name + nf.read_text()
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def synthesize_notes(notes_dir: Path, index_dir: Path) -> tuple[str, CacheStats]:
    """Synthesize all notes into a cross-video summary. Returns (markdown, CacheStats)."""
    note_files = sorted(notes_dir.glob("*.md"))
    if not note_files:
        raise ValueError("No notes found to synthesize.")

    combined = "\n\n---\n\n".join(
        f"## {nf.stem}\n\n{nf.read_text()[:3000]}" for nf in note_files
    )
    user_msg = f"Notes from {len(note_files)} videos:\n\n{combined}"

    raw, stats = chat_json(
        system=_SYSTEM_PROMPT,
        user=user_msg,
        model="deepseek-v4-flash",
        temperature=0.0,
    )

    lines: list[str] = ["# Cross-Video Synthesis\n"]
    lines.append(f"**Videos analyzed:** {len(note_files)}\n\n---\n")

    synthesis: str = raw.get("synthesis", "")
    if synthesis:
        lines.append(f"## Synthesis\n\n{synthesis}\n")

    themes: list[str] = raw.get("cross_cutting_themes", [])
    if themes:
        lines.append("## Cross-Cutting Themes\n")
        lines.extend(f"- {t}\n" for t in themes)
        lines.append("")

    contradictions: list[dict[str, Any]] = raw.get("contradictions", [])
    if contradictions:
        lines.append("## Contradictions\n")
        for c in contradictions:
            lines.append(f"**{c.get('topic', '')}**\n")
            for v in c.get("views", []):
                lines.append(f"- {v}\n")
        lines.append("")

    complementary: list[str] = raw.get("complementary_ideas", [])
    if complementary:
        lines.append("## Complementary Ideas\n")
        lines.extend(f"- {c}\n" for c in complementary)
        lines.append("")

    order: list[str] = raw.get("recommended_watch_order", [])
    if order:
        lines.append("## Recommended Watch Order\n")
        lines.extend(f"{i + 1}. `{vid}`\n" for i, vid in enumerate(order))
        lines.append("")

    questions: list[str] = raw.get("further_questions", [])
    if questions:
        lines.append("## Further Questions\n")
        lines.extend(f"- {q}\n" for q in questions)

    return "".join(lines), stats
