from __future__ import annotations
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, model_validator


class TranscriptSource(str, Enum):
    YOUTUBE_TRANSCRIPT_API = "youtube_transcript_api"
    YTDLP_SUBS = "ytdlp_subs"
    WHISPER = "whisper"


class TranscriptSegment(BaseModel):
    text: str
    start: float
    duration: float


class VideoMeta(BaseModel):
    title: str
    channel: str
    duration_seconds: int


class TranscriptCache(BaseModel):
    video_id: str
    url: str
    source: TranscriptSource
    fetched_at: datetime
    title: str
    channel: str
    duration_seconds: int
    segments: list[TranscriptSegment]


class KeyClaim(BaseModel):
    text: str
    timestamp_seconds: float


class WorthRewatching(BaseModel):
    timestamp_seconds: float
    reason: str


class GemType(str, Enum):
    """The audit's five hidden-gem loss classes plus `other` (ADR-0009).

    Metadata only: no pipeline stage may branch on the type; it exists for
    per-loss-class retention metrics and taxonomy growth.
    """

    CAUSAL_LAW = "causal_law"
    CONDITIONAL_BET = "conditional_bet"
    DEFINITIONAL_ASIDE = "definitional_aside"
    EMBEDDED_QUOTE = "embedded_quote"
    CLOSING_PREDICTION = "closing_prediction"
    OTHER = "other"


class Gem(BaseModel):
    """A near-verbatim, timestamped transcript fragment that must survive
    into the study note (the gem data contract consumed by tickets 3-6)."""

    id: str
    type: GemType
    quote: str
    timestamp_start: float
    why: str
    rank: int


class VerdictStatus(str, Enum):
    """Per-gem verification verdicts (issue #18): covered / missing / inapplicable."""

    COVERED = "covered"
    MISSING = "missing"
    INAPPLICABLE = "inapplicable"


class Verdict(BaseModel):
    """One gem's verification outcome against the finished note.

    Covered verdicts must cite the note span where the gem's idea survives;
    missing and inapplicable verdicts carry a one-line reason.
    """

    gem_id: str
    verdict: VerdictStatus
    reason: str
    note_span: str = ""

    @model_validator(mode="after")
    def covered_requires_note_span(self) -> Verdict:
        if self.verdict is VerdictStatus.COVERED and not self.note_span.strip():
            raise ValueError("covered verdict must cite a note span")
        return self


class ExtractedContent(BaseModel):
    summary: str
    key_claims: list[KeyClaim]
    frameworks_and_mental_models: list[str]
    definitions: dict[str, str]
    worth_rewatching: list[WorthRewatching]
    counterpoints_and_caveats: list[str]
    open_questions: list[str]
