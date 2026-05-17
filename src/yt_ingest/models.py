from __future__ import annotations
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


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


class ExtractedContent(BaseModel):
    summary: str
    key_claims: list[KeyClaim]
    frameworks_and_mental_models: list[str]
    definitions: dict[str, str]
    worth_rewatching: list[WorthRewatching]
    counterpoints_and_caveats: list[str]
    open_questions: list[str]
