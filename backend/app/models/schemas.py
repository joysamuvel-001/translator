from datetime import datetime
from pydantic import BaseModel, Field


class SpeakerEnrollRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)


class SpeakerOut(BaseModel):
    id: str
    name: str
    enrolled_at: datetime
    sample_duration_sec: float


class TranscriptTurn(BaseModel):
    id: str
    session_id: str
    speaker_id: str | None
    speaker_name: str
    match_confidence: float | None  # None when speaker is unknown / below threshold
    detected_language: str
    source_text: str          # raw IndicConformer output, original language
    translated_text: str      # IndicTrans2 output, English
    start_sec: float
    end_sec: float
    created_at: datetime


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: datetime
    turn_count: int
    speakers: dict[str, int]  # speaker_name -> turn count


class TranscribeResponse(BaseModel):
    turn: TranscriptTurn | None = None
    turns: list[TranscriptTurn] = []
    correction_applied: bool = False