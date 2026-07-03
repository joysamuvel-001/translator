from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services import speaker_service
from app.utils.audio import load_waveform
from app.models.schemas import SpeakerOut

router = APIRouter(prefix="/api/speakers", tags=["speakers"])


@router.get("", response_model=list[SpeakerOut])
async def get_speakers():
    return speaker_service.list_speakers()


@router.post("/enroll", response_model=SpeakerOut)
async def enroll(name: str = Form(...), sample: UploadFile = File(...)):
    raw = await sample.read()
    try:
        waveform, sr = load_waveform(raw)
        record = speaker_service.enroll_speaker(name.strip(), waveform, sr)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return record
