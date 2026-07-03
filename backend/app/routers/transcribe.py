import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form

from app.services import asr_service, translation_service, speaker_service, session_store
from app.utils.audio import load_waveform, resample_waveform, duration_sec
from app.utils.vad import segment_by_voice_activity
from app.utils.diarize import resplit_by_speaker_change
from app.models.schemas import TranscribeResponse

logger = logging.getLogger("medtranscribe.transcribe")
router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])


@router.post("", response_model=TranscribeResponse)
async def transcribe_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    language_hint: str = Form(...),
):
    raw = await audio.read()
    waveform, sr = load_waveform(raw)
    waveform, sr = resample_waveform(waveform, sr, target_sr=16000)

    vad_segments = segment_by_voice_activity(waveform, sr)
    if not vad_segments:
        vad_segments = [(0.0, duration_sec(waveform, sr))]

    # Re-split each VAD segment further wherever the speaker embedding
    # drifts mid-segment — catches back-to-back speaker changes VAD misses.
    segments = []
    for start_sec, end_sec in vad_segments:
        segments.extend(resplit_by_speaker_change(waveform, sr, start_sec, end_sec))

    logger.info("Final segments after diarization re-split: %s", segments)

    created_turns = []

    for start_sec, end_sec in segments:
        start_idx = int(start_sec * sr)
        end_idx = int(end_sec * sr)
        chunk = waveform[start_idx:end_idx]

        if len(chunk) < int(0.3 * sr):
            continue

        speaker_record, confidence = speaker_service.identify_speaker(chunk, sr)
        speaker_name = speaker_record["name"] if speaker_record else "Unknown speaker"
        speaker_id = speaker_record["id"] if speaker_record else None

        asr_result = asr_service.transcribe(chunk, sr, language_hint=language_hint)
        logger.info("Segment %.1f-%.1f -> speaker=%s text=%r", start_sec, end_sec, speaker_name, asr_result["text"])

        english_text = translation_service.translate_to_english(
            asr_result["text"], asr_result["language"]
        )

        turn = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "speaker_id": speaker_id,
            "speaker_name": speaker_name,
            "match_confidence": round(confidence, 3) if speaker_record else None,
            "detected_language": asr_result["language"],
            "source_text": asr_result["text"],
            "translated_text": english_text,
            "start_sec": round(start_sec, 2),
            "end_sec": round(end_sec, 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        session_store.add_turn(session_id, turn)
        created_turns.append(turn)

    return {"turn": created_turns[-1] if created_turns else None, "turns": created_turns}