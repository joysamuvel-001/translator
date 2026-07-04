import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from app.services import asr_service, translation_service, speaker_service, session_store, medgemma_service
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
        created_turns.append(turn)

    # ── MedGemma correction pass ────────────────────────────────────────
    # Best-effort: a RunPod failure must never lose a transcript that
    # ASR/diarization already produced. Runs in a threadpool since
    # medgemma_service is synchronous (blocking polling loop).
    correction_applied = False
    if created_turns:
        slim_conversation = [
            {"speaker": t["speaker_name"], "text": t["source_text"]}
            for t in created_turns
        ]
        try:
            logger.info("Sending %d turns to MedGemma for correction", len(slim_conversation))
            result = await run_in_threadpool(medgemma_service.correct_transcript, slim_conversation)
            corrected = result["corrected_conversation"]

            if len(corrected) == len(created_turns):
                for turn, fixed in zip(created_turns, corrected):
                    turn["speaker_name"] = fixed.get("speaker", turn["speaker_name"])
                    turn["source_text"] = fixed.get("text", turn["source_text"])
                correction_applied = True
                logger.info("MedGemma correction applied to %d turns", len(created_turns))
            else:
                logger.warning(
                    "MedGemma turn count mismatch (%d vs %d) — correction skipped",
                    len(corrected), len(created_turns)
                )
        except Exception as exc:
            logger.warning("MedGemma correction failed (%s) — returning uncorrected transcript", exc)

    # Persist turns (post-correction, so stored transcript has the fix applied)
    for turn in created_turns:
        session_store.add_turn(session_id, turn)

    return {
        "turn": created_turns[-1] if created_turns else None,
        "turns": created_turns,
        "correction_applied": correction_applied,
    }