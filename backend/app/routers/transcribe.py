import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from app.services import asr_service, speaker_service, session_store, medgemma_service, diarization_service, translation_service
from app.utils.audio import load_waveform, resample_waveform, duration_sec
from app.utils.merge import merge_by_identity
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

    # Real diarization: detects actual speaker-change boundaries, not just
    # silence gaps. Runs in a threadpool since pyannote's pipeline call is
    # a blocking, CPU/GPU-heavy synchronous operation.
    diarized_segments = await run_in_threadpool(diarization_service.diarize, waveform, sr)

    if not diarized_segments:
        diarized_segments = [{
            "start": 0.0,
            "end": duration_sec(waveform, sr),
            "diarized_label": "SPEAKER_00",
        }]

    logger.info("Diarized segments: %s", diarized_segments)

    # Identify each diarized segment against enrolled voiceprints via TitaNet.
    identified = []
    for seg in diarized_segments:
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        chunk = waveform[start_idx:end_idx]

        if len(chunk) < int(0.3 * sr):
            continue

        speaker_record, confidence = speaker_service.identify_speaker(chunk, sr)
        identified.append({
            "start": seg["start"],
            "end": seg["end"],
            "diarized_label": seg["diarized_label"],
            "speaker_id": speaker_record["id"] if speaker_record else None,
            # Fallback to the diarized label (SPEAKER_00 etc.) rather than a
            # generic "Unknown speaker" for everyone — keeps different
            # unidentified people visually distinct instead of collapsing
            # them into one indistinguishable "Unknown".
            "speaker_name": speaker_record["name"] if speaker_record else seg["diarized_label"],
            "confidence": confidence,
        })

    # Merge consecutive segments that pyannote diarized as the same speaker
    # AND TitaNet also identified as the same enrolled person — with a small
    # gap allowance for natural pauses within a continuous turn.
    merged_segments = merge_by_identity(identified)
    logger.info("Segments after identity-based merge: %s", merged_segments)

    created_turns = []
    for seg in merged_segments:
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        chunk = waveform[start_idx:end_idx]

        asr_result = asr_service.transcribe(chunk, sr, language_hint=language_hint)
        logger.info(
            "Segment %.1f-%.1f speaker=%s (diarized=%s) text=%r",
            seg["start"], seg["end"], seg["speaker_name"], seg["diarized_label"], asr_result["text"]
        )

        translated = await run_in_threadpool(
            translation_service.translate_to_english, asr_result["text"], asr_result["language"]
        )
        logger.info("IndicTrans2 raw translation: %r", translated)

        turn = {
            "id": str(uuid.uuid4()),
            "session_id": session_id,
            "speaker_id": seg["speaker_id"],
            "speaker_name": seg["speaker_name"],
            "match_confidence": round(seg["confidence"], 3) if seg["speaker_id"] else None,
            "detected_language": asr_result["language"],
            "source_text": asr_result["text"],       # raw ASR, original language — never overwritten
            "translated_text": translated,           # IndicTrans2 output — MedGemma corrects THIS below
            "start_sec": round(seg["start"], 2),
            "end_sec": round(seg["end"], 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        created_turns.append(turn)

    correction_applied = False
    if created_turns:
        # MedGemma corrects the English translation (medical terms, drug
        # names, dosages) — NOT the original-language source_text. Runs
        # once per recording as a single batched job.
        slim_conversation = [
            {"speaker": t["speaker_name"], "text": t["translated_text"]}
            for t in created_turns
        ]
        try:
            logger.info("Sending %d turns to MedGemma", len(slim_conversation))
            result = await run_in_threadpool(medgemma_service.correct_transcript, slim_conversation)
            corrected = result["corrected_conversation"]
            if len(corrected) == len(created_turns):
                for turn, fixed in zip(created_turns, corrected):
                    turn["speaker_name"] = fixed.get("speaker", turn["speaker_name"])
                    turn["translated_text"] = fixed.get("text", turn["translated_text"])
                correction_applied = True
                logger.info("MedGemma correction applied to %d turns", len(created_turns))
            else:
                logger.warning("MedGemma turn count mismatch (%d vs %d)", len(corrected), len(created_turns))
        except Exception as exc:
            logger.warning("MedGemma correction failed (%s)", exc)

    for turn in created_turns:
        session_store.add_turn(session_id, turn)

    return {
        "turn": created_turns[-1] if created_turns else None,
        "turns": created_turns,
        "correction_applied": correction_applied,
    }