import uuid
import logging
from datetime import datetime, timezone
import numpy as np

from fastapi import APIRouter, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.services import asr_service, speaker_service, session_store, medgemma_service, diarization_service, translation_service
from app.utils.audio import load_waveform, resample_waveform, duration_sec
from app.utils.merge import merge_by_identity
from app.models.schemas import TranscribeResponse

logger = logging.getLogger("medtranscribe.transcribe")
router = APIRouter(prefix="/api/transcribe", tags=["transcribe"])

# Below this much pooled speech a label can't be identified reliably, so it
# falls through to the unenrolled name rather than guessing.
MIN_SPEECH_FOR_ID_SEC = 0.5


@router.post("", response_model=TranscribeResponse)
async def transcribe_turn(
    audio: UploadFile = File(...),
    session_id: str = Form(...),
    language_hint: str = Form(...),
):
    settings = get_settings()
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

    # Pool each label's speech into one waveform and embed that once, rather
    # than embedding each segment separately and averaging. A short backchannel
    # ("mm-hmm") carries too little signal to embed alone, and pooling mirrors
    # enrollment, which also embeds concatenated clean speech.
    label_chunks: dict[str, list] = {}
    for seg in diarized_segments:
        start_idx = int(seg["start"] * sr)
        end_idx = int(seg["end"] * sr)
        label_chunks.setdefault(seg["diarized_label"], []).append(waveform[start_idx:end_idx])

    label_vectors = {}
    for label, chunks in label_chunks.items():
        pooled = np.concatenate(chunks)
        if len(pooled) < int(MIN_SPEECH_FOR_ID_SEC * sr):
            logger.info(
                "Label %s has only %.2fs of speech — too little to identify",
                label, len(pooled) / sr,
            )
            continue
        try:
            label_vectors[label] = speaker_service.embed_normalized(pooled, sr)
        except Exception as e:
            logger.error("Failed to embed label %s: %s", label, e)

    label_identity = speaker_service.assign_labels_to_speakers(label_vectors)

    # Name whatever no enrolled voiceprint claimed. Only the doctor is
    # enrolled, so any other voice in the room is the patient.
    label_order = list(dict.fromkeys(seg["diarized_label"] for seg in diarized_segments))
    unenrolled = [lbl for lbl in label_order if label_identity.get(lbl, (None, 0.0))[0] is None]
    base_name = settings.unenrolled_speaker_name
    patient_names = {
        label: (base_name if len(unenrolled) == 1 else f"{base_name} {i}")
        for i, label in enumerate(unenrolled, start=1)
    }

    identified = []
    for seg in diarized_segments:
        label = seg["diarized_label"]
        speaker_record, confidence = label_identity.get(label, (None, 0.0))
        identified.append({
            "start": seg["start"],
            "end": seg["end"],
            "diarized_label": label,
            "speaker_id": speaker_record["id"] if speaker_record else None,
            "speaker_name": speaker_record["name"] if speaker_record else patient_names[label],
            "confidence": confidence,
        })

    # Rejoin consecutive segments pyannote gave the same label, with a small
    # gap allowance so a natural pause mid-turn isn't split into two turns.
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
            "correction_applied": False,              # frontend can show a "correcting..." badge until this flips true
            "start_sec": round(seg["start"], 2),
            "end_sec": round(seg["end"], 2),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        created_turns.append(turn)

    correction_applied = False
    if created_turns and not settings.pause_medgemma:
        # MedGemma corrects the English translation (medical terms, drug
        # names, dosages) — NOT the original-language source_text. Runs
        # once per recording as a single batched job, and MUST complete
        # (or fail) before we respond — the frontend should only ever see
        # the final, corrected text, never an intermediate uncorrected one.
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
    elif created_turns and settings.pause_medgemma:
        logger.info("MedGemma correction bypassed (paused dynamically). Returning translated output directly.")

    for turn in created_turns:
        session_store.add_turn(session_id, turn)

    return {
        "turn": created_turns[-1] if created_turns else None,
        "turns": created_turns,
        "correction_applied": correction_applied,
    }