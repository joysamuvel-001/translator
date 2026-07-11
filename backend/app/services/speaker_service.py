"""
Wraps NVIDIA's TitaNet-Large speaker embedding model. Enrollment stores one
512-d embedding per person (averaged from their 5-10s sample); every
transcribed turn is scored against all enrolled embeddings by cosine
similarity, and the closest match above the configured threshold is
attached to that turn.

Model card: nvidia/speakerverification_en_titanet_large
"""
import json
import logging
import uuid
from datetime import datetime, timezone

import numpy as np
import torch
import torchaudio

from app.core.config import get_settings
from app.utils.audio import duration_sec

logger = logging.getLogger("medtranscribe.speaker")
settings = get_settings()

_model = None
_device = None
TARGET_SR = 16000


def _resolve_device() -> str:
    if settings.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    global _model, _device
    if _model is None:
        import nemo.collections.asr as nemo_asr  # heavy import, deferred

        _device = _resolve_device()
        logger.info("Loading TitaNet (%s) on %s", settings.speaker_model_name, _device)
        _model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
            model_name=settings.speaker_model_name
        ).to(_device)
        _model.eval()
    return _model


def _prepare_waveform(waveform: np.ndarray, sr: int) -> torch.Tensor:
    """Resample to 16kHz mono, matching what TitaNet was trained on."""
    tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        tensor = resampler(tensor)
    return tensor


def embed(waveform: np.ndarray, sr: int) -> np.ndarray:
    model = get_model()
    tensor = _prepare_waveform(waveform, sr).to(_device)
    lengths = torch.tensor([tensor.shape[1]]).to(_device)
    with torch.no_grad():
        _, embedding = model.forward(input_signal=tensor, input_signal_length=lengths)
    return embedding.squeeze(0).cpu().numpy()


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def get_clean_speech_waveform(waveform: np.ndarray, sr: int) -> np.ndarray:
    """
    Runs diarization (VAD) on the waveform and returns a concatenated waveform
    containing only the active speech segments, stripping out silence.
    """
    from app.services import diarization_service
    try:
        segments = diarization_service.diarize(waveform, sr)
        if not segments:
            return waveform
        
        chunks = []
        for seg in segments:
            start_idx = int(seg["start"] * sr)
            end_idx = int(seg["end"] * sr)
            chunks.append(waveform[start_idx:end_idx])
            
        if chunks:
            return np.concatenate(chunks)
    except Exception as e:
        logger.warning("Could not perform speech cleaning on enrollment: %s", e)
    return waveform


def enroll_speaker(name: str, waveform: np.ndarray, sr: int) -> dict:
    # Clean the enrollment waveform (VAD silence stripping) to get a pure voiceprint
    cleaned_waveform = get_clean_speech_waveform(waveform, sr)
    dur = duration_sec(cleaned_waveform, sr)
    
    # Fallback to the original waveform if the cleaned one is too short
    if dur < settings.enrollment_min_seconds:
        logger.warning("Cleaned enrollment audio is too short (%.2fs), falling back to original waveform (%.2fs)",
                       dur, duration_sec(waveform, sr))
        cleaned_waveform = waveform
        dur = duration_sec(waveform, sr)

    if dur < settings.enrollment_min_seconds:
        raise ValueError(
            f"Sample is {dur}s — record at least {settings.enrollment_min_seconds}s for a reliable voiceprint."
        )

    vector = embed(cleaned_waveform, sr)

    # Look for an existing speaker with this exact name — average into it
    # instead of creating a duplicate entry, so repeated enrollments improve
    # one voiceprint rather than fragmenting into several partial ones.
    existing_path = None
    existing_record = None
    for path in settings.speakers_dir.glob("*.json"):
        record = json.loads(path.read_text())
        if record["name"].strip().lower() == name.strip().lower():
            existing_path = path
            existing_record = record
            break

    if existing_record:
        old_vector = np.array(existing_record["embedding"])
        prev_count = existing_record.get("sample_count", 1)
        new_count = prev_count + 1
        averaged = (old_vector * prev_count + vector) / new_count
        averaged = averaged / (np.linalg.norm(averaged) + 1e-9)

        existing_record["embedding"] = averaged.tolist()
        existing_record["sample_count"] = new_count
        existing_record["sample_duration_sec"] = existing_record.get("sample_duration_sec", 0) + dur
        existing_path.write_text(json.dumps(existing_record))
        return {k: v for k, v in existing_record.items() if k != "embedding"}

    speaker_id = str(uuid.uuid4())
    record = {
        "id": speaker_id,
        "name": name,
        "embedding": vector.tolist(),
        "sample_count": 1,
        "enrolled_at": datetime.now(timezone.utc).isoformat(),
        "sample_duration_sec": dur,
    }
    out_path = settings.speakers_dir / f"{speaker_id}.json"
    out_path.write_text(json.dumps(record))
    return {k: v for k, v in record.items() if k != "embedding"}


def list_speakers() -> list[dict]:
    speakers = []
    for path in sorted(settings.speakers_dir.glob("*.json")):
        record = json.loads(path.read_text())
        speakers.append({k: v for k, v in record.items() if k != "embedding"})
    return speakers


def identify_speaker_vector(query_vector: np.ndarray, duration: float = 5.0) -> tuple[dict | None, float]:
    scores = []

    for path in settings.speakers_dir.glob("*.json"):
        record = json.loads(path.read_text())
        score = cosine_similarity(query_vector, np.array(record["embedding"]))
        logger.info("Speaker match candidate: %s scored %.3f", record["name"], score)
        scores.append((record, score))

    if not scores:
        return None, 0.0

    # Sort scores descending
    scores.sort(key=lambda x: x[1], reverse=True)
    best_record, best_score = scores[0]

    # Scale threshold dynamically based on speech duration.
    # Short segments naturally yield lower cosine similarity scores.
    base_threshold = settings.speaker_match_threshold
    if duration < 4.0:
        scale_factor = 0.75 + 0.25 * (max(duration, 0.5) / 4.0)
        threshold = base_threshold * scale_factor
    else:
        threshold = base_threshold

    # Standard check: if it passes the dynamic threshold, it's a solid match
    if best_score >= threshold:
        logger.info("Best match %s accepted via dynamic threshold %.3f (score: %.3f)", 
                    best_record["name"], threshold, best_score)
        return best_record, best_score

    # Relative match strategy for borderline cases (between min_threshold of 0.45 and scaled threshold)
    # This resolves matching for short utterances by evaluating the confidence margin
    # against other enrolled speakers.
    min_threshold = 0.45
    if best_score >= min_threshold:
        if len(scores) == 1:
            logger.info("Best match %s accepted via single-speaker fallback (score: %.3f >= %.3f)", 
                        best_record["name"], best_score, min_threshold)
            return best_record, best_score
        else:
            second_best_record, second_best_score = scores[1]
            margin = best_score - second_best_score
            if margin >= 0.15:
                logger.info("Best match %s accepted via confidence margin of %.3f over %s (score: %.3f, runner-up: %.3f)", 
                            best_record["name"], margin, second_best_record["name"], best_score, second_best_score)
                return best_record, best_score
            else:
                logger.info("Best match %s rejected: score %.3f below threshold %.3f, and margin %.3f is insufficient (< 0.15)",
                            best_record["name"], best_score, threshold, margin)
    else:
        logger.info("Best match %s rejected: score %.3f is below absolute minimum threshold %.3f",
                    best_record["name"] if best_record else None, best_score, min_threshold)

    return None, max(best_score, 0.0)


def identify_speaker(waveform: np.ndarray, sr: int) -> tuple[dict | None, float]:
    query_vector = embed(waveform, sr)
    dur = duration_sec(waveform, sr)
    return identify_speaker_vector(query_vector, duration=dur)