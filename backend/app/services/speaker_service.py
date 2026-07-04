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


def enroll_speaker(name: str, waveform: np.ndarray, sr: int) -> dict:
    dur = duration_sec(waveform, sr)
    if dur < settings.enrollment_min_seconds:
        raise ValueError(
            f"Sample is {dur}s — record at least {settings.enrollment_min_seconds}s for a reliable voiceprint."
        )

    vector = embed(waveform, sr)

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


def identify_speaker(waveform: np.ndarray, sr: int) -> tuple[dict | None, float]:
    query_vector = embed(waveform, sr)
    best_record, best_score = None, -1.0

    for path in settings.speakers_dir.glob("*.json"):
        record = json.loads(path.read_text())
        score = cosine_similarity(query_vector, np.array(record["embedding"]))
        logger.info("Speaker match: %s scored %.3f", record["name"], score)  # add this
        if score > best_score:
            best_record, best_score = record, score

    logger.info("Best match: %s (%.3f), threshold=%.3f", 
                best_record["name"] if best_record else None, best_score, settings.speaker_match_threshold)  # add this

    if best_record is None or best_score < settings.speaker_match_threshold:
        return None, max(best_score, 0.0)
    return best_record, best_score