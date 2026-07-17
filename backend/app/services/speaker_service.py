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


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def embed_normalized(waveform: np.ndarray, sr: int) -> np.ndarray:
    """TitaNet returns unnormalized vectors; normalize so they can be averaged."""
    return _normalize(embed(waveform, sr))


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

    vector = embed_normalized(cleaned_waveform, sr)

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
        # Normalize on read as well: records written before embeddings were
        # normalized at enrollment are still raw, and averaging a raw vector
        # against a unit one lets whichever is larger dominate the voiceprint.
        old_vector = _normalize(np.array(existing_record["embedding"]))
        prev_count = existing_record.get("sample_count", 1)
        new_count = prev_count + 1
        averaged = _normalize(old_vector * prev_count + vector)

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


def assign_labels_to_speakers(
    label_vectors: dict[str, np.ndarray],
) -> dict[str, tuple[dict | None, float]]:
    """
    Decides which diarized label belongs to which enrolled speaker, by
    comparing the labels against each other rather than against a fixed
    threshold each.

    This is a *relative* choice on purpose. With one voiceprint enrolled (the
    doctor), an absolute threshold cannot express "this label is more
    doctor-like than that one" — every voice either clears the bar or doesn't,
    so an unenrolled patient scoring moderately well gets stamped as the
    doctor. Scoring the labels against each other and letting the best one win
    is what actually answers "which of these people is the doctor".

    Returns {label: (speaker_record | None, score)}. Labels no enrolled
    speaker claims come back as None for the caller to name.
    """
    assignment: dict[str, tuple[dict | None, float]] = {
        label: (None, 0.0) for label in label_vectors
    }

    enrolled = [json.loads(p.read_text()) for p in settings.speakers_dir.glob("*.json")]
    if not enrolled or not label_vectors:
        return assignment

    by_id = {r["id"]: r for r in enrolled}

    candidates = []
    for record in enrolled:
        ref = _normalize(np.array(record["embedding"]))
        for label, vec in label_vectors.items():
            score = cosine_similarity(ref, vec)
            logger.info("Label %s vs %s: %.3f", label, record["name"], score)
            candidates.append((score, record["id"], label))

    candidates.sort(key=lambda c: c[0], reverse=True)

    floor = settings.speaker_presence_floor
    claimed_labels: set[str] = set()
    claimed_speakers: set[str] = set()
    primary_score: dict[str, float] = {}

    # Pass 1: each enrolled speaker takes the single label that scores highest
    # against their voiceprint. The floor is what keeps argmax from crowning
    # someone when the enrolled speaker never actually speaks in this audio.
    for score, sid, label in candidates:
        if score < floor:
            break
        if sid in claimed_speakers or label in claimed_labels:
            continue
        assignment[label] = (by_id[sid], score)
        claimed_labels.add(label)
        claimed_speakers.add(sid)
        primary_score[sid] = score
        logger.info("Label %s -> %s (primary, score %.3f)", label, by_id[sid]["name"], score)

    # Pass 2: pyannote sometimes splits one person across several labels. A
    # leftover label scoring nearly as high as a speaker's primary label is
    # that same person, not a new one — without this, a doctor talking alone
    # gets a phantom second speaker invented from their own split speech.
    for score, sid, label in candidates:
        if label in claimed_labels or sid not in primary_score:
            continue
        if score < floor:
            continue
        margin = primary_score[sid] - score
        if margin <= settings.speaker_same_voice_margin:
            assignment[label] = (by_id[sid], score)
            claimed_labels.add(label)
            logger.info(
                "Label %s -> %s (same voice over-segmented, score %.3f, %.3f below primary)",
                label, by_id[sid]["name"], score, margin,
            )

    for label in label_vectors:
        if label not in claimed_labels:
            best = max((c[0] for c in candidates if c[2] == label), default=0.0)
            logger.info("Label %s unclaimed (best score %.3f) — caller will name it", label, best)

    return assignment