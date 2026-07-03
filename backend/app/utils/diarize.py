"""
Re-splits long VAD segments wherever the speaker embedding changes
mid-segment — i.e. lightweight diarization. WebRTC VAD only detects
speech-vs-silence, so two people talking back-to-back with a short pause
get merged into one segment, producing a blended embedding that doesn't
match either speaker. This scans a long segment in overlapping windows,
comparing consecutive window embeddings, and cuts a new boundary wherever
similarity drops sharply — a likely speaker change.
"""
import numpy as np
from app.services.speaker_service import embed, cosine_similarity

WINDOW_SEC = 2.0
STEP_SEC = 1.0
CHANGE_THRESHOLD = 0.45  # similarity below this between adjacent windows = likely speaker change
MIN_SPLIT_SEGMENT_SEC = 1.5


def resplit_by_speaker_change(waveform: np.ndarray, sr: int, start_sec: float, end_sec: float) -> list[tuple[float, float]]:
    duration = end_sec - start_sec
    if duration <= WINDOW_SEC * 1.5:
        return [(start_sec, end_sec)]  # too short to bother re-splitting

    window_samples = int(WINDOW_SEC * sr)
    step_samples = int(STEP_SEC * sr)
    seg_start_idx = int(start_sec * sr)
    seg_end_idx = int(end_sec * sr)

    embeddings = []
    positions = []
    idx = seg_start_idx
    while idx + window_samples <= seg_end_idx:
        chunk = waveform[idx: idx + window_samples]
        vec = embed(chunk, sr)
        embeddings.append(vec)
        positions.append(idx / sr)
        idx += step_samples

    if len(embeddings) < 2:
        return [(start_sec, end_sec)]

    boundaries = [start_sec]
    for i in range(1, len(embeddings)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        if sim < CHANGE_THRESHOLD:
            candidate = positions[i]
            if candidate - boundaries[-1] >= MIN_SPLIT_SEGMENT_SEC:
                boundaries.append(candidate)
    boundaries.append(end_sec)

    return [(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)
            if boundaries[i + 1] - boundaries[i] >= MIN_SPLIT_SEGMENT_SEC]