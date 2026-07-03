"""
Splits a continuous recording into individual speech segments using WebRTC's
voice activity detector. This exists because feeding one long multi-speaker
blob straight into ASR causes hallucination on silences/speaker-changes, and
because speaker identification needs to run per-utterance, not once for an
entire mixed recording.

Requires 16kHz mono int16 PCM — webrtcvad does not accept float32 or other
sample rates.
"""
import webrtcvad
import numpy as np


def _float_to_pcm16(waveform: np.ndarray) -> bytes:
    clipped = np.clip(waveform, -1.0, 1.0)
    return (clipped * 32767).astype(np.int16).tobytes()


def _frame_generator(frame_duration_ms: int, pcm_bytes: bytes, sample_rate: int):
    bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    duration = frame_duration_ms / 1000.0
    offset = 0
    timestamp = 0.0
    while offset + bytes_per_frame <= len(pcm_bytes):
        yield pcm_bytes[offset:offset + bytes_per_frame], timestamp
        timestamp += duration
        offset += bytes_per_frame


def segment_by_voice_activity(
    waveform: np.ndarray,
    sr: int,
    frame_duration_ms: int = 30,
    aggressiveness: int = 2,
    min_segment_sec: float = 0.6,
    max_silence_sec: float = 0.6,
) -> list[tuple[float, float]]:
    """
    Returns [(start_sec, end_sec), ...] for each contiguous speech region.
    `waveform` MUST already be 16kHz mono — resample before calling this.

    aggressiveness: 0 (least aggressive filtering) to 3 (most aggressive).
    max_silence_sec: how long a pause must be before we treat it as a
        segment boundary (i.e. likely speaker turn change or sentence end).
    """
    if sr != 16000:
        raise ValueError(f"VAD requires 16kHz audio, got {sr}Hz — resample first")

    vad = webrtcvad.Vad(aggressiveness)
    pcm = _float_to_pcm16(waveform)
    frames = list(_frame_generator(frame_duration_ms, pcm, sr))
    frame_dur = frame_duration_ms / 1000.0

    segments = []
    seg_start = None
    silence_run = 0.0

    for frame_bytes, ts in frames:
        is_speech = vad.is_speech(frame_bytes, sr)
        if is_speech:
            if seg_start is None:
                seg_start = ts
            silence_run = 0.0
        elif seg_start is not None:
            silence_run += frame_dur
            if silence_run >= max_silence_sec:
                seg_end = ts - silence_run + frame_dur
                if seg_end - seg_start >= min_segment_sec:
                    segments.append((seg_start, seg_end))
                seg_start = None
                silence_run = 0.0

    if seg_start is not None and frames:
        seg_end = frames[-1][1] + frame_dur
        if seg_end - seg_start >= min_segment_sec:
            segments.append((seg_start, seg_end))

    return segments