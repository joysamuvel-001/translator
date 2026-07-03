import io
import numpy as np
import soundfile as sf
import torch
import torchaudio
from app.core.config import get_settings

settings = get_settings()


def load_waveform(raw_bytes: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data.astype(np.float32), sr


def resample_waveform(waveform: np.ndarray, sr: int, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Resample to target_sr (mono). Used to normalize audio once, centrally,
    before it's handed to VAD, ASR, or the speaker embedding model — all of
    which expect 16kHz."""
    if sr == target_sr:
        return waveform, sr
    tensor = torch.from_numpy(waveform).float().unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
    resampled = resampler(tensor).squeeze(0).numpy()
    return resampled.astype(np.float32), target_sr


def duration_sec(waveform: np.ndarray, sr: int) -> float:
    return round(len(waveform) / float(sr), 2)