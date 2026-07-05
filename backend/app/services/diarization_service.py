import logging
import os
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

from app.core.config import get_settings

logger = logging.getLogger("medtranscribe.diarization")
settings = get_settings()

# ─────────────────────────────────────────────────────────────────────────
# Compatibility shim: newer torchaudio (2.1+) removed several APIs that
# pyannote.audio still references at import time.
# ─────────────────────────────────────────────────────────────────────────
if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *args, **kwargs: None
    logger.info("Patched torchaudio.set_audio_backend (removed in this torchaudio version)")

if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]
    logger.info("Patched torchaudio.list_audio_backends (removed in this torchaudio version)")

if not hasattr(torchaudio, "AudioMetaData"):
    @dataclass
    class _AudioMetaDataShim:
        sample_rate: int = 0
        num_frames: int = 0
        num_channels: int = 0
        bits_per_sample: int = 0
        encoding: str = ""

    torchaudio.AudioMetaData = _AudioMetaDataShim
    logger.info("Patched torchaudio.AudioMetaData (removed in this torchaudio version)")

# ─────────────────────────────────────────────────────────────────────────
# Bug workaround: speechbrain's LazyModule.__getattr__ raises ImportError
# when an optional dependency (k2, flair, etc.) isn't installed. But
# inspect.getmodule() (called deep inside pytorch_lightning's checkpoint
# loader, via is_scripting's inspect.stack() walk) uses
# hasattr(module, '__file__'), and hasattr() only swallows AttributeError
# — not ImportError — so the crash propagates all the way up and kills
# the whole request. This has nothing to do with actually needing
# k2/flair/etc — we never touch speechbrain's NLP/k2 integrations at all.
# Patch __getattr__ so failed lazy imports raise AttributeError instead,
# matching what hasattr()/getattr() actually expect.
# ─────────────────────────────────────────────────────────────────────────
try:
    from speechbrain.utils.importutils import LazyModule

    _original_lazy_getattr = LazyModule.__getattr__

    def _patched_lazy_getattr(self, attr):
        try:
            return _original_lazy_getattr(self, attr)
        except ImportError as e:
            raise AttributeError(str(e)) from e

    LazyModule.__getattr__ = _patched_lazy_getattr
    logger.info("Patched speechbrain LazyModule.__getattr__ (ImportError -> AttributeError)")
except Exception as _patch_exc:
    logger.warning("Could not patch speechbrain LazyModule: %s", _patch_exc)

from pyannote.audio import Pipeline

_pipeline = None


def _resolve_device() -> str:
    if settings.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_pipeline():
    global _pipeline
    if _pipeline is None:

        hf_token = settings.hf_token or os.environ.get("HF_TOKEN")
        if not hf_token:
            raise EnvironmentError(
                "HF_TOKEN is required to load pyannote/speaker-diarization-3.1 "
                "(gated model) — add it to your .env file."
            )

        logger.info("Loading pyannote speaker-diarization-3.1 on %s", _resolve_device())

        # pyannote's checkpoint predates PyTorch 2.6's weights_only=True default,
        # and references several internal pyannote/omegaconf classes not in the
        # safe-globals allowlist. We trust this checkpoint (official HF model),
        # so force weights_only=False just for this load rather than
        # allowlisting classes one at a time.
        _original_torch_load = torch.load

        def _patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _original_torch_load(*args, **kwargs)

        torch.load = _patched_load
        try:
            _pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token,
            )
        finally:
            torch.load = _original_torch_load

        _pipeline.to(torch.device(_resolve_device()))
    return _pipeline


def diarize(waveform: np.ndarray, sr: int) -> list[dict]:
    """
    Returns [{"start": float, "end": float, "diarized_label": "SPEAKER_00"}, ...].
    """
    pipeline = get_pipeline()

    tensor = torch.from_numpy(waveform).float().unsqueeze(0)  # -> (1, time)
    audio_input = {"waveform": tensor, "sample_rate": sr}
    diarization = pipeline(audio_input)

    segments = []
    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
        segments.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "diarized_label": speaker_label,
        })

    segments.sort(key=lambda s: s["start"])
    logger.info("pyannote found %d speaker turns", len(segments))
    return segments