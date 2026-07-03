"""
Wraps AI4Bharat's IndicConformer-600M-Multi ASR model — a single checkpoint
covering all 22 official Indian languages, loaded via `transformers.AutoModel`
(NOT the NeMo per-language checkpoints). Loaded once, lazily, and reused for
every request.

Model card: ai4bharat/indic-conformer-600m-multilingual
Note: this repo is gated — you must accept its terms while logged in on
huggingface.co, and have a valid HF token available (HF_TOKEN env var or
`huggingface-cli login`) before the first load will succeed.
"""
import logging
import torch
import torchaudio
import numpy as np

from app.core.config import get_settings

logger = logging.getLogger("medtranscribe.asr")
settings = get_settings()

_model = None
_device = None

MODEL_ID = "ai4bharat/indic-conformer-600m-multilingual"
TARGET_SR = 16000

# Supported language codes per the model card
SUPPORTED_LANGS = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri",
    "gu": "Gujarati", "hi": "Hindi", "kn": "Kannada", "kok": "Konkani",
    "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi",
    "sa": "Sanskrit", "sat": "Santali", "sd": "Sindhi", "ta": "Tamil",
    "te": "Telugu", "ur": "Urdu",
}

DEFAULT_LANG = getattr(settings, "default_asr_lang", "hi")
DECODING = "ctc" if getattr(settings, "asr_decoding_strategy", "rnnt") == "ctc" else "rnnt"


def _resolve_device() -> str:
    if settings.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_model():
    """Lazily load the single multilingual checkpoint (downloaded once, cached by HF)."""
    global _model, _device
    if _model is None:
        from transformers import AutoModel  # heavy import, deferred

        _device = _resolve_device()
        logger.info("Loading IndicConformer-600M-Multi (%s) on %s", MODEL_ID, _device)
        _model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
        _model = _model.to(_device)
        _model.eval()
    return _model


def _prepare_waveform(waveform: np.ndarray, sr: int) -> torch.Tensor:
    """Convert numpy waveform to a mono, 16kHz torch tensor as the model expects."""
    wav = torch.from_numpy(waveform).float()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)  # (1, T)
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)  # downmix to mono

    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        wav = resampler(wav)

    return wav


def transcribe(waveform: np.ndarray, sr: int, language_hint: str | None = None) -> dict:
    """
    Returns {"text": str, "language": str}. This model has no true auto-detect
    mode either — you must pass a language code — but unlike the per-language
    NeMo checkpoints, switching languages here is free: no new download, no
    new model load, just a different `lang` argument on the same instance.
    """
    lang = (language_hint or DEFAULT_LANG).lower()
    if lang not in SUPPORTED_LANGS:
        logger.warning("Unknown language hint %r, falling back to %s", lang, DEFAULT_LANG)
        lang = DEFAULT_LANG

    model = get_model()
    wav = _prepare_waveform(waveform, sr).to(_device)

    with torch.no_grad():
        text = model(wav, lang, DECODING)

    # model() may return a str directly, or occasionally a list/tuple depending
    # on version — normalize defensively.
    if isinstance(text, (list, tuple)):
        text = text[0]

    return {"text": str(text).strip(), "language": SUPPORTED_LANGS[lang]}