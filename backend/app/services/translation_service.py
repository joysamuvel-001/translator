"""
Wraps AI4Bharat's IndicTrans2 (Indic -> English) HuggingFace model. Used to
produce the English line shown under each transcript turn so a multilingual
consult is readable by anyone reviewing the chart later.

Model card: ai4bharat/indictrans2-indic-en-1B
Requires: pip install IndicTransToolkit
"""
import logging
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor

from app.core.config import get_settings

logger = logging.getLogger("medtranscribe.translate")
settings = get_settings()

_model = None
_tokenizer = None
_processor = None
_device = None

# IndicTrans2 FLORES-style source tags for the languages IndicConformer covers.
SRC_LANG_TAGS = {
    "Hindi": "hin_Deva", "Tamil": "tam_Taml", "Telugu": "tel_Telu",
    "Kannada": "kan_Knda", "Malayalam": "mal_Mlym", "Marathi": "mar_Deva",
    "Gujarati": "guj_Gujr", "Bengali": "ben_Beng", "Punjabi": "pan_Guru",
    "Odia": "ory_Orya", "Assamese": "asm_Beng", "Urdu": "urd_Arab",
    "English": "eng_Latn",
}

TGT_LANG_TAG = "eng_Latn"


def _resolve_device() -> str:
    if settings.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _load():
    global _model, _tokenizer, _processor, _device
    if _model is None:
        _device = _resolve_device()
        logger.info("Loading IndicTrans2 (%s) on %s", settings.translation_model_name, _device)
        _tokenizer = AutoTokenizer.from_pretrained(
            settings.translation_model_name, trust_remote_code=True
        )
        _model = AutoModelForSeq2SeqLM.from_pretrained(
            settings.translation_model_name, trust_remote_code=True
        ).to(_device)
        _model.eval()
        _processor = IndicProcessor(inference=True)
    return _model, _tokenizer, _processor


# add this function anywhere in translation_service.py, e.g. right after _load()
def get_model():
    """Public entrypoint to force-load the model, e.g. for startup warmup."""
    _load()


def translate_to_english(text: str, source_language: str) -> str:
    if not text.strip():
        return ""
    if source_language == "English":
        return text  # nothing to do, already English

    model, tokenizer, ip = _load()
    src_tag = SRC_LANG_TAGS.get(source_language, "hin_Deva")

    # IndicProcessor handles normalization + placeholder tagging for numerals,
    # dates, etc. — required preprocessing, not just string concatenation.
    batch_preprocessed = ip.preprocess_batch([text], src_lang=src_tag, tgt_lang=TGT_LANG_TAG)

    inputs = tokenizer(
        batch_preprocessed,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=256,
    ).to(_device)

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_length=256,
            num_beams=5,
            early_stopping=True,
            use_cache=False,  # avoids the past_key_values None crash on transformers 4.57+
        )

    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    # IndicProcessor also postprocesses to restore placeholders (numbers/dates)
    final_output = ip.postprocess_batch(decoded, lang=TGT_LANG_TAG)
    return final_output[0].strip()