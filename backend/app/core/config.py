"""
Central app configuration. Reads from environment / .env so model paths,
storage locations and thresholds aren't hardcoded across the codebase.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # --- General ---
    app_name: str = "MedTranscribe API"
    cors_origins: list[str] = ["http://localhost:5173"]

    # --- Storage ---
    storage_dir: Path = BASE_DIR / "storage"
    speakers_dir: Path = BASE_DIR / "storage" / "speakers"
    sessions_dir: Path = BASE_DIR / "storage" / "sessions"
    audio_dir: Path = BASE_DIR / "storage" / "audio"

    # --- ASR: AI4Bharat IndicConformer (multilingual) ---
    # Hybrid CTC/RNNT NeMo checkpoint, supports 22 Indic languages + English.
    asr_model_name: str = "ai4bharat/indicconformer_stt_multilingual"
    asr_decoding_strategy: str = "rnnt"  # "ctc" also available on the hybrid model
    sample_rate: int = 16000

    # --- Translation: AI4Bharat IndicTrans2 (Indic -> English) ---
    translation_model_name: str = "ai4bharat/indictrans2-indic-en-1B"
    translation_target_lang: str = "eng_Latn"

    # --- Speaker ID: NVIDIA TitaNet-Large ---
    speaker_model_name: str = "nvidia/speakerverification_en_titanet_large"
    speaker_match_threshold: float = 0.65  # cosine similarity floor to accept an identity
    enrollment_min_seconds: float = 5.0
    enrollment_max_seconds: float = 10.0

    # --- Device ---
    device: str = "cuda"  # falls back to "cpu" automatically if unavailable

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()

