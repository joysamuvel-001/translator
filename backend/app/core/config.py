"""
Central app configuration. Reads from environment / .env so model paths,
storage locations and thresholds aren't hardcoded across the codebase.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Storage lives OUTSIDE the backend/ folder (sibling to backend/, not inside
# it) so `uvicorn --reload` never watches it. Writing session/speaker JSON
# files mid-request was triggering WatchFiles restarts that killed the
# server before the HTTP response could be sent, causing ERR_EMPTY_RESPONSE
# in the browser even though the backend had actually succeeded.
DATA_DIR = BASE_DIR.parent / "data"


class Settings(BaseSettings):
    # --- General ---
    app_name: str = "MedTranscribe API"
    cors_origins: list[str] = ["http://localhost:5173"]

    # --- Storage ---
    storage_dir: Path = DATA_DIR
    speakers_dir: Path = DATA_DIR / "speakers"
    sessions_dir: Path = DATA_DIR / "sessions"
    audio_dir: Path = DATA_DIR / "audio"

    # --- ASR: AI4Bharat IndicConformer (multilingual) ---
    asr_model_name: str = "ai4bharat/indicconformer_stt_multilingual"
    asr_decoding_strategy: str = "rnnt"
    sample_rate: int = 16000

    # --- Translation: AI4Bharat IndicTrans2 (Indic -> English) ---
    translation_model_name: str = "ai4bharat/indictrans2-indic-en-1B"
    translation_target_lang: str = "eng_Latn"

    # --- Speaker ID: NVIDIA TitaNet-Large ---
    speaker_model_name: str = "nvidia/speakerverification_en_titanet_large"
    speaker_match_threshold: float = 0.65
    enrollment_min_seconds: float = 5.0
    enrollment_max_seconds: float = 10.0

    # --- Device ---
    device: str = "cuda"

    # --- Correction: MedGemma via RunPod serverless (optional — disabled
    # by default in transcribe.py due to hallucinated diagnosis
    # substitutions; see README Known Limitations). Made optional here so
    # the app can start without a RunPod endpoint configured.
    runpod_api_key: str = ""
    runpod_medgemma_endpoint_id: str = ""
    hf_token: str = ""

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()