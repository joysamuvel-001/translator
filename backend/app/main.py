import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import enrollment, transcribe, sessions

logging.basicConfig(level=logging.INFO)
settings = get_settings()

for d in (settings.storage_dir, settings.speakers_dir, settings.sessions_dir, settings.audio_dir):
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(enrollment.router)
app.include_router(transcribe.router)
app.include_router(sessions.router)


@app.on_event("startup")
async def warm_models():
    logging.info("Warming models — this may take a couple minutes on CPU...")
    from app.services import speaker_service, asr_service, translation_service

    speaker_service.get_model()
    logging.info("TitaNet loaded")

    asr_service.get_model()
    logging.info("IndicConformer loaded")

    translation_service.get_model()
    logging.info("IndicTrans2 loaded")

    logging.info("All models warmed — ready to serve requests")


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": settings.app_name}