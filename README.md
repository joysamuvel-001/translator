# MedTranscribe

MedTranscribe is a full-stack speech-to-text and speaker-aware transcription application for clinical or meeting-style conversations. It records audio, enrolls speakers, transcribes speech, identifies speakers, and translates the transcript into English.

## Overview

This project combines:
- a FastAPI backend for audio processing, speaker enrollment, transcription, diarization, and translation
- a React + Vite frontend for recording, speaker management, and transcript viewing

## Key Features

- Record audio and submit it for transcription
- Enroll speaker voices for later identification
- Detect speaker turns and assign speaker labels
- Transcribe speech in multiple Indic languages and English
- Translate recognized speech to English
- Store sessions and transcript turns for review

## Tech Stack

### Backend
- Python
- FastAPI
- PyTorch / torchaudio
- NeMo ASR models
- webrtcvad for voice activity detection
- Pydantic settings

### Frontend
- React
- Vite
- Tailwind CSS
- Lucide icons

## Project Structure

- backend/app — FastAPI application, routers, services, and models
- backend/storage — persisted sessions, audio, and speaker data
- frontend/src — React UI components and hooks

## Prerequisites

- Python 3.10+ recommended
- Node.js 18+ and npm
- A working internet connection for downloading AI models on first run
- GPU is recommended for faster inference, but CPU is supported

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate   # On Windows
pip install -r requirements.txt
```

Start the API server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at:
- http://localhost:8000/docs for Swagger UI
- http://localhost:8000/api/health for health checks

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

Then open:
- http://localhost:5173

## Usage

1. Start the backend and frontend.
2. Open the frontend in your browser.
3. Enroll one or more speakers by uploading short voice samples.
4. Start a recording session and submit audio for transcription.
5. Review the generated transcript turns and speaker assignments.

## API Highlights

- POST /api/speakers/enroll — enroll a new speaker
- GET /api/speakers — list enrolled speakers
- POST /api/transcribe — upload audio and receive transcript turns
- POST /api/sessions — create a new session
- GET /api/sessions — list sessions

## Notes

The first startup can take several minutes because the backend downloads and loads the speech and speaker models. If you are on CPU, expect slower performance than on GPU.

## License

This project is intended for local development and demonstration purposes.
