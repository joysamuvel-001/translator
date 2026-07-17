# MedTranscribe

Multilingual consult transcription for clinical settings. Records a doctor–patient conversation, identifies who's speaking, transcribes it in the original Indian language, corrects medical terminology with a domain-tuned LLM, and translates it to English — all in one pipeline.

## Features

- **Multilingual ASR** — transcribes Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Marathi, Gujarati, Punjabi, and other Indic languages via AI4Bharat's IndicConformer.
- **Speaker identification** — enroll a doctor/patient's voice once (5–10s sample); TitaNet recognizes them in every future session. Unenrolled speakers are labeled "Unknown speaker" rather than misattributed.
- **Voice-activity + speaker-change segmentation** — a single recording is automatically split into per-utterance segments, so overlapping or back-to-back speakers don't get merged into one garbled block, while natural pauses from the *same* speaker don't fragment into unnecessary extra turns.
- **Medical term correction (MedGemma)** — raw ASR output is passed to a MedGemma 4B model deployed on RunPod serverless, which corrects medical terminology, drug names, and dosages that general-purpose ASR frequently mishears or hallucinates.
- **English translation** — AI4Bharat's IndicTrans2 translates the *corrected* transcript to English for review, charting, or handoff.
- **Session history** — each consult is saved with full turn-by-turn detail (speaker, language, timing, confidence).

## Architecture

```
1. Receive audio blob (webm) + session_id + language_hint
        │
        ▼
2. Decode to waveform, resample to 16kHz mono
   (load_waveform + resample_waveform)
        │
        ▼
3. Diarization (diarization_service.diarize)
   → pyannote 3.1 returns speaker turns with labels
     (SPEAKER_00, SPEAKER_01, …). These labels are the ONLY
     authority on where the voice changes.
        │
        ▼
4. Label identification (speaker_service.assign_labels_to_speakers)
   → pools each label's speech, embeds it once with TitaNet, and
     assigns labels to enrolled speakers by comparing the labels
     AGAINST EACH OTHER, not against a fixed threshold each.
     Only the doctor is enrolled, so the best-scoring label is the
     doctor and unclaimed labels become "Patient".
        │
        ▼
5. Merge (merge.merge_by_identity)
   → rejoins consecutive segments sharing a pyannote label across
     short pauses. Never merges across different labels.
        │
        ▼
   FOR EACH final segment:
        │
        ├─► 6. ASR transcription (asr_service.transcribe)
        │      → forced to `language_hint`, produces source_text
        │      → asr_result["language"] = detected_language
        │
        ├─► 7. Translation (translation_service.translate_to_english)
        │      → IndicTrans2 translates source_text → translated_text
        │      → THIS RUNS BEFORE CORRECTION
        │
        └─► 8. Turn dict built and appended to created_turns
              { speaker_name, source_text (original lang),
                translated_text (English, UNCORRECTED),
                detected_language, timing, confidence }
        │
        ▼
9. Build slim_conversation = [{speaker, text: source_text}, ...]
   for ALL turns from this recording (batched, one call)
        │
        ▼
10. MedGemma correction (medgemma_service.correct_transcript)
    → sends slim_conversation to RunPod as ONE job
    → polls until COMPLETED
    → returns corrected {speaker, text} per turn
        │
        ▼
11. Overwrite: turn["source_text"] = fixed["text"]
    (MedGemma's output REPLACES source_text)
    → translated_text is NEVER touched again — it keeps
      whatever IndicTrans2 produced in step 7, from the
      UNCORRECTED source_text
        │
        ▼
12. Persist all turns to session_store
        │
        ▼
13. Return { turn, turns, correction_applied } to frontend
```

## Project Structure

```
medtranscribe/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI app, CORS, router wiring, model warmup
│   │   ├── core/
│   │   │   └── config.py               # env-driven settings
│   │   ├── models/
│   │   │   └── schemas.py              # Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── enrollment.py           # POST /api/speakers/enroll, GET /api/speakers
│   │   │   ├── transcribe.py           # POST /api/transcribe (core pipeline)
│   │   │   └── sessions.py             # session CRUD
│   │   ├── services/
│   │   │   ├── asr_service.py          # IndicConformer wrapper
│   │   │   ├── translation_service.py  # IndicTrans2 wrapper
│   │   │   ├── speaker_service.py      # TitaNet enroll/identify
│   │   │   ├── medgemma_service.py     # RunPod MedGemma correction client
│   │   │   └── session_store.py        # JSON-file session/turn persistence
│   │   └── utils/
│   │       ├── audio.py                # webm→wav decode, resampling
│   │       ├── vad.py                  # voice-activity segmentation
│   │       ├── diarize.py              # speaker-change re-splitting
│   │       └── merge.py                # merge same-speaker segments across pauses
│   ├── storage/
│   │   ├── speakers/                   # one JSON per enrolled voiceprint
│   │   ├── sessions/                   # one JSON per session (with turns)
│   │   └── audio/                      # optional raw audio archive
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx                     # layout + state orchestration
        ├── components/
        │   ├── Sidebar.jsx
        │   ├── EnrollmentPanel.jsx     # record + enroll a voice sample
        │   ├── SpeakerList.jsx
        │   ├── RecordingPanel.jsx      # mic control + language selector
        │   ├── TranscriptFeed.jsx
        │   └── TranscriptCard.jsx      # one turn: speaker, language, corrected text
        ├── hooks/
        │   └── useRecorder.js          # MediaRecorder wrapper
        ├── lib/
        │   └── api.js                  # fetch wrapper for backend calls
        └── styles/
            └── index.css
```

## Setup

### Prerequisites

- Python 3.10+ (tested on Anaconda `indic` environment)
- Node.js 18+
- A [Hugging Face](https://huggingface.co) account with access accepted for `ai4bharat/indic-conformer-600m-multilingual` (gated model)
- A [RunPod](https://runpod.io) account with a deployed MedGemma serverless endpoint

### Backend

```bash
cd backend
pip install -r requirements.txt --break-system-packages   # if using a managed env
```

Create `backend/.env` (copy from `.env.example`) and fill in:

```env
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_MEDGEMMA_ENDPOINT_ID=your_runpod_endpoint_id
HF_TOKEN=your_huggingface_token   # required for gated IndicConformer checkpoint
```

Run the server:

```bash
uvicorn app.main:app --reload --port 8000
```

On first startup, three models (TitaNet, IndicConformer, IndicTrans2) are loaded once and kept warm in memory — this can take 1–3 minutes on CPU. Subsequent requests are fast.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173`.

## Usage

1. **Enroll speakers.** In the sidebar, enter a name and record a 5–10 second sample per person. Enrolling the same name again averages a new sample into the existing voiceprint rather than creating a duplicate.
2. **Select the speaker language** from the dropdown before recording — the ASR model requires an explicit language and has no reliable auto-detection.
3. **Tap the mic** to start recording, tap again to stop. The recording is automatically segmented, transcribed, corrected, translated, and matched to enrolled speakers.
4. **Review the transcript feed** — each card shows the identified speaker (or "Unknown speaker"), detected language, match confidence, and the MedGemma-corrected transcript text.

## Known Limitations

- **No true language auto-detection.** IndicConformer requires an explicit language code per request; mixing languages mid-sentence (code-switching) will be transcribed using whichever language was selected, which can produce phonetic mistranscription for words in a different language.
- **English is not supported by IndicConformer.** The multilingual checkpoint covers 22 Indic languages, not English. English speech will be misrecognized if selected.
- **Rapid enumeration (e.g. reading out a list of lab test abbreviations) is prone to ASR hallucination.** Short pauses between items help segmentation and generally improve accuracy.
- **Speaker identification accuracy scales with enrollment length and sample count.** A single short (5s) enrollment produces a noisier voiceprint than multiple longer samples; re-enroll the same person a few times for best results.
- **MedGemma correction runs once per recording**, batching all turns from that recording into a single RunPod job. If a RunPod job fails or times out, the pipeline falls back to the uncorrected transcript rather than losing the turn.

## Tech Stack

| Component | Model / Library |
|---|---|
| ASR | AI4Bharat IndicConformer-600M-Multilingual |
| Translation | AI4Bharat IndicTrans2 (Indic → English) |
| Speaker verification | NVIDIA TitaNet-Large |
| Medical correction | MedGemma 4B (via RunPod serverless) |
| Voice activity detection | WebRTC VAD |
| Backend | FastAPI, PyTorch, NeMo, Transformers |
| Frontend | React, Vite, Tailwind CSS |