"""
Medical ASR correction using MedGemma 4B deployed on RunPod serverless.
Ported from the working transcribe-model pipeline. Uses RunPod's async
/run + /status polling API — this module is intentionally synchronous
(blocking with time.sleep), matching the proven working version; the
FastAPI route calls it via run_in_threadpool so it doesn't block the
event loop.
"""

import re
import json
import time
from datetime import datetime

import requests

from app.core.config import get_settings

settings = get_settings()

RUNPOD_API_KEY     = settings.runpod_api_key
RUNPOD_ENDPOINT_ID = settings.runpod_medgemma_endpoint_id

BASE_URL       = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"
RUN_URL        = f"{BASE_URL}/run"
STATUS_URL_TPL = f"{BASE_URL}/status/{{job_id}}"
CANCEL_URL_TPL = f"{BASE_URL}/cancel/{{job_id}}"

POLL_INTERVAL    = 2
MAX_WAIT_SECONDS = 300


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(tag: str, msg: str):
    print(f"[{_ts()}] [{tag}] {msg}", flush=True)


def _log_separator():
    print("─" * 70, flush=True)


def _headers() -> dict:
    if not RUNPOD_API_KEY:
        raise EnvironmentError("RUNPOD_API_KEY is not set. Add it to your .env file.")
    return {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }


def _submit_job(conversation: list) -> str:
    slim_conversation = [
        {"speaker": t["speaker"], "text": t["text"]} for t in conversation
    ]
    payload = {"input": {"conversation": slim_conversation}}

    _log_separator()
    _log("RUNPOD", f"Submitting job to endpoint: {RUNPOD_ENDPOINT_ID}")
    _log("RUNPOD", f"POST {RUN_URL}")
    _log("RUNPOD", f"Conversation turns: {len(conversation)}")

    resp = requests.post(RUN_URL, json=payload, headers=_headers(), timeout=30)
    _log("RUNPOD", f"HTTP {resp.status_code} received")

    if resp.status_code != 200:
        _log("RUNPOD", f"ERROR — response body: {resp.text}")
        resp.raise_for_status()

    data = resp.json()
    job_id = data.get("id")
    initial_status = data.get("status", "unknown")

    _log("RUNPOD", f"Job ID        : {job_id}")
    _log("RUNPOD", f"Initial status: {initial_status}")
    _log_separator()

    return job_id


def _cancel_job(job_id: str):
    try:
        resp = requests.post(CANCEL_URL_TPL.format(job_id=job_id), headers=_headers(), timeout=10)
        _log("RUNPOD", f"Cancel requested for job {job_id} — HTTP {resp.status_code}")
    except requests.RequestException as exc:
        _log("RUNPOD", f"Cancel request failed for job {job_id}: {exc}")


def _poll_until_complete(job_id: str) -> dict:
    status_url = STATUS_URL_TPL.format(job_id=job_id)
    elapsed = 0
    poll_count = 0

    _log("RUNPOD", f"Polling: {status_url}")
    _log("RUNPOD", f"Interval: {POLL_INTERVAL}s | Timeout: {MAX_WAIT_SECONDS}s")
    _log_separator()

    STATUS_EMOJI = {
        "IN_QUEUE": "⏳", "IN_PROGRESS": "🔄", "COMPLETED": "✅",
        "FAILED": "❌", "CANCELLED": "🚫", "TIMED_OUT": "⏰",
    }

    while elapsed < MAX_WAIT_SECONDS:
        try:
            resp = requests.get(status_url, headers=_headers(), timeout=30)
        except requests.RequestException as exc:
            _log("POLL", f"Network error ({exc.__class__.__name__}) — retrying...")
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue

        if resp.status_code != 200:
            _log("POLL", f"HTTP {resp.status_code} — retrying...")
            time.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            continue

        data = resp.json()
        status = data.get("status", "unknown")
        poll_count += 1

        delay_info = f" | queue delay: {data['delayTime']}ms" if "delayTime" in data else ""
        exec_info = f" | exec time: {data['executionTime']}ms" if "executionTime" in data else ""
        emoji = STATUS_EMOJI.get(status, "❓")

        _log("POLL", f"#{poll_count:03d} elapsed={elapsed:>4}s  {emoji} {status}{delay_info}{exec_info}")

        if status == "COMPLETED":
            _log_separator()
            _log("RUNPOD", "Job COMPLETED successfully")
            if "executionTime" in data:
                _log("RUNPOD", f"Total execution time: {data['executionTime']}ms")
            return data

        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            _log_separator()
            error_msg = data.get("error", data.get("output", "No error details returned"))
            _log("RUNPOD", f"Job ended with status: {status} — {error_msg}")
            raise RuntimeError(f"RunPod job {job_id} ended with status '{status}': {error_msg}")

        time.sleep(POLL_INTERVAL)
        elapsed += POLL_INTERVAL

    _log("RUNPOD", f"Timed out after {MAX_WAIT_SECONDS}s — cancelling job {job_id}")
    _cancel_job(job_id)
    raise TimeoutError(f"RunPod job {job_id} did not complete within {MAX_WAIT_SECONDS}s")


def _find_conversation_list(obj):
    if isinstance(obj, list) and obj and all(isinstance(t, dict) and "text" in t for t in obj):
        return obj
    if isinstance(obj, dict):
        for key in ("conversation", "corrected_conversation", "turns"):
            found = _find_conversation_list(obj.get(key))
            if found:
                return found
        for value in obj.values():
            found = _find_conversation_list(value)
            if found:
                return found
    return None


def _extract_corrected_text(result: dict) -> str:
    output = result.get("output", {})
    _log("RUNPOD", f"Raw output type: {type(output).__name__} | keys: {list(output.keys()) if isinstance(output, dict) else 'n/a'}")

    if isinstance(output, str):
        return output.strip()

    if isinstance(output, dict):
        for key in ("corrected_text", "text", "result", "response", "output"):
            if key in output:
                _log("RUNPOD", f"Extracted from output['{key}']")
                return str(output[key]).strip()

    _log("RUNPOD", "WARNING: unexpected output shape — stringifying whole output")
    return str(output).strip()


def _parse_corrected_output(raw: str, original: list) -> list:
    cleaned = raw.strip()

    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL | re.IGNORECASE)
    if fence_match:
        cleaned = fence_match.group(1).strip()
        _log("PARSE", "Stripped markdown code fence")

    def _try_json(text: str):
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return None

    parsed = _try_json(cleaned)
    if isinstance(parsed, list) and all(isinstance(t, dict) and "text" in t for t in parsed):
        _log("PARSE", "Parsed as direct JSON list")
        return parsed

    if isinstance(parsed, dict):
        for key in ("conversation", "corrected_conversation", "turns", "output", "result"):
            candidate = parsed.get(key)
            if isinstance(candidate, list) and all(isinstance(t, dict) and "text" in t for t in candidate):
                _log("PARSE", f"Parsed as JSON object, unwrapped key '{key}'")
                return candidate

    for match in re.finditer(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL):
        candidate = _try_json(match.group(1))
        if isinstance(candidate, list) and all(isinstance(t, dict) and "text" in t for t in candidate):
            _log("PARSE", "Parsed as embedded JSON array found in text")
            return candidate
        if isinstance(candidate, dict):
            for key in ("conversation", "corrected_conversation", "turns"):
                inner = candidate.get(key)
                if isinstance(inner, list) and all(isinstance(t, dict) and "text" in t for t in inner):
                    _log("PARSE", f"Parsed as embedded JSON object, unwrapped key '{key}'")
                    return inner

    lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
    line_turns = []
    for line in lines:
        match = re.match(r"^(Doctor|Patient|Speaker[_\s]\d+)\s*:\s*(.+)$", line, re.IGNORECASE)
        if match:
            line_turns.append({"speaker": match.group(1), "text": match.group(2).strip()})

    if len(line_turns) == len(original):
        _log("PARSE", "Parsed as Speaker: text lines")
        return line_turns

    _log("PARSE", "WARNING: could not parse structured output — returning original turns unchanged")
    _log("PARSE", f"RAW OUTPUT (first 500 chars): {cleaned[:500]!r}")
    return [{"speaker": t["speaker"], "text": t["text"]} for t in original]


def correct_transcript(conversation: list) -> dict:
    """
    conversation: list of {"speaker": str, "text": str, ...any extra fields}.
    Extra fields (start_sec, end_sec, match_confidence, etc.) are preserved
    and re-merged after correction, since MedGemma only sees speaker+text.
    """
    _log("MEDGEMMA", f"Starting correction | turns: {len(conversation)}")
    _log_separator()

    job_id = _submit_job(conversation)
    result = _poll_until_complete(job_id)

    corrected = _find_conversation_list(result.get("output"))
    if corrected:
        _log("MEDGEMMA", f"Found structured conversation in output ({len(corrected)} turns)")
    else:
        raw = _extract_corrected_text(result)
        _log("MEDGEMMA", f"Raw output length: {len(raw)} chars")
        corrected = _parse_corrected_output(raw, conversation)

    if len(corrected) == len(conversation):
        merged = []
        for original, fixed in zip(conversation, corrected):
            merged.append({
                **original,
                "speaker": fixed.get("speaker", original["speaker"]),
                "text": fixed.get("text", original["text"]),
            })
        corrected = merged
    else:
        _log("MEDGEMMA", f"WARNING: turn count mismatch ({len(corrected)} vs {len(conversation)}) — metadata not merged")

    _log_separator()
    _log("MEDGEMMA", "Correction complete")
    _log_separator()

    return {
        "corrected_conversation": corrected,
        "job_id": job_id,
        "execution_time_ms": result.get("executionTime"),
    }