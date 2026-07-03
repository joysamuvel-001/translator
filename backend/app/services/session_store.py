"""
Minimal JSON-file persistence for sessions and turns. Swappable for Postgres
later — every call here is async-friendly and isolated behind this module so
routers never touch the file format directly.
"""
import json
import uuid
from datetime import datetime, timezone
from collections import defaultdict

from app.core.config import get_settings

settings = get_settings()


def _session_path(session_id: str):
    return settings.sessions_dir / f"{session_id}.json"


def create_session(title: str = "Untitled session") -> dict:
    session_id = str(uuid.uuid4())
    session = {
        "id": session_id,
        "title": title,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "turns": [],
    }
    _session_path(session_id).write_text(json.dumps(session))
    return session


def get_session(session_id: str) -> dict | None:
    path = _session_path(session_id)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def add_turn(session_id: str, turn: dict) -> dict:
    session = get_session(session_id)
    if session is None:
        session = create_session()
        session_id = session["id"]
    session["turns"].append(turn)
    _session_path(session_id).write_text(json.dumps(session))
    return session


def list_sessions() -> list[dict]:
    out = []
    for path in sorted(settings.sessions_dir.glob("*.json"), reverse=True):
        session = json.loads(path.read_text())
        counts = defaultdict(int)
        for t in session["turns"]:
            counts[t["speaker_name"]] += 1
        out.append({
            "id": session["id"],
            "title": session["title"],
            "created_at": session["created_at"],
            "turn_count": len(session["turns"]),
            "speakers": dict(counts),
        })
    return out
