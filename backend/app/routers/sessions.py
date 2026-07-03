from fastapi import APIRouter, HTTPException

from app.services import session_store
from app.models.schemas import SessionOut

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("")
async def new_session(title: str = "Untitled session"):
    return session_store.create_session(title)


@router.get("", response_model=list[SessionOut])
async def get_sessions():
    return session_store.list_sessions()


@router.get("/{session_id}")
async def get_session(session_id: str):
    session = session_store.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
