"""Per-session conversation memory (in-memory; Redis would slot in here).

Keeps the last few turns per session so the router can resolve follow-ups
("what about the second one?"). Sessions go quiet-expire after an hour.
"""
import threading
import time
from collections import deque
from dataclasses import dataclass, field

MAX_TURNS = 6
SESSION_TTL_S = 3600


@dataclass
class Session:
    turns: deque = field(default_factory=lambda: deque(maxlen=MAX_TURNS))
    last_seen: float = field(default_factory=time.time)


_sessions: dict[str, Session] = {}
_lock = threading.Lock()


def _get(session_id: str) -> Session:
    with _lock:
        now = time.time()
        # opportunistic expiry
        expired = [k for k, v in _sessions.items() if now - v.last_seen > SESSION_TTL_S]
        for k in expired:
            del _sessions[k]
        sess = _sessions.get(session_id)
        if sess is None:
            sess = _sessions[session_id] = Session()
        sess.last_seen = now
        return sess


def history_text(session_id: str | None, max_chars: int = 1500) -> str:
    """Recent turns formatted for the router prompt ('' when none)."""
    if not session_id:
        return ""
    turns = list(_get(session_id).turns)
    if not turns:
        return ""
    lines = [f"Q: {q}\nA: {a[:400]}" for q, a in turns]
    text = "\n".join(lines)
    return text[-max_chars:]


def add_turn(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return
    _get(session_id).turns.append((question, answer))


def clear_session(session_id: str) -> bool:
    with _lock:
        return _sessions.pop(session_id, None) is not None
