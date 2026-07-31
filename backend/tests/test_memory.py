"""Conversation memory tests (pure in-memory, no DB)."""
from app import memory


def test_add_and_read_history():
    sid = "test-sess-1"
    memory.clear_session(sid)
    assert memory.history_text(sid) == ""
    memory.add_turn(sid, "Where did cricket originate?", "In England.")
    memory.add_turn(sid, "What about its equipment?", "Bat and ball.")
    h = memory.history_text(sid)
    assert "England" in h and "Bat and ball" in h


def test_max_turns_bounded():
    sid = "test-sess-2"
    memory.clear_session(sid)
    for i in range(20):
        memory.add_turn(sid, f"q{i}", f"a{i}")
    h = memory.history_text(sid)
    assert "q0" not in h  # evicted by maxlen
    assert "q19" in h


def test_clear_unknown_session():
    assert memory.clear_session("no-such-session") is False


def test_no_session_no_crash():
    memory.add_turn(None, "q", "a")
    assert memory.history_text(None) == ""
    assert memory.history_text("") == ""
