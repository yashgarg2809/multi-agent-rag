"""API contract tests: TestClient + monkeypatched DB/LLMs (no external calls)."""
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

import app.main as main  # noqa: E402


def _client(monkeypatch) -> TestClient:
    monkeypatch.setattr(main.registry, "create_document", lambda name: "doc-1")
    monkeypatch.setattr(main.registry, "mark_ready", lambda *a, **k: None)
    monkeypatch.setattr(main.registry, "mark_error", lambda *a, **k: None)
    monkeypatch.setattr(main, "ingest_pdf", lambda *a, **k: 3)
    return TestClient(main.app)


def test_health():
    assert TestClient(main.app).get("/health").json() == {"status": "ok"}


def test_rejects_non_pdf(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/upload-pdf", files={"file": ("x.txt", io.BytesIO(b"hi"), "text/plain")})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "INVALID_FILE"


def test_upload_starts_job_and_status_flows(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/upload-pdf", files={"file": ("a.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")})
    assert r.status_code == 202
    body = r.json()
    assert body["doc_id"] == "doc-1"
    s = c.get("/index-status", params={"doc_id": "doc-1"}).json()
    assert s["status"] in ("running", "done")
    assert s["doc_id"] == "doc-1"


def test_index_status_unknown_job():
    r = TestClient(main.app).get("/index-status", params={"doc_id": "nope"})
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "UNKNOWN_JOB"


def test_ask_empty_rejected(monkeypatch):
    c = _client(monkeypatch)
    r = c.post("/ask", json={"question": "   "})
    assert r.status_code == 400
    assert r.json()["detail"]["code"] == "EMPTY_QUESTION"


def test_ask_no_docs(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(main.registry, "ready_doc_ids", lambda *a, **k: [])
    r = c.post("/ask", json={"question": "hi there?"})
    assert r.status_code == 409
    assert r.json()["detail"]["code"] == "NO_DOCS_INDEXED"


def test_ask_happy_path(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(main.registry, "ready_doc_ids", lambda *a, **k: ["d1"])
    fake = {"answer": "In England.", "route": "answer", "grounding_score": 0.9,
            "approved": True, "timings_ms": {}}
    monkeypatch.setattr(main, "run_rag", lambda q, session_id="", doc_ids=None: fake)
    r = c.post("/ask", json={"question": "Where?", "session_id": "s1", "doc_ids": ["d1"]})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "In England."
    assert body["doc_ids"] == ["d1"]
    assert "request_id" in body
    assert "X-Request-ID" in r.headers


def test_documents_and_delete(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(main.registry, "list_documents",
                        lambda: [{"id": "d1", "filename": "a.pdf", "status": "ready",
                                  "chunks": 3, "error": "", "created_at": "t"}])
    assert c.get("/documents").json()["documents"][0]["id"] == "d1"
    monkeypatch.setattr(main, "clear_document", lambda doc_id: None)
    monkeypatch.setattr(main.registry, "delete_document", lambda doc_id: True)
    r = c.delete("/documents/d1")
    assert r.json() == {"status": "deleted", "doc_id": "d1"}


def test_delete_unknown_doc(monkeypatch):
    c = _client(monkeypatch)
    monkeypatch.setattr(main, "clear_document", lambda doc_id: None)
    monkeypatch.setattr(main.registry, "delete_document", lambda doc_id: False)
    r = c.delete("/documents/nope")
    assert r.status_code == 404
    assert r.json()["detail"]["code"] == "UNKNOWN_DOCUMENT"
