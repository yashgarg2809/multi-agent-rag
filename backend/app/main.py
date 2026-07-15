import os
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import registry
from app.agents.orchestrator import run_rag
from app.ingest import clear_document, ingest_pdf
from app.memory import clear_session

MAX_PDF_BYTES = 50 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        registry.init_db()
        print("DB registry ready", flush=True)
    except Exception as e:  # DB down: /ready says so, endpoints 503
        print(f"DB registry unavailable at startup: {e}", flush=True)
    yield


app = FastAPI(title="PersonalProject RAG Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


def err(code: str, message: str, status: int) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


@app.middleware("http")
async def request_id_mw(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
    request.state.rid = rid
    resp = await call_next(request)
    resp.headers["X-Request-ID"] = rid
    return resp


def _db_or_503():
    try:
        return registry.list_documents()
    except Exception as e:
        raise err("DB_UNAVAILABLE", f"Database unreachable: {str(e)[:200]}", 503) from e


# One index job per document (uploads of different PDFs run in parallel)
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _new_job(doc_id: str, filename: str) -> dict:
    job = {"status": "running", "stage": "starting", "done": 0, "total": 0,
           "filename": filename, "chunks_indexed": 0, "error": ""}
    with _jobs_lock:
        _jobs[doc_id] = job
    return job


def _run_index(path: str, filename: str, doc_id: str):
    job = _new_job(doc_id, filename)
    try:
        def cb(stage: str, done: int, total: int):
            job.update(stage=stage, done=done, total=total)
        n = ingest_pdf(path, source_name=filename, doc_id=doc_id, on_progress=cb)
        job.update(status="done", stage="done", chunks_indexed=n)
        registry.mark_ready(doc_id, n)
    except Exception as e:  # surfaced via /index-status, never crashes the server
        job.update(status="error", error=str(e)[:1000])
        try:
            registry.mark_error(doc_id, str(e)[:1000])
        except Exception:
            pass
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


class AskRequest(BaseModel):
    question: str
    session_id: str = ""
    doc_ids: list[str] = Field(default_factory=list,
                               description="Scope: [] or ['all'] = all ready docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Dependency check: DB reachable + required API keys present (names only)."""
    deps: dict = {}
    try:
        registry.list_documents()
        deps["db"] = "ok"
    except Exception as e:
        deps["db"] = f"error: {str(e)[:150]}"
    for key in ("GROQ_API_KEY", "GOOGLE_API_KEY", "CEREBRAS_API_KEY"):
        deps[key] = "set" if os.getenv(key) else "missing"
    ok = deps["db"] == "ok" and all(deps[k] == "set" for k in
                                    ("GROQ_API_KEY", "GOOGLE_API_KEY", "CEREBRAS_API_KEY"))
    return JSONResponse({"ready": ok, "deps": deps}, status_code=200 if ok else 503)


@app.post("/upload-pdf", status_code=202)
async def upload_pdf(file: UploadFile = File(...)):
    name = os.path.basename(file.filename or "")
    if not name.lower().endswith(".pdf"):
        raise err("INVALID_FILE", "Only .pdf accepted", 400)
    blob = await file.read()
    if len(blob) > MAX_PDF_BYTES:
        raise err("FILE_TOO_LARGE", f"Limit is {MAX_PDF_BYTES // 1024 // 1024}MB", 413)
    try:
        doc_id = registry.create_document(name)
    except Exception as e:
        raise err("DB_UNAVAILABLE", f"Database unreachable: {str(e)[:200]}", 503) from e
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(blob)
        path = tmp.name
    t = threading.Thread(target=_run_index, args=(path, name, doc_id), daemon=True)
    t.start()
    return {"status": "started", "doc_id": doc_id, "filename": name,
            "poll": f"/index-status?doc_id={doc_id}"}


@app.get("/index-status")
def index_status(doc_id: str = ""):
    with _jobs_lock:
        if doc_id:
            job = _jobs.get(doc_id)
            if job is None:
                raise err("UNKNOWN_JOB", "No index job for this doc_id", 404)
            return {"doc_id": doc_id, **job}
        if not _jobs:  # backwards compat: nothing indexed yet
            return {"status": "idle", "stage": "", "done": 0, "total": 0,
                    "filename": "", "chunks_indexed": 0, "error": ""}
        latest = next(reversed(_jobs))  # dicts keep insertion order
        return {"doc_id": latest, **_jobs[latest]}


@app.get("/documents")
def documents():
    return {"documents": _db_or_503()}


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    with _jobs_lock:
        if _jobs.get(doc_id, {}).get("status") == "running":
            raise err("INDEX_RUNNING", "Wait for indexing to finish first", 409)
    try:
        clear_document(doc_id)
    except Exception as e:
        raise err("DB_UNAVAILABLE", f"Vector delete failed: {str(e)[:200]}", 503) from e
    try:
        if not registry.delete_document(doc_id):
            raise err("UNKNOWN_DOCUMENT", "No such document", 404)
    except HTTPException:
        raise
    except Exception as e:
        raise err("DB_UNAVAILABLE", f"Database unreachable: {str(e)[:200]}", 503) from e
    with _jobs_lock:
        _jobs.pop(doc_id, None)
    return {"status": "deleted", "doc_id": doc_id}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if not clear_session(session_id):
        raise err("UNKNOWN_SESSION", "No such session", 404)
    return {"status": "cleared", "session_id": session_id}


@app.post("/ask")
def ask(req: AskRequest, request: Request):
    if not req.question.strip():
        raise err("EMPTY_QUESTION", "Empty question", 400)
    try:
        scope = registry.ready_doc_ids(req.doc_ids or ["all"])
    except KeyError as e:
        raise err("UNKNOWN_DOCUMENT", str(e), 404) from e
    except ValueError as e:
        raise err("DOC_NOT_READY", str(e), 409) from e
    except Exception as e:
        raise err("DB_UNAVAILABLE", f"Database unreachable: {str(e)[:200]}", 503) from e
    if not scope:
        raise err("NO_DOCS_INDEXED", "Upload and index a PDF first", 409)
    t0 = time.perf_counter()
    try:
        out = run_rag(req.question, session_id=req.session_id, doc_ids=scope)
    except Exception as e:  # all LLM fallbacks exhausted - honest 502, not a crash
        raise err("UPSTREAM_FAILED", f"All models failed: {str(e)[:300]}", 502) from e
    out["request_id"] = getattr(request.state, "rid", "")
    out["doc_ids"] = scope
    out["api_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return out
