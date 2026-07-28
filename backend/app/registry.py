"""Document catalogue for the multi-PDF library.

Vectors live in one shared pgvector collection, namespaced by ``doc_id``.
This table tracks one row per document so the API can list/scope/delete
without scanning the vector store.
"""
import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from app.config import settings


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "rag_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512))
    status: Mapped[str] = mapped_column(String(32), default="indexing")  # indexing|ready|error
    chunks: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


_engine = None
_engines: dict[str, object] = {}

# Fail fast: a dead DB must surface as 503s, never hang boot/requests.
CONNECT_TIMEOUT_S = 5


def _engine_kwargs(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"connect_args": {"check_same_thread": False}}
    return {"connect_args": {"connect_timeout": CONNECT_TIMEOUT_S}}


def get_engine(url: str | None = None):
    """Engine for the registry. ``url`` override exists so tests can use sqlite."""
    global _engine
    if url is not None:
        if url not in _engines:
            _engines[url] = create_engine(url, **_engine_kwargs(url))
        return _engines[url]
    if _engine is None:
        conn = settings.POSTGRES_CONNECTION_STRING
        _engine = create_engine(conn, **_engine_kwargs(conn))
    return _engine


def close_engines() -> None:
    """Dispose cached engines (releases file locks; used in tests)."""
    global _engine
    for e in list(_engines.values()):
        e.dispose()
    _engines.clear()
    if _engine is not None:
        _engine.dispose()
        _engine = None


def init_db(url: str | None = None) -> None:
    Base.metadata.create_all(get_engine(url))


def _session(url: str | None = None) -> Session:
    return Session(get_engine(url))


def create_document(filename: str, url: str | None = None) -> str:
    doc_id = str(uuid.uuid4())
    with _session(url) as s:
        s.add(Document(id=doc_id, filename=filename, status="indexing"))
        s.commit()
    return doc_id


def mark_ready(doc_id: str, chunks: int, url: str | None = None) -> None:
    with _session(url) as s:
        doc = s.get(Document, doc_id)
        if doc is not None:
            doc.status, doc.chunks, doc.error = "ready", chunks, ""
            s.commit()


def mark_error(doc_id: str, error: str, url: str | None = None) -> None:
    with _session(url) as s:
        doc = s.get(Document, doc_id)
        if doc is not None:
            doc.status, doc.error = "error", error[:1000]
            s.commit()


def delete_document(doc_id: str, url: str | None = None) -> bool:
    with _session(url) as s:
        doc = s.get(Document, doc_id)
        if doc is None:
            return False
        s.delete(doc)
        s.commit()
        return True


def list_documents(url: str | None = None) -> list[dict]:
    with _session(url) as s:
        rows = s.scalars(select(Document).order_by(Document.created_at.desc())).all()
        return [
            {"id": d.id, "filename": d.filename, "status": d.status,
             "chunks": d.chunks, "error": d.error,
             "created_at": d.created_at.isoformat() if d.created_at else None}
            for d in rows
        ]


def ready_doc_ids(doc_ids: list[str] | None, url: str | None = None) -> list[str]:
    """Resolve a scope request to concrete ready doc ids.

    ``None``/empty/``["all"]`` means every ready document.
    Raises KeyError if a requested id is unknown, ValueError if not ready.
    """
    docs = {d["id"]: d for d in list_documents(url)}
    if not doc_ids or doc_ids == ["all"]:
        return [i for i, d in docs.items() if d["status"] == "ready"]
    out = []
    for i in doc_ids:
        if i not in docs:
            raise KeyError(f"unknown document: {i}")
        if docs[i]["status"] != "ready":
            raise ValueError(f"document not ready: {docs[i]['filename']} ({docs[i]['status']})")
        out.append(i)
    return out
