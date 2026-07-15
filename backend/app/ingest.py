"""Indexing: each upload gets a doc_id namespace.

One shared pgvector collection; documents are isolated by ``doc_id`` metadata.
Re-uploading replaces only that document's vectors.
"""
import time
from collections.abc import Callable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings
from app.vectorstore import get_vectorstore

ProgressCb = Callable[[str, int, int], None]


def _log(stage: str, done: int = 0, total: int = 0):
    ts = time.strftime("%H:%M:%S")
    extra = f" {done}/{total}" if total else ""
    print(f"[{ts}] INGEST {stage}{extra}", flush=True)


def clear_document(doc_id: str):
    """Delete one document's vectors. Raw SQL because PGVector.delete()
    silently ignores metadata filters in this langchain-postgres version."""
    from sqlalchemy import create_engine, text

    from app.registry import CONNECT_TIMEOUT_S
    eng = create_engine(settings.POSTGRES_CONNECTION_STRING,
                        connect_args={"connect_timeout": CONNECT_TIMEOUT_S})
    with eng.begin() as c:
        c.execute(
            text("DELETE FROM langchain_pg_embedding "
                 "WHERE cmetadata->>'doc_id' = :doc_id"),
            {"doc_id": doc_id},
        )


def ingest_pdf(pdf_path: str, source_name: str | None = None,
               doc_id: str | None = None,
               on_progress: ProgressCb | None = None) -> int:
    cb = on_progress or (lambda s, d, t: None)
    t0 = time.time()
    store = get_vectorstore()

    _log(f"clearing vectors for doc {doc_id}")
    cb("clearing", 0, 0)
    if doc_id:
        clear_document(doc_id)

    _log("loading PDF")
    cb("loading", 0, 0)
    reader = PdfReader(pdf_path)
    docs = [Document(page_content=page.extract_text() or "",
                     metadata={"page": i})
            for i, page in enumerate(reader.pages)]
    _log(f"loaded {len(docs)} pages")
    cb("loading", len(docs), len(docs))
    for d in docs:
        d.metadata["source"] = source_name or pdf_path
        if doc_id:
            d.metadata["doc_id"] = doc_id

    _log("chunking")
    cb("chunking", 0, 0)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    _log(f"chunked into {len(chunks)} chunks")
    cb("chunking", len(chunks), len(chunks))

    # Batched embed+insert so free-tier rate limits survive and progress is visible.
    # ~50 chunks per batch keeps Gemini embedding calls small.
    batch, total = 50, len(chunks)
    for i in range(0, total, batch):
        part = chunks[i:i + batch]
        _log("embedding", min(i + batch, total), total)
        cb("embedding", min(i + batch, total), total)
        store.add_documents(part)

    dt = time.time() - t0
    _log(f"done {total} chunks in {dt:.1f}s")
    cb("done", total, total)
    return total
