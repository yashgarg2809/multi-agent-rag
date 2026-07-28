"""Hybrid retrieval: BM25 keywords + pgvector dense, fused with RRF.

Dense search gets meaning but misses exact terms; BM25 is the reverse.
Running both and fusing ranks gets most of the benefit of each, with no
score normalization to tune:

    score(doc) = sum(1 / (K + rank)) over each ranking   (K = 60)
"""
import re
import time

from langchain_core.documents import Document
from sqlalchemy import create_engine, text

from app.config import settings

RRF_K = 60
BM25_TOP_N = 20
DENSE_TOP_N = 20
CORPUS_LIMIT = 5000


def tokenize(s: str) -> list[str]:
    return re.findall(r"\w+", (s or "").lower())


def rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse ranked key-lists into [(key, score)] sorted by score desc."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, key in enumerate(ranking):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def _doc_key(d: Document) -> str:
    m = d.metadata or {}
    return f"{m.get('doc_id', '')}|{m.get('page', '')}|{hash(d.page_content)}"


def load_corpus(doc_ids: list[str]) -> list[Document]:
    """Load raw chunks for docs straight from the pgvector tables (no embedding)."""
    from app.registry import CONNECT_TIMEOUT_S
    eng = create_engine(settings.POSTGRES_CONNECTION_STRING,
                        connect_args={"connect_timeout": CONNECT_TIMEOUT_S})
    q = text(
        "SELECT e.document, e.cmetadata FROM langchain_pg_embedding e "
        "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
        "WHERE c.name = :name AND e.cmetadata->>'doc_id' = ANY(:doc_ids) "
        "LIMIT :limit"
    )
    with eng.connect() as conn:
        rows = conn.execute(
            q, {"name": settings.POSTGRES_COLLECTION_NAME,
                "doc_ids": doc_ids, "limit": CORPUS_LIMIT}
        ).all()
    return [Document(page_content=r[0], metadata=dict(r[1] or {})) for r in rows]


def bm25_rank(corpus: list[Document], query: str, top_n: int = BM25_TOP_N) -> list[str]:
    """Rank corpus with BM25, return ordered doc keys. Needs ``rank-bm25``."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        raise RuntimeError("rank-bm25 not installed (pip install rank-bm25)") from e
    tokenized = [tokenize(d.page_content) for d in corpus]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(corpus)), key=lambda i: scores[i], reverse=True)
    return [_doc_key(corpus[i]) for i in ranked[:top_n] if scores[i] > 0]


def hybrid_search(store, query: str, doc_ids: list[str],
                  k_final: int = 5) -> tuple[list[Document], dict]:
    """Dense + BM25 -> RRF -> top-k Documents. Falls back to dense-only."""
    t0 = time.perf_counter()
    info: dict = {"hybrid": False}
    filt = {"doc_id": {"$in": doc_ids}} if doc_ids else None
    dense_docs: list[Document] = store.similarity_search(query, k=DENSE_TOP_N, filter=filt)
    by_key = {_doc_key(d): d for d in dense_docs}
    dense_ranking = list(by_key)
    info["dense_hits"] = len(dense_ranking)

    rankings = [dense_ranking]
    try:
        corpus = load_corpus(doc_ids)
        for d in corpus:
            by_key.setdefault(_doc_key(d), d)
        bm25_ranking = bm25_rank(corpus, query)
        rankings.append(bm25_ranking)
        info["hybrid"] = True
        info["bm25_hits"] = len(bm25_ranking)
        info["corpus_size"] = len(corpus)
    except Exception as e:  # noqa: BLE001 - BM25 is best-effort, dense decides
        info["bm25_error"] = f"{type(e).__name__}: {str(e)[:120]}"

    fused = rrf_fuse(rankings)
    docs = [by_key[key] for key, _ in fused[:k_final] if key in by_key]
    info["fused"] = len(docs)
    info["hybrid_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return docs, info
