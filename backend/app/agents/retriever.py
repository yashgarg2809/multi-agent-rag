"""Agent 2: find evidence (hybrid search) and draft the answer."""
import time

from langchain_core.prompts import ChatPromptTemplate

from app.config import make_chat_model, settings
from app.hybrid import hybrid_search
from app.vectorstore import get_vectorstore

# 429 backoff schedule (seconds). Free tiers throttle constantly; a short wait
# almost always clears it. Non-quota errors (401/402/404) are raised immediately.
RETRY_WAITS = (5, 15, 30)


def _is_quota_error(e: Exception) -> bool:
    sig = f"{type(e).__name__} {str(e)}"
    return "429" in sig or "RESOURCE_EXHAUSTED" in sig or "RateLimit" in sig or "rate_limit" in sig


def _is_daily_quota(e: Exception) -> bool:
    # Gemini's 429 body names the quota: per-minute limits recover with a short
    # wait, but "PerDay" means the allowance is spent until reset - retrying the
    # 5s/15s/30s schedule is guaranteed dead time (~50s per question).
    sig = str(e)
    return "PerDay" in sig or "per_day" in sig or "per day" in sig.lower() or "daily" in sig.lower()


def _with_retry(label: str, fn):
    last = None
    for attempt, wait in enumerate([0, *RETRY_WAITS]):
        if attempt:
            print(f"RETRIEVER {label} hit quota, retry {attempt}/3 after {wait}s", flush=True)
            time.sleep(wait)
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - inspected below, re-raised if fatal
            last = e
            if not _is_quota_error(e):
                raise
            if _is_daily_quota(e):
                print(f"RETRIEVER {label} daily quota spent - skipping retries, straight to fallback", flush=True)
                raise
            print(f"RETRIEVER {label} quota error: {str(e)[:150]}", flush=True)
    raise last

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are the RETRIEVER-ANSWERER. Answer ONLY from context. Cite as [source p.page]. "
               "If insufficient, say \"I don't have enough information in the uploaded PDF.\""),
    ("human", "Context:\n{context}\n\nQuestion: {question}\nAnswer:"),
])

_llms: dict[str, object] = {}
last_retriever_model: str = ""

def get_retriever_llm(model_string: str | None = None):
    m = model_string or settings.RETRIEVER_MODEL
    if m not in _llms:
        _llms[m] = make_chat_model(m)
    return _llms[m]


def _as_text(content) -> str:
    # Reasoning models (gpt-oss etc.) return a LIST of blocks
    # [{'type':'text','text':'...'}, {'extras':{'signature':'...'}}] -
    # join only the text parts, drop signatures.
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str):
                parts.append(b["text"])
            elif hasattr(b, "text"):
                parts.append(str(b.text))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts).strip()
    return str(content)


def retrieve_and_answer(query: str, top_k: int | None = None,
                        doc_ids: list[str] | None = None):
    k = top_k or settings.TOP_K
    store = get_vectorstore()
    t0 = time.perf_counter()
    docs, hybrid_info = _with_retry(
        "hybrid-search", lambda: hybrid_search(store, query, doc_ids or [], k_final=k))
    embed_ms = (time.perf_counter() - t0) * 1000
    context = "\n\n".join(
        f"[{d.metadata.get('source', 'pdf')} p.{d.metadata.get('page', '?')}] {d.page_content}"
        for d in docs
    ) or "(no results - no PDF indexed yet)"
    t1 = time.perf_counter()
    candidates = [settings.RETRIEVER_MODEL]
    if settings.RETRIEVER_FALLBACK_MODEL and settings.RETRIEVER_FALLBACK_MODEL not in candidates:
        candidates.append(settings.RETRIEVER_FALLBACK_MODEL)
    answer, last_err = "", None
    for m in candidates:
        try:
            def _call(m=m):
                msg = (PROMPT | get_retriever_llm(m)).invoke(
                    {"context": context, "question": query})
                u = getattr(msg, "usage_metadata", {}) or {}
                print(f"RETRIEVER chat[{m}] usage={u} ctx_chars={len(context)}", flush=True)
                return msg.content
            answer = _as_text(_with_retry(f"chat[{m}]", _call))
            global last_retriever_model
            last_retriever_model = m
            if m != candidates[0]:
                print(f"RETRIEVER fallback in use: {m} (primary failed: {last_err})", flush=True)
            break
        except Exception as e:  # noqa: BLE001 - tried as fallback below, raised if all fail
            last_err = f"{type(e).__name__}: {str(e)[:150]}"
    else:
        raise last_err if isinstance(last_err, Exception) else RuntimeError(str(last_err))
    chat_ms = (time.perf_counter() - t1) * 1000
    return answer, docs, {"embed_ms": round(embed_ms, 1), "chat_ms": round(chat_ms, 1),
                          "hybrid": hybrid_info}
