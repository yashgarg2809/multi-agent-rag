"""LangGraph chain: router -> retriever -> critic, one retry allowed."""
import time
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.critic import verify_answer
from app.agents.retriever import retrieve_and_answer
from app.agents.router import route_query
from app.config import settings


class RAGState(TypedDict):
    user_query: str
    session_id: str
    doc_ids: list[str]
    rewritten: str
    route: str
    answer: str
    context: str
    approved: bool
    score: float
    retries: int
    router_ms: float
    retriever_ms: float
    embed_ms: float
    chat_ms: float
    critic_ms: float
    hybrid: dict


def node_router(s: RAGState) -> RAGState:
    t0 = time.perf_counter()
    from app.memory import history_text
    d = route_query(s["user_query"], history=history_text(s.get("session_id", "")))
    s["rewritten"], s["route"] = d.rewritten_query, d.route
    if d.route == "reject":
        s.update(answer="That doesn't look answerable from the uploaded PDFs. Please ask about PDF content.",
                 context="", approved=True, score=1.0)
    s["router_ms"] = (time.perf_counter() - t0) * 1000
    return s


def node_retriever(s: RAGState) -> RAGState:
    t0 = time.perf_counter()
    if s.get("route") == "reject":
        s["retriever_ms"] = 0.0
        s["embed_ms"] = 0.0
        s["chat_ms"] = 0.0
        return s
    ans, docs, timings = retrieve_and_answer(
        s.get("rewritten") or s["user_query"], doc_ids=s.get("doc_ids", []))
    s["answer"] = ans
    s["context"] = "\n\n".join(d.page_content for d in docs)
    s["retriever_ms"] = (time.perf_counter() - t0) * 1000
    s["embed_ms"] = timings["embed_ms"]
    s["chat_ms"] = timings["chat_ms"]
    s["hybrid"] = timings.get("hybrid", {})
    return s


def node_critic(s: RAGState) -> RAGState:
    t0 = time.perf_counter()
    if s.get("route") == "reject":
        s["critic_ms"] = 0.0
        return s
    v = verify_answer(s.get("rewritten") or s["user_query"], s["answer"], s["context"])
    # Clamp: some fallback models ignore the 0-1 scale (we saw 10.0 once).
    # A score outside 0-1 would corrupt every average and trace display.
    s["approved"], s["score"] = v.approved, min(max(float(v.score), 0.0), 1.0)
    if not v.approved and v.corrected_answer:
        s["answer"], s["approved"] = v.corrected_answer, True
    elif not v.approved:
        s["retries"] = s.get("retries", 0) + 1
    s["critic_ms"] = (time.perf_counter() - t0) * 1000
    return s


def should_retry(s: RAGState) -> str:
    if s.get("route") == "reject" or s.get("approved"):
        return "end"
    return "end" if s.get("retries", 0) > settings.CRITIC_MAX_RETRIES else "retry"


def build_graph():
    g = StateGraph(RAGState)
    g.add_node("router", node_router)
    g.add_node("retriever", node_retriever)
    g.add_node("critic", node_critic)
    g.set_entry_point("router")
    g.add_edge("router", "retriever")
    g.add_edge("retriever", "critic")
    g.add_conditional_edges("critic", should_retry, {"retry": "retriever", "end": END})
    return g.compile()


_graph = None

def run_rag(user_query: str, session_id: str = "",
            doc_ids: list[str] | None = None) -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    t0 = time.perf_counter()
    out = _graph.invoke({"user_query": user_query, "session_id": session_id,
                         "doc_ids": doc_ids or [], "retries": 0, "approved": False,
                         "score": 0.0, "rewritten": "", "route": "", "answer": "", "context": "",
                         "router_ms": 0.0, "retriever_ms": 0.0, "embed_ms": 0.0,
                         "chat_ms": 0.0, "critic_ms": 0.0, "hybrid": {}})
    from app.agents.critic import last_critic_model
    from app.agents.retriever import last_retriever_model
    from app.memory import add_turn
    if out.get("route") != "reject":
        add_turn(session_id, user_query, out["answer"])
    return {"answer": out["answer"], "rewritten_query": out.get("rewritten", ""),
            "route": out.get("route", ""), "grounding_score": out.get("score", 0.0),
            "approved": out.get("approved", False), "critic_model": last_critic_model,
            "retriever_model": last_retriever_model,
            "hybrid": out.get("hybrid", {}),
            "timings_ms": {"router": round(out.get("router_ms", 0.0), 1),
                           "retriever": round(out.get("retriever_ms", 0.0), 1),
                           "embed": round(out.get("embed_ms", 0.0), 1),
                           "chat": round(out.get("chat_ms", 0.0), 1),
                           "critic": round(out.get("critic_ms", 0.0), 1),
                           "total": round((time.perf_counter() - t0) * 1000, 1)}}
