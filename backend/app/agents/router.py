"""Agent 1: rewrite the question, decide if it's worth answering."""
from typing import Literal

from pydantic import BaseModel, Field

from app.config import make_chat_model, settings


class RouteDecision(BaseModel):
    rewritten_query: str = Field(description="Search-optimized rewrite")
    route: Literal["answer", "reject"] = Field(description="answer or reject")
    reason: str = Field(description="Short reason")


_llm = None

def get_router_llm():
    global _llm
    if _llm is None:
        _llm = make_chat_model(settings.ROUTER_MODEL).with_structured_output(RouteDecision)
    return _llm


def route_query(user_query: str, history: str = "") -> RouteDecision:
    history_block = (f"\nConversation so far (resolve follow-ups like 'it'/'that'/'the second one' "
                     f"against this):\n{history}\n" if history.strip() else "")
    prompt = (
        "You are the ROUTER in a multi-PDF RAG system. Map the user question.\n"
        "Rules (follow strictly):\n"
        "- ALWAYS rewrite the question into a standalone search query."
        " If it is a follow-up, fold in the referenced entity from history"
        " (e.g. 'what about its sequel?' -> '... sequel of <film from history>').\n"
        "- Default to route='answer' for ANY question that could plausibly relate to a document's contents"
        " (what/who/when/where/why/how/list/explain/summarize/compare questions, even with typos).\n"
        "- Use route='reject' ONLY for messages with zero question content: pure greetings"
        " ('hi'), thanks ('thanks!'), goodbyes, or obvious non-questions.\n"
        "- NEVER reject just because you personally can't answer it - the retriever decides that"
        " from the PDFs. When in doubt, route='answer'.\n"
        "Examples:\n"
        "- 'what are differnt kinds of cricket equipemnts?' -> route='answer' (typos don't matter)\n"
        "- 'hi' -> route='reject'\n"
        f"{history_block}"
        f"\nUser question: {user_query}"
    )
    d = get_router_llm().invoke(prompt)
    print(f"ROUTER route={d.route} reason={d.reason} rewritten={d.rewritten_query!r}", flush=True)
    return d
