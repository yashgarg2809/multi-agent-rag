"""Agent 3: check the draft answer against the evidence."""
from pydantic import BaseModel

from app.config import make_chat_model, settings


class Verdict(BaseModel):
    approved: bool
    score: float
    feedback: str = ""
    corrected_answer: str = ""


_llms: dict[str, object] = {}
last_critic_model: str = ""


def get_critic_llm(model_string: str):
    if model_string not in _llms:
        _llms[model_string] = make_chat_model(model_string).with_structured_output(Verdict)
    return _llms[model_string]


def verify_answer(question: str, answer: str, context: str) -> Verdict:
    global last_critic_model
    prompt = (
        "You are the CRITIC. Verify the RAG answer against PDF context.\n"
        "Approve only if every claim is supported. If 'not enough info' and context is empty, approve.\n"
        f"\nQuestion: {question}\nAnswer: {answer}\nContext:\n{context}"
    )
    # Primary first (independent provider), fallback on quota/dead-model errors
    # so one exhausted free tier can never kill the whole chain.
    candidates = [settings.CRITIC_MODEL]
    if settings.CRITIC_FALLBACK_MODEL and settings.CRITIC_FALLBACK_MODEL not in candidates:
        candidates.append(settings.CRITIC_FALLBACK_MODEL)
    last_err: Exception | None = None
    for m in candidates:
        try:
            verdict = get_critic_llm(m).invoke(prompt)
            last_critic_model = m
            if m != candidates[0]:
                print(f"CRITIC fallback in use: {m} (primary {candidates[0]} failed: {last_err})", flush=True)
            return verdict
        except Exception as e:
            last_err = e
            print(f"CRITIC model {m} failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
    raise last_err  # all candidates failed - surface the error honestly
