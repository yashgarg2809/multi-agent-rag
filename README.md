# PDF Q&A (RAG)

Ask questions over your PDFs. Upload one or more documents, then chat with them.
Under the hood it's a FastAPI backend running a small team of LLM agents
(router → retriever → critic) on top of Postgres + pgvector, with a Streamlit
frontend.

```
Streamlit  --->  FastAPI  --->  Router -> Retriever -> Critic
                                     |         |
                                  pgvector   BM25 + dense (RRF)
```

## Run it

You need Python 3.12, API keys (Groq, Google, Cerebras — all have free tiers),
and a Postgres with pgvector. Easiest is Supabase: create a project, run
`create extension if not exists vector;` in its SQL editor, and grab the
**Session pooler** connection string.

```bash
# backend
cd backend
cp .env.example .env        # fill in keys + connection string
pip install -r requirements.txt
python -m uvicorn app.main:app --reload     # :8000, /docs for Swagger

# frontend (another terminal)
cd frontend
pip install -r requirements.txt
streamlit run app.py                        # :8501
```

`GET /ready` tells you if everything (DB + keys) is actually working.

Prefer fully local? `docker compose up -d` in `backend/` gives you pgvector on
localhost — point `.env` at it instead.

## How it works

- **Upload** a PDF → it's split into overlapping chunks, embedded, and stored.
  Each document gets its own namespace, so uploads never overwrite each other.
  Indexing runs in the background; the UI polls for progress.
- **Ask** → the router rewrites your question (follow-ups included, using chat
  history), the retriever searches by meaning *and* keywords, fuses both
  rankings, and drafts a cited answer. A separate critic model checks the
  answer against the passages and gets one retry if it fails.
- If a provider's free quota dies mid-answer, the chain falls over to a backup
  model instead of erroring out.

## API sketch

`POST /upload-pdf` → `GET /index-status?doc_id=` → `GET /documents` →
`POST /ask {question, session_id, doc_ids}` → answer plus a trace
(rewrite, route, score, models used, timings). Errors come back as
`{"detail": {"code", "message"}}`, e.g. `NO_DOCS_INDEXED`.

## Evals and tests

```bash
cd backend
python -m pytest                      # unit + API tests, no network needed
ruff check .                          # lint
python eval/run_eval.py --limit 3     # cheap smoke test (~10 API calls)
python eval/run_eval.py --tag v1 --min-pass 0.8   # full 12-question grading
```

The full eval costs ~40 API calls, so save it for milestones. Tests and lint
run locally whenever you want.

## Knobs that matter

All in `backend/.env`: `ROUTER_MODEL`, `RETRIEVER_MODEL` (+ fallback),
`CRITIC_MODEL` (+ fallback), `EMBEDDING_MODEL`, `CHUNK_SIZE`,
`CHUNK_OVERLAP`, `TOP_K`, `CRITIC_MAX_RETRIES`. See `.env.example`.
