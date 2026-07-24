"""Frontend only - talks to FastAPI backend. No LangChain here."""
import os
import uuid
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
# Streamlit Cloud: set BACKEND_URL in App -> Settings -> Secrets instead of env
try:
    if "BACKEND_URL" in st.secrets:
        BACKEND_URL = st.secrets["BACKEND_URL"]
except Exception:
    pass


def api(method: str, path: str, **kw):
    kw.setdefault("timeout", 30)
    return requests.request(method, f"{BACKEND_URL}{path}", **kw)


def fetch_docs() -> list[dict]:
    try:
        r = api("GET", "/documents")
        if r.ok:
            return r.json().get("documents", [])
    except Exception:
        pass
    return []


st.set_page_config(page_title="PersonalProject PDF Q&A", layout="wide")
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex[:12]
if "history" not in st.session_state:
    st.session_state.history = []

st.title("PersonalProject - PDF Q&A")
st.caption("Frontend (Streamlit) -> Backend (FastAPI: Router + Retriever + Critic + pgvector)")

# Backend health
try:
    h = api("GET", "/health", timeout=5).json()
    st.caption(f"Backend: {h.get('status', h)}")
except Exception as e:
    st.error(f"Backend not reachable at {BACKEND_URL}. Start it first. Error: {e}")

# ---- Sidebar: document library ----
with st.sidebar:
    st.header("Document library")
    docs = fetch_docs()
    ready = [d for d in docs if d["status"] == "ready"]
    if not docs:
        st.info("No documents yet. Upload one below.")
    for d in docs:
        label = f"{d['filename']} ({d['status']}, {d['chunks']} chunks)"
        c1, c2 = st.columns([4, 1])
        c1.write(label)
        if d["status"] != "indexing" and c2.button("Delete", key=f"del_{d['id']}"):
            dr = api("DELETE", f"/documents/{d['id']}")
            if dr.ok:
                st.success(f"Deleted {d['filename']}")
                st.rerun()
            else:
                st.error(dr.text)
    if st.button("Refresh library"):
        st.rerun()
    st.divider()
    if st.button("New conversation"):
        try:
            api("DELETE", f"/sessions/{st.session_state.session_id}")
        except Exception:
            pass
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.history = []
        st.rerun()
    st.caption(f"session: {st.session_state.session_id}")

st.divider()
st.subheader("1. Upload PDF (multi-PDF library: each upload is a separate document)")
up = st.file_uploader("Choose PDF", type=["pdf"])
if up and st.button("Index PDF"):
    with st.spinner("Uploading PDF..."):
        try:
            r = api("POST", "/upload-pdf",
                    files={"file": (up.name, up.getvalue(), "application/pdf")}, timeout=120)
        except Exception as e:
            st.error(f"Upload failed: backend not reachable at {BACKEND_URL}. "
                     f"Is the backend running? Error: {e}")
            st.stop()
    if not r.ok:
        st.error(r.text)
    else:
        doc_id = r.json()["doc_id"]
        import time
        bar = st.progress(0, text="Starting index...")
        status_box = st.empty()
        # poll backend until done/error (embedding a big PDF can take minutes on free tier)
        while True:
            time.sleep(2)
            try:
                j = api("GET", "/index-status", params={"doc_id": doc_id}, timeout=10).json()
            except Exception as e:
                status_box.warning(f"Waiting for backend... ({e})")
                continue
            st_state, stage, done, total = j.get("status"), j.get("stage", ""), j.get("done", 0), j.get("total", 0)
            if total:
                bar.progress(min(done / total, 1.0), text=f"{stage}: {done}/{total} chunks")
            else:
                status_box.info(f"Stage: {stage or st_state}...")
            if st_state == "done":
                bar.progress(1.0, text="Done")
                st.success(f"Indexed {j.get('chunks_indexed')} chunks from {j.get('filename')}")
                st.rerun()
            if st_state == "error":
                st.error(f"Index failed: {j.get('error')}")
                break

st.divider()
st.subheader("2. Ask a question")
scope_opts = {"All documents": ["all"]}
for d in ready:
    scope_opts[f"{d['filename']} (only)"] = [d["id"]]
scope_label = st.selectbox("Search scope", list(scope_opts.keys()))
q = st.text_input("Your question (follow-ups use conversation memory)")
if q and st.button("Ask"):
    with st.spinner("Router -> Retriever -> Critic..."):
        try:
            r = api("POST", "/ask", json={"question": q,
                                          "session_id": st.session_state.session_id,
                                          "doc_ids": scope_opts[scope_label]}, timeout=300)
        except Exception as e:
            st.error(f"Ask failed: backend not reachable at {BACKEND_URL}. "
                     f"Is the backend running? Error: {e}")
            st.stop()
        if r.ok:
            d = r.json()
            st.session_state.history.append((q, d))
            st.write(d["answer"])
            with st.expander("Agent trace"):
                st.json({k: d.get(k) for k in ("rewritten_query", "route", "grounding_score",
                                               "approved", "retriever_model", "critic_model",
                                               "hybrid", "doc_ids", "request_id")})
        else:
            st.error(r.text)

if st.session_state.history:
    st.divider()
    st.subheader("Chat history (this session)")
    for qq, dd in reversed(st.session_state.history):
        st.markdown(f"**Q:** {qq}\n\n**A:** {dd['answer']}")
