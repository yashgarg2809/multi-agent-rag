from langchain_postgres import PGVector

from app.config import make_embedding_model, settings
from app.registry import CONNECT_TIMEOUT_S

_store = None


def get_vectorstore() -> PGVector:
    global _store
    if _store is None:
        _store = PGVector(
            embeddings=make_embedding_model(),
            collection_name=settings.POSTGRES_COLLECTION_NAME,
            connection=settings.POSTGRES_CONNECTION_STRING,
            # fail fast: a dead DB surfaces as 503s, never hangs requests
            engine_args={"connect_args": {"connect_timeout": CONNECT_TIMEOUT_S}},
            use_jsonb=True,
        )
    return _store
