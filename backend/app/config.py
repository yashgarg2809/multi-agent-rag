from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()  # export .env to os.environ so LangChain provider SDKs see API keys


class Settings(BaseSettings):
    # Supabase (pgvector) - use pooler URL for serverless/Vercel:
    # postgresql+psycopg://postgres.PROJECT_REF:PASSWORD@aws-0-region.pooler.supabase.com:6543/postgres
    POSTGRES_CONNECTION_STRING: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    POSTGRES_COLLECTION_NAME: str = "pdf_docs"
    EMBEDDING_MODEL: str = "gemini:models/gemini-embedding-001"
    ROUTER_MODEL: str = "groq:openai/gpt-oss-20b"
    # Reasoning effort for Groq gpt-oss models (low/medium/high). Low slashes the
    # ~40s thinking traces to a few seconds; extractive RAG answers barely suffer.
    GROQ_REASONING_EFFORT: str = "low"
    RETRIEVER_MODEL: str = "gemini:gemini-3.6-flash"
    # Fallback when primary retriever hits quota/dead-model errors.
    # Embeddings stay on Gemini (separate quota); only the chat call falls back.
    # Must be verified working - tested 200 OK on Groq free tier (30 RPM, 14.4k/day).
    RETRIEVER_FALLBACK_MODEL: str = "groq:openai/gpt-oss-120b"
    CRITIC_MODEL: str = "cerebras:gpt-oss-120b"
    # Fallback when the primary critic's free quota is exhausted (402 etc).
    # Must be a model verified working on YOUR keys - tested 200 OK.
    CRITIC_FALLBACK_MODEL: str = "groq:openai/gpt-oss-120b"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    CRITIC_MAX_RETRIES: int = 1

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def make_chat_model(model_string: str):
    provider, _, model = model_string.partition(":")
    if not model:
        model, provider = provider, "openai"
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=0)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=0)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=0)
    if provider == "groq":
        from langchain_groq import ChatGroq
        kwargs = {"model": model, "temperature": 0}
        # gpt-oss reasoning models only; ignored otherwise
        if "gpt-oss" in model:
            kwargs["reasoning_effort"] = settings.GROQ_REASONING_EFFORT
        return ChatGroq(**kwargs)
    if provider in ("gemini", "google", "google-genai"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        # retries=0: the Google SDK retries 429s internally with long backoffs
        # (~30s hidden latency). Our code owns retry policy (retriever._with_retry)
        # and fallbacks, so fail fast here instead.
        return ChatGoogleGenerativeAI(model=model, temperature=0, retries=0)
    if provider == "cerebras":
        try:
            from langchain_cerebras import ChatCerebras
            return ChatCerebras(model=model, temperature=0)
        except ImportError:
            import os

            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=model,
                temperature=0,
                api_key=os.getenv("CEREBRAS_API_KEY"),
                base_url="https://api.cerebras.ai/v1",
            )
    raise ValueError(f"Unknown provider in {model_string!r}")


def make_embedding_model():
    provider, _, model = settings.EMBEDDING_MODEL.partition(":")
    if provider == "gemini" or provider in ("google", "google-genai") or "text-embedding" in settings.EMBEDDING_MODEL and provider != "openai":
        # default free path: gemini:models/text-embedding-004
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        emb_model = model or "models/text-embedding-004"
        return GoogleGenerativeAIEmbeddings(model=emb_model)
    if provider == "openai" or not model:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model or settings.EMBEDDING_MODEL)
    # fallback: treat as Gemini
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    return GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
