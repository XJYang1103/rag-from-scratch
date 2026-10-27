"""Shared Paratera configuration for the RAG notebooks (no embedded secrets)."""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_config():
    """Reload the project .env, including values changed while a kernel is running."""
    load_dotenv(ENV_PATH, override=True)
    os.environ.setdefault("USER_AGENT", "rag-from-scratch/1.0")


def _required(name):
    value = os.getenv(name, "").strip()
    if not value or "*" in value or value.startswith("<"):
        raise ValueError(f"Please set {name} in {ENV_PATH.name}.")
    return value


def _connection():
    load_config()
    base_url = _required("PARATERA_BASE_URL").rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    return {"api_key": _required("PARATERA_API_KEY"), "base_url": base_url}


def make_chat_model(**kwargs):
    """Qwen via Paratera's OpenAI-compatible Chat Completions endpoint."""
    config = _connection()
    config.update(
        model=_required("PARATERA_CHAT_MODEL"),
        temperature=0,
        timeout=90,
        max_retries=1,
        use_responses_api=False,
        extra_body={"enable_thinking": False},
    )
    config.update(kwargs)
    return ChatOpenAI(**config)


def make_embeddings():
    """Use the same GLM embedding model for documents and retrieval queries.

    Paratera currently exposes GLM-Embedding-3, not Qwen embedding models.
    Send raw strings, not OpenAI token IDs; keep batches below the 64-item limit.
    """
    config = _connection()
    return OpenAIEmbeddings(
        **config,
        model=_required("PARATERA_EMBEDDING_MODEL"),
        check_embedding_ctx_length=False,
        chunk_size=16,
        model_kwargs={"encoding_format": "float"},
        request_timeout=90,
        max_retries=1,
    )


def show_config():
    """Print configuration without exposing API keys."""
    load_config()
    for name in (
        "PARATERA_BASE_URL", "PARATERA_CHAT_MODEL", "PARATERA_EMBEDDING_MODEL"
    ):
        print(f"{name}: {os.getenv(name, '(not set)')}")
    print("PARATERA_API_KEY configured:", bool(os.getenv("PARATERA_API_KEY")))


def rag_prompt():
    """Local equivalent of the tutorial's rlm/rag-prompt; no Hub request needed."""
    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question}\nContext: {context}\nAnswer:"
    )
