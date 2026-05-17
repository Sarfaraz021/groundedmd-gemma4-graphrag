"""
Model backend — Gemma 4 via Ollama + nomic-embed-text via Ollama.
All models are served by the local Ollama daemon; no external API calls.
"""

import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "gemma4:e4b")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

_EMBED_DIMS: dict[str, int] = {
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "all-minilm": 384,
}


def get_embedding_dimensions() -> int:
    return _EMBED_DIMS.get(OLLAMA_EMBED_MODEL, 768)


def get_embed_model_name() -> str:
    return OLLAMA_EMBED_MODEL


def get_llm_model_name() -> str:
    return OLLAMA_LLM_MODEL


def build_llm():
    from neo4j_graphrag.llm import OllamaLLM
    host = OLLAMA_BASE_URL.rstrip("/") if OLLAMA_BASE_URL else None
    kwargs = {"model_name": OLLAMA_LLM_MODEL, "model_params": {"temperature": 0, "num_ctx": 8192}}
    if host:
        kwargs["host"] = host
    return OllamaLLM(**kwargs)


def build_extraction_llm():
    """LLM for entity extraction — same as build_llm() but with JSON format mode enforced."""
    from neo4j_graphrag.llm import OllamaLLM
    host = OLLAMA_BASE_URL.rstrip("/") if OLLAMA_BASE_URL else None
    kwargs = {"model_name": OLLAMA_LLM_MODEL, "model_params": {"temperature": 0, "num_ctx": 8192, "format": "json"}}
    if host:
        kwargs["host"] = host
    return OllamaLLM(**kwargs)


def build_embedder():
    """
    Return neo4j-graphrag's ``OllamaEmbeddings`` (subclass of ``Embedder``).

    SimpleKGPipeline and ``VectorCypherRetriever`` validate the embedder with
    Pydantic; a plain duck-typed class is rejected. The official wrapper uses
    the ``ollama`` Python client (same daemon as ``OllamaLLM``).
    """
    try:
        from neo4j_graphrag.embeddings.ollama import OllamaEmbeddings
    except ImportError as exc:
        raise ImportError(
            'Missing Ollama embedding support. Install with: pip install "neo4j-graphrag[ollama]"'
        ) from exc

    host = OLLAMA_BASE_URL.rstrip("/") if OLLAMA_BASE_URL else None
    if host:
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, host=host)
    return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
