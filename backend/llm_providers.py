"""
Model backend — Gemma 4 26B MoE via Ollama + nomic-embed-text via Ollama.

LLM: gemma4:26b — 25.8B parameter Mixture-of-Experts model (4B active per token),
     Q4_K_M quantization, 256K context window, served on NVIDIA L40S 48 GB.
Embeddings: nomic-embed-text — 768-D local embeddings, no external API calls.
"""

import os

OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "gemma4:26b")
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
    # 26B Q4_K_M: ~17 GB weights on L40S 46 GB → ~29 GB free for KV cache
    # 32768 ctx uses ~12 GB KV cache → ~29 GB total, well within limits
    kwargs = {"model_name": OLLAMA_LLM_MODEL, "model_params": {"temperature": 0, "num_ctx": 32768}}
    if host:
        kwargs["host"] = host
    return OllamaLLM(**kwargs)


def build_extraction_llm():
    """LLM for entity extraction — same as build_llm() but with JSON format mode enforced."""
    from neo4j_graphrag.llm import OllamaLLM
    host = OLLAMA_BASE_URL.rstrip("/") if OLLAMA_BASE_URL else None
    kwargs = {"model_name": OLLAMA_LLM_MODEL, "model_params": {"temperature": 0, "num_ctx": 32768, "format": "json"}}
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
