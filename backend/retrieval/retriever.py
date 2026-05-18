"""
GraphRAG retrieval layer.

Sets up the vector index, VectorCypherRetriever (graph-aware), and the
GraphRAG chain that enforces evidence-grounded, cited responses.

LangSmith: ``@traceable`` spans record retrieval and generation when optional
local tracing is enabled. The default model path uses Ollama.
"""

import langsmith_env  # noqa: F401 — ensure .env + LangSmith before LLM usage

import asyncio
import os
import re
from collections.abc import AsyncGenerator

import neo4j
import numpy as np
from langsmith import traceable
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from neo4j_graphrag.indexes import create_vector_index
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm_providers import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, build_embedder, build_llm, get_embed_model_name, get_embedding_dimensions
from prompts import GRAPH_RETRIEVAL_QUERY, RAG_PROMPT
from skills import get_skill_context

# ---------------------------------------------------------------------------
# Post-retrieval config (all tunable via env vars)
# ---------------------------------------------------------------------------
_MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))          # 0=diversity, 1=relevance
_MMR_FETCH_MULT = int(os.getenv("MMR_FETCH_MULTIPLIER", "3")) # fetch top_k × N from Neo4j
_MMR_KEEP_MULT = int(os.getenv("MMR_KEEP_MULTIPLIER", "2"))   # keep top_k × N after MMR
_RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
_LOCAL_RERANK_ENABLED = os.getenv("LOCAL_RERANK_ENABLED", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Lazy-loaded — only allocated on first rerank call, not at server startup.
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(_RERANKER_MODEL)
    return _reranker


def _record_formatter(record: neo4j.Record) -> RetrieverResultItem:
    """Extract the `info` string field from the Cypher retrieval query result.

    Without this, the default formatter does str(record) which produces
    '<Record info='...'>' — a Neo4j Record repr, not the plain text content.
    """
    return RetrieverResultItem(content=record.get("info", ""), metadata=None)

@traceable(name="mmr_filter", run_type="chain")
def _mmr_filter(
    items: list[RetrieverResultItem],
    query: str,
    top_k: int,
    lambda_mult: float = _MMR_LAMBDA,
) -> list[RetrieverResultItem]:
    """Maximal Marginal Relevance — balance relevance to query with diversity among chunks.

    Uses TF-IDF cosine similarity so no extra API calls or models are needed.

    How it works:
      1. Build a TF-IDF matrix for [query] + all chunk texts.
      2. Greedily select chunks that maximise:
             lambda × sim(chunk, query)  −  (1−lambda) × max sim(chunk, already_selected)
         i.e. high relevance AND low redundancy with what we already picked.
    """
    if len(items) <= top_k:
        return items

    texts = [query] + [item.content or "" for item in items]
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(texts)

    query_vec = tfidf[0]           # shape (1, vocab)
    chunk_vecs = tfidf[1:]         # shape (n_chunks, vocab)

    relevance = cosine_similarity(chunk_vecs, query_vec).flatten()  # (n,)
    redundancy = np.full(len(items), -np.inf)                       # max sim to selected set

    selected: list[int] = []
    candidates = list(range(len(items)))

    while len(selected) < top_k and candidates:
        # Score = relevance − (1−λ) × redundancy (redundancy starts at -inf → no penalty yet)
        scores = [
            lambda_mult * relevance[i] - (1 - lambda_mult) * max(redundancy[i], 0.0)
            for i in candidates
        ]
        best_pos = int(np.argmax(scores))
        best_idx = candidates[best_pos]
        selected.append(best_idx)
        candidates.pop(best_pos)

        # Update redundancy: sim of remaining candidates to the newly selected chunk
        new_sims = cosine_similarity(chunk_vecs[np.array(candidates)], chunk_vecs[[best_idx]]).flatten()
        for pos, cand_idx in enumerate(candidates):
            redundancy[cand_idx] = max(redundancy[cand_idx], new_sims[pos])

    return [items[i] for i in selected]


@traceable(name="local_rerank", run_type="chain")
async def _local_rerank(
    items: list[RetrieverResultItem],
    query: str,
    top_n: int,
) -> list[RetrieverResultItem]:
    """Re-rank chunks using BAAI/bge-reranker-v2-m3 (local cross-encoder, air-gapped).

    Cross-encoder reads the full (query, chunk) pair rather than comparing vectors
    independently, catching semantic nuances that embedding similarity misses.
    Runs on CPU; ~300MB model loaded once and cached for the process lifetime.
    """
    import logging
    logger = logging.getLogger(__name__)

    if not items:
        return items

    if not _LOCAL_RERANK_ENABLED:
        return items[:top_n]

    try:
        reranker = _get_reranker()
        pairs = [(query, item.content or "") for item in items]
        scores = await asyncio.to_thread(reranker.predict, pairs)
        ranked = sorted(zip(scores, items), key=lambda x: x[0], reverse=True)
        reranked = [item for _, item in ranked[:top_n]]
        logger.debug(
            "Local rerank (%s): %d → %d chunks. Top score: %.4f",
            _RERANKER_MODEL, len(items), len(reranked), float(scores[0]) if len(scores) else 0,
        )
        return reranked
    except Exception as exc:
        logger.warning("Local rerank failed (%s) — falling back to MMR order: %s", type(exc).__name__, exc)
        return items[:top_n]


VECTOR_INDEX_NAME = "text_embeddings"
TOP_K = 10


def setup_vector_index(driver: neo4j.Driver) -> None:
    """Create vector index if it does not already exist."""
    import logging
    logger = logging.getLogger(__name__)
    try:
        create_vector_index(
            driver,
            name=VECTOR_INDEX_NAME,
            label="Chunk",
            embedding_property="embedding",
            dimensions=get_embedding_dimensions(),
            similarity_fn="cosine",
        )
        logger.info("Vector index '%s' created.", VECTOR_INDEX_NAME)
    except Exception as exc:
        msg = str(exc).lower()
        if "already exists" in msg or "equivalent index already exists" in msg:
            logger.debug("Vector index '%s' already exists — skipping creation.", VECTOR_INDEX_NAME)
        else:
            logger.error("Failed to create vector index '%s': %s", VECTOR_INDEX_NAME, exc)
            raise RuntimeError(
                f"Vector index setup failed — RAG queries will not work. Cause: {exc}"
            ) from exc


def build_graph_rag(driver: neo4j.Driver) -> GraphRAG:
    embedder = build_embedder()

    retriever = VectorCypherRetriever(
        driver=driver,
        index_name=VECTOR_INDEX_NAME,
        embedder=embedder,
        retrieval_query=GRAPH_RETRIEVAL_QUERY,
        result_formatter=_record_formatter,
    )

    llm = build_llm()

    rag_template = RagTemplate(
        template=RAG_PROMPT,
        expected_inputs=["query_text", "context", "examples"],
    )

    return GraphRAG(llm=llm, retriever=retriever, prompt_template=rag_template)


def list_pipelines(driver: neo4j.Driver) -> list[dict]:
    """Return distinct pipeline_ids with doc counts from Document nodes."""
    q = """
    MATCH (d:Document)
    WHERE d.pipeline_id IS NOT NULL
    WITH d.pipeline_id AS id, count(d) AS doc_count
    RETURN id, doc_count ORDER BY id
    """
    with driver.session() as session:
        result = session.run(q)
        return [{"id": r["id"], "doc_count": r["doc_count"]} for r in result]


@traceable(name="vector_retrieval", run_type="retriever")
def _vector_retrieval(
    graph_rag: GraphRAG,
    query_text: str,
    top_k: int,
    owner_user_id: str | None = None,
    pipeline_id: str | None = None,
):
    """Neo4j vector + Cypher graph context (child run in LangSmith)."""
    return graph_rag.retriever.search(
        query_text=query_text,
        top_k=top_k,
        query_params={"owner_user_id": owner_user_id, "pipeline_id": pipeline_id},
    )


@traceable(name="llm_generation", run_type="llm", metadata={"model": OLLAMA_LLM_MODEL})
def _llm_generation(
    graph_rag: GraphRAG,
    query_text: str,
    context: str,
    examples: str,
) -> str:
    """Format RAG prompt and call the configured GraphRAG LLM (Ollama by default)."""
    prompt = graph_rag.prompt_template.format(
        query_text=query_text,
        context=context,
        examples=examples,
    )
    llm_response = graph_rag.llm.invoke(
        input=prompt,
        message_history=None,
        system_instruction=graph_rag.prompt_template.system_instructions,
    )
    return (llm_response.content or "").strip()


@traceable(name="graphrag_search")
async def search(
    graph_rag: GraphRAG,
    query: str,
    skill_name: str = "tbi_evidence_interpretation",
    top_k: int = TOP_K,
    owner_user_id: str | None = None,
    pipeline_id: str | None = None,
) -> dict:
    """
    Run retrieval + generation (equivalent to ``GraphRAG.search``) with nested traces.

    Returns:
        answer         — LLM-generated response with inline [N] citations.
        source_chunks — raw retrieved chunks with source metadata.
    """
    skill_context = get_skill_context(skill_name)

    fetch_k = top_k * _MMR_FETCH_MULT
    retriever_result = _vector_retrieval(graph_rag, query, fetch_k, owner_user_id, pipeline_id)

    items = retriever_result.items or []
    items = _mmr_filter(items, query, top_k=top_k * _MMR_KEEP_MULT)
    items = await _local_rerank(items, query, top_n=top_k)
    retriever_result.items = items

    context = "\n".join(item.content for item in items)
    answer = _llm_generation(graph_rag, query, context, skill_context)
    source_chunks = _parse_chunks(retriever_result)

    return {
        "answer": answer,
        "source_chunks": source_chunks,
    }


def _retrieval_graph_stats(retriever_result) -> tuple[int, int]:
    """Count chunks that include expanded graph text and total graph segments."""
    if retriever_result is None or not getattr(retriever_result, "items", None):
        return 0, 0
    with_graph = 0
    segments = 0
    for item in retriever_result.items:
        content = item.content or ""
        if "[GRAPH:" not in content:
            continue
        with_graph += 1
        rest = content.split("[GRAPH:", 1)[1]
        end = rest.find("]")
        g = rest[:end] if end != -1 else rest
        segments += g.count(";") + (1 if g.strip() else 0)
    return with_graph, segments


async def search_stream_events(
    graph_rag: GraphRAG,
    query: str,
    skill_name: str = "tbi_evidence_interpretation",
    top_k: int = TOP_K,
    owner_user_id: str | None = None,
    pipeline_id: str | None = None,
    skill_context_override: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Stream logical pipeline steps for the UI (embedding → vector + graph → context → LLM → result).

    Vector search and Cypher graph expansion run in a single retriever call; steps describe that work.
    """
    skill_context = skill_context_override if skill_context_override else get_skill_context(skill_name)
    fetch_k = top_k * _MMR_FETCH_MULT

    yield {
        "type": "step",
        "phase": "embedding",
        "title": "Query embedding",
        "detail": f"{get_embed_model_name()} · {get_embedding_dimensions()}-D · cosine similarity",
    }
    await asyncio.sleep(0)

    yield {
        "type": "step",
        "phase": "vector_index",
        "title": "Vector similarity search",
        "detail": f"Neo4j vector index `{VECTOR_INDEX_NAME}` on :Chunk.embedding · fetching top {fetch_k} candidates"
            + (f" · pipeline={pipeline_id}" if pipeline_id else ""),
    }
    await asyncio.sleep(0)

    retriever_result = await asyncio.to_thread(
        _vector_retrieval, graph_rag, query, fetch_k, owner_user_id, pipeline_id
    )

    n_fetched = len(retriever_result.items) if retriever_result and retriever_result.items else 0
    with_graph, rel_segments = _retrieval_graph_stats(retriever_result)

    yield {
        "type": "step",
        "phase": "graph_traversal",
        "title": "Graph expansion (Cypher)",
        "detail": (
            "Per hit chunk: OPTIONAL MATCH document via FROM_DOCUMENT; "
            "OPTIONAL MATCH entities via FROM_CHUNK and entity–entity rels (excludes FROM_CHUNK). "
            f"Retrieved passages: {n_fetched}; {with_graph} include linked graph context"
            + (f" (~{rel_segments} relationship mention(s))" if rel_segments else "")
        ),
    }
    await asyncio.sleep(0)

    # --- MMR diversity filter ---
    items = retriever_result.items or []
    mmr_keep = top_k * _MMR_KEEP_MULT
    items = _mmr_filter(items, query, top_k=mmr_keep)

    yield {
        "type": "step",
        "phase": "mmr",
        "title": "MMR diversity filter",
        "detail": (
            f"Maximal Marginal Relevance (λ={_MMR_LAMBDA}) · {n_fetched} → {len(items)} chunks · "
            "balances relevance to query with diversity among selected passages"
        ),
    }
    await asyncio.sleep(0)

    # --- Local rerank ---
    n_before_rerank = len(items)
    items = await _local_rerank(items, query, top_n=top_k)

    yield {
        "type": "step",
        "phase": "rerank",
        "title": "Cross-encoder rerank",
        "detail": f"{_RERANKER_MODEL} · {n_before_rerank} → {len(items)} chunks · local cross-encoder re-scoring (air-gapped)",
    }
    await asyncio.sleep(0)

    retriever_result.items = items

    yield {
        "type": "step",
        "phase": "context",
        "title": "Context assembly",
        "detail": f"Merged {len(items)} evidence chunk(s) for the prompt (source headers + optional [GRAPH: …] snippets)",
    }
    await asyncio.sleep(0)

    context = "\n".join(item.content for item in items)

    yield {
        "type": "step",
        "phase": "llm",
        "title": "Answer generation",
        "detail": f"Streaming grounded completion · {OLLAMA_LLM_MODEL} · inline [n] citations",
    }
    await asyncio.sleep(0)

    # Stream tokens directly from Ollama so the UI shows live text immediately.
    # Falls back to blocking invoke() if streaming fails.
    source_chunks = _parse_chunks(retriever_result)
    prompt = graph_rag.prompt_template.format(
        query_text=query,
        context=context,
        examples=skill_context,
    )

    answer_tokens: list[str] = []
    streamed_ok = False
    try:
        import httpx, json as _json
        base = (OLLAMA_BASE_URL or "").rstrip("/")
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "POST",
                f"{base}/api/chat",
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True,
                    "options": {"temperature": 0, "num_ctx": 32768},
                },
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = _json.loads(raw_line)
                    except Exception:
                        continue
                    token = (chunk.get("message") or {}).get("content") or ""
                    if token:
                        answer_tokens.append(token)
                        yield {"type": "token", "token": token}
                    if chunk.get("done"):
                        break
        streamed_ok = True
    except Exception as exc:
        logger.warning("Streaming generation failed (%s) — falling back to blocking invoke", exc)

    if streamed_ok:
        answer = "".join(answer_tokens).strip()
    else:
        answer = await asyncio.to_thread(
            _llm_generation, graph_rag, query, context, skill_context
        )
        answer = (answer or "").strip()

    yield {
        "type": "result",
        "answer": answer,
        "source_chunks": source_chunks,
        "context": context,
    }


_RE_SOURCE_HEAD = re.compile(r"^\[SOURCE: ([^\]]*)\]\n")
_RE_CHUNK_HEAD = re.compile(r"^\[CHUNK: ([^\]]+)\]\n")


def _parse_chunk_meta_line(inner: str) -> tuple[int | None, str | None]:
    chunk_index: int | None = None
    chunk_node_id: str | None = None
    for part in inner.split(","):
        part = part.strip()
        if part.startswith("index="):
            raw = part.removeprefix("index=").strip()
            if raw == "null" or raw == "-1":
                chunk_index = None
            else:
                try:
                    chunk_index = int(raw)
                except ValueError:
                    chunk_index = None
        elif part.startswith("id="):
            chunk_node_id = part.removeprefix("id=").strip() or None
    return chunk_index, chunk_node_id


def _parse_chunks(retriever_result) -> list[dict]:
    """
    Extract source path, display filename, Neo4j chunk index / id, and body text.

    Expected prefix (from ``GRAPH_RETRIEVAL_QUERY``)::

        [SOURCE: <coalesce(source, path, file_path)>]
        [CHUNK: index=<chunk.index>, id=<elementId(chunk)>]
        <chunk.text>
        [GRAPH: ...]  # optional
    """
    chunks = []
    if retriever_result is None:
        return chunks

    for i, item in enumerate(retriever_result.items, start=1):
        content: str = str(item.content) if item.content is not None else ""
        source = "unknown"
        doc_name = "unknown"
        chunk_index: int | None = None
        chunk_node_id: str | None = None
        text = content

        rest = content
        m_src = _RE_SOURCE_HEAD.match(rest)
        if m_src:
            source = m_src.group(1).strip() or "unknown"
            doc_name = source.split("/")[-1] if source != "unknown" else "unknown"
            rest = rest[m_src.end() :]

        m_ch = _RE_CHUNK_HEAD.match(rest)
        if m_ch:
            chunk_index, chunk_node_id = _parse_chunk_meta_line(m_ch.group(1))
            rest = rest[m_ch.end() :]

        text = rest
        graph_at = text.find("\n[GRAPH:")
        if graph_at != -1:
            text = text[:graph_at]

        chunks.append(
            {
                "index": i,
                "document": doc_name,
                "source_path": source,
                "text": text.strip(),
                "chunk_index": chunk_index,
                "chunk_node_id": chunk_node_id,
            }
        )

    return chunks
