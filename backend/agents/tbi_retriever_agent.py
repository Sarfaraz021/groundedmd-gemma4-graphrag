"""
TBI Retriever subagent — handles clinical and research questions about TBI.

Delegates to the full GraphRAG pipeline:
  vector similarity search → Cypher graph expansion → MMR diversity filter
  → cross-encoder rerank → context assembly → Gemma 4 generation

All pipeline steps are streamed as SSE events so the UI shows live progress.
LangSmith tracing: inherits the supervisor span via contextvars; the nested
spans (vector_retrieval, llm_generation, graphrag_search) appear as children.
"""

import logging
from collections.abc import AsyncGenerator

logger = logging.getLogger(__name__)


async def run(
    graph_rag,
    query: str,
    top_k: int = 10,
    owner_user_id: str | None = None,
    pipeline_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Proxy all events from search_stream_events() and tag the final
    result event with subagent='retriever-tbi'.
    """
    from retrieval.retriever import search_stream_events

    async for event in search_stream_events(
        graph_rag=graph_rag,
        query=query,
        top_k=top_k,
        owner_user_id=owner_user_id,
        pipeline_id=pipeline_id,
    ):
        if event.get("type") == "result":
            event["subagent"] = "retriever-tbi"
        yield event
