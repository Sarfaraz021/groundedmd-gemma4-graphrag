"""
TBI Retriever subagent — handles clinical and research questions about TBI.

Delegates to the full GraphRAG pipeline:
  vector similarity search → Cypher graph expansion → MMR diversity filter
  → cross-encoder rerank → context assembly → Gemma 4 generation

All pipeline steps are streamed as SSE events so the UI can show live progress.
LangSmith tracing: inherits the nested spans from retrieval.retriever
  (vector_retrieval, llm_generation, graphrag_search).
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
    Async generator matching the subagent stream contract.

    Proxies all events from search_stream_events() and tags the final
    ``result`` event with subagent='retriever-tbi' for observability.
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
