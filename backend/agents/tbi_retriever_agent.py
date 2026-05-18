"""
TBI Retriever subagent — handles clinical and research questions about TBI.

Delegates to the full GraphRAG pipeline:
  vector similarity search → Cypher graph expansion → MMR diversity filter
  → cross-encoder rerank → context assembly → Gemma 4 26B MoE generation

All pipeline steps are streamed as SSE events so the UI shows live progress.
LangSmith tracing: inherits the supervisor span via contextvars; the nested
spans (vector_retrieval, llm_generation, graphrag_search) appear as children.
"""

import logging
from collections.abc import AsyncGenerator

from agents.skills.tbi_retriever_skills import SKILLS, ANSWER_STYLE

logger = logging.getLogger(__name__)

_SKILL_NAME = "tbi_evidence_interpretation"


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

    Skill: tbi_evidence_interpretation — detailed clinical guidance from skills.py
    ANSWER_STYLE from tbi_retriever_skills is appended to the skill context so the
    model follows the correct citation and brevity constraints.
    """
    from retrieval.retriever import search_stream_events
    from skills import get_skill_context

    # Merge base skill guidance with agent-level answer style constraints
    skill_context = get_skill_context(_SKILL_NAME) + f"\n\n## Answer Style\n{ANSWER_STYLE}"

    async for event in search_stream_events(
        graph_rag=graph_rag,
        query=query,
        top_k=top_k,
        owner_user_id=owner_user_id,
        pipeline_id=pipeline_id,
        skill_context_override=skill_context,
    ):
        if event.get("type") == "result":
            event["subagent"] = "retriever-tbi"
        yield event
