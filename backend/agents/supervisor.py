"""
Supervisor agent for GroundedMD.

Three-way query classification → subagent routing:
  greeting      → general_agent (mode='greeting')      — warm social reply
  tbi           → tbi_retriever_agent                  — full GraphRAG pipeline
  out_of_domain → general_agent (mode='out_of_domain') — instant decline + redirect

LangSmith tracing:
  _classify_query is @traceable(name='supervisor_route') — root span per request.
  _generate_greeting in general_agent is @traceable(name='general_subagent').
  search_stream_events in retriever is @traceable(name='graphrag_search') with
  nested vector_retrieval and llm_generation children.
  All spans inherit contextvars propagated via asyncio.to_thread.
"""

import langsmith_env  # noqa: F401 — must be first

import asyncio
import logging
from collections.abc import AsyncGenerator

from langsmith import traceable, trace as ls_trace

from llm_providers import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
from agents.skills.supervisor_skills import SKILLS, ROUTING_EXAMPLES

logger = logging.getLogger(__name__)

def _build_routing_system() -> str:
    caps = "\n".join(f"  - {c}" for c in SKILLS["capabilities"][:3])
    examples = "\n".join(
        f'  user: "{q}" → {{"label": "{label}"}}'
        for q, label in ROUTING_EXAMPLES
    )
    return (
        f"You are a strict query classifier for GroundedMD ({SKILLS['description']}).\n"
        "Classify the user query into exactly one label and respond with ONLY valid JSON.\n\n"
        "Labels:\n"
        "  greeting      — any greeting, farewell, thanks, or purely social message\n"
        "  tbi           — any question about traumatic brain injury, concussion, TBI biomarkers,\n"
        "                  TBI treatment, neurorehabilitation, brain imaging, ICP, or related clinical topics\n"
        "  out_of_domain — anything that is neither a greeting nor about TBI\n\n"
        f"Examples:\n{examples}\n\n"
        'Respond with ONLY JSON: {"label": "greeting"} or {"label": "tbi"} or {"label": "out_of_domain"}'
    )

_ROUTING_SYSTEM = _build_routing_system()


@traceable(name="supervisor_route", run_type="chain")
def _classify_query(query: str) -> str:
    """
    Classify query into 'greeting', 'tbi', or 'out_of_domain'.
    Uses JSON format mode to force structured output — avoids thinking-token cutoff issues.
    Falls back to 'tbi' on any error so clinical queries are never dropped.
    """
    import json as _json
    import httpx

    try:
        base = (OLLAMA_BASE_URL or "").rstrip("/")
        if not base:
            return "tbi"

        with httpx.Client(timeout=20.0) as client:
            resp = client.post(
                f"{base}/api/chat",
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _ROUTING_SYSTEM},
                        {"role": "user", "content": query},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0, "num_predict": 50},
                },
            )
            resp.raise_for_status()

        raw = ((resp.json().get("message") or {}).get("content") or "").strip()
        parsed = _json.loads(raw)
        label = str(parsed.get("label", "tbi")).strip().lower()

        if label not in ("greeting", "tbi", "out_of_domain"):
            label = "tbi"

        logger.debug("Supervisor: '%s…' → %s (raw=%r)", query[:60], label, raw)
        return label

    except Exception as exc:
        logger.warning("Supervisor routing failed (%s) — defaulting to tbi", exc)
        return "tbi"


_ROUTE_DETAILS = {
    "greeting": "General subagent (greeting mode): warm conversational reply, no retrieval",
    "tbi": "TBI retriever subagent: vector search → graph expansion → rerank → generation",
    "out_of_domain": "General subagent (out-of-domain mode): instant decline, redirects to TBI topics",
}


async def supervisor_stream_events(
    graph_rag,
    query: str,
    top_k: int = 10,
    owner_user_id: str | None = None,
    pipeline_id: str | None = None,
) -> AsyncGenerator[dict, None]:
    """
    Entry point for /chat/stream.

    `ls_trace("supervisor")` creates the root LangSmith span.  All @traceable
    calls inside (classify → subagent LLM / retrieval chain) inherit this parent
    via contextvars, producing the full supervisor → subagent → model hierarchy.

    Important: do NOT call run.end() manually inside the with block — the
    context manager's __exit__ ends the run automatically when the block exits.
    """
    import agents.general_agent as general_agent
    import agents.tbi_retriever_agent as tbi_retriever_agent

    label = "tbi"  # safe default — initialised before with block

    with ls_trace(
        "supervisor",
        run_type="chain",
        inputs={"query": query},
        metadata={"top_k": top_k, "pipeline_id": pipeline_id},
    ):
        yield {
            "type": "step",
            "phase": "supervisor",
            "title": "Supervisor routing",
            "detail": "Classifying query → greeting / TBI retriever / out-of-domain",
        }
        await asyncio.sleep(0)

        # @traceable + asyncio.to_thread: contextvars propagated to thread → nests under supervisor
        label = await asyncio.to_thread(_classify_query, query)

        yield {
            "type": "step",
            "phase": "supervisor",
            "title": f"Routed → {label}",
            "detail": _ROUTE_DETAILS.get(label, ""),
        }
        await asyncio.sleep(0)

        if label == "tbi":
            async for event in tbi_retriever_agent.run(
                graph_rag=graph_rag,
                query=query,
                top_k=top_k,
                owner_user_id=owner_user_id,
                pipeline_id=pipeline_id,
            ):
                yield event
        else:
            mode = "greeting" if label == "greeting" else "out_of_domain"
            async for event in general_agent.run(query, mode=mode):
                yield event
    # ls_trace __exit__ fires here — ends the supervisor span cleanly
