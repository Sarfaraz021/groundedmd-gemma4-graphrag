"""
General subagent — handles greetings and out-of-domain redirect responses.

Two modes driven by the supervisor's classification:
  greeting      — warm, brief conversational reply (hello, thanks, bye, etc.)
  out_of_domain — politely declines and steers the user back to TBI research

No RAG retrieval is performed in either mode.
LangSmith tracing: each LLM call is recorded as a 'general_subagent' span.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from langsmith import traceable

from llm_providers import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL

logger = logging.getLogger(__name__)

_GREETING_SYSTEM = (
    "You are GroundedMD, a clinical evidence assistant specialising in TBI (Traumatic Brain Injury) research. "
    "The user has sent a greeting or casual message. Reply warmly and briefly — one or two sentences. "
    "End by inviting them to ask a TBI research question."
)

_OUT_OF_DOMAIN_SYSTEM = (
    "You are GroundedMD, a clinical evidence assistant strictly focused on TBI (Traumatic Brain Injury) research. "
    "The user has asked a question outside your domain. "
    "Politely explain that you can only answer questions related to TBI — covering topics such as "
    "biomarkers, AI diagnostics, outcome prediction, neurorehabilitation, imaging, and clinical management. "
    "Do NOT answer the off-topic question. Keep your reply to two sentences maximum, then invite a TBI question."
)


@traceable(name="general_subagent", run_type="llm")
def _generate(query: str, system_prompt: str) -> str:
    try:
        import httpx

        base = (OLLAMA_BASE_URL or "").rstrip("/")
        if not base:
            return "Hello! Ask me anything about TBI research — biomarkers, diagnostics, outcomes, or neurorehabilitation."

        with httpx.Client(timeout=30.0) as client:
            resp = client.post(
                f"{base}/api/chat",
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 128},
                },
            )
            resp.raise_for_status()

        return ((resp.json().get("message") or {}).get("content") or "").strip()

    except Exception as exc:
        logger.warning("General subagent LLM call failed: %s", exc)
        return (
            "I'm specialised in TBI clinical evidence. "
            "Ask me anything about traumatic brain injury research, biomarkers, or outcomes."
        )


async def run(query: str, mode: str = "greeting") -> AsyncGenerator[dict, None]:
    """
    Async generator matching the subagent stream contract.

    Args:
        query: the user's message
        mode:  'greeting' for social messages, 'out_of_domain' for off-topic questions
    """
    system_prompt = _GREETING_SYSTEM if mode == "greeting" else _OUT_OF_DOMAIN_SYSTEM
    answer = await asyncio.to_thread(_generate, query, system_prompt)
    yield {
        "type": "result",
        "answer": answer,
        "source_chunks": [],
        "context": "",
        "subagent": "general",
        "mode": mode,
    }
