"""
General subagent — handles greetings and out-of-domain redirect responses.

Two modes driven by the supervisor's classification:
  greeting      — warm, brief conversational reply (hello, thanks, bye, etc.)
  out_of_domain — instant hardcoded decline; does NOT call the LLM

LangSmith tracing: the Ollama call is recorded as a 'general_subagent' span
and nests under the supervisor span via inherited contextvars.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from langsmith import traceable

from llm_providers import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL

logger = logging.getLogger(__name__)

_GREETING_SYSTEM = (
    "You are GroundedMD, a clinical evidence assistant specialising in TBI (Traumatic Brain Injury). "
    "The user has sent a greeting or social message. Reply warmly in 1-2 sentences maximum. "
    "Do NOT list features or capabilities. End with a simple invitation to ask a TBI question. "
    "Example: 'Hello! I'm GroundedMD, your TBI evidence assistant. What would you like to know about TBI?'"
)

_OUT_OF_DOMAIN_DECLINE = (
    "Sorry, I can't help with that — I'm specialised exclusively in TBI "
    "(Traumatic Brain Injury) research. Feel free to ask me about TBI biomarkers, "
    "AI diagnostics, outcome prediction, neurorehabilitation, or clinical management."
)


@traceable(name="general_subagent", run_type="llm")
def _generate_greeting(query: str) -> str:
    """LLM call for greeting responses. @traceable nests under the supervisor span."""
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
                        {"role": "system", "content": _GREETING_SYSTEM},
                        {"role": "user", "content": query},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.3, "num_ctx": 2048, "num_predict": 128},
                },
            )
            resp.raise_for_status()

        content = ((resp.json().get("message") or {}).get("content") or "").strip()
        return content or "Hello! Ask me anything about TBI research."

    except Exception as exc:
        logger.warning("General subagent Ollama call failed: %s", exc)
        return "Hello! I'm your TBI clinical evidence assistant. Ask me anything about TBI research, biomarkers, or outcomes."


async def run(query: str, mode: str = "greeting") -> AsyncGenerator[dict, None]:
    """
    Async generator matching the subagent stream contract.

    Args:
        query: the user's message
        mode:  'greeting' — call LLM for a warm reply
               'out_of_domain' — return instant hardcoded decline, no LLM call
    """
    if mode == "out_of_domain":
        answer = _OUT_OF_DOMAIN_DECLINE
    else:
        # asyncio.to_thread propagates contextvars so @traceable nests correctly
        answer = await asyncio.to_thread(_generate_greeting, query)

    yield {
        "type": "result",
        "answer": answer,
        "source_chunks": [],
        "context": "",
        "subagent": "general",
        "mode": mode,
    }
