"""
LLM-as-Judge evaluation for GraphRAG responses.

Scores each response on 4 dimensions (0–10) using GPT-4o-mini.
Called after every /chat/stream response and emitted as an ``evaluation`` SSE event.

Metrics
-------
faithfulness     — every claim in the answer is grounded in the retrieved chunks
completeness     — all relevant information from the context is covered
relevance        — the answer directly addresses the question asked
context_quality  — the retrieved chunks were actually useful for the question
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

_JUDGE_SYSTEM = (
    "You are an expert evaluator for a medical RAG (Retrieval-Augmented Generation) system "
    "specialising in TBI (Traumatic Brain Injury) clinical evidence. "
    "Score the answer strictly based on the retrieved context provided — not on your own knowledge."
)

_JUDGE_PROMPT = """Evaluate the following RAG response across 4 dimensions. Score each 0–10 (integers only).

Scoring guide:
- 9–10: excellent, no issues
- 7–8:  good, minor gaps
- 5–6:  acceptable, noticeable gaps or issues
- 3–4:  poor, significant problems
- 0–2:  very poor or completely wrong — reserve 0 ONLY for answers that directly contradict the context

IMPORTANT — context format notes:
- Context chunks are extracted from medical PDFs. Tables may appear as irregular whitespace-separated text rather than proper rows/columns.
- Judge faithfulness SEMANTICALLY — ask "is this claim inferable from the context?" not "does this exact phrase appear verbatim?"
- If table data is present in the context (even if garbled), and the answer correctly interprets it, treat those claims as faithful.
- Do NOT penalise for claims that are reasonable interpretations of PDF-parsed table structure.

Dimensions:
1. faithfulness     — Are claims SEMANTICALLY supported by the retrieved context? A claim is faithful if it can be reasonably inferred from the context, even if the wording differs or the source is a parsed table.
2. completeness     — Does the answer cover all key information available in the context relevant to the question? Penalise only for significant omissions.
3. relevance        — Does the answer directly address the question asked?
4. context_quality  — Were the retrieved chunks useful and sufficient to answer the question well?

Question:
{question}

Retrieved Context (PDF-extracted chunks — tables may be formatted irregularly):
{context}

Answer to evaluate:
{answer}

Respond ONLY with valid JSON, no explanation outside the JSON:
{{
  "faithfulness": <0-10>,
  "completeness": <0-10>,
  "relevance": <0-10>,
  "context_quality": <0-10>,
  "reasoning": "<one concise sentence explaining the lowest score>"
}}"""


async def judge_response(
    question: str,
    answer: str,
    context_chunks: list[str],
    context_str: str | None = None,
) -> dict:
    """
    Score a RAG response using Gemma 4 26B MoE via Ollama as judge.

    Returns a dict with keys: faithfulness, completeness, relevance,
    context_quality (all int 0–10), reasoning (str), and overall (float avg).

    Falls back to a neutral score dict on any failure so the stream is never blocked.
    """

    if not answer.strip() or not context_chunks:
        return _fallback("empty answer or context")

    import re

    def _clean_chunk(text: str) -> str:
        """Strip retrieval metadata headers so the judge sees only content text."""
        text = re.sub(r'^\[SOURCE:[^\]]*\]\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\[CHUNK:[^\]]*\]\n?', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n\[GRAPH:[^\]]*\]', '', text)
        text = re.sub(r'\[CONTINUES:', '\n[CONTINUES:', text)
        return text.strip()

    # Prefer the pre-joined context string when available — it preserves full
    # chunk content without per-chunk truncation, which is critical for large
    # tables that span many characters in a single chunk.
    if context_str and context_str.strip():
        raw_context = _clean_chunk(context_str)
    else:
        cleaned = [_clean_chunk(c) for c in context_chunks if c.strip()]
        raw_context = "\n\n---\n\n".join(
            f"[Chunk {i+1}]: {c}" for i, c in enumerate(cleaned)
        )

    # Cap at 80k chars — large enough to include full tables while staying
    # within model context limits. Truncate from the end to keep the most
    # relevant chunks (which retrieval already ranked highest) intact.
    context_text = raw_context[:80_000]

    prompt = _JUDGE_PROMPT.format(
        question=question.strip(),
        context=context_text,
        answer=answer.strip()[:6_000],
    )

    try:
        import httpx
        from llm_providers import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL
        judge_model = os.environ.get("EVAL_JUDGE_MODEL", OLLAMA_LLM_MODEL)
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": _JUDGE_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0},
                },
            )
            resp.raise_for_status()
        raw = (resp.json().get("message", {}).get("content") or "").strip()
        scores = json.loads(raw)

        # Clamp all numeric scores to 0–10
        for key in ("faithfulness", "completeness", "relevance", "context_quality"):
            scores[key] = max(0, min(10, int(scores.get(key, 5))))

        scores["overall"] = round(
            (scores["faithfulness"] + scores["completeness"] +
             scores["relevance"] + scores["context_quality"]) / 4, 1
        )
        scores.setdefault("reasoning", "")
        logger.info(
            "judge_response: overall=%.1f  faith=%d  complete=%d  rel=%d  ctx=%d",
            scores["overall"], scores["faithfulness"], scores["completeness"],
            scores["relevance"], scores["context_quality"],
        )
        return scores

    except Exception as exc:
        logger.warning("judge_response: evaluation failed (%s), using fallback.", exc)
        return _fallback(str(exc))


def _fallback(reason: str) -> dict:
    return {
        "faithfulness": -1,
        "completeness": -1,
        "relevance": -1,
        "context_quality": -1,
        "overall": -1,
        "reasoning": f"Evaluation unavailable: {reason}",
    }
