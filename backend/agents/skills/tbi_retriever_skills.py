"""
Skill definition for the TBI Retriever agent.

Drives the full GraphRAG pipeline:
  vector search → graph expansion → MMR filter → rerank → Gemma 4 generation
"""

SKILLS = {
    "name": "tbi_retriever",
    "description": (
        "Evidence-grounded TBI Q&A via GraphRAG. Retrieves relevant chunks from the "
        "Neo4j knowledge graph and generates answers strictly grounded in those chunks."
    ),
    "capabilities": [
        "Vector similarity search over TBI publication embeddings",
        "Cypher graph expansion to surface related entities and passages",
        "MMR diversity filtering to reduce redundant chunks",
        "Cross-encoder reranking for precision chunk selection",
        "Grounded answer generation using Gemma 4 26B MoE via Ollama",
        "Inline citation of source chunks in every answer",
    ],
    "constraints": [
        "Must answer ONLY from retrieved evidence chunks — never from training knowledge",
        "Must cite source chunks inline using [n] notation",
        "Must refuse to speculate beyond the retrieved context",
        "If context is insufficient, must say so explicitly rather than guessing",
    ],
}

ANSWER_STYLE = (
    "Answer directly and concisely. "
    "Lead with the key finding, then supporting detail. "
    "Cite each claim with [n] referring to the chunk number. "
    "If the evidence is inconclusive or absent, say so explicitly."
)
