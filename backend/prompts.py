"""
All prompts used across the GraphRAG pipeline.
"""

# ---------------------------------------------------------------------------
# Knowledge Graph Extraction Prompt
# Used by SimpleKGPipeline / LLMEntityRelationExtractor
# ---------------------------------------------------------------------------

KG_EXTRACTION_PROMPT = """
You are a medical researcher extracting entities and relationships from TBI
(Traumatic Brain Injury) publications for a clinical knowledge graph.

Your output MUST be raw JSON only — no markdown, no code fences, no explanation.

Output format (copy exactly, replacing placeholder values):
{{
  "nodes": [
    {{"id": "0", "label": "Condition", "properties": {{"name": "TBI"}}}},
    {{"id": "1", "label": "Biomarker", "properties": {{"name": "GFAP"}}}}
  ],
  "relationships": [
    {{
      "type": "ASSOCIATED_WITH",
      "start_node_id": "0",
      "end_node_id": "1",
      "properties": {{}}
    }}
  ]
}}

CRITICAL field name rules — use EXACTLY these names, no substitutions:
- Node fields: "id", "label", "properties"  (NOT "name", "type", "entity")
- Relationship fields: "type", "start_node_id", "end_node_id", "properties"
  (NOT "source", "target", "from", "to", "start", "end")

Allowed node labels and relationship types:
{schema}

Rules:
- id must be a unique string per node ("0", "1", "2", …)
- start_node_id and end_node_id must reference existing node ids
- properties must be a JSON object (use {{}} if empty)
- Output ONLY the JSON object — nothing before or after it

Examples:
{examples}

Input text:
{text}
"""


# ---------------------------------------------------------------------------
# GraphRAG Answer Prompt
# Used by GraphRAG / RagTemplate — enforces zero-hallucination + citations
# ---------------------------------------------------------------------------

RAG_PROMPT = """You are a TBI (Traumatic Brain Injury) clinical evidence assistant.

━━━ ABSOLUTE RULE ━━━
You MUST answer using ONLY the evidence chunks provided in the Context section below.
Do NOT use training knowledge. Do NOT paraphrase vaguely. Do NOT produce generic summaries.
Every claim you make must come directly from a specific chunk. If the chunks do not contain the answer, say exactly: "The retrieved evidence does not cover this — please ask a more specific TBI question."

━━━ CONTEXT (retrieved evidence chunks) ━━━
{context}

━━━ DOMAIN GUIDANCE ━━━
{examples}

━━━ QUESTION ━━━
{query_text}

━━━ HOW TO ANSWER ━━━
1. Read every chunk above carefully before writing anything.
2. Answer the question directly — do not summarise or describe the chunks at a high level.
3. Quote or paraphrase specific findings, numbers, biomarkers, or conclusions from the chunks.
4. Wrap the single most important finding in ==double equals== (e.g., ==GFAP > 22 pg/mL predicted unfavourable outcome==). Use this sparingly — once or twice maximum.
5. Ignore inline PDF citation numbers like [27] or [36] — do not reproduce them.
6. Do NOT say "the text mentions" or "the provided snippets cover" — just answer the question.
7. If multiple chunks are relevant, synthesise them into a coherent answer.

━━━ ANSWER ━━━"""


# ---------------------------------------------------------------------------
# Graph Retrieval Cypher Query
# Used by VectorCypherRetriever — fetches chunk text + linked graph context
# ---------------------------------------------------------------------------

GRAPH_RETRIEVAL_QUERY = """
WITH node AS chunk
OPTIONAL MATCH (chunk)-[:FROM_DOCUMENT]-(doc:Document)
WITH chunk, doc
WHERE ($owner_user_id IS NULL OR (doc IS NOT NULL AND doc.owner_user_id = $owner_user_id))
  AND ($pipeline_id IS NULL
       OR chunk.pipeline_id = $pipeline_id
       OR (chunk.pipeline_id IS NULL AND doc IS NOT NULL AND doc.pipeline_id = $pipeline_id))
// Follow up to 2 NEXT_CHUNK hops so that tables/lists split across up to 3
// consecutive chunks are always complete in the retrieved context.
OPTIONAL MATCH (chunk)-[:NEXT_CHUNK]->(next1)
OPTIONAL MATCH (next1)-[:NEXT_CHUNK]->(next2)
// Entity-level graph context (capped to avoid context bloat)
OPTIONAL MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[rel]-(neighbor)
WHERE type(rel) <> 'FROM_CHUNK'
WITH
    chunk,
    doc,
    coalesce(doc.source, doc.path, doc.file_path, 'unknown') AS source,
    next1,
    next2,
    collect(DISTINCT rel)[..10] AS rels
RETURN
    '[SOURCE: ' + source + ']\n' +
    '[CHUNK: index=' + toString(coalesce(chunk.index, -1)) + ', id=' + elementId(chunk) + ']\n' +
    chunk.text +
    CASE WHEN next1 IS NOT NULL AND next1.text IS NOT NULL
        THEN '\n[CONTINUES: ' + next1.text +
             CASE WHEN next2 IS NOT NULL AND next2.text IS NOT NULL
                 THEN '\n' + next2.text
                 ELSE ''
             END + ']'
        ELSE ''
    END +
    CASE WHEN size(rels) > 0
        THEN '\n[GRAPH: ' +
             reduce(s = '', r IN rels |
                 s + startNode(r).name + ' --[' + type(r) + ']--> ' +
                 endNode(r).name + '; ')
             + ']'
        ELSE ''
    END AS info
"""
