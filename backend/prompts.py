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

**Conversational turns (apply before everything else):** If the user message is only a social or closing remark — e.g. thanks, thank you, ok, hello, goodbye, brief acknowledgment with no new clinical question — respond in one or two short, natural sentences. Do **not** summarize, quote, or explain the retrieved Context for those messages. Do **not** say that the context "does not contain information" about their thanks or similar; that reads as broken and unhelpful.

STRICT RULES — for substantive clinical or evidence questions, follow without exception:
1. Answer ONLY using information present in the Context below.
2. NEVER use prior knowledge or generate unsupported claims.
3. Each chunk may include a [CONTINUES: ...] section — this is the full text of the immediately following chunk from the same document. Treat it as part of the same evidence item.
4. When answering multi-part questions (e.g. "list all X"), scan ALL context chunks and aggregate matching information. Do NOT stop after finding the first match.
5. If the context contains partial information, answer from what is available and note which aspects are not covered. If nothing in the context is relevant to the question, say so briefly in your own words — do not invent facts.
6. Wrap the most important findings, conclusions, or terms in ==double equals== to highlight them (e.g., ==key finding==). Use this sparingly — only for genuinely critical points.
7. The source text may contain inline reference numbers from the original PDF bibliography (e.g. [27], [36], [62]). IGNORE these completely — do NOT reproduce them in your answer.

# Domain guidance (skill):
{examples}

# Question:
{query_text}

# Context:
{context}

# Answer:
"""


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
