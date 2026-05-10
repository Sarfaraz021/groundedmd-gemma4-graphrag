"""
All prompts used across the GraphRAG pipeline.
"""

# ---------------------------------------------------------------------------
# Knowledge Graph Extraction Prompt
# Used by SimpleKGPipeline / LLMEntityRelationExtractor
# ---------------------------------------------------------------------------

KG_EXTRACTION_PROMPT = """
You are a medical researcher tasked with extracting information from TBI
(Traumatic Brain Injury) publications and structuring it as a property graph
for clinical Q&A.

Extract entities (nodes) and their types from the Input text.
Extract relationships between those nodes.
Relationship direction: start node → end node.

Return ONLY valid JSON in this exact format:
{{
  "nodes": [
    {{"id": "0", "label": "<node type>", "properties": {{"name": "<entity name>"}}}}
  ],
  "relationships": [
    {{
      "type": "<RELATIONSHIP_TYPE>",
      "start_node_id": "0",
      "end_node_id": "1",
      "properties": {{"details": "<brief description of the relationship>"}}
    }}
  ]
}}

Use ONLY the following nodes and relationships:
{schema}

Rules:
- Assign a unique string ID to each node.
- Reuse node IDs in relationships.
- Respect source/target types and relationship direction from the schema.
- Do not return anything outside the JSON.

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
