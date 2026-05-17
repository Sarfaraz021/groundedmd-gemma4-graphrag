"""
Request and response Pydantic models for the GraphRAG API.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    top_k: int = 5
    pipeline_id: str | None = None


class SourceChunk(BaseModel):
    index: int
    document: str
    source_path: str
    text: str
    chunk_index: int | None = None  # Neo4j Chunk.index in document
    chunk_node_id: str | None = None  # Neo4j element id for Chunk node


class ChatResponse(BaseModel):
    answer: str
    source_chunks: list[SourceChunk]
    session_id: str | None = None


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    results: list[dict]


class HealthResponse(BaseModel):
    status: str
    neo4j: str
    message: str = ""


class IngestConfigResponse(BaseModel):
    """Feature flags for ingest UI."""

    paddle_ocr_preview: bool  # kept for backwards compat; mirrors docling_ocr_available
    docling_ocr_available: bool
    chunk_preview: bool
    layout_analysis: bool  # torch + transformers available for LayoutReader / LayoutLMv3


class KnowledgeBaseDocument(BaseModel):
    """One row in the Neo4j lexical graph ``Document`` list."""

    id: str
    name: str
    path: str = ""
    chunk_count: int = 0


class KnowledgeBaseListResponse(BaseModel):
    documents: list[KnowledgeBaseDocument]


class DeleteDocumentResponse(BaseModel):
    deleted: bool


class KnowledgeBaseGraphNode(BaseModel):
    id: str
    name: str
    kind: str
    group: int = 0


class KnowledgeBaseGraphLink(BaseModel):
    source: str
    target: str
    rel_type: str


class KnowledgeBaseGraphResponse(BaseModel):
    """Neo4j subgraph for one document (chunks, entities, relationships)."""

    document_id: str
    document_name: str
    nodes: list[KnowledgeBaseGraphNode]
    links: list[KnowledgeBaseGraphLink]


class IngestFromUrlRequest(BaseModel):
    """Request body for ``POST /ingest/from-url``."""

    url: str
    session_id: str | None = None


class PipelineInfo(BaseModel):
    id: str
    doc_count: int


class PipelinesResponse(BaseModel):
    pipelines: list[PipelineInfo]
