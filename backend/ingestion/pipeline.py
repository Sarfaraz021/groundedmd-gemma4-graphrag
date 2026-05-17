"""
Knowledge Graph construction pipeline using neo4j-graphrag SimpleKGPipeline.

Ingests TBI publication PDFs → extracts entities/relationships → writes to Neo4j.
Each chunk retains source document metadata for downstream citation.
"""

import langsmith_env  # noqa: F401 — load .env + LangSmith before Neo4j GraphRAG

import inspect
import os
from pathlib import Path

import neo4j
from langchain_text_splitters import RecursiveCharacterTextSplitter as _RCTS
from neo4j_graphrag.experimental.components.text_splitters.base import TextChunks, TextSplitter
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import TextChunk
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

from llm_providers import build_embedder, build_extraction_llm
from prompts import KG_EXTRACTION_PROMPT
from schema import NODE_LABELS, PATTERNS, RELATIONSHIP_TYPES

# Unified semantic chunk sizes — used for PDF/Markdown and local OCR ingestion.
# Soft target: 4000 chars.  Hard max: 6000 chars (1.5×).
# Overlap: 600 chars prepended from the previous chunk for cross-boundary context.
# breakpoint_percentile_threshold=85 — only very distinct topic shifts trigger a split,
# keeping clinical methodology sections and findings intact. Test 75/80/85 for ablation.
CHUNK_SIZE = 4000          # kept for backward-compat imports; same value as SEMANTIC_CHUNK_SIZE
CHUNK_OVERLAP = 600        # kept for backward-compat imports; same value as SEMANTIC_CHUNK_OVERLAP
SEMANTIC_CHUNK_SIZE = 4000
SEMANTIC_CHUNK_OVERLAP = 600

# Priority-ordered separators for RecursiveCharacterTextSplitter.
# RCTS tries each in order, falling back to the next only when a candidate
# split would still exceed chunk_size.  Page markers come first so the
# splitter never crosses a [Page N] boundary unless a single page exceeds
# the chunk limit.
SEPARATORS: list[str] = [
    "\n\n[Page ",   # OCR/page boundary markers
    "\n\n## ",      # Markdown H2 section headings
    "\n\n### ",     # Markdown H3 sub-sections
    "\n\n#### ",    # Markdown H4
    "\n\n",         # Paragraph breaks
    "\n",           # Line breaks
    " ",            # Word boundaries
    "",             # Character-level last resort
]


class _RecursiveCharacterSplitter(TextSplitter):
    """
    Wraps LangChain RecursiveCharacterTextSplitter as a neo4j-graphrag
    TextSplitter Component.  Implements the required async ``run`` interface
    so ComponentType validation in SimpleKGPipeline passes.
    """

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str],
    ) -> None:
        self._rcts = _RCTS(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            keep_separator=True,
        )

    async def run(self, text: str) -> TextChunks:
        pieces = self._rcts.split_text(text)
        return TextChunks(chunks=[TextChunk(text=t, index=i) for i, t in enumerate(pieces)])


LEXICAL_GRAPH_CONFIG = LexicalGraphConfig(
    chunk_node_label="Chunk",
    document_node_label="Document",
    chunk_to_document_relationship_type="FROM_DOCUMENT",
    next_chunk_relationship_type="NEXT_CHUNK",
    node_to_chunk_relationship_type="FROM_CHUNK",
    chunk_embedding_property="embedding",
)


def _simple_kg_pipeline_file_mode_kwarg(from_file: bool) -> dict[str, bool]:
    """
    Older neo4j-graphrag builds use ``from_pdf``; current builds use ``from_file``.
    Semantics match: True = load from disk path, False = use ``text`` + metadata path.
    """
    sig = inspect.signature(SimpleKGPipeline.__init__)
    params = sig.parameters
    if "from_file" in params:
        return {"from_file": from_file}
    if "from_pdf" in params:
        return {"from_pdf": from_file}
    raise RuntimeError(
        "SimpleKGPipeline has neither 'from_file' nor 'from_pdf'. "
        "Upgrade neo4j-graphrag: pip install -U 'neo4j-graphrag[ollama]'"
    )


def build_pipeline(driver: neo4j.Driver, *, from_file: bool = True) -> SimpleKGPipeline:
    """
    Chunking uses RecursiveCharacterTextSplitter (no external embedding API for splits).

    ``from_file=True``: PdfLoader / MarkdownLoader reads from disk, then split.
    ``from_file=False``: Pre-parsed markdown from local OCR, then split.

    LLM + chunk embeddings for the graph pipeline come from ``llm_providers`` (Ollama).
    """
    llm = build_extraction_llm()
    embedder = build_embedder()

    file_mode = _simple_kg_pipeline_file_mode_kwarg(from_file)

    splitter = _RecursiveCharacterSplitter(
        chunk_size=SEMANTIC_CHUNK_SIZE,
        chunk_overlap=SEMANTIC_CHUNK_OVERLAP,
        separators=SEPARATORS,
    )
    schema_dict = {
        "node_types": NODE_LABELS,
        "relationship_types": RELATIONSHIP_TYPES,
        "patterns": PATTERNS,
        "additional_node_types": False,
    }
    if "from_file" in file_mode:
        return SimpleKGPipeline(
            llm=llm,
            driver=driver,
            embedder=embedder,
            text_splitter=splitter,
            schema=schema_dict,
            prompt_template=KG_EXTRACTION_PROMPT,
            lexical_graph_config=LEXICAL_GRAPH_CONFIG,
            perform_entity_resolution=True,
            on_error="IGNORE",
            from_file=file_mode["from_file"],  # type: ignore[call-arg]
        )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=embedder,
        text_splitter=splitter,
        schema=schema_dict,
        prompt_template=KG_EXTRACTION_PROMPT,
        lexical_graph_config=LEXICAL_GRAPH_CONFIG,
        perform_entity_resolution=True,
        on_error="IGNORE",
        from_pdf=file_mode["from_pdf"],
    )


async def ingest_pdfs(driver: neo4j.Driver, pdf_dir: Path) -> dict:
    import asyncio

    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        return {"status": "error", "message": f"No PDFs found in {pdf_dir}"}

    results = []
    for path in pdf_paths:
        print(f"  Ingesting: {path.name}")
        try:
            meta = {"source": path.name, "file_path": str(path)}
            pipeline = build_pipeline(driver, from_file=True)
            result = await pipeline.run_async(
                file_path=str(path),
                document_metadata=meta,
            )
            results.append({"file": path.name, "status": "success", "result": str(result)})
            print(f"  Done:  {path.name}")
        except Exception as exc:
            results.append({"file": path.name, "status": "error", "error": str(exc)})
            print(f"  Error: {path.name} — {exc}")

    return {"status": "complete", "files_processed": len(results), "results": results}
