"""
FastAPI route handlers.
"""

import ipaddress
import json
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

from typing import List

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

logger = logging.getLogger(__name__)

# 50 MB upload limit — prevents resource exhaustion from huge uploads
_MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_ALLOWED_URL_SCHEMES = {"http", "https"}
_ALLOWED_URL_EXTENSIONS = {".pdf", ".md", ".markdown"}
# Hostnames that must never be reached regardless of IP resolution
_FORBIDDEN_HOSTS = {"localhost", "metadata.google.internal", "169.254.169.254"}


def _validate_ingest_url(raw: str) -> str:
    """
    Return a sanitized URL or raise HTTPException.

    Blocks:
    - Non-http/https schemes
    - Private, loopback, link-local, and reserved IP addresses (SSRF)
    - Known cloud metadata endpoints
    - Unsupported file extensions
    """
    url = raw.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL must not be empty.")
    if len(url) > 2048:
        raise HTTPException(status_code=400, detail="URL exceeds maximum length.")

    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        raise HTTPException(status_code=400, detail="URL must use http or https.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise HTTPException(status_code=400, detail="URL is missing a hostname.")
    if hostname in _FORBIDDEN_HOSTS:
        raise HTTPException(status_code=400, detail="URL points to a forbidden host.")

    try:
        addr = ipaddress.ip_address(hostname)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
            raise HTTPException(
                status_code=400,
                detail="URL points to a private or reserved IP address.",
            )
    except ValueError:
        pass  # hostname is a domain name, not a raw IP — allowed

    path_lower = parsed.path.lower().split("?")[0]
    if not any(path_lower.endswith(ext) for ext in _ALLOWED_URL_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"URL must point to a supported file type: {', '.join(sorted(_ALLOWED_URL_EXTENSIONS))}.",
        )

    return url


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload file enforcing the size limit."""
    body = await file.read()
    if len(body) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        )
    return body


_RESTRUCTURE_SYSTEM_PROMPT = (
    "You clean OCR markdown for a clinical evidence viewer. "
    "Convert HTML tables into GitHub-flavoured markdown tables when possible. "
    "Remove unsafe HTML attributes. Preserve page markers, headings, paragraphs, and lists. "
    "Return only cleaned markdown, with no explanation and no code fences."
)


def _restructure_markdown_for_preview(markdown: str) -> str:
    """Use the local Ollama LLM to clean OCR markdown for preview."""
    import re

    if not markdown.strip():
        return markdown
    if not re.search(r"<(table|tr|td|th)\b", markdown, re.IGNORECASE):
        return markdown

    try:
        import httpx
        from llm_providers import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL

        base = (OLLAMA_BASE_URL or "").rstrip("/")
        if not base:
            return markdown
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{base}/api/chat",
                json={
                    "model": OLLAMA_LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _RESTRUCTURE_SYSTEM_PROMPT},
                        {"role": "user", "content": markdown},
                    ],
                    "stream": False,
                    "options": {"temperature": 0},
                },
            )
            resp.raise_for_status()
        content = (resp.json().get("message") or {}).get("content") or ""
        return str(content).strip() or markdown
    except Exception as exc:
        logger.warning("Markdown restructure failed; returning original preview: %s", exc)
        return markdown

from api.auth import get_owner_user_id
from api.models import (
    ChatRequest,
    ChatResponse,
    DeleteDocumentResponse,
    HealthResponse,
    IngestConfigResponse,
    IngestFromUrlRequest,
    IngestResponse,
    KnowledgeBaseDocument,
    KnowledgeBaseGraphLink,
    KnowledgeBaseGraphNode,
    KnowledgeBaseGraphResponse,
    KnowledgeBaseListResponse,
    PipelineInfo,
    PipelinesResponse,
    SourceChunk,
)

router = APIRouter()

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "publications"


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    driver = request.app.state.neo4j_driver
    try:
        driver.verify_connectivity()
        neo4j_status = "connected"
    except Exception as exc:
        neo4j_status = f"error: {exc}"

    return HealthResponse(
        status="ok" if neo4j_status == "connected" else "degraded",
        neo4j=neo4j_status,
    )


@router.get("/health/models")
async def health_models() -> JSONResponse:
    """Return active model backend config — shown in the UI status bar."""
    import httpx
    from llm_providers import (
        OLLAMA_BASE_URL,
        get_embed_model_name,
        get_embedding_dimensions,
        get_llm_model_name,
    )

    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            ollama_status = "connected" if resp.status_code == 200 else f"http {resp.status_code}"
    except Exception as exc:
        ollama_status = f"unreachable: {exc}"

    return JSONResponse({
        "backend": "ollama",
        "llm_model": get_llm_model_name(),
        "embed_model": get_embed_model_name(),
        "embed_dimensions": get_embedding_dimensions(),
        "ollama_url": OLLAMA_BASE_URL,
        "ollama_status": ollama_status,
    })


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: Request,
    body: ChatRequest,
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> ChatResponse:
    graph_rag = request.app.state.graph_rag

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    try:
        from retrieval.retriever import search

        result = await search(
            graph_rag=graph_rag,
            query=body.message,
            top_k=body.top_k,
            owner_user_id=owner_user_id,
            pipeline_id=body.pipeline_id,
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred while processing your request.")

    return ChatResponse(
        answer=result["answer"],
        source_chunks=[SourceChunk(**c) for c in result["source_chunks"]],
        session_id=body.session_id,
    )


@router.post("/chat/stream")
async def chat_stream(
    request: Request,
    body: ChatRequest,
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> StreamingResponse:
    """
    Server-Sent Events stream: pipeline ``step`` events, then a final ``result`` with answer + chunks.
    Each line is ``data: <json>\\n\\n`` for easy consumption with fetch + ReadableStream.
    """
    graph_rag = request.app.state.graph_rag

    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    from retrieval.retriever import search_stream_events

    async def event_generator():
        try:
            async for event in search_stream_events(
                graph_rag=graph_rag,
                query=body.message,
                top_k=body.top_k,
                owner_user_id=owner_user_id,
                pipeline_id=body.pipeline_id,
            ):
                line = json.dumps(event, ensure_ascii=False)
                yield f"data: {line}\n\n"
        except Exception as exc:
            err = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/ingest/config", response_model=IngestConfigResponse)
async def ingest_config() -> IngestConfigResponse:
    from ingestion.layout_analysis import layout_analysis_available
    from ingestion.docling_ocr import docling_ocr_enabled

    try:
        import fitz  # noqa: F401
        chunk_preview = True
    except Exception:
        chunk_preview = False

    _docling = docling_ocr_enabled()
    return IngestConfigResponse(
        paddle_ocr_preview=_docling,
        docling_ocr_available=_docling,
        chunk_preview=chunk_preview,
        layout_analysis=layout_analysis_available(),
    )


@router.post("/ingest/chunk-preview")
async def ingest_chunk_preview(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """
    Extract text (no OCR) and return fixed-size chunks for UI review.
    """
    from ingestion.chunk_preview import chunk_preview_pdf

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Chunk preview supports PDF only.")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        body = await _read_upload(file)
        with open(tmp_path, "wb") as f:
            f.write(body)
        payload = chunk_preview_pdf(Path(tmp_path))
        payload["source_filename"] = file.filename
        return JSONResponse(content=payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Chunk preview error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Chunk preview failed.") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/ingest/restructure")
async def ingest_restructure(raw_markdown: str = Form(...)) -> JSONResponse:
    """
    Run local LLM restructuring on demand: convert HTML tables and multi-column prose
    from OCR output into clean GitHub-flavoured markdown.
    Called only when the user explicitly clicks "Structured View" in the preview modal.
    """
    import asyncio

    if not raw_markdown.strip():
        raise HTTPException(status_code=400, detail="raw_markdown is empty.")

    try:
        structured = await asyncio.to_thread(_restructure_markdown_for_preview, raw_markdown)
        return JSONResponse(content={"markdown": structured})
    except Exception as exc:
        logger.error("Restructure error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Restructuring failed.") from exc


@router.get("/pipelines", response_model=PipelinesResponse)
async def list_pipelines_endpoint(request: Request) -> PipelinesResponse:
    """Return distinct pipeline IDs with document counts (for ablation study pipeline selection)."""
    from retrieval.retriever import list_pipelines
    driver = request.app.state.neo4j_driver
    pipelines = list_pipelines(driver)
    return PipelinesResponse(pipelines=[PipelineInfo(**p) for p in pipelines])


@router.post("/ingest/continue")
async def ingest_continue_stream(
    request: Request,
    markdown: UploadFile = File(...),
    source_filename: str = Form(...),
    ocr_source: str = Form("docling"),
    session_id: str | None = Form(None),
    pipeline_id: str | None = Form(None),
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> StreamingResponse:
    """
    Continue KG ingest using pre-parsed Markdown from the local OCR preview step.
    """
    driver = request.app.state.neo4j_driver
    from ingestion.ingest_stream import ingest_markdown_stream

    if not source_filename.strip():
        raise HTTPException(status_code=400, detail="source_filename is required.")

    try:
        raw = await markdown.read()
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="Markdown must be UTF-8.") from exc

    suffix = Path(source_filename).suffix.lower() or ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        Path(tmp_path).touch()
    except OSError:
        pass

    meta: dict = {"source": source_filename, "session_id": session_id}
    if owner_user_id:
        meta["owner_user_id"] = owner_user_id

    async def event_generator():
        try:
            async for item in ingest_markdown_stream(driver, tmp_path, text, meta, ocr_source=ocr_source, pipeline_id=pipeline_id):
                line = json.dumps(item, ensure_ascii=False)
                yield f"data: {line}\n\n"
        except Exception as exc:
            logger.exception(
                "ingest_continue_stream: SSE failed source_filename=%r ocr_source=%r pipeline_id=%r",
                source_filename,
                ocr_source,
                pipeline_id,
            )
            err = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest/stream")
async def ingest_upload_stream(
    request: Request,
    file: UploadFile = File(...),
    session_id: str | None = Form(None),
    pipeline_id: str | None = Form(None),
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> StreamingResponse:
    """
    Upload a PDF or Markdown file and stream SimpleKGPipeline events (Neo4j GraphRAG Pipeline.stream),
    then a ``done`` event with a small Neo4j graph preview for the document.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in (".pdf", ".md", ".markdown"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported format. Use .pdf, .md, or .markdown.",
        )

    driver = request.app.state.neo4j_driver
    from ingestion.ingest_stream import ingest_file_stream

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        body = await _read_upload(file)
        with open(tmp_path, "wb") as f:
            f.write(body)
    except HTTPException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    except Exception as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        logger.error("File read error for %s: %s", file.filename, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to read uploaded file.") from exc

    meta: dict = {"source": file.filename, "session_id": session_id}
    if owner_user_id:
        meta["owner_user_id"] = owner_user_id

    async def event_generator():
        try:
            async for item in ingest_file_stream(
                driver,
                tmp_path,
                meta,
                skip_external_parse=True,
                pipeline_id=pipeline_id,
            ):
                line = json.dumps(item, ensure_ascii=False)
                yield f"data: {line}\n\n"
        except Exception as exc:
            logger.exception(
                "ingest_upload_stream: SSE failed filename=%r pipeline_id=%r",
                file.filename,
                pipeline_id,
            )
            err = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest/batch")
async def ingest_batch_upload_stream(
    request: Request,
    files: List[UploadFile] = File(...),
    session_id: str | None = Form(None),
    pipeline_id: str | None = Form(None),
    ocr_mode: str = Form("none"),
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> StreamingResponse:
    """
    Upload multiple PDF/Markdown files and stream concurrent SimpleKGPipeline events.
    Each SSE event is tagged with ``file_index`` and ``file_name`` for per-file routing.
    OCR is skipped for speed; use the single-file endpoint for OCR preview workflows.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    allowed = {".pdf", ".md", ".markdown"}
    driver = request.app.state.neo4j_driver
    from ingestion.ingest_stream import ingest_batch_stream

    tmp_paths: list[str] = []
    files_meta: list[tuple[str, dict]] = []

    for f in files:
        if not f.filename:
            continue
        suffix = Path(f.filename).suffix.lower()
        if suffix not in allowed:
            continue
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        try:
            body = await _read_upload(f)
            with open(tmp_path, "wb") as fp:
                fp.write(body)
        except HTTPException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        except Exception as exc:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            logger.error("Batch file read error for %s: %s", f.filename, exc, exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to read uploaded file.") from exc
        tmp_paths.append(tmp_path)
        meta: dict = {"source": f.filename, "session_id": session_id, "ocr_mode": ocr_mode}
        if owner_user_id:
            meta["owner_user_id"] = owner_user_id
        if pipeline_id:
            meta["pipeline_id"] = pipeline_id
        files_meta.append((tmp_path, meta))

    if not files_meta:
        raise HTTPException(status_code=400, detail="No supported files (.pdf, .md, .markdown).")

    async def event_generator():
        try:
            async for item in ingest_batch_stream(driver, files_meta):
                line = json.dumps(item, ensure_ascii=False)
                yield f"data: {line}\n\n"
        except Exception as exc:
            names = [m.get("source") for _, m in files_meta if m.get("source")]
            logger.exception(
                "ingest_batch_upload_stream: SSE outer failure (%d files: %s) ocr_mode=%r",
                len(names),
                names,
                ocr_mode,
            )
            err = json.dumps({"type": "error", "message": str(exc)}, ensure_ascii=False)
            yield f"data: {err}\n\n"
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest/from-url")
async def ingest_from_url(
    request: Request,
    body: IngestFromUrlRequest,
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> StreamingResponse:
    """
    Ingest a document from a remote URL and stream SimpleKGPipeline events.

    Supported extensions: .pdf, .md, .markdown
    SSRF-safe: private IPs, loopback, and cloud metadata hosts are blocked.
    """
    url = _validate_ingest_url(body.url)
    driver = request.app.state.neo4j_driver
    from ingestion.ingest_stream import ingest_url_stream

    meta: dict = {"source": url, "session_id": body.session_id}
    if owner_user_id:
        meta["owner_user_id"] = owner_user_id

    async def event_generator():
        try:
            async for item in ingest_url_stream(driver, url, meta):
                line = json.dumps(item, ensure_ascii=False)
                yield f"data: {line}\n\n"
        except Exception as exc:
            logger.error("URL ingest error for %s: %s", url, exc, exc_info=True)
            err = json.dumps({"type": "error", "message": "Ingest from URL failed."}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest/docling-ocr-preview")
@router.post("/ingest/paddle-ocr-preview")  # legacy alias
async def ingest_paddle_ocr_preview(
    request: Request,
    file: UploadFile = File(...),
    use_layout_reader: bool = Form(False),
) -> JSONResponse:
    """
    Run Docling OCR on EC2 for a PDF and return OCR markdown for UI preview.
    Endpoint name kept for frontend compatibility.
    """
    from ingestion.docling_ocr import docling_ocr_pdf

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename.")

    suffix = Path(file.filename).suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(status_code=400, detail="Docling OCR preview supports PDF only.")

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    try:
        body = await _read_upload(file)
        with open(tmp_path, "wb") as f:
            f.write(body)
        import asyncio
        payload = await asyncio.to_thread(docling_ocr_pdf, Path(tmp_path))
        payload["source_filename"] = file.filename
        return JSONResponse(content=payload)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Docling OCR preview error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Docling OCR preview failed.") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request) -> IngestResponse:
    driver = request.app.state.neo4j_driver

    if not DATA_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Data directory not found: {DATA_DIR}")

    from ingestion.pipeline import ingest_pdfs

    result = await ingest_pdfs(driver=driver, pdf_dir=DATA_DIR)

    return IngestResponse(
        status=result["status"],
        files_processed=result.get("files_processed", 0),
        results=result.get("results", []),
    )


@router.get("/knowledge-base/documents", response_model=KnowledgeBaseListResponse)
async def knowledge_base_list(
    request: Request,
    pipeline_id: str | None = None,
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> KnowledgeBaseListResponse:
    """List ingested documents (Neo4j ``Document`` nodes) for the Knowledge Base UI."""
    from api.knowledge_base import list_documents

    driver = request.app.state.neo4j_driver
    rows = list_documents(driver, owner_user_id=owner_user_id, pipeline_id=pipeline_id or None)
    return KnowledgeBaseListResponse(
        documents=[KnowledgeBaseDocument(**r) for r in rows],
    )


@router.get("/knowledge-base/documents/graph", response_model=KnowledgeBaseGraphResponse)
async def knowledge_base_graph(
    request: Request,
    document_id: str = Query(..., description="Neo4j elementId of the Document node"),
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> KnowledgeBaseGraphResponse:
    """
    Neo4j subgraph for one document (Document, Chunk, extracted entities, and relationships).
    Distinct from ``Pipeline.draw()`` in neo4j-graphrag, which renders the ingest *pipeline* DAG.
    """
    from api.knowledge_base import get_document_graph

    driver = request.app.state.neo4j_driver
    raw = get_document_graph(driver, document_id=document_id, owner_user_id=owner_user_id)
    if not raw:
        raise HTTPException(status_code=404, detail="Document not found.")
    return KnowledgeBaseGraphResponse(
        document_id=raw["document_id"],
        document_name=raw["document_name"],
        nodes=[KnowledgeBaseGraphNode(**n) for n in raw["nodes"]],
        links=[KnowledgeBaseGraphLink(**x) for x in raw["links"]],
    )


@router.delete("/knowledge-base/documents", response_model=DeleteDocumentResponse)
async def knowledge_base_delete(
    request: Request,
    document_id: str = Query(..., description="Neo4j elementId of the Document node"),
    owner_user_id: str | None = Depends(get_owner_user_id),
) -> DeleteDocumentResponse:
    """
    Delete a document and all associated chunks; remove extracted entities that only
    linked to those chunks, then remove the Document node.
    """
    from api.knowledge_base import delete_document

    driver = request.app.state.neo4j_driver
    ok = delete_document(driver, document_id=document_id, owner_user_id=owner_user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found or could not be deleted.")
    return DeleteDocumentResponse(deleted=True)
