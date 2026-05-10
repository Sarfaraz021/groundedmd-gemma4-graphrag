"""
Stream SimpleKGPipeline execution events (neo4j-graphrag Pipeline.stream) + Neo4j graph preview.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import urllib.request
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import neo4j
from neo4j.exceptions import TransientError as Neo4jTransientError
from pydantic.v1.utils import deep_update

logger = logging.getLogger(__name__)


def _ingest_exc_summary(exc: BaseException, max_len: int = 800) -> str:
    """Short, log- and UI-safe summary (no traceback)."""
    s = f"{type(exc).__name__}: {exc}"
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


# ---------------------------------------------------------------------------
# Retry helper — Neo4j deadlocks + transient upstream (429 / rate limit) errors.
# ---------------------------------------------------------------------------
_RETRY_EXCEPTIONS = (Neo4jTransientError,)
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 2.0  # seconds; doubles each attempt


def _is_retryable(exc: BaseException) -> bool:
    """Return True for Neo4j deadlocks and retryable rate-limit / overload errors."""
    if isinstance(exc, _RETRY_EXCEPTIONS):
        return True
    cls_name = type(exc).__name__
    if "RateLimit" in cls_name or "ServiceUnavailable" in cls_name:
        return True
    # Catch by message as a fallback
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "deadlock" in msg


async def _run_with_retry(coro_fn, *args, **kwargs):
    """
    Call ``coro_fn(*args, **kwargs)`` (an async generator) up to _MAX_RETRIES times.

    Yields events from the generator.  On a retryable exception the generator is
    restarted from scratch after an exponential back-off delay.  On the final
    attempt the exception propagates.
    """
    delay = _RETRY_BASE_DELAY
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async for event in coro_fn(*args, **kwargs):
                yield event
            return  # success
        except Exception as exc:
            if attempt == _MAX_RETRIES or not _is_retryable(exc):
                raise
            logger.warning(
                "Batch ingest attempt %d/%d failed (%s: %s) — retrying in %.1fs",
                attempt, _MAX_RETRIES, type(exc).__name__, exc, delay,
            )
            yield {
                "type": "warning",
                "message": f"Attempt {attempt} failed ({type(exc).__name__}), retrying in {delay:.0f}s…",
            }
            await asyncio.sleep(delay)
            delay *= 2

from neo4j_graphrag.experimental.pipeline.notification import Event, TaskEvent

from api.knowledge_base import _session

from ingestion.pipeline import CHUNK_OVERLAP, CHUNK_SIZE, SEMANTIC_CHUNK_OVERLAP, SEMANTIC_CHUNK_SIZE, build_pipeline
from llm_providers import get_embed_model_name


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if hasattr(obj, "model_dump"):
        try:
            return _json_safe(obj.model_dump(mode="python"))
        except Exception:
            pass
    return str(obj)


def event_to_dict(ev: Event) -> dict[str, Any]:
    d: dict[str, Any] = {
        "event_type": ev.event_type.value,
        "run_id": ev.run_id,
        "timestamp": ev.timestamp.isoformat(),
        "message": ev.message,
        "payload": _json_safe(ev.payload),
    }
    if isinstance(ev, TaskEvent):
        d["task_name"] = ev.task_name
    return d


def _task_hint(task_name: str | None, pre_parsed: bool = False) -> dict[str, str]:
    """Human-readable stage hints for the UI."""
    if not task_name:
        return {}
    splitter_detail = (
        f"RecursiveCharacterSplitter · chunk_size={SEMANTIC_CHUNK_SIZE} · overlap={SEMANTIC_CHUNK_OVERLAP}"
        if pre_parsed
        else f"RecursiveCharacterSplitter · chunk_size={CHUNK_SIZE} · overlap={CHUNK_OVERLAP}"
    )
    hints: dict[str, dict[str, str]] = {
        "file_loader": {
            "title": "Document loader",
            "detail": "PdfLoader / MarkdownLoader (by file extension)",
        },
        "splitter": {
            "title": "Chunking",
            "detail": splitter_detail,
        },
        "chunk_embedder": {
            "title": "Chunk embeddings",
            "detail": f"{__import__('llm_providers').get_embed_model_name()} on each chunk",
        },
        "schema": {
            "title": "Schema",
            "detail": "SchemaBuilder / schema from config",
        },
        "extractor": {
            "title": "Entity & relation extraction",
            "detail": "LLMEntityRelationExtractor (structured graph)",
        },
        "pruner": {
            "title": "Graph pruning",
            "detail": "GraphPruning vs schema",
        },
        "writer": {
            "title": "Write to Neo4j",
            "detail": "Neo4jWriter — lexical + semantic graph",
        },
        "resolver": {
            "title": "Entity resolution",
            "detail": "SinglePropertyExactMatchResolver (merge same label+name)",
        },
    }
    return hints.get(task_name, {"title": task_name, "detail": ""})


def _log_kg_pipeline_task_event(source: str, d: dict[str, Any]) -> None:
    """
    INFO logs at coarse ingest milestones: document load, LLM extraction, Neo4j write.

    neo4j-graphrag emits TASK_* events with ``task_name`` matching pipeline step ids.
    """
    et = str(d.get("event_type") or "")
    task = d.get("task_name")
    if not task:
        return
    if et not in ("TASK_STARTED", "TASK_FINISHED"):
        return

    run_id = d.get("run_id")
    rid = f" run_id={run_id}" if run_id else ""
    msg = d.get("message")
    tail = f" message={msg!r}" if msg else ""

    if task == "file_loader":
        phase = "start" if et == "TASK_STARTED" else "complete"
        logger.info(
            "ingest: source=%r stage=document_load (%s)%s%s",
            source,
            phase,
            rid,
            tail,
        )
    elif task == "extractor":
        phase = "start" if et == "TASK_STARTED" else "complete"
        logger.info(
            "ingest: source=%r stage=entity_relation_extraction (%s)%s%s",
            source,
            phase,
            rid,
            tail,
        )
    elif task == "writer":
        phase = "start" if et == "TASK_STARTED" else "complete"
        logger.info(
            "ingest: source=%r stage=neo4j_graph_write (%s)%s%s",
            source,
            phase,
            rid,
            tail,
        )
    elif task == "resolver":
        phase = "start" if et == "TASK_STARTED" else "complete"
        logger.info(
            "ingest: source=%r stage=neo4j_entity_resolution (%s)%s%s",
            source,
            phase,
            rid,
            tail,
        )


def _tag_pipeline_nodes(
    driver: neo4j.Driver,
    exact_path: str,
    basename: str,
    pipeline_id: str,
) -> None:
    """Set pipeline_id on the Document and all its Chunk nodes after ingest."""
    b = Path(basename).name
    q = """
    MATCH (d:Document)
    WHERE (d.path = $exact OR coalesce(d.path, '') CONTAINS $b OR coalesce(d.path, '') ENDS WITH $b)
    WITH d ORDER BY elementId(d) DESC LIMIT 1
    SET d.pipeline_id = $pipeline_id
    WITH d
    OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
    SET c.pipeline_id = $pipeline_id
    RETURN count(c) AS tagged
    """
    with _session(driver) as session:
        rec = session.run(q, exact=exact_path, b=b, pipeline_id=pipeline_id).single()
        tagged = int(rec["tagged"]) if rec else 0
        logger.info(
            "ingest: Neo4j tagged pipeline_id on Document + Chunk nodes basename=%r pipeline_id=%r chunks_tagged=%d",
            b,
            pipeline_id,
            tagged,
        )
        logger.debug(
            "_tag_pipeline_nodes: exact=%r b=%r",
            exact_path,
            b,
        )
        if tagged == 0:
            logger.warning("_tag_pipeline_nodes: 0 chunks tagged — document path may not match. "
                           "exact_path=%r  basename=%r", exact_path, b)


def _ensure_document_owner(
    driver: neo4j.Driver,
    exact_path: str,
    basename: str,
    owner_user_id: str,
) -> None:
    """Set ``owner_user_id`` on the ingested Document (latest match by path) for multi-tenant isolation."""
    b = Path(basename).name
    q = """
    MATCH (d:Document)
    WHERE (d.path = $exact OR coalesce(d.path, '') CONTAINS $b OR coalesce(d.path, '') ENDS WITH $b)
    WITH d ORDER BY elementId(d) DESC LIMIT 1
    SET d.owner_user_id = $owner
    """
    with _session(driver) as session:
        session.run(q, exact=exact_path, b=b, owner=owner_user_id)
    logger.info(
        "ingest: Neo4j updated Document.owner_user_id basename=%r path_suffix=%r",
        b,
        Path(exact_path).name,
    )


def _graph_preview_sync(
    driver: neo4j.Driver,
    exact_path: str,
    basename: str,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    """Stats + sample nodes/edges for UI after ingest (Document.path is the temp file path used during ingest)."""
    b = Path(basename).name
    out: dict[str, Any] = {
        "basename": b,
        "stats": {"chunks": 0, "entities": 0, "relationships": 0},
        "nodes": [],
        "links": [],
    }
    stats_q = """
    MATCH (d:Document)
    WHERE (d.path = $exact OR coalesce(d.path, '') CONTAINS $b OR coalesce(d.path, '') ENDS WITH $b)
      AND ($owner_user_id IS NULL OR d.owner_user_id = $owner_user_id)
    WITH d ORDER BY id(d) DESC LIMIT 1
    OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
    OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
    RETURN count(DISTINCT c) AS chunks, count(DISTINCT e) AS entities
    """
    rel_q = """
    MATCH (d:Document)
    WHERE (d.path = $exact OR coalesce(d.path, '') CONTAINS $b OR coalesce(d.path, '') ENDS WITH $b)
      AND ($owner_user_id IS NULL OR d.owner_user_id = $owner_user_id)
    WITH d ORDER BY id(d) DESC LIMIT 1
    MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
    MATCH (e)-[:FROM_CHUNK]->(c)
    OPTIONAL MATCH (e)-[r]-(o)
    WHERE o IS NOT NULL AND type(r) <> 'FROM_CHUNK'
    RETURN DISTINCT
      id(e) AS eid,
      coalesce(e.name, '') AS ename,
      head(labels(e)) AS elabel,
      type(r) AS rel,
      id(o) AS oid,
      coalesce(o.name, '') AS oname,
      head(labels(o)) AS olabel
    LIMIT 60
    """
    q_params = {"exact": exact_path, "b": b, "owner_user_id": owner_user_id}
    with _session(driver) as session:
        rec = session.run(stats_q, **q_params).single()
        if rec:
            out["stats"]["chunks"] = int(rec["chunks"] or 0)
            out["stats"]["entities"] = int(rec["entities"] or 0)
        seen: set[tuple[int, str, int]] = set()
        for row in session.run(rel_q, **q_params):
            eid, oid = row["eid"], row["oid"]
            rel = row["rel"]
            if eid is None or oid is None or rel is None:
                continue
            key = (eid, rel, oid)
            if key in seen:
                continue
            seen.add(key)
            out["links"].append(
                {
                    "source": int(eid),
                    "target": int(oid),
                    "type": rel,
                    "source_name": row["ename"] or "?",
                    "target_name": row["oname"] or "?",
                }
            )
        out["stats"]["relationships"] = len(out["links"])
        # Node list for force-free viz (unique ids)
        nmap: dict[int, dict[str, Any]] = {}
        for link in out["links"]:
            for nid, nm, lab in (
                (link["source"], link["source_name"], "Entity"),
                (link["target"], link["target_name"], "Entity"),
            ):
                if nid not in nmap:
                    nmap[nid] = {"id": nid, "name": nm, "kind": lab}
        out["nodes"] = list(nmap.values())[:80]
    return out


async def ingest_file_stream(
    driver: neo4j.Driver,
    file_path: str,
    document_metadata: dict[str, Any] | None = None,
    skip_external_parse: bool = False,
    pipeline_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Yield serialized pipeline events, then a ``done`` payload with graph preview.

    Uses ``Pipeline.stream`` from the SimpleKGPipeline runner (TASK_STARTED / TASK_PROGRESS / …).
    """
    document_metadata = dict(document_metadata or {})
    if document_metadata.get("owner_user_id") is not None:
        document_metadata["owner_user_id"] = str(document_metadata["owner_user_id"])
    path = Path(file_path)
    # Auto-tag so every document always has a pipeline_id for filtering
    if not pipeline_id:
        pipeline_id = "no_ocr"

    _src = document_metadata.get("source") or path.name
    logger.info(
        "ingest_file_stream: start source=%r path=%s",
        _src,
        file_path,
    )

    kg = build_pipeline(driver, from_file=True)
    user_input = {
        "file_path": file_path,
        "document_metadata": document_metadata,
    }

    runner = kg.runner
    run_param = deep_update(
        runner.run_params,
        runner.config.get_run_params(user_input),
    )

    yield {
        "type": "meta",
        "loader": "PdfLoader or MarkdownLoader (extension-based)",
        "chunk_config": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
        "embedding_model": get_embed_model_name(),
    }

    try:
        async for event in runner.pipeline.stream(data=run_param, raise_exception=True):
            d = event_to_dict(event)
            d["hint"] = _task_hint(d.get("task_name"), pre_parsed=False)
            _log_kg_pipeline_task_event(_src, d)
            yield {"type": "event", "event": d}
    except Exception as exc:
        logger.error(
            "ingest_file_stream: pipeline stream failed source=%r path=%s — %s",
            _src,
            file_path,
            _ingest_exc_summary(exc),
            exc_info=True,
        )
        raise

    logger.info("ingest_file_stream: pipeline stream completed source=%r", _src)

    basename = str(document_metadata.get("source") or Path(file_path).name)
    _owner = document_metadata.get("owner_user_id")
    logger.info("ingest: source=%r post_pipeline=neo4j_bookkeeping (owner / pipeline_id / preview)", _src)
    if _owner is not None:
        await asyncio.to_thread(
            _ensure_document_owner, driver, file_path, basename, str(_owner)
        )
    if pipeline_id:
        await asyncio.to_thread(_tag_pipeline_nodes, driver, file_path, basename, pipeline_id)
    _owner = str(_owner) if _owner is not None else None
    preview = await asyncio.to_thread(_graph_preview_sync, driver, file_path, basename, _owner)
    logger.info(
        "ingest: source=%r post_pipeline=graph_preview chunks=%s entities=%s rels=%s",
        _src,
        preview.get("stats", {}).get("chunks"),
        preview.get("stats", {}).get("entities"),
        preview.get("stats", {}).get("relationships"),
    )
    yield {"type": "done", "graph_preview": preview}


async def ingest_url_stream(
    driver: neo4j.Driver,
    url: str,
    document_metadata: dict[str, Any] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Ingest a document from a remote URL and stream KG pipeline events.

    ``document_metadata["source"]`` is set to the URL so the Document node in
    Neo4j records where the file came from.
    """
    document_metadata = dict(document_metadata or {})
    if document_metadata.get("owner_user_id") is not None:
        document_metadata["owner_user_id"] = str(document_metadata["owner_user_id"])
    document_metadata.setdefault("source", url)

    # Download to a temp file, then run the standard file pipeline.
    suffix = Path(url.split("?")[0]).suffix.lower() or ".pdf"
    if suffix not in (".pdf", ".md", ".markdown"):
        suffix = ".pdf"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        yield {
            "type": "meta",
            "loader": "URL download → PdfLoader or MarkdownLoader",
            "chunk_config": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP},
            "embedding_model": get_embed_model_name(),
            "source_url": url,
        }
        await asyncio.to_thread(urllib.request.urlretrieve, url, tmp_path)
        async for event in ingest_file_stream(
            driver, tmp_path, document_metadata, skip_external_parse=True
        ):
            yield event
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
def _batch_max_concurrent() -> int:
    """
    Maximum files processed in parallel.

    Defaults to 1 (sequential) to avoid Neo4j MERGE deadlocks and concurrent
    pipeline pressure on Ollama / Neo4j.  Override with ``BATCH_MAX_CONCURRENT``
    env var (max 5).
    """
    try:
        val = int(os.environ.get("BATCH_MAX_CONCURRENT", "1"))
        return max(1, min(val, 5))
    except ValueError:
        return 1


async def _batch_paddle_stream(
    driver: neo4j.Driver,
    file_path: str,
    meta: dict[str, Any],
    pipeline_id: str | None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run Paddle OCR on a single PDF (no preview), then ingest the markdown."""
    from ingestion.paddle_ocr import paddle_ocr_preview_pdf  # type: ignore[import]

    path = Path(file_path)
    try:
        logger.info("batch_paddle_ocr: starting path=%s meta_source=%r", path, meta.get("source"))
        result = await asyncio.to_thread(paddle_ocr_preview_pdf, path, use_layout_reader=False)
        markdown: str = result.get("markdown") or ""
    except Exception as exc:
        logger.error(
            "batch_paddle_ocr: Paddle OCR raised for %r: %s",
            meta.get("source"),
            _ingest_exc_summary(exc),
            exc_info=True,
        )
        yield {"type": "error", "message": f"Paddle OCR failed: {exc}"}
        return

    if not markdown.strip():
        logger.warning(
            "batch_paddle_ocr: empty markdown after OCR for %r path=%s",
            meta.get("source"),
            path,
        )
        yield {"type": "error", "message": "Paddle OCR produced no text."}
        return

    async for event in ingest_markdown_stream(
        driver,
        file_path,
        markdown,
        meta,
        ocr_source="paddle",
        pipeline_id=pipeline_id,
    ):
        yield event


async def ingest_batch_stream(
    driver: neo4j.Driver,
    files_meta: list[tuple[str, dict[str, Any]]],
    max_concurrent: int | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Ingest multiple files, streaming tagged SSE events per file.

    Concurrency is controlled by ``max_concurrent`` (default: ``BATCH_MAX_CONCURRENT``
    env var, which defaults to **1** — sequential — to reduce Neo4j deadlocks and
    concurrent load on Ollama when multiple pipelines write at once).

    Each event is tagged with ``file_index`` and ``file_name`` so the frontend
    can route events to the correct per-file ingest job.
    Transient errors (deadlocks, 429s) are retried with exponential back-off.
    """
    if not files_meta:
        return

    concurrency = max_concurrent if max_concurrent is not None else _batch_max_concurrent()
    total = len(files_meta)
    semaphore = asyncio.Semaphore(concurrency)
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

    async def _process(idx: int, file_path: str, meta: dict[str, Any]) -> None:
        file_name = meta.get("source", "")
        _pipeline_id: str | None = meta.get("pipeline_id")
        _ocr_mode: str = (meta.get("ocr_mode") or "none").lower()
        async with semaphore:
            # Announce queue position before starting
            await queue.put({
                "type": "progress",
                "file_index": idx,
                "file_name": file_name,
                "completed": idx,
                "total": total,
            })
            try:
                logger.info(
                    "ingest_batch: start file_index=%d file_name=%r ocr_mode=%r pipeline_id=%r temp_path=%s",
                    idx,
                    file_name,
                    _ocr_mode,
                    _pipeline_id,
                    file_path,
                )
                if _ocr_mode == "paddle" and file_name.lower().endswith(".pdf"):
                    async for event in _batch_paddle_stream(
                        driver, file_path, meta, _pipeline_id
                    ):
                        tagged = dict(event)
                        tagged["file_index"] = idx
                        tagged["file_name"] = file_name
                        await queue.put(tagged)
                    logger.info("ingest_batch: finished file_index=%d file_name=%r (paddle path)", idx, file_name)
                    return
                async for event in _run_with_retry(
                    ingest_file_stream, driver, file_path, meta,
                    skip_external_parse=True, pipeline_id=_pipeline_id
                ):
                    tagged = dict(event)
                    tagged["file_index"] = idx
                    tagged["file_name"] = file_name
                    await queue.put(tagged)
                logger.info("ingest_batch: finished file_index=%d file_name=%r (file stream path)", idx, file_name)
            except Exception as exc:
                summary = _ingest_exc_summary(exc)
                logger.error(
                    "ingest_batch: failed file_index=%d file_name=%r after retries — %s",
                    idx,
                    file_name,
                    summary,
                    exc_info=True,
                )
                await queue.put({
                    "type": "error",
                    "file_index": idx,
                    "file_name": file_name,
                    "message": f"Ingest failed: {summary}",
                })
            finally:
                await queue.put(None)  # per-task sentinel

    # Store task references to prevent garbage collection before completion.
    tasks = [
        asyncio.create_task(_process(i, fp, meta))
        for i, (fp, meta) in enumerate(files_meta)
    ]

    pending = len(tasks)
    while pending > 0:
        item = await queue.get()
        if item is None:
            pending -= 1
        else:
            yield item


async def ingest_markdown_stream(
    driver: neo4j.Driver,
    file_path: str,
    markdown: str,
    document_metadata: dict[str, Any] | None = None,
    ocr_source: str = "paddle",
    pipeline_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """
    Run SimpleKGPipeline with pre-parsed Markdown (``from_file=False``), e.g. after local OCR preview.
    ``file_path`` is stored on the Document node (use a temp path unique to this ingest).
    """
    document_metadata = dict(document_metadata or {})
    if document_metadata.get("owner_user_id") is not None:
        document_metadata["owner_user_id"] = str(document_metadata["owner_user_id"])
    kg = build_pipeline(driver, from_file=False)
    user_input = {
        "file_path": file_path,
        "text": markdown,
        "document_metadata": document_metadata,
    }
    runner = kg.runner
    run_param = deep_update(
        runner.run_params,
        runner.config.get_run_params(user_input),
    )

    # Auto-tag so every document always has a pipeline_id for filtering
    if not pipeline_id:
        pipeline_id = "paddle_ocr"

    _src = document_metadata.get("source") or Path(file_path).name
    logger.info(
        "ingest_markdown_stream: start source=%r path=%s ocr_source=%r markdown_chars=%d pipeline_id=%r",
        _src,
        file_path,
        ocr_source,
        len(markdown),
        pipeline_id,
    )

    yield {
        "type": "meta",
        "loader": "Pre-parsed Markdown (Paddle OCR) → in-graph pipeline",
        "markdown_chars": len(markdown),
        "chunk_config": {"size": SEMANTIC_CHUNK_SIZE, "overlap": SEMANTIC_CHUNK_OVERLAP},
        "embedding_model": get_embed_model_name(),
        "parse_preview_continue": True,
    }

    try:
        async for event in runner.pipeline.stream(data=run_param, raise_exception=True):
            d = event_to_dict(event)
            d["hint"] = _task_hint(d.get("task_name"), pre_parsed=True)
            _log_kg_pipeline_task_event(_src, d)
            yield {"type": "event", "event": d}
    except Exception as exc:
        logger.error(
            "ingest_markdown_stream: pipeline stream failed source=%r ocr_source=%r — %s",
            _src,
            ocr_source,
            _ingest_exc_summary(exc),
            exc_info=True,
        )
        raise

    logger.info("ingest_markdown_stream: pipeline stream completed source=%r", _src)

    basename = str(document_metadata.get("source") or Path(file_path).name)
    _owner = document_metadata.get("owner_user_id")
    logger.info("ingest: source=%r post_pipeline=neo4j_bookkeeping (owner / pipeline_id / preview)", _src)
    if _owner is not None:
        await asyncio.to_thread(
            _ensure_document_owner, driver, file_path, basename, str(_owner)
        )
    if pipeline_id:
        await asyncio.to_thread(_tag_pipeline_nodes, driver, file_path, basename, pipeline_id)
    _owner = str(_owner) if _owner is not None else None
    preview = await asyncio.to_thread(_graph_preview_sync, driver, file_path, basename, _owner)
    logger.info(
        "ingest: source=%r post_pipeline=graph_preview chunks=%s entities=%s rels=%s",
        _src,
        preview.get("stats", {}).get("chunks"),
        preview.get("stats", {}).get("entities"),
        preview.get("stats", {}).get("relationships"),
    )
    yield {"type": "done", "graph_preview": preview}

