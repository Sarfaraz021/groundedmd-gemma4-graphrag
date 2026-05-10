"""
List and delete Document nodes and dependent lexical / extracted graph data in Neo4j.
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from typing import Any, TypeVar

import neo4j
from neo4j.exceptions import ServiceUnavailable, SessionExpired, TransientError

T = TypeVar("T")

# Load balancers may close idle sockets; the pool can still hand out a dead connection once.
_RETRYABLE = (SessionExpired, ServiceUnavailable, TransientError)


def _with_session_retry(driver: neo4j.Driver, work: Callable[[neo4j.Session], T]) -> T:
    """Run ``work(session)`` with a fresh session; retry once on stale pool disconnects."""
    try:
        with _session(driver) as session:
            return work(session)
    except _RETRYABLE:
        with _session(driver) as session:
            return work(session)


def _node_props(node: Any) -> dict[str, Any]:
    try:
        return dict(node)
    except Exception:
        return {}


def _session(driver: neo4j.Driver) -> neo4j.Session:
    db = os.environ.get("NEO4J_DATABASE", "").strip()
    if db:
        return driver.session(database=db)
    return driver.session()


def list_documents(
    driver: neo4j.Driver,
    *,
    owner_user_id: str | None,
    pipeline_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return Document nodes, optionally filtered by pipeline_id."""
    q = """
    MATCH (d:Document)
    WHERE ($owner_user_id IS NULL OR d.owner_user_id = $owner_user_id)
      AND ($pipeline_id IS NULL OR d.pipeline_id = $pipeline_id)
    OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
    WITH d, count(c) AS chunk_count
    RETURN
      elementId(d) AS id,
      coalesce(d.source, d.path, d.file_path, '') AS name,
      coalesce(d.path, '') AS path,
      chunk_count AS chunk_count
    ORDER BY toLower(coalesce(d.source, d.path, d.file_path, ''))
    """

    def work(session: neo4j.Session) -> list[dict[str, Any]]:
        rows = session.run(q, owner_user_id=owner_user_id, pipeline_id=pipeline_id)
        out: list[dict[str, Any]] = []
        for r in rows:
            name = (r["name"] or "").strip() or (r["path"] or "").strip() or (r["id"] or "")
            out.append(
                {
                    "id": r["id"],
                    "name": name,
                    "path": r["path"] or "",
                    "chunk_count": int(r["chunk_count"] or 0),
                }
            )
        return out

    return _with_session_retry(driver, work)


def _delete_document_tx(
    tx: neo4j.ManagedTransaction,
    *,
    doc_eid: str,
    owner_user_id: str | None,
) -> bool:
    r = tx.run(
        """
        MATCH (d:Document) WHERE elementId(d) = $doc_eid
          AND ($owner_user_id IS NULL OR d.owner_user_id = $owner_user_id)
        OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
        OPTIONAL MATCH (e)-[:FROM_CHUNK]->(c)
        RETURN collect(DISTINCT elementId(c)) AS chunk_ids,
               collect(DISTINCT elementId(e)) AS entity_ids
        """,
        doc_eid=doc_eid,
        owner_user_id=owner_user_id,
    )
    rec = r.single()
    if not rec:
        return False

    chunk_ids = [x for x in rec["chunk_ids"] if x]
    entity_ids = [x for x in rec["entity_ids"] if x]

    if chunk_ids:
        tx.run(
            """
            UNWIND $chunk_ids AS ceid
            MATCH (c:Chunk) WHERE elementId(c) = ceid
            DETACH DELETE c
            """,
            chunk_ids=chunk_ids,
        )

    if entity_ids:
        tx.run(
            """
            UNWIND $entity_ids AS eid
            MATCH (n) WHERE elementId(n) = eid AND NOT (n)-[:FROM_CHUNK]->(:Chunk)
            DETACH DELETE n
            """,
            entity_ids=entity_ids,
        )

    doc_del = tx.run(
        """
        MATCH (d:Document) WHERE elementId(d) = $doc_eid
          AND ($owner_user_id IS NULL OR d.owner_user_id = $owner_user_id)
        DETACH DELETE d
        RETURN count(d) AS deleted
        """,
        doc_eid=doc_eid,
        owner_user_id=owner_user_id,
    )
    dr = doc_del.single()
    return bool(dr and int(dr["deleted"] or 0) > 0)


_ELEMENT_ID_RE = re.compile(r'^[\w\-:]+$')
_ELEMENT_ID_MAX_LEN = 256


def _validate_element_id(raw: str) -> str:
    """Return the stripped element ID or raise ValueError if it looks malformed."""
    eid = raw.strip()
    if not eid:
        raise ValueError("document_id must not be empty.")
    if len(eid) > _ELEMENT_ID_MAX_LEN:
        raise ValueError("document_id exceeds maximum allowed length.")
    if not _ELEMENT_ID_RE.match(eid):
        raise ValueError("document_id contains invalid characters.")
    return eid


def get_document_graph(
    driver: neo4j.Driver,
    *,
    document_id: str,
    owner_user_id: str | None,
) -> dict[str, Any] | None:
    """
    Return nodes and links for the Neo4j subgraph tied to one ``Document`` (chunks, entities, relationships).
    Used by the Knowledge Base graph viewer (distinct from ``Pipeline.draw``, which visualizes the ingest DAG).
    """
    try:
        doc_eid = _validate_element_id(document_id)
    except ValueError:
        return None

    def _node_eid(node: Any) -> str:
        eid = getattr(node, "element_id", None)
        if eid is not None:
            return str(eid)
        # Neo4j Python driver 4.x fallback
        return str(getattr(node, "id", ""))

    def work(session: neo4j.Session) -> dict[str, Any] | None:
        rec = session.run(
            """
            MATCH (d:Document) WHERE elementId(d) = $doc_id
              AND ($owner_user_id IS NULL OR d.owner_user_id = $owner_user_id)
            OPTIONAL MATCH (c:Chunk)-[:FROM_DOCUMENT]->(d)
            WITH d, collect(DISTINCT c) AS chunks
            OPTIONAL MATCH (e)-[:FROM_CHUNK]->(cc)
            WHERE cc IN chunks
            WITH d, chunks, collect(DISTINCT e) AS all_entities
            RETURN d, chunks, all_entities[..500] AS entities
            """,
            doc_id=doc_eid,
            owner_user_id=owner_user_id,
        ).single()
        if not rec or rec["d"] is None:
            return None

        d_node = rec["d"]
        doc_id = _node_eid(d_node)
        dp = _node_props(d_node)
        doc_name = str(dp.get("source") or dp.get("path") or dp.get("file_path") or "").strip() or "Document"

        chunks = [c for c in rec["chunks"] if c is not None]
        entities = [e for e in rec["entities"] if e is not None]

        nodes: list[dict[str, Any]] = [
            {"id": doc_id, "name": doc_name[:200], "kind": "Document", "group": 0},
        ]

        for c in chunks:
            cid = _node_eid(c)
            cp = _node_props(c)
            idx = cp.get("index")
            raw = (cp.get("text") or "").replace("\n", " ").strip()
            snip = raw[:56] + ("…" if len(raw) > 56 else "")
            if idx is not None:
                title = f"Chunk {idx}"
            else:
                title = "Chunk"
            if snip:
                title = f"{title}: {snip}"
            nodes.append({"id": cid, "name": title[:160], "kind": "Chunk", "group": 1})

        for e in entities:
            eid = _node_eid(e)
            labels = list(e.labels) if hasattr(e, "labels") else []
            lab = labels[0] if labels else "Entity"
            ep = _node_props(e)
            nm = str(ep.get("name") or lab)[:120]
            nodes.append({"id": eid, "name": nm, "kind": lab, "group": 2})

        id_set = {n["id"] for n in nodes}
        links: list[dict[str, Any]] = []

        if len(id_set) > 1:
            rel_rows = session.run(
                """
                MATCH (a)-[r]-(b)
                WHERE elementId(a) IN $ids AND elementId(b) IN $ids
                  AND elementId(a) < elementId(b)
                  AND NOT type(r) IN ['NEXT_CHUNK']
                RETURN elementId(a) AS source, elementId(b) AS target, type(r) AS rel_type
                LIMIT 500
                """,
                ids=list(id_set),
            )
            for rr in rel_rows:
                links.append(
                    {
                        "source": rr["source"],
                        "target": rr["target"],
                        "rel_type": rr["rel_type"],
                    }
                )

        return {
            "document_id": doc_id,
            "document_name": doc_name,
            "nodes": nodes,
            "links": links,
        }

    return _with_session_retry(driver, work)


def delete_document(
    driver: neo4j.Driver,
    *,
    document_id: str,
    owner_user_id: str | None,
) -> bool:
    """
    Remove a  Document, its Chunks, extracted nodes that only linked to those chunks, and the Document.
    ``document_id`` is the Neo4j element id string from ``list_documents``.
    """
    try:
        doc_eid = _validate_element_id(document_id)
    except ValueError:
        return False

    def write_tx(tx: neo4j.ManagedTransaction) -> bool:
        return _delete_document_tx(tx, doc_eid=doc_eid, owner_user_id=owner_user_id)

    def session_work(session: neo4j.Session) -> bool:
        return bool(session.execute_write(write_tx))

    return bool(_with_session_retry(driver, session_work))
