"""
API authentication stub.

This project runs **without** per-user auth: every request uses a shared Neo4j view
(``owner_user_id`` is always ``None`` in retrieval Cypher).

To add JWT-based isolation later, replace this module with verification of your
issuer’s tokens and return ``sub`` from ``get_owner_user_id``.
"""

from __future__ import annotations

from fastapi import Header


def get_owner_user_id(authorization: str | None = Header(None)) -> str | None:
    """No authenticated user — shared graph for all clients."""
    _ = authorization
    return None
