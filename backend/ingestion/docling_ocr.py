"""
Docling OCR client — calls the Docling microservice running on EC2.

The service runs on EC2 at DOCLING_OCR_URL (default: http://44.248.251.171:8001).
Falls back gracefully if the service is unreachable.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DOCLING_OCR_URL: str = os.getenv("DOCLING_OCR_URL", "http://44.248.251.171:8001")


def docling_ocr_enabled() -> bool:
    """True when the Docling OCR service on EC2 is reachable."""
    try:
        import requests
        r = requests.get(f"{DOCLING_OCR_URL}/health", timeout=5)
        return r.ok
    except Exception:
        return False


def docling_ocr_pdf(path: Path, *, timeout: int = 600) -> dict[str, Any]:
    """
    Send a PDF to the EC2 Docling OCR service and return a payload
    compatible with the ingest pipeline.

    Returns dict with keys:
        markdown         : str   — full document as Markdown
        grounding        : dict  — empty (Docling doesn't expose bounding boxes via this API)
        tokens           : list  — empty
        reading_order    : dict  — empty
        layout_available : bool  — False (no UI overlay data)
    """
    import requests

    if not path.exists():
        raise FileNotFoundError(str(path))

    logger.info("docling_ocr: sending %s to %s", path.name, DOCLING_OCR_URL)

    with open(path, "rb") as f:
        response = requests.post(
            f"{DOCLING_OCR_URL}/ocr",
            files={"file": (path.name, f, "application/pdf")},
            timeout=timeout,
        )

    response.raise_for_status()
    data = response.json()
    markdown = data.get("markdown", "")

    logger.info(
        "docling_ocr: received %d chars for %s",
        len(markdown),
        path.name,
    )

    return {
        "markdown": markdown,
        "grounding": {},
        "tokens": [],
        "reading_order": {},
        "layout_available": False,
    }
