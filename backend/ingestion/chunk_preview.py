"""
Non-OCR PDF chunk preview.

Goal: show users extracted text chunks before they decide whether local OCR is needed.

Uses the same RecursiveCharacterTextSplitter configuration as the ingest
pipeline so the preview faithfully represents what Neo4j Chunks will look like.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ingestion.pipeline import CHUNK_OVERLAP, CHUNK_SIZE, SEPARATORS


def chunk_preview_pdf(path: Path, *, max_pages: int | None = None, max_chunks: int = 80) -> dict[str, Any]:
    """
    Extract text from PDF pages (no OCR) and return RecursiveCharacter chunks for UI preview.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    doc = fitz.open(path)
    pages = doc.page_count
    if max_pages is not None:
        pages = min(pages, max_pages)

    parts: list[str] = []
    for i in range(pages):
        page = doc.load_page(i)
        txt = (page.get_text("text") or "").strip()
        if txt:
            parts.append(txt)

    full_text = "\n\n".join(parts).strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )
    chunks = splitter.split_text(full_text) if full_text else []

    # UI should remain responsive; return only first N chunks.
    limited = chunks[:max_chunks]
    return {
        "chunk_config": {"size": CHUNK_SIZE, "overlap": CHUNK_OVERLAP, "splitter": "recursive"},
        "extracted_pages": pages,
        "extracted_chars": len(full_text),
        "chunk_count_total": len(chunks),
        "chunks": [
            {
                "index": idx,
                "text": c,
            }
            for idx, c in enumerate(limited, start=1)
        ],
    }

