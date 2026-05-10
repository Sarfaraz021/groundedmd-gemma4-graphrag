"""
PP-StructureV3 document parsing for PDFs.

Uses PaddleOCR 3.0's PPStructureV3 pipeline which includes:
- PP-OCRv5 text detection + recognition
- Layout analysis (PP-DocLayout-plus): multi-column, headings, paragraphs
- Table recognition (PP-TableMagic): full HTML/markdown table structure
- Reading order recovery (enhanced X-Y cut)

Per the PaddleOCR 3.0 technical report, preprocessing models
(doc orientation, unwarping, textline orientation) are disabled for PDFs
since digital PDFs are always upright and never physically warped.

The grounding dict is compatible with ParsePreviewModal (normalized 0..1 boxes).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# Module-level pipeline cache — avoids reloading models per request.
_ocr_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def paddle_ocr_preview_enabled() -> bool:
    """True when PPStructureV3 (paddleocr + paddlex[ocr]) is importable."""
    try:
        from paddleocr import PPStructureV3  # noqa: F401
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _json_safe(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    return str(obj)


@dataclass(frozen=True)
class OcrBox:
    left: float
    top: float
    right: float
    bottom: float

    @staticmethod
    def from_points(points: list[list[float]] | list[tuple[float, float]]) -> "OcrBox":
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        return OcrBox(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))


def _normalize_box(box: OcrBox, *, w: int, h: int) -> dict[str, float]:
    if w <= 0 or h <= 0:
        return {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
    return {
        "left":   max(0.0, min(1.0, box.left   / w)),
        "top":    max(0.0, min(1.0, box.top    / h)),
        "right":  max(0.0, min(1.0, box.right  / w)),
        "bottom": max(0.0, min(1.0, box.bottom / h)),
    }


def _render_pdf_page_to_image(page: fitz.Page, *, scale: float) -> tuple[Any, int, int]:
    """Render a PDF page to a numpy RGB array (H, W, 3)."""
    import numpy as np

    pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.n >= 3:
        img = img.reshape(pix.height, pix.width, pix.n)[:, :, :3]
    else:
        img = np.stack([img.reshape(pix.height, pix.width)] * 3, axis=-1)
    return img, pix.width, pix.height


# ---------------------------------------------------------------------------
# Main preview function
# ---------------------------------------------------------------------------


def paddle_ocr_preview_pdf(
    path: Path,
    *,
    max_pages: int | None = None,
    scale: float | None = None,
    max_boxes_per_page: int = 600,
    use_layout_reader: bool = False,  # kept for API compat; PPStructureV3 has its own reading order
) -> dict[str, Any]:
    """
    Run PPStructureV3 on a PDF and return a preview payload compatible with
    the ``ParsePreviewModal`` UI component.

    PPStructureV3 (PaddleOCR 3.0) replaces plain PaddleOCR with a full
    document parsing pipeline that adds layout analysis, table recognition,
    and proper reading order recovery on top of PP-OCRv5.

    Parameters
    ----------
    path : Path
        Local PDF file path.
    max_pages : int | None
        Cap on pages to process (None = all pages).
    scale : float | None
        Render DPI scale for page-to-image conversion.  Defaults to
        ``PADDLE_OCR_RENDER_SCALE`` env var or 1.2.
    max_boxes_per_page : int
        Hard cap on layout-block entries per page.
    use_layout_reader : bool
        Legacy parameter — PPStructureV3 has built-in reading order recovery
        so this is a no-op. Kept for backward API compatibility.

    Returns
    -------
    dict with keys:
        markdown       : str   — structured markdown (tables, headings preserved)
        grounding      : dict  — {gid: {page, type, box, score, text, reading_rank, label}}
        tokens         : list  — [{id, page, text, bbox_01, score, reading_rank}]
        reading_order  : dict  — per-page reading order indices
        layout_available : bool
    """
    if not path.exists():
        raise FileNotFoundError(str(path))

    try:
        from paddleocr import PPStructureV3
    except Exception as exc:
        raise RuntimeError(
            "PPStructureV3 is not available. Install with:\n"
            "  pip install paddleocr paddlepaddle\n"
            "  pip install 'paddlex[ocr]'"
        ) from exc

    import numpy as np  # noqa: F401

    # Render scale — 1.2 is sufficient for digital PDFs; override via env var.
    render_scale = scale if scale is not None else float(
        os.environ.get("PADDLE_OCR_RENDER_SCALE", "1.2")
    )

    # Cache the pipeline — loading all models takes ~30 s on first call.
    cache_key = "ppstructurev3"
    if cache_key not in _ocr_cache:
        logger.info("Initialising PPStructureV3 — first call only")
        _ocr_cache[cache_key] = PPStructureV3(
            # Disable preprocessing models — not needed for digital PDFs
            # (per PaddleOCR 3.0 technical report, Appendix B)
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            # Enable document understanding features relevant for research PDFs
            use_table_recognition=True,    # tables are common in medical/research docs
            use_formula_recognition=False, # skip: adds significant overhead
            use_chart_recognition=False,   # skip: not needed for text extraction
            use_seal_recognition=False,    # skip: no seals in TBI research papers
        )
    pipeline = _ocr_cache[cache_key]

    doc = fitz.open(path)
    n_pages = doc.page_count
    if max_pages is not None:
        n_pages = min(n_pages, max_pages)

    markdown_parts: list[str] = []
    grounding: dict[str, dict[str, Any]] = {}
    all_tokens: list[dict[str, Any]] = []
    reading_order_by_page: dict[str, list[int]] = {}

    for page_index in range(n_pages):
        page = doc.load_page(page_index)
        img, w, h = _render_pdf_page_to_image(page, scale=render_scale)

        # PPStructureV3.predict() accepts a numpy image and returns a list with
        # one LayoutParsingResultV2 per input image.
        page_results = list(pipeline.predict(input=img))
        if not page_results:
            markdown_parts.append(f"## Page {page_index + 1}\n")
            reading_order_by_page[str(page_index)] = []
            continue

        page_result = page_results[0]

        # --- Structured markdown (tables preserved, headings detected) -------
        try:
            md_data = page_result.markdown
            page_markdown = (
                md_data.get("markdown_texts", "")
                if isinstance(md_data, dict)
                else str(md_data)
            )
        except Exception as exc:
            logger.warning("Page %d: failed to get markdown — %s", page_index, exc)
            page_markdown = ""

        # --- Layout blocks → grounding overlay -------------------------------
        # PPStructureV3 returns layout blocks (paragraphs, tables, headings…)
        # each with a bbox in image-pixel coordinates and content text.
        try:
            parsing_res_list = page_result.get("parsing_res_list", []) or []
        except Exception:
            parsing_res_list = []

        parsing_res_list = parsing_res_list[:max_boxes_per_page]
        # Will be rebuilt after processing blocks using order_index values.
        page_reading_order: list[int] = []

        for block_idx, block in enumerate(parsing_res_list):
            try:
                # bbox = [x_min, y_min, x_max, y_max] in rendered image pixels
                bbox = list(block.bbox)
            except Exception:
                continue

            if len(bbox) < 4:
                continue

            text  = getattr(block, "content", "") or ""
            label = getattr(block, "label",   "text") or "text"
            # order_index is 1-based reading order from PPStructureV3's X-Y cut;
            # fall back to insertion index if not set.
            raw_order = getattr(block, "order_index", None)
            reading_rank = (raw_order - 1) if raw_order is not None else block_idx

            nb = {
                "left":   max(0.0, min(1.0, bbox[0] / w)),
                "top":    max(0.0, min(1.0, bbox[1] / h)),
                "right":  max(0.0, min(1.0, bbox[2] / w)),
                "bottom": max(0.0, min(1.0, bbox[3] / h)),
            }

            gid = f"p{page_index}-b{block_idx}"
            grounding[gid] = {
                "page":         page_index,
                "type":         "chunkText",
                "box":          nb,
                "score":        1.0,
                "text":         text,
                "reading_rank": reading_rank,
                "label":        label,
            }
            all_tokens.append({
                "id":           gid,
                "page":         page_index,
                "text":         text,
                "bbox_01":      nb,
                "score":        1.0,
                "reading_rank": reading_rank,
            })

        # Build reading_order: list of block indices sorted by reading_rank.
        token_ranks = [
            (t["reading_rank"], i) for i, t in enumerate(all_tokens)
            if t["page"] == page_index
        ]
        page_reading_order = [i for _, i in sorted(token_ranks)]
        reading_order_by_page[str(page_index)] = page_reading_order
        markdown_parts.append(
            f"## Page {page_index + 1}\n"
            + (page_markdown.strip() if page_markdown else "")
        )

        logger.debug(
            "Page %d: %d layout blocks, %d markdown chars",
            page_index, len(parsing_res_list), len(page_markdown),
        )

    markdown = "\n\n".join(markdown_parts).strip()

    return {
        "markdown":          markdown,
        "grounding":         _json_safe(grounding),
        "tokens":            _json_safe(all_tokens),
        "reading_order":     _json_safe(reading_order_by_page),
        "layout_available":  True,
    }
