"""
LayoutReader + LayoutLMv3 pipeline for reading order and structure inference.

Models
------
hantian/layoutreader
    LayoutLMv3 fine-tuned on permutation labels. Predicts the reading order
    rank for each OCR line from its bounding box position alone.
    Input:  words (list[str]) + boxes in 0..1000 integer [x0,y0,x1,y1] format.
    Output: per-token logits → argmax gives predicted reading-order rank.

microsoft/layoutlmv3-base
    General-purpose document understanding backbone.  Used here as an
    end-to-end sanity test to confirm the token+bbox pipeline is wired
    correctly.  Fine-tuned heads (title / paragraph / table) can be swapped
    in later; for now the backbone produces contextual embeddings.

Device
------
Defaults to CPU (M4 Mac).  Set LAYOUT_DEVICE=mps to use Apple MPS or
LAYOUT_DEVICE=cuda for GPU.

Usage
-----
from ingestion.layout_analysis import layout_analysis_available, analyze_paddle_output

if layout_analysis_available():
    result = analyze_paddle_output(tokens, use_layoutreader=True)
    # result["reading_order"]     — indices sorted in reading order
    # result["tokens_with_order"] — tokens annotated with reading_rank
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature flag
# ---------------------------------------------------------------------------


def layout_analysis_available() -> bool:
    """True when torch + transformers can be imported (lazy check)."""
    try:
        import torch        # noqa: F401
        import transformers  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_device(override: str | None = None) -> str:
    """Pick the best available device unless explicitly overridden."""
    if override:
        return override
    env = os.environ.get("LAYOUT_DEVICE", "").strip().lower()
    if env:
        return env
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Box helpers
# ---------------------------------------------------------------------------


def boxes_to_1000(boxes_01: list[dict[str, float]]) -> list[list[int]]:
    """
    Convert normalized 0..1 boxes to LayoutLM integer format [x0, y0, x1, y1]
    in the 0..1000 range.  Guarantees x0 <= x1 and y0 <= y1.
    """
    out: list[list[int]] = []
    for b in boxes_01:
        x0 = int(max(0, min(1000, round(float(b.get("left",  0)) * 1000))))
        y0 = int(max(0, min(1000, round(float(b.get("top",   0)) * 1000))))
        x1 = int(max(0, min(1000, round(float(b.get("right", 0)) * 1000))))
        y1 = int(max(0, min(1000, round(float(b.get("bottom",0)) * 1000))))
        out.append([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
    return out


# ---------------------------------------------------------------------------
# Module-level model cache (avoid reloading on each request)
# ---------------------------------------------------------------------------

_model_cache: dict[str, tuple[Any, Any]] = {}


def _load_token_cls(model_name: str, device: str) -> tuple[Any, Any]:
    """Load (tokenizer, LayoutLMv3ForTokenClassification) with caching."""
    key = f"cls:{model_name}:{device}"
    if key not in _model_cache:
        from transformers import AutoTokenizer, LayoutLMv3ForTokenClassification
        logger.info("Loading LayoutReader tokenizer + model: %s  device=%s", model_name, device)
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        mdl.eval()
        mdl.to(device)
        _model_cache[key] = (tok, mdl)
    return _model_cache[key]


def _load_backbone(model_name: str, device: str) -> tuple[Any, Any]:
    """Load (tokenizer, LayoutLMv3Model) with caching."""
    key = f"bb:{model_name}:{device}"
    if key not in _model_cache:
        from transformers import AutoTokenizer, LayoutLMv3Model
        logger.info("Loading LayoutLMv3 backbone: %s  device=%s", model_name, device)
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = LayoutLMv3Model.from_pretrained(model_name)
        mdl.eval()
        mdl.to(device)
        _model_cache[key] = (tok, mdl)
    return _model_cache[key]


# ---------------------------------------------------------------------------
# LayoutReader — reading order prediction
# ---------------------------------------------------------------------------


def run_layoutreader(
    texts: list[str],
    boxes_01: list[dict[str, float]],
    *,
    model_name: str | None = None,
    max_seq_length: int = 512,
    device: str | None = None,
) -> list[int]:
    """
    Predict reading order using hantian/layoutreader.

    Parameters
    ----------
    texts : list[str]
        One string per OCR line (or word).
    boxes_01 : list[dict]
        Normalized boxes ``{left, top, right, bottom}`` in 0..1 per text.
    model_name : str | None
        HuggingFace checkpoint.  Defaults to the env LAYOUTREADER_MODEL or
        ``hantian/layoutreader``.
    max_seq_length : int
        Tokenizer truncation limit (capped at 512 for LayoutLMv3).
    device : str | None
        Torch device.  Auto-detected when None.

    Returns
    -------
    list[int]
        Original indices sorted in reading order.
        E.g. ``[2, 0, 1]`` → index 2 reads first, then 0, then 1.
        Returns identity order on any failure.
    """
    import torch

    if not texts:
        return []

    _model = model_name or os.environ.get("LAYOUTREADER_MODEL", "hantian/layoutreader")
    _device = _resolve_device(device)
    max_seq = min(max_seq_length, 512)

    n_total = len(texts)
    # Reserve 2 positions for [CLS] + [SEP]
    n = min(n_total, max_seq - 2)

    texts_in   = [str(t).strip() or "." for t in texts[:n]]
    boxes_1000 = boxes_to_1000(boxes_01[:n])

    tokenizer, model = _load_token_cls(_model, _device)

    encoding = tokenizer(
        texts_in,
        boxes=boxes_1000,
        truncation=True,
        max_length=max_seq,
        return_tensors="pt",
        padding="max_length",
    )

    inputs = {k: v.to(_device) for k, v in encoding.items()
              if k != "overflow_to_sample_mapping"}

    with torch.no_grad():
        outputs = model(**inputs)

    # logits: [1, seq_len, num_labels] — predicted reading-order rank per token
    logits = outputs.logits[0].cpu()  # [seq_len, num_labels]

    # Aggregate to word level using the first subword token for each word
    word_rank: dict[int, int] = {}
    try:
        word_ids = encoding.word_ids(batch_index=0)
        for tok_idx, word_id in enumerate(word_ids):
            if word_id is None or word_id in word_rank:
                continue
            word_rank[word_id] = int(logits[tok_idx].argmax(-1).item())
    except Exception:
        # Fallback: scan attention mask for non-padding tokens
        attention = encoding.get("attention_mask")
        if attention is not None:
            seq_len = int(attention[0].sum().item())
            for i in range(min(n, seq_len)):
                if i not in word_rank:
                    word_rank[i] = int(logits[i].argmax(-1).item())

    # Sort word indices by their predicted reading-order rank
    pairs = [(word_rank.get(i, i * 10), i) for i in range(n)]
    pairs.sort(key=lambda p: (p[0], p[1]))  # tie-break: original position
    reading_order = [p[1] for p in pairs]

    # Append any truncated items (preserve original relative order)
    if n_total > n:
        reading_order.extend(range(n, n_total))

    return reading_order


# ---------------------------------------------------------------------------
# LayoutLMv3 backbone — sanity / debug forward pass
# ---------------------------------------------------------------------------


def run_layoutlmv3_sanity(
    texts: list[str],
    boxes_01: list[dict[str, float]],
    *,
    model_name: str | None = None,
    max_seq_length: int = 512,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Forward pass through microsoft/layoutlmv3-base (no fine-tuned head).

    Returns embedding statistics as a debug payload confirming the
    token+bbox pipeline is wired end-to-end.
    """
    import torch

    if not texts:
        return {"status": "no_input"}

    _model  = model_name or os.environ.get("LAYOUTLMV3_MODEL", "microsoft/layoutlmv3-base")
    _device = _resolve_device(device)
    max_seq = min(max_seq_length, 512)

    n = min(len(texts), max_seq - 2)
    texts_in   = [str(t).strip() or "." for t in texts[:n]]
    boxes_1000 = boxes_to_1000(boxes_01[:n])

    tokenizer, model = _load_backbone(_model, _device)

    encoding = tokenizer(
        texts_in,
        boxes=boxes_1000,
        truncation=True,
        max_length=max_seq,
        return_tensors="pt",
        padding="max_length",
    )
    inputs = {k: v.to(_device) for k, v in encoding.items()
              if k != "overflow_to_sample_mapping"}

    with torch.no_grad():
        outputs = model(**inputs)

    last_hidden = outputs.last_hidden_state[0].cpu()  # [seq_len, hidden_size]

    return {
        "status": "ok",
        "model": _model,
        "device": _device,
        "input_words": n,
        "seq_len": int(last_hidden.shape[0]),
        "hidden_size": int(last_hidden.shape[1]),
        "mean_embedding_norm": round(float(last_hidden.norm(dim=-1).mean().item()), 4),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def analyze_paddle_output(
    tokens: list[dict[str, Any]],
    *,
    use_layoutreader: bool = True,
    use_layoutlmv3_debug: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    """
    Apply LayoutReader (and optionally LayoutLMv3) to a PaddleOCR token list.

    Parameters
    ----------
    tokens : list[dict]
        Each dict: ``{text, bbox_01: {left, top, right, bottom}, page, ...}``
        (one entry per OCR line on a *single* page).
    use_layoutreader : bool
        Run hantian/layoutreader to predict reading order.
    use_layoutlmv3_debug : bool
        Run microsoft/layoutlmv3-base backbone pass for embedding sanity check.
    device : str | None
        Torch device override.

    Returns
    -------
    dict with keys:
        reading_order       : list[int]   — indices in reading order
        tokens_with_order   : list[dict]  — tokens annotated with reading_rank
        layout_debug        : dict        — LayoutLMv3 stats (only if requested)
        error               : str | None  — non-fatal error message
    """
    n = len(tokens)
    result: dict[str, Any] = {
        "reading_order": list(range(n)),
        "tokens_with_order": list(tokens),
        "error": None,
    }

    if n == 0:
        return result

    texts    = [t.get("text", "") or "." for t in tokens]
    boxes_01 = [
        t.get("bbox_01") or {"left": 0.0, "top": 0.0, "right": 0.0, "bottom": 0.0}
        for t in tokens
    ]

    if use_layoutreader and layout_analysis_available():
        try:
            order = run_layoutreader(texts, boxes_01, device=device)
            result["reading_order"] = order
        except Exception as exc:
            logger.warning("LayoutReader failed (falling back to position order)", exc_info=True)
            result["error"] = f"LayoutReader: {exc}"

    # Annotate each token with its reading_rank (0-based position in reading order)
    order = result["reading_order"]
    rank_map = {orig_idx: rank for rank, orig_idx in enumerate(order)}
    result["tokens_with_order"] = [
        {**t, "reading_rank": rank_map.get(i, i)}
        for i, t in enumerate(tokens)
    ]

    if use_layoutlmv3_debug and layout_analysis_available():
        try:
            result["layout_debug"] = run_layoutlmv3_sanity(texts, boxes_01, device=device)
        except Exception as exc:
            logger.warning("LayoutLMv3 sanity pass failed", exc_info=True)
            result["layout_debug"] = {"error": str(exc)}

    return result
