"""
Ingest all publications from backend/data/publications/ through the full pipeline.

Each PDF is processed sequentially:
  1. PaddleOCR (PP-StructureV3) — layout + text extraction
  2. Chunking + embedding (nomic-embed-text)
  3. Entity extraction (Gemma 4)
  4. Schema pruning + Neo4j write
  5. Entity resolution

Usage (from the backend/ directory with venv active):
    python scripts/ingest_all.py
    python scripts/ingest_all.py --publications-dir data/publications
    python scripts/ingest_all.py --skip-ocr   # skip PaddleOCR, use plain PDF loader
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Make backend/ importable when running from project root or backend/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load .env from backend/ or project root before any other imports
from dotenv import load_dotenv
_backend_dir = Path(__file__).resolve().parents[1]
load_dotenv(_backend_dir / ".env")
load_dotenv(_backend_dir.parent / ".env")

import neo4j
from ingestion.ingest_stream import ingest_file_stream
from ingestion.docling_ocr import docling_ocr_enabled, docling_ocr_pdf


async def warmup_ollama() -> None:
    """Send a tiny request to Ollama and wait until the model is fully loaded."""
    import os, httpx
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = os.environ.get("OLLAMA_LLM_MODEL", "gemma4:e4b")
    print(f"  Warming up Ollama ({model}) — waiting for model to load into GPU...")
    for attempt in range(30):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(f"{base}/api/generate",
                    json={"model": model, "prompt": "Hi", "stream": False, "options": {"num_predict": 1}})
                if r.status_code == 200:
                    print("  Ollama ready.")
                    return
        except Exception:
            pass
        print(f"  Still loading... ({attempt + 1}/30)")
        await asyncio.sleep(10)
    print("  Warmup timed out — proceeding anyway.")


def get_driver() -> neo4j.Driver:
    import os
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "password")
    return neo4j.GraphDatabase.driver(uri, auth=(user, password))


async def ingest_pdf(driver: neo4j.Driver, pdf_path: Path, skip_ocr: bool) -> bool:
    print(f"\n{'='*60}")
    print(f"  Ingesting: {pdf_path.name}")
    print(f"{'='*60}")

    pipeline_id = "docling_ocr" if (not skip_ocr and docling_ocr_enabled()) else "no_ocr"
    meta = {"source": pdf_path.name}

    if not skip_ocr and docling_ocr_enabled():
        print("  [1/2] Running Docling OCR on EC2 (timeout: 600s)...")
        import tempfile, os
        try:
            payload = await asyncio.to_thread(docling_ocr_pdf, pdf_path)
            markdown_text = payload.get("markdown", "")
            fd, tmp_md = tempfile.mkstemp(suffix=".md")
            os.close(fd)
            with open(tmp_md, "w", encoding="utf-8") as f:
                f.write(markdown_text)
            print(f"  [1/2] Docling OCR complete — {len(markdown_text):,} chars extracted")
            ingest_path = tmp_md
            cleanup = tmp_md
        except Exception as e:
            print(f"  [1/2] Docling OCR failed ({e}) — falling back to plain PDF loader")
            ingest_path = str(pdf_path)
            cleanup = None
            pipeline_id = "no_ocr"
    else:
        print("  [1/2] Skipping Docling OCR — using plain PDF loader")
        ingest_path = str(pdf_path)
        cleanup = None

    print("  [2/2] Running ingest pipeline (embed → entities → Neo4j write)...")
    success = True
    try:
        async for event in ingest_file_stream(driver, ingest_path, meta, pipeline_id=pipeline_id):
            t = event.get("type", "")
            if t == "TASK_STARTED":
                hint = event.get("hint", event.get("task_name", ""))
                print(f"        → {hint}")
            elif t == "error":
                print(f"        ERROR: {event.get('message')}")
                success = False
            elif t == "done":
                nodes = event.get("node_count", "?")
                rels = event.get("rel_count", "?")
                print(f"        Done — {nodes} nodes, {rels} relationships written")
    except Exception as e:
        print(f"  Pipeline error: {e}")
        success = False
    finally:
        if cleanup:
            try:
                import os
                os.unlink(cleanup)
            except OSError:
                pass

    status = "OK" if success else "FAILED"
    print(f"  Result: {status}")
    return success


async def main(publications_dir: Path, skip_ocr: bool) -> None:
    pdfs = sorted(publications_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in {publications_dir}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) to ingest:")
    for p in pdfs:
        print(f"  - {p.name}")

    ocr_available = docling_ocr_enabled()
    print(f"\nDocling OCR (EC2) available: {ocr_available}")
    print(f"Skip OCR flag: {skip_ocr}")

    await warmup_ollama()

    driver = get_driver()
    results = {}

    try:
        for pdf in pdfs:
            ok = await ingest_pdf(driver, pdf, skip_ocr=skip_ocr or not ocr_available)
            results[pdf.name] = ok
    finally:
        driver.close()

    print(f"\n{'='*60}")
    print("  INGEST SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} file(s) failed. Check logs above.")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} files ingested successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest all TBI publications into Neo4j")
    parser.add_argument(
        "--publications-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "publications",
        help="Directory containing PDF files (default: backend/data/publications/)",
    )
    parser.add_argument(
        "--skip-ocr",
        action="store_true",
        help="Skip PaddleOCR and use plain PDF text extraction",
    )
    args = parser.parse_args()

    if not args.publications_dir.exists():
        print(f"Directory not found: {args.publications_dir}")
        sys.exit(1)

    asyncio.run(main(args.publications_dir, args.skip_ocr))
