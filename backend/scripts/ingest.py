"""
CLI script — ingest TBI PDFs into the Neo4j knowledge graph.

Usage (from backend/):
    # Ingest all PDFs
    python scripts/ingest.py

    # Ingest a single PDF for testing
    python scripts/ingest.py --file blood-biomarkers-ich-outcome-moderate-severe-tbi-anderson.pdf
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import langsmith_env  # noqa: F401 — load .env + LangSmith before pipeline

import neo4j

from ingestion.pipeline import build_pipeline, ingest_pdfs
from retrieval.retriever import setup_vector_index

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "publications"


async def ingest_single(driver: neo4j.Driver, filename: str) -> None:
    path = DATA_DIR / filename
    if not path.exists():
        print(f"  File not found: {path}")
        print(f"  Available PDFs:")
        for p in sorted(DATA_DIR.glob("*.pdf")):
            print(f"    - {p.name}")
        return

    pipeline = build_pipeline(driver)
    print(f"  Ingesting: {filename}")
    try:
        result = await pipeline.run_async(
            file_path=str(path),
            document_metadata={"source": filename, "file_path": str(path)},
        )
        print(f"  Done: {filename}")
        print(f"  Result: {result}")
    except Exception as exc:
        print(f"  Error: {exc}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest TBI PDFs into Neo4j")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Ingest a single PDF by filename (e.g. blood-biomarkers-ich-outcome-moderate-severe-tbi-anderson.pdf)",
    )
    args = parser.parse_args()

    print("GroundedMD — TBI Knowledge Graph Ingestion")
    print("=" * 50)

    driver = neo4j.GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )

    print("Setting up vector index...")
    setup_vector_index(driver)
    print("Vector index ready.\n")

    if args.file:
        print(f"Mode: single file — {args.file}\n")
        await ingest_single(driver, args.file)
    else:
        print(f"Mode: all PDFs in {DATA_DIR}\n")
        result = await ingest_pdfs(driver=driver, pdf_dir=DATA_DIR)

        print("\n" + "=" * 50)
        print(f"Status         : {result['status']}")
        print(f"Files processed: {result.get('files_processed', 0)}")
        for r in result.get("results", []):
            icon = "✓" if r["status"] == "success" else "✗"
            print(f"  {icon} {r['file']}")
            if r["status"] == "error":
                print(f"      Error: {r['error']}")

    driver.close()
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
