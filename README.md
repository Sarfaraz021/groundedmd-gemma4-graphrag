# GroundedMD

Evidence-grounded clinical Q&A for traumatic brain injury, powered by **Gemma 4 via Ollama**, **Neo4j GraphRAG**, local embeddings, and visible source citations.

GroundedMD is built for the Google DeepMind / Kaggle Gemma 4 Good Hackathon. The default demo path is self-hosted: React, FastAPI, Neo4j, Ollama, Gemma 4, and `nomic-embed-text`.

## Why It Matters

Clinical teams in rural or low-connectivity settings need decision-support tools that can run close to the point of care and show exactly where each answer came from. GroundedMD answers questions from a curated TBI evidence corpus and surfaces the source chunks used for each response.

This is not a diagnostic device and does not replace clinical judgment. It is a research demo for grounded, inspectable clinical knowledge retrieval.

## What It Does

- Answers TBI questions using a Neo4j knowledge graph built from NINDS TBI Common Data Element publications.
- Runs generation locally with Gemma 4 through Ollama.
- Runs embeddings locally with `nomic-embed-text` through Ollama.
- Retrieves evidence with vector search plus graph context.
- Shows source chunks, document names, and graph previews.
- Uses no hosted LLM or hosted auth provider in the default path.

## Architecture

```text
React UI
  -> FastAPI backend
  -> Neo4j graph + vector index
  -> Ollama
       -> Gemma 4 LLM
       -> nomic-embed-text embeddings
```

## Quick Start

Prerequisites:

- Docker Desktop or Docker Engine
- Enough disk space for Ollama models

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f ollama-setup
```

Open:

```text
http://localhost
```

Useful health checks:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/health/models
```

After the stack is healthy, ingest the bundled TBI publications from the UI or API, then ask a question such as:

```text
Which blood-based biomarkers are discussed for acute TBI, and what roles do they play?
```

## Local Development

Backend:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Ollama models:

```bash
ollama pull gemma4:e4b
ollama pull nomic-embed-text
```

## Configuration

The active runtime configuration is:

- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- `OLLAMA_BASE_URL`
- `OLLAMA_LLM_MODEL`
- `OLLAMA_EMBED_MODEL`
- `LOCAL_RERANK_ENABLED`

LangSmith tracing is optional for developer debugging and is disabled by default:

```env
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=GroundedMD-Hackathon
```

Do not commit real `.env` files or credentials.

## Evidence Corpus

The demo corpus is composed of public NINDS TBI Common Data Element publications, including:

- `blood-based-biomarkers-tbi-wg-508c.pdf`
- `clinical-symptoms-days1-14-wg-508c.pdf`
- `tbi-imaging-wg-508c.pdf`
- `TBI-classification-nomenclature-workshop-agenda-508c.pdf`
- `knowlege-to-practice-wg-508c.pdf`

## License

Project code is released under Apache-2.0.

The included source documents and model weights remain subject to their original licenses and terms. This repository's Apache-2.0 license applies only to the project code.