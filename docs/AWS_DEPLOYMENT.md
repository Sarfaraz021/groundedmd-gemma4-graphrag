# GroundedMD Live Demo Deployment

Use this checklist when exposing a temporary live demo for judges. The default architecture remains self-hosted: React, FastAPI, Neo4j, Ollama, Gemma 4, and local embeddings.

## Recommended Shape

```text
Browser
  -> frontend nginx
  -> FastAPI backend
  -> Neo4j endpoint
  -> Ollama on the same VM or private network
```

## Environment

Start from `.env.example` and set only the values needed by your deployment:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=replace-me

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gemma4:e4b
OLLAMA_EMBED_MODEL=nomic-embed-text
LOCAL_RERANK_ENABLED=false

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=GroundedMD-Hackathon
```

Enable LangSmith only for your own debugging sessions. Keep it disabled for the judged path unless you intentionally want traces.

## Docker Compose

```bash
cp .env.example .env
docker compose up -d --build
docker compose logs -f ollama-setup
```

Then verify:

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/health/models
```

## Security Notes

- Do not commit `.env` files.
- Do not expose Neo4j or Ollama ports publicly unless a firewall restricts access.
- For a public demo, expose only the frontend and route API traffic through the frontend or a reverse proxy.
- Rotate any credentials that were ever pasted into chat, logs, screenshots, or commits.

## Demo Readiness

Before recording or submitting:

```bash
cd frontend
npm run lint
npm run build
npm audit
```

```bash
cd backend
python -m compileall api ingestion retrieval scripts
```

Confirm the UI can:

- Show model/health status.
- Ingest the bundled TBI publications.
- Answer a TBI question with citations.
- Open source chunks and the document graph preview.
