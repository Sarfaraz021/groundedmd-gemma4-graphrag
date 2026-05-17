# GroundedMD

Evidence-grounded clinical Q&A for traumatic brain injury, powered by **Gemma 4 via Ollama**, **Neo4j GraphRAG**, local embeddings, and visible source citations.

GroundedMD is built for the Google DeepMind / Kaggle Gemma 4 Good Hackathon. The default demo path is self-hosted: React, FastAPI, Neo4j, Ollama, Gemma 4, and `nomic-embed-text`.

## Why It Matters

Clinical teams in rural or low-connectivity settings need decision-support tools that can run close to the point of care and show exactly where each answer came from. GroundedMD answers questions from a curated TBI evidence corpus and surfaces the source chunks used for each response.

This is not a diagnostic device and does not replace clinical judgment. It is a research demo for grounded, inspectable clinical knowledge retrieval.

## What It Does

- Answers TBI questions using a Neo4j knowledge graph built from peer-reviewed TBI research publications covering AI diagnostics, biomarkers, outcome prediction, and neurorehabilitation.
- Runs generation locally with Gemma 4 through Ollama.
- Runs embeddings locally with `nomic-embed-text` through Ollama.
- Retrieves evidence with vector search plus graph context.
- Shows source chunks, document names, and graph previews.
- Uses no hosted LLM or hosted auth provider in the default path.

## Architecture

```text
React UI
  -> FastAPI backend (local)
       -> Docling OCR service (EC2 GPU)  — document understanding: tables, figures, diagrams
       -> Neo4j Aura                     — graph + vector index
       -> Ollama (EC2 GPU)
            -> Gemma 4 E4B LLM           — entity extraction + answer generation
            -> nomic-embed-text          — chunk embeddings
```

### Ingestion Pipeline

1. **Docling OCR** (IBM, EC2 A10G GPU) — AI-powered document parsing with table, figure, and diagram understanding
2. **Chunking** — RecursiveCharacterTextSplitter (4000 chars, 600 overlap)
3. **Embeddings** — nomic-embed-text via Ollama
4. **Entity extraction** — Gemma 4 E4B extracts biomarkers, conditions, methods into Neo4j nodes
5. **Graph write** — lexical + semantic graph written to Neo4j Aura
6. **Entity resolution** — merges duplicate entities across documents

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
- `OLLAMA_BASE_URL` — Ollama endpoint (EC2: `http://<EC2_IP>:11434`)
- `OLLAMA_LLM_MODEL` — defaults to `gemma4:e4b`
- `OLLAMA_EMBED_MODEL` — defaults to `nomic-embed-text`
- `DOCLING_OCR_URL` — Docling OCR service endpoint (EC2: `http://<EC2_IP>:8001`)
- `LOCAL_RERANK_ENABLED`

LangSmith tracing is optional for developer debugging and is disabled by default:

```env
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=GroundedMD-Hackathon
```

Do not commit real `.env` files or credentials.

## Evidence Corpus

The corpus is composed of peer-reviewed TBI research publications covering AI diagnostics, blood biomarkers, outcome prediction, and neurorehabilitation:

| File | Paper |
|------|-------|
| `ai-for-tbi-imaging-translational-review-furst-2026.pdf` | Artificial Intelligence for TBI Imaging: A Translational Review from Algorithm Development to Clinical Implementation — Frontiers in Neurology, 2026 |
| `conformal-prediction-ich-detection-gamble-2024.pdf` | Applying Conformal Prediction to a Deep Learning Model for Intracranial Hemorrhage Detection to Improve Trustworthiness — Gamble et al., Radiology: AI, 2024 |
| `trust-gap-conformal-prediction-ich-ngum-filippi-2025.pdf` | Bridging the Trust Gap: Conformal Prediction for AI-based Intracranial Hemorrhage Detection — Ngum & Filippi, Radiology: AI, 2025 |
| `blood-biomarkers-ich-outcome-moderate-severe-tbi-anderson.pdf` | Blood-Based Biomarkers for Prediction of Intracranial Hemorrhage and Outcome in Patients with Moderate or Severe TBI — Anderson et al. |
| `outcome-prediction-severe-tbi-deep-learning-ct.pdf` | Outcome Prediction in Patients with Severe TBI Using Deep Learning from Head CT Scans — Radiology |
| `data-driven-prognosis-tbi-interpretable-ml-tritt-2023.pdf` | Data-Driven Distillation and Precision Prognosis in TBI with Interpretable Machine Learning — Tritt et al., Scientific Reports, 2023 |
| `outcome-prediction-tbi-machine-learning-bark-2024.pdf` | Refining Outcome Prediction After TBI with Machine Learning Algorithms — Bark et al., Scientific Reports, 2024 |
| `tbi-and-ai-neurorehabilitation-review-orenuga-2025.pdf` | Traumatic Brain Injury and Artificial Intelligence: Shaping the Future of Neurorehabilitation — Orenuga et al., Life, 2025 |

## License

Project code is released under **CC-BY 4.0** (Creative Commons Attribution 4.0 International), as required by the Gemma 4 Good Hackathon competition rules.

The included source documents (PDF publications) and model weights (Gemma 4, nomic-embed-text, BAAI/bge-reranker-v2-m3) remain subject to their original licenses and terms. The CC-BY 4.0 license applies only to the GroundedMD project source code.