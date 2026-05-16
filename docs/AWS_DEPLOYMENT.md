# GroundedMD — AWS GPU Deployment Guide

For a smooth demo, run on a GPU instance so Gemma 4 responses are fast.
Neo4j is hosted on Aura (cloud) — no local database needed.

## Recommended Instance

| Field | Value |
|-------|-------|
| Instance type | `g4dn.xlarge` |
| GPU | 1x NVIDIA T4 (16GB VRAM) |
| Cost | ~$0.526/hr |
| AMI | Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) |
| Storage | 50GB gp3 |

Security group — open only these ports:
- `22` — SSH, your IP only
- `80` — HTTP, anywhere (the demo URL)

---

## Step 1 — Connect to the instance

```bash
ssh -i your-key.pem ubuntu@<ec2-public-ip>
```

---

## Step 2 — Install NVIDIA Container Toolkit

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify GPU is visible:
```bash
nvidia-smi
```

---

## Step 3 — Clone the repo

```bash
git clone https://github.com/<your-repo>/groundedmd-gemma4-graphrag.git
cd groundedmd-gemma4-graphrag
```

---

## Step 4 — Set up .env

```bash
cp .env.example .env
nano .env
```

Fill in your Neo4j Aura credentials and leave Ollama pointing to localhost:

```env
NEO4J_URI=neo4j+s://<your-aura-instance-id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-aura-password>

OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_LLM_MODEL=gemma4:e4b
OLLAMA_EMBED_MODEL=nomic-embed-text
LOCAL_RERANK_ENABLED=false

LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=GroundedMD-Hackathon
```

---

## Step 5 — Launch with GPU override

```bash
docker compose -f docker-compose.yml -f docker-compose.aws.yml up -d --build
```

Watch model download (first run only — ~3GB):
```bash
docker compose logs -f ollama-setup
```

---

## Step 6 — Verify everything is up

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/health/models
```

Expected:
```json
{"status": "ok", "neo4j": "connected"}
{"backend": "ollama", "llm_model": "gemma4:e4b", "ollama_status": "connected"}
```

---

## Step 7 — Ingest the TBI publications

Open the UI at `http://<ec2-public-ip>` and use the Ingest page to load all 8 publications,
or run via the API:

```bash
curl -X POST http://localhost:8000/ingest
```

---

## Step 8 — Open the demo

```
http://<ec2-public-ip>
```

---

## Security Notes

- Do not expose ports `7687` (Neo4j), `11434` (Ollama) publicly.
- Expose only port `80` (frontend/nginx) to the internet.
- Do not commit `.env` files.
- Rotate Neo4j Aura credentials after the demo if shared.

## Cost Estimate

| Item | Rate |
|------|------|
| g4dn.xlarge | ~$0.53/hr |
| 8 hours demo day | ~$4.24 |
| Storage 50GB | ~$0.10/day |

Well within the AWS Activate credits.
