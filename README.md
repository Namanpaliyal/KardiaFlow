# KardiaFlow

Lightweight RAG (Retrieval-Augmented Generation) demo using FastAPI, Chroma (vector DB), Sentence-Transformers embeddings, and Ollama LLM (`gemma2:2b`). This repository contains the FastAPI app, helper deployment scripts, and CI for container-based Azure deployment.
<img width="1783" height="892" alt="image" src="https://github.com/user-attachments/assets/1ea62918-c674-41f7-9a4e-a0f8b5cf2f11" />


## Contents

- `app/` — FastAPI application and RAG implementation (`app/main.py`, `app/rag.py`).
- `Dockerfile` — container image to run the app (uvicorn on port 8000).
- `requirements.txt` — Python dependencies.
- `deploy_acr.sh`, `deploy_azure.ps1`, `deploy_azure.sh` — helper deployment scripts.
- `.github/workflows/azure-acr-deploy.yml` — example GitHub Actions workflow for building/pushing to ACR and updating a Web App.

## Quick Start (Local)

1. Create and activate a virtual environment (Windows PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Note: heavy ML packages (transformers, sentence-transformers, chromadb) can take time and may require a C compiler or platform-specific wheels. If installation fails, paste the pip output here so it can be diagnosed.

3. Set environment variables (example defaults):

```powershell
$env:OLLAMA_BASE_URL = 'http://localhost:11434'
$env:PERSIST_DIR = './chroma_db'
```

4. Run the server:

```powershell
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

5. Endpoints

- `GET /health` — basic health check.
- `POST /api/upload-pdf` — upload a PDF to ingest into the vector store.
- `POST /api/ask` — ask a question (JSON body: `{"question": "...", "top_k": 3}`).

## RAG Details

- Vector DB: Chroma persisted to `PERSIST_DIR` (default `./chroma_db`).
- Embeddings: `sentence-transformers` (all-MiniLM-L6-v2) via `SentenceTransformerEmbeddings`.
- LLM: Ollama client — model `gemma2:2b`. Ensure an Ollama server is reachable at `OLLAMA_BASE_URL` or responses will error but the API will stay up.

## Docker (recommended for Azure)

Build locally:

```powershell
docker build -t kardiaflow:latest .
docker run -p 8000:8000 -e OLLAMA_BASE_URL='http://host.docker.internal:11434' kardiaflow:latest
```

Notes:
- When running Ollama from the host with Docker on Windows, `host.docker.internal` often routes to the host for the container.

## Azure Deployment (recommended: container)

Why container?
- App Service build environment may fail installing heavy ML dependencies. Building an image locally or in ACR avoids those issues.

Quick container deploy steps (high-level):

1. Use the provided `deploy_acr.sh` (or `deploy_azure.ps1`) to build and push the image to Azure Container Registry and configure a Web App for Containers.
2. Ensure the Web App setting `WEBSITES_PORT` is set to `8000` and the startup command is `python -m uvicorn app.main:app --host 0.0.0.0 --port 8000`.
3. Set App Settings for `OLLAMA_BASE_URL` and `PERSIST_DIR` in the Web App configuration.

See: `deploy_acr.sh` and `.github/workflows/azure-acr-deploy.yml` for CI automation.

## Troubleshooting

- Container/app not responding on Azure: ensure `WEBSITES_PORT=8000` and Web App startup command is correct.
- Pip install failures: collect pip logs and system platform info; consider using a container image with pre-built wheels.
- Ollama unreachable: ensure `OLLAMA_BASE_URL` points to a reachable Ollama server; else the `/api/ask` handler will return an error string.

## Development Notes

- Persisted Chroma DB is in `PERSIST_DIR` (default `./chroma_db`). Back this up if you need to preserve vectors.
- If you change embeddings or re-ingest PDFs, you may need to delete or re-create the `PERSIST_DIR` to rebuild the vector store.

## Files of interest

- [deploy_acr.sh](deploy_acr.sh) — ACR + Web App helper script
- [deploy_azure.ps1](deploy_azure.ps1) — PowerShell deploy helper
- [.github/workflows/azure-acr-deploy.yml](.github/workflows/azure-acr-deploy.yml) — CI workflow

## Contributing

Create an issue or PR. For troubleshooting installation issues, include OS, Python version, and the pip error output.




