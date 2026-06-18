# Multi-Modal Diagnostic & Research Copilot

A stateful, multi-agent medical copilot built to ingest scanned patient clinical history documents and medical imaging (MRIs/X-rays), analyze findings using a Vision Transformer/U-Net, query localized clinical guidelines via RAG with cross-encoder reranking, and synthesize a Pydantic-validated diagnostic summary report.

---

## Architecture Overview

The system uses a stateful agent orchestration pattern directed by a **Supervisor Agent**:

```
                  ┌──────────────────────┐
                  │   Supervisor Agent   │
                  └──────────┬───────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
   ┌────────────────┐┌───────────────┐┌───────────────┐
   │Document Ingest ││Vision-Inference││ RAG Retrieval │
   │   (OCR/PDF)    ││ (ViT/U-Net)   ││(FAISS/Chroma) │
   └────────────────┘└───────────────┘└───────────────┘
```

1. **Supervisor Routing (LangGraph style)**: Manages state, sequences workers dynamically based on the uploaded inputs, and aggregates JSON outputs.
2. **Document Ingestion Agent**: Uses `pytesseract` and `pdf2image` to perform optical character recognition (OCR) on clinical notes, with a fallback text-parser if system libraries are not present.
3. **Vision Inference Agent**: Employs a Vision Transformer (ViT) & U-Net architecture (via PyTorch) to segment/classify scans (Glioma, Meningioma, Pituitary Adenoma, or Normal) and calculate lesion sizing, with a pixel density analysis fallback.
4. **Retrieval Agent**: Matches findings against a local medical guidelines database using vector embeddings (`sentence-transformers`) and a Cross-Encoder for reranking to prevent hallucinations.
5. **FastAPI Streaming Endpoint**: Provides execution progress logging and final report streaming using Server-Sent Events (SSE).

---

## Tech Stack

- **Backend**: FastAPI, Uvicorn, Pydantic (v2), PyTorch, ChromaDB/FAISS, Sentence-Transformers, PyTesseract, PDF2Image.
- **Frontend**: React (Vite), TypeScript, custom premium dark-mode Vanilla CSS design system.
- **Deployment**: Docker, Docker Compose.

---

## Security & Protection Features

- **Input Validation**: Hard constraints on model fields using Pydantic schemas.
- **Upload Guards**:
  - Max file size limits enforced at 10 MB.
  - Strict path traversal guards using `os.path.basename` and absolute path verification to confirm files remain inside the sandbox folder.
  - Magic byte validation to prevent MIME type spoofing (blocks scripts disguised as images or documents).
  - Restricted directory execution permissions (chmod 600 for files, chmod 750 for folders).
- **CORS Allowlist**: Restricted to trusted development origins (no wildcard `*` allowed).
- **Rate Limiting**: Custom middleware implements in-memory IP rate limiting to mitigate DoS threats.
- **Security Headers**: Standard headers injected (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, strict Content Security Policy).
- **Secrets Management**: Ephemeral secret fallback generator prevents hardcoded credential exposure.

---

## Getting Started

### Option A: Running with Docker Compose

Ensure Docker and Docker Compose are installed, then execute:

```bash
docker-compose up --build
```

- **Frontend UI**: Open `http://localhost:5173`
- **FastAPI backend**: Served at `http://localhost:8000`

---

### Option B: Running Locally (Without Docker)

#### 1. Backend Setup
Create a Python virtual environment and install dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Start the FastAPI application. By default, it binds strictly to `127.0.0.1` for local safety:

```bash
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

#### 2. Frontend Setup
Make sure Node.js (v18+) is installed:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.
