# SynapseMed -- Stateful Multi-Agent Clinical Diagnostic Platform

SynapseMed is a stateful multi-agent medical diagnostic and clinical decision support system. It ingests clinical notes via OCR, processes medical imaging (MRI/CT scans) via Vision Transformer and U-Net pipelines, queries evidence-based medical guidelines through RAG (Retrieval-Augmented Generation), and synthesizes validated Pydantic diagnostic reports.

---

## Architecture Topology

```mermaid
graph TB
    subgraph INPUTS["Clinical Inputs"]
        NOTE["Clinical Notes PDF and Images"]
        MRI["Medical Imaging MRI and CT"]
    end

    subgraph INGESTION["Agent 1: Ingestion and OCR"]
        TESS["Tesseract OCR Engine"]
        TEXT["Structured Note Extractor"]
    end

    subgraph VISION["Agent 2: Vision Analysis"]
        VIT["Vision Transformer ViT"]
        UNET["U-Net Segmentation Engine"]
        FEAT["Imaging Feature Map"]
    end

    subgraph RAG["Agent 3: Clinical Guideline RAG"]
        FAISS["FAISS Vector Store"]
        MED["Medical Guidelines DB"]
        EMB["SentenceTransformer Embeddings"]
    end

    subgraph SUPERVISOR["Agent 4: Supervisor and Synthesizer"]
        GRAPH["LangGraph Stateful Supervisor"]
        PYD["Pydantic Clinical Report Generator"]
        VAL["Diagnostic Validation Guard"]
    end

    subgraph OUTPUT["Diagnostic Output"]
        REPORT["Structured Pydantic Diagnostic Report JSON and PDF"]
    end

    NOTE --> TESS
    TESS --> TEXT
    MRI --> VIT
    MRI --> UNET
    VIT --> FEAT
    UNET --> FEAT
    TEXT --> GRAPH
    FEAT --> GRAPH
    GRAPH -->|query context| EMB
    EMB --> FAISS
    EMB --> MED
    FAISS --> RAG
    MED --> RAG
    RAG -->|guideline context| GRAPH
    GRAPH --> PYD
    PYD --> VAL
    VAL --> REPORT

    style INPUTS fill:#18181b,stroke:#a1a1aa,color:#fff
    style INGESTION fill:#18181b,stroke:#ffffff,color:#fff
    style VISION fill:#18181b,stroke:#e4e4e7,color:#fff
    style RAG fill:#18181b,stroke:#d4d4d8,color:#fff
    style SUPERVISOR fill:#000000,stroke:#ffffff,color:#fff
    style OUTPUT fill:#18181b,stroke:#a1a1aa,color:#fff
```

---

## Multi-Agent Workflow Sequence Diagram

```mermaid
sequenceDiagram
    participant Physician as Physician / User Interface
    participant Ingestion as Ingestion Agent (OCR)
    participant Vision as Vision Agent (ViT/U-Net)
    participant Retrieval as Retrieval Agent (RAG)
    participant Supervisor as Supervisor Agent (LangGraph)

    Physician->>Supervisor: Upload Clinical Note + MRI Scan
    par Parallel Ingestion and Vision Analysis
        Supervisor->>Ingestion: Extract and parse clinical text
        Ingestion-->>Supervisor: Returns structured clinical entities
    and
        Supervisor->>Vision: Segment and analyze MRI scan features
        Vision-->>Supervisor: Returns bounding boxes and lesion classifications
    end
    Supervisor->>Retrieval: Query clinical guidelines (FAISS vector search)
    Retrieval-->>Supervisor: Relevant clinical protocols and risk scores
    Supervisor->>Supervisor: Synthesize Pydantic diagnostic schema
    Supervisor-->>Physician: Render Pydantic Diagnostic Report
```

---

## Agent System Overview

| Agent | Responsibility | Underlying Technology |
|-------|----------------|----------------------|
| **Ingestion Agent** | OCR text extraction from clinical notes, vitals parsing, lab results formatting | Tesseract OCR, PyPDF2, Regex NLP |
| **Vision Agent** | MRI/CT scan feature extraction, tumor/lesion segmentation, heatmap visualization | PyTorch, Vision Transformer (ViT), U-Net |
| **Retrieval Agent** | Semantic search over clinical practice guidelines & drug interaction databases | LangChain, FAISS Vector DB, HuggingFace Embeddings |
| **Supervisor Agent** | Multi-agent coordination, state management, Pydantic report synthesis | LangGraph, Pydantic v2, FastAPI |

---

## Directory Structure

```
SynapseMed/
|-- docker-compose.yml          # Multi-container orchestration (Backend + Frontend + Vector DB)
|-- README.md                   # ASCII Architecture & User Documentation
|-- backend/
|   |-- Dockerfile              # PyTorch + Tesseract OCR FastAPI container
|   |-- requirements.txt        # FastAPI, LangGraph, PyTorch, FAISS dependencies
|   |-- uploads/                # Temporary file upload storage
|   |-- vector_db/              # FAISS vector database indices
|   `-- app/
|       |-- __init__.py
|       |-- main.py             # FastAPI entry point & WebSocket endpoints
|       |-- config.py           # Environment variables & model thresholds
|       |-- schemas.py          # Pydantic v2 diagnostic schemas
|       |-- agents/
|       |   |-- __init__.py
|       |   |-- ingestion.py    # OCR & note parsing agent
|       |   |-- vision.py       # ViT & U-Net MRI scan agent
|       |   |-- retrieval.py    # RAG guideline search agent
|       |   `-- supervisor.py   # LangGraph multi-agent coordinator
|       `-- utils/              # Helper functions & image preprocessors
`-- frontend/                   # Web User Interface
```

---

## Quick Start Guide

### Prerequisites
- Docker & Docker Compose **OR** Python 3.10+ with PyTorch and Tesseract OCR installed.

### Running with Docker Compose

1. **Clone Repository**:
   ```bash
   git clone https://github.com/siddarth1872004/SynapseMed.git
   cd SynapseMed
   ```

2. **Launch Services**:
   ```bash
   docker-compose up --build
   ```

3. **Access Application**:
   - Backend API Docs: `http://localhost:8000/docs`
   - Frontend Dashboard: `http://localhost:3000`

### Local Development Setup

1. **Navigate to Backend**:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Start FastAPI Application**:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

---

## License

Distributed under the **MIT License**. See `LICENSE` for details.
