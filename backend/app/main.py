import json
import logging
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from datetime import datetime, timedelta, timezone

from app.config import settings, BASE_DIR
from app.utils.security import secure_save_file
from app.agents.supervisor import SupervisorOrchestrator

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_server")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Stateful Multi-Agent Clinical Ingestion & MRI Vision RAG Copilot",
    version="1.0.0"
)

# 1. Rate Limiting Middleware (Simple In-Memory IP Limiter)
# Resolves: MUST rate limit all APIs.
IP_REQUESTS = {}
RATE_LIMIT_WINDOW_SEC = 60
MAX_REQUESTS_PER_WINDOW = 60

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = datetime.now(timezone.utc)
    
    # Prune expired entries
    IP_REQUESTS[client_ip] = [
        t for t in IP_REQUESTS.get(client_ip, [])
        if now - t < timedelta(seconds=RATE_LIMIT_WINDOW_SEC)
    ]
    
    if len(IP_REQUESTS[client_ip]) >= MAX_REQUESTS_PER_WINDOW:
        logger.warning(f"Rate limit exceeded for IP: {client_ip}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
        
    IP_REQUESTS[client_ip].append(now)
    return await call_next(request)

# 2. Secure Security Headers Middleware
# Resolves: MUST configure clickjacking, nosniff, and frame-ancestors
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none';"
    return response

# 3. CORS Configuration
# Resolves: MUST avoid wildcard CORS (*) and only allow trusted development origins.
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:8000",
    "http://127.0.0.1:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
def read_root():
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": settings.PROJECT_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# 4. Stream endpoint: uploads files and initiates supervisor workflow
@app.post("/api/copilot/run")
async def run_copilot(
    patient_id: str = Form(..., min_length=3, max_length=50),
    age: Optional[int] = Form(None, ge=0, le=125),
    gender: Optional[str] = Form(None),
    document: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Directly handles uploaded files, validates them, and streams the LangGraph 
    supervisor multi-agent execution steps and the final diagnostic report JSON in real time.
    """
    saved_doc_path = None
    saved_img_path = None
    
    # Securely save uploaded document
    if document and document.filename:
        logger.info(f"Processing clinical document upload: {document.filename}")
        doc_bytes = await document.read()
        saved_doc_path = secure_save_file(
            content=doc_bytes,
            original_filename=document.filename,
            upload_dir=settings.UPLOAD_DIR
        )
        
    # Securely save uploaded MRI image
    if image and image.filename:
        logger.info(f"Processing scan image upload: {image.filename}")
        img_bytes = await image.read()
        saved_img_path = secure_save_file(
            content=img_bytes,
            original_filename=image.filename,
            upload_dir=settings.UPLOAD_DIR
        )
        
    # Build supervisor execution state
    execution_state = {
        "patient_id": patient_id,
        "age": age,
        "gender": gender,
        "document_path": str(saved_doc_path) if saved_doc_path else None,
        "image_path": str(saved_img_path) if saved_img_path else None
    }
    
    # Initialize the supervisor
    supervisor = SupervisorOrchestrator(execution_state)
    
    # Streaming SSE generator
    def event_generator():
        try:
            for step_update in supervisor.execute_step():
                yield f"data: {json.dumps(step_update)}\n\n"
        except Exception as e:
            logger.error(f"Supervisor execution failed: {e}")
            err_msg = {"step": "ERROR", "log": f"Supervisor run aborted: {str(e)}"}
            yield f"data: {json.dumps(err_msg)}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Serve frontend static assets if built
frontend_dist = BASE_DIR.parent / "frontend" / "dist"
if frontend_dist.exists():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
