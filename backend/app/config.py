import os
import secrets
import logging
from pathlib import Path
from pydantic_settings import BaseSettings

# Set up simple logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

# Base directory of the backend package (works regardless of deployment location,
# e.g. Docker's /app or a developer's local checkout).
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # API Configurations
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "Multi-Modal Medical Diagnostic & Research Copilot"

    # Secret Key for sessions/tokens
    # Resolves: Environment Variable -> Local File -> Ephemeral Random Key (with warning)
    JWT_SECRET_KEY: str = ""

    # Path & File System Settings
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    VECTOR_DB_DIR: Path = BASE_DIR / "vector_db"
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB limit
    ALLOWED_IMAGE_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "tiff", "tif"}
    ALLOWED_DOC_EXTENSIONS: set[str] = {"pdf", "txt", "docx"}
    
    # Model configuration
    # Can set path to real models or default to mock pipelines if not found
    VISION_MODEL_PATH: str = ""
    RAG_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    RAG_RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": True,
    }

# Create base directories safely
def init_directories(settings: Settings):
    try:
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
        # Enforce non-executable permissions on upload directory (chmod 700 / 750)
        os.chmod(settings.UPLOAD_DIR, 0o750)
    except Exception as e:
        logger.error(f"Failed to create secure folders: {e}")

# Instantiate and resolve secret
settings = Settings()

if not settings.JWT_SECRET_KEY:
    env_secret = os.getenv("JWT_SECRET_KEY")
    if env_secret:
        settings.JWT_SECRET_KEY = env_secret
    else:
        # Check for local file
        secret_file = BASE_DIR / "jwt_secret.txt"
        if secret_file.exists():
            settings.JWT_SECRET_KEY = secret_file.read_text().strip()
        else:
            # Fallback to random ephemeral key
            logger.warning("Generating ephemeral JWT secret. This instance is isolated and secret will reset on restart!")
            settings.JWT_SECRET_KEY = secrets.token_hex(32)
            # Try to write it locally to persist
            try:
                secret_file.parent.mkdir(parents=True, exist_ok=True)
                secret_file.write_text(settings.JWT_SECRET_KEY)
                os.chmod(secret_file, 0o600)  # User read/write only
            except Exception as e:
                logger.warning(f"Could not persist ephemeral secret key: {e}")

init_directories(settings)
