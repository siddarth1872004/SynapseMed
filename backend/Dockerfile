FROM python:3.11-slim

# Install system dependencies for OCR and PDF image conversion
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend app files
COPY . .

# Expose backend port
EXPOSE 8000

# Create uploads and DB folder inside container with restricted permissions
RUN mkdir -p /app/uploads /app/vector_db && chmod 750 /app/uploads

# Run FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
