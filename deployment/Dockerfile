# Multi-stage build for optimized image size
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY requirements-api.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt
RUN pip install --no-cache-dir --user -r requirements-api.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy source code
COPY src/ ./src/

# Copy model files (CRITICAL for predictions to work!)
COPY models/ ./models/
COPY models_enhanced/ ./models_enhanced/

# Create data directory
RUN mkdir -p data

# Set environment variables
ENV PORT=8000
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import os, requests; requests.get(f'http://localhost:{os.environ.get(\"PORT\", \"8000\")}/health')" || exit 1

# Run the application (Railway sets PORT env var)
CMD ["sh", "-c", "python -m uvicorn src.api.api_server:app --host 0.0.0.0 --port ${PORT:-8000}"]