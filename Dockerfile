# ── Stage 1: build the React frontend ───────────────────────────────────────
FROM node:20-slim AS frontend-build

WORKDIR /frontend

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend, serving the built frontend ─────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch — saves ~1.3GB vs default CUDA build
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding and reranker models at build time — avoids cold-start delays
# and works in air-gapped environments after the image is built.
ENV SENTENCE_TRANSFORMERS_HOME=/app/.model_cache
ENV TRANSFORMERS_CACHE=/app/.model_cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY core/ core/
COPY api/ api/
COPY --from=frontend-build /frontend/dist/ frontend/dist/

RUN mkdir -p docs data chroma_db .model_cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
