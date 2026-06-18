# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml .
COPY src/ src/

# Install runtime deps only (no dev/eval extras) into an isolated prefix
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Create non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1000 appuser

# Copy application source and scripts
COPY --chown=appuser:appgroup src/ src/
COPY --chown=appuser:appgroup scripts/ scripts/

# Data directory (FAISS index written here at runtime via PVC mount)
RUN mkdir -p data && chown appuser:appgroup data

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "rag_assistant.api.app:create_app", \
     "--factory", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
