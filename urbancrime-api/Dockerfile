# ── Stage 1: builder ──────────────────────────────────────────────────────────
# We install deps in a separate stage so the final image doesn't carry pip cache
FROM python:3.9-slim AS builder

WORKDIR /install

COPY requirements.txt .

# CPU-only torch saves ~600 MB vs the default CUDA wheel
RUN pip install --no-cache-dir --prefix=/install/packages \
    torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --prefix=/install/packages \
    -r requirements.txt --no-deps torch


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.9-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/packages /usr/local

# Copy application code
COPY app/ ./app/

# Data is mounted at runtime (Docker volume or bind mount) — not baked into the
# image.  This keeps the image small and lets you update models without rebuild.
# Expected mount: -v $(pwd)/../data:/data

ENV PYTHONUNBUFFERED=1 \
    DATA_DIR=/data/processed \
    MODELS_DIR=/data/models

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
