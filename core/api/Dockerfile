# Build from core/ (the uv workspace root): docker build -f api/Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependency layer: every workspace member's pyproject.toml must be present
# for uv to resolve the workspace, but --package scopes what's *installed* —
# this never pulls in fs-explorer-indexer's docling/langextract.
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml shared/pyproject.toml
COPY api/pyproject.toml api/pyproject.toml
COPY indexer/pyproject.toml indexer/pyproject.toml
RUN uv sync --frozen --no-dev --package fs-explorer-api --no-install-workspace

COPY shared/src shared/src
COPY api/src api/src
RUN uv sync --frozen --no-dev --package fs-explorer-api

EXPOSE 8000

# Run the venv's uvicorn directly rather than `uv run` — avoids an implicit
# `uv sync` (and its PyPI round-trip) on every container start.
CMD ["/app/.venv/bin/uvicorn", "fs_explorer_api.server:app", "--host", "0.0.0.0", "--port", "8000"]
