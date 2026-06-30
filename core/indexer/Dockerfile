# Build from core/ (the uv workspace root): docker build -f indexer/Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv

WORKDIR /app

RUN pip install --no-cache-dir uv

# Dependency layer: every workspace member's pyproject.toml must be present
# for uv to resolve the workspace, but --package scopes what's *installed* —
# this is the one image that legitimately needs Docling/langextract.
COPY pyproject.toml uv.lock ./
COPY shared/pyproject.toml shared/pyproject.toml
COPY api/pyproject.toml api/pyproject.toml
COPY indexer/pyproject.toml indexer/pyproject.toml
RUN uv sync --frozen --no-dev --package fs-explorer-indexer --no-install-workspace

COPY shared/src shared/src
COPY indexer/src indexer/src
RUN uv sync --frozen --no-dev --package fs-explorer-indexer

EXPOSE 8001

# Run the venv's uvicorn directly rather than `uv run` — avoids an implicit
# `uv sync` (and its PyPI round-trip) on every container start.
CMD ["/app/.venv/bin/uvicorn", "fs_explorer_indexer.indexer_server:app", "--host", "0.0.0.0", "--port", "8001"]
