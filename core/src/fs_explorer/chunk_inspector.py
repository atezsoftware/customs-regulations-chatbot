"""
Standalone chunk inspector app.

This is deliberately separate from the main FsExplorer UI and indexing pipeline.
Run it when you want to inspect regulatory chunk boundaries for a single file.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from .fs import SUPPORTED_EXTENSIONS
from .indexing.regulatory_chunker import chunk_file

app = FastAPI(
    title="Chunk Inspector",
    description="Standalone regulatory chunk preview tool",
)


class ChunkRequest(BaseModel):
    """Request body for standalone chunk preview."""

    file_path: str
    root_path: str | None = None
    max_chunk_chars: int = 2400


def _iter_supported_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.name.startswith("~$"):
            continue
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path.resolve())
    return sorted(files)


@app.get("/", response_class=HTMLResponse)
async def get_ui() -> HTMLResponse:
    html_path = Path(__file__).parent / "chunk_inspector.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Chunk inspector UI not found</h1>", status_code=404)
    return HTMLResponse(html_path.read_text(encoding="utf-8"), status_code=200)


@app.get("/api/files", response_model=None)
async def list_files(
    folder: str = "data/customs_test",
) -> dict[str, Any] | JSONResponse:
    try:
        root = Path(folder).expanduser().resolve()
        if not root.exists():
            return JSONResponse({"error": "Folder not found"}, status_code=404)
        if not root.is_dir():
            return JSONResponse({"error": "Path is not a folder"}, status_code=400)
        files = _iter_supported_files(root)
        return {
            "folder": str(root),
            "files": [
                {
                    "name": path.name,
                    "path": str(path),
                    "relative_path": str(path.relative_to(root)),
                    "size_bytes": path.stat().st_size,
                }
                for path in files
            ],
        }
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/api/chunk", response_model=None)
async def preview_chunks(request: ChunkRequest) -> dict[str, Any] | JSONResponse:
    try:
        max_chunk_chars = max(int(request.max_chunk_chars or 2400), 500)
        result = await asyncio.to_thread(
            chunk_file,
            request.file_path,
            root_path=request.root_path,
            max_chunk_chars=max_chunk_chars,
        )
        return result.to_dict()
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except PermissionError:
        return JSONResponse({"error": "Permission denied"}, status_code=403)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


def run_server(host: str = "127.0.0.1", port: int = 8123) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standalone chunk inspector.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8123, type=int)
    args = parser.parse_args()
    run_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
