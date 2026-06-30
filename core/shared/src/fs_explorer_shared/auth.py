"""
Shared internal-call gate for the api and indexer FastAPI services.

Both services are meant to be called only by `backend`, never directly from a
browser. If `CORE_INTERNAL_TOKEN` isn't configured, the gate is a no-op
(local/CLI usage without a bridge in front of it keeps working).
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException


def internal_token_valid(provided: str | None) -> bool:
    """Check a caller-supplied token against `CORE_INTERNAL_TOKEN`."""
    expected = os.getenv("CORE_INTERNAL_TOKEN")
    if not expected:
        return True
    return provided == expected


async def require_internal_token(
    x_internal_token: str | None = Header(default=None),
) -> None:
    """FastAPI dependency gating REST endpoints meant only for the backend bridge."""
    if not internal_token_valid(x_internal_token):
        raise HTTPException(
            status_code=403, detail="Invalid or missing internal token."
        )
