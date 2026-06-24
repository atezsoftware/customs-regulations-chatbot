"""
Configuration helpers for index storage (Postgres).
"""

from __future__ import annotations

import os

ENV_DATABASE_URL = "DATABASE_URL"


def resolve_database_url(override_url: str | None = None) -> str:
    """
    Resolve the Postgres connection string from CLI override or env var.

    Precedence:
    1) explicit override_url
    2) DATABASE_URL

    Raises:
        ValueError: If no connection string is available from either source.
    """
    resolved = override_url or os.getenv(ENV_DATABASE_URL)
    if not resolved:
        raise ValueError(
            "No database connection string found: pass --database-url or set "
            "the DATABASE_URL environment variable (e.g. "
            "postgresql://user:pass@localhost:5432/db)."
        )
    return resolved
