"""
Shared pytest fixtures for FsExplorer tests.

Provides the `database_url` fixture for tests that need a real Postgres
(indexing/search/storage) — these require a running Postgres reachable via
the DATABASE_URL env var (the same instance `db/` brings up; point it at a
throwaway test database) and skip otherwise.

Package-specific fixtures (e.g. the mock GenAI client used by api tests)
live in that package's own `tests/<package>/conftest.py` so this file stays
free of any `fs_explorer_api`/`fs_explorer_indexer` imports — it's loaded
for every test run regardless of which packages are installed.
"""

import os

import pytest


@pytest.fixture
def database_url() -> str:
    """Postgres connection string for storage/indexing/search tests."""
    url = os.getenv("DATABASE_URL")
    if not url:
        pytest.skip("DATABASE_URL not set — skipping test that needs Postgres.")
    return url
