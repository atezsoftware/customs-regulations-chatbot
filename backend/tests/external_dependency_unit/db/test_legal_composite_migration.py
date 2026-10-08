import pytest

from onyx.db.legal_composite_migration_validation import validate_source_kind_migration


@pytest.mark.asyncio
async def test_source_kind_upgrade_and_downgrade_with_asyncpg() -> None:
    await validate_source_kind_migration()
