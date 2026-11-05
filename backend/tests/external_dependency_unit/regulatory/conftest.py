"""Owned, migrated schemas for regulatory workflow tests."""

from collections.abc import Generator

import pytest

from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingDatabase,
    migrated_labeling_database,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


@pytest.fixture(scope="session")
def intake_database() -> Generator[LabelingDatabase, None, None]:
    # Each intake test rolls back its outer transaction, including worker commits.
    yield from migrated_labeling_database()
