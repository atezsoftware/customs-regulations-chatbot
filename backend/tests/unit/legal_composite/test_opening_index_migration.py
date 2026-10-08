import importlib.util
from pathlib import Path
from types import ModuleType
from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy import Table
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex

from onyx.db.models import RegulatoryTemporalProjection


@pytest.fixture
def migration() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[3]
        / "alembic/versions/ca61f30b9e72_index_canonical_source_openings.py"
    )
    spec = importlib.util.spec_from_file_location("opening_indexes", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def connection(migration: ModuleType, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    bind = MagicMock()
    bind.dialect = postgresql.dialect()
    bind.execute.return_value.scalar_one.return_value = 'tenant"quoted'
    conn = MagicMock()
    conn.dialect = postgresql.dialect()
    bind.engine.connect.return_value.execution_options.return_value.__enter__.return_value = conn
    monkeypatch.setattr(migration.op, "get_bind", lambda: bind)
    return conn


def test_existing_valid_indexes_make_upgrade_idempotent(
    migration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = connection(migration, monkeypatch)
    conn.execute.return_value.one_or_none.return_value = (True,)
    migration.upgrade()
    assert conn.execute.call_count == 2
    conn.exec_driver_sql.assert_not_called()


def test_invalid_owned_index_is_repaired_on_a_schema_qualified_connection(
    migration: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    conn = connection(migration, monkeypatch)
    conn.execute.return_value.one_or_none.side_effect = [(False,), None]
    migration.upgrade()
    commands = [call.args[0] for call in conn.exec_driver_sql.call_args_list]
    assert len(commands) == 3
    assert commands[0] == (
        'DROP INDEX CONCURRENTLY "tenant""quoted".'
        "ix_temporal_projection_file_qualification"
    )
    assert all(" CONCURRENTLY " in command for command in commands)
    assert all('"tenant""quoted".' in command for command in commands)
    assert "canonical_opening" in commands[2]
    assert not any("UPDATE " in command or "DELETE " in command for command in commands)


def test_model_indexes_keep_retired_qualification_and_exact_opening_predicates(
    migration: ModuleType,
) -> None:
    indexes = {
        str(index.name): index
        for index in cast(Table, RegulatoryTemporalProjection.__table__).indexes
        if index.name in migration.INDEXES
    }
    assert indexes.keys() == migration.INDEXES.keys()
    qualification = str(
        CreateIndex(indexes["ix_temporal_projection_file_qualification"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert "(user_file_id)" in qualification and " WHERE " not in qualification
    opening = str(
        CreateIndex(indexes["ix_temporal_projection_canonical_opening"]).compile(
            dialect=postgresql.dialect()
        )
    )
    assert migration.INDEXES["ix_temporal_projection_canonical_opening"] in opening
