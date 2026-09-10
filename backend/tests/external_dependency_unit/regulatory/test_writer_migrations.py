"""Round-trip the additive writer schema on one owned disposable local tenant."""

import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.schema import DropSchema

from onyx.configs.app_configs import POSTGRES_HOST, POSTGRES_PORT
from onyx.db.engine.sql_engine import SqlEngine


def test_additive_writer_migrations_round_trip() -> None:
    assert POSTGRES_HOST in {"localhost", "127.0.0.1"} and str(POSTGRES_PORT) == "25432"
    schema = "tenant_5d_migrations_" + uuid4().hex
    backend = Path(__file__).resolve().parents[3]
    command = [sys.executable, "-m", "alembic", "-x", f"schemas={schema}"]
    SqlEngine.init_engine(pool_size=2, max_overflow=0)
    try:
        for operation, revision in (
            ("upgrade", "head"),
            ("downgrade", "1fa78a83fa88"),
            ("upgrade", "head"),
        ):
            result = subprocess.run(
                [*command, operation, revision],
                cwd=backend,
                env=os.environ.copy(),
                text=True,
                capture_output=True,
            )
            print(result.stdout)
            print(result.stderr)
            assert result.returncode == 0
        with SqlEngine.get_engine().connect() as connection:
            assert (
                connection.scalar(
                    text(f'SELECT version_num FROM "{schema}".alembic_version')
                )
                == "5d85320c4e35"
            )
            columns = set(
                connection.scalars(
                    text(
                        "SELECT column_name FROM information_schema.columns WHERE table_schema=:schema AND table_name='regulatory_indexing_item'"
                    ),
                    {"schema": schema},
                )
            )
            assert {
                "projection_id",
                "projection_ordinal",
                "projection_input",
            } <= columns
    finally:
        with SqlEngine.get_engine().begin() as connection:
            connection.execute(DropSchema(schema, cascade=True, if_exists=True))
