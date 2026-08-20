import json
from collections.abc import Generator
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.orm import Session

_LEGACY_REVISION = "c8f1a6d4e2b7"
_CHUNK_IDENTITY_REVISION = "d2a9c7e4b1f6"


def _alembic_config(engine: Engine, schema_name: str) -> Config:
    backend_root = Path(__file__).resolve().parents[3]
    config = Config(str(backend_root / "alembic.ini"))
    config.set_main_option("script_location", str(backend_root / "alembic"))
    config.attributes["connection"] = engine
    config.attributes["schema_name"] = schema_name
    config.attributes["configure_logger"] = False
    return config


@pytest.fixture
def legacy_regulatory_schema(
    db_session: Session,
) -> Generator[tuple[Engine, str], None, None]:
    engine = db_session.get_bind()
    assert isinstance(engine, Engine)
    schema_name = f"regulatory_migration_{uuid4().hex}"
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
        connection.execute(text(f'SET search_path TO "{schema_name}"'))
        connection.execute(
            text(
                """
                CREATE TABLE user_file (
                    id uuid PRIMARY KEY
                );
                CREATE TABLE regulatory_chunk (
                    id text PRIMARY KEY
                );
                CREATE TABLE regulatory_indexing_job (
                    id uuid PRIMARY KEY,
                    user_file_id uuid NOT NULL
                        REFERENCES user_file(id) ON DELETE CASCADE,
                    content_hash varchar(128) NOT NULL,
                    search_settings_id integer NOT NULL,
                    prompt_hash varchar(128) NOT NULL,
                    config_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb,
                    status varchar(32) NOT NULL,
                    stage varchar(32) NOT NULL,
                    lease_generation integer NOT NULL DEFAULT 0,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now(),
                    CONSTRAINT uq_regulatory_indexing_job_idempotency UNIQUE (
                        user_file_id,
                        content_hash,
                        search_settings_id,
                        prompt_hash
                    )
                );
                CREATE TABLE regulatory_indexing_item (
                    id uuid PRIMARY KEY,
                    job_id uuid NOT NULL REFERENCES regulatory_indexing_job(id)
                        ON DELETE CASCADE,
                    regulatory_chunk_id text NOT NULL REFERENCES regulatory_chunk(id)
                        ON DELETE CASCADE
                );
                CREATE TABLE alembic_version (
                    version_num varchar(32) PRIMARY KEY
                );
                INSERT INTO alembic_version (version_num)
                VALUES ('c8f1a6d4e2b7');
                """
            )
        )
    try:
        yield engine, schema_name
    finally:
        # Alembic sets search_path on pooled connections. Dispose those idle
        # connections so later external-dependency tests reopen on public.
        engine.dispose()
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
        engine.dispose()


def test_two_generation_downgrade_keeps_current_job_and_cascades_old_items(
    legacy_regulatory_schema: tuple[Engine, str],
) -> None:
    engine, schema_name = legacy_regulatory_schema
    config = _alembic_config(engine, schema_name)
    user_file_id = uuid4()
    chunk_id = f"chunk-{uuid4().hex}"
    first_job_id = uuid4()
    second_job_id = uuid4()
    first_item_id = uuid4()
    second_item_id = uuid4()
    content_hash = "a" * 64

    with engine.begin() as connection:
        connection.execute(text(f'SET search_path TO "{schema_name}"'))
        connection.execute(
            text("INSERT INTO user_file (id) VALUES (:id)"),
            {"id": user_file_id},
        )
        connection.execute(
            text("INSERT INTO regulatory_chunk (id) VALUES (:id)"),
            {"id": chunk_id},
        )
        connection.execute(
            text(
                """
                INSERT INTO regulatory_indexing_job (
                    id, user_file_id, content_hash, search_settings_id,
                    prompt_hash, config_snapshot, status, stage,
                    lease_generation, created_at, updated_at
                ) VALUES (
                    :id, :user_file_id, :content_hash, 17, 'prompt-v1',
                    '{"pre_migration": true}'::jsonb, 'SUCCEEDED', 'PUBLISH',
                    1, '2026-08-19T10:00:00Z', '2026-08-19T10:00:00Z'
                )
                """
            ),
            {
                "id": first_job_id,
                "user_file_id": user_file_id,
                "content_hash": content_hash,
            },
        )

    command.upgrade(config, "head")
    with engine.begin() as connection:
        connection.execute(text(f'SET search_path TO "{schema_name}"'))
        migrated_snapshot = connection.scalar(
            text("SELECT config_snapshot FROM regulatory_indexing_job WHERE id = :id"),
            {"id": first_job_id},
        )
        assert migrated_snapshot["input_content_hash"] == content_hash
        assert migrated_snapshot["input_hash_version"] == "legacy-v1"
        first_generation_hash = migrated_snapshot["chunk_generation_hash"]

        connection.execute(
            text(
                """
                INSERT INTO regulatory_indexing_job (
                    id, user_file_id, content_hash, chunk_generation_hash,
                    search_settings_id, prompt_hash, config_snapshot, status,
                    stage, lease_generation, created_at, updated_at
                ) VALUES (
                    :id, :user_file_id, :content_hash, :generation_hash,
                    17, 'prompt-v1', CAST(:snapshot AS jsonb),
                    'SUCCEEDED', 'PUBLISH', 2,
                    '2026-08-20T10:00:00Z', '2026-08-20T10:00:00Z'
                )
                """
            ),
            {
                "id": second_job_id,
                "user_file_id": user_file_id,
                "content_hash": content_hash,
                "generation_hash": "b" * 64,
                "snapshot": json.dumps(
                    {
                        "input_content_hash": content_hash,
                        "input_hash_version": "canonical-v2",
                        "chunk_generation_hash": "b" * 64,
                    }
                ),
            },
        )
        connection.execute(
            text(
                """
                INSERT INTO regulatory_indexing_item (
                    id, job_id, regulatory_chunk_id
                ) VALUES
                    (:first_item_id, :first_job_id, :chunk_id),
                    (:second_item_id, :second_job_id, :chunk_id)
                """
            ),
            {
                "first_item_id": first_item_id,
                "first_job_id": first_job_id,
                "second_item_id": second_item_id,
                "second_job_id": second_job_id,
                "chunk_id": chunk_id,
            },
        )

    command.downgrade(config, _LEGACY_REVISION)
    with engine.begin() as connection:
        connection.execute(text(f'SET search_path TO "{schema_name}"'))
        assert connection.scalars(
            text("SELECT id FROM regulatory_indexing_job")
        ).all() == [second_job_id]
        assert connection.scalars(
            text("SELECT id FROM regulatory_indexing_item")
        ).all() == [second_item_id]
        assert connection.scalar(
            text(
                """
                SELECT count(*) = 0
                FROM information_schema.columns
                WHERE table_schema = :schema_name
                  AND table_name = 'regulatory_indexing_job'
                  AND column_name = 'chunk_generation_hash'
                """
            ),
            {"schema_name": schema_name},
        )

    command.upgrade(config, "head")
    with engine.begin() as connection:
        connection.execute(text(f'SET search_path TO "{schema_name}"'))
        upgraded = connection.execute(
            text(
                """
                SELECT id, chunk_generation_hash,
                       config_snapshot ->> 'input_hash_version'
                FROM regulatory_indexing_job
                """
            )
        ).one()
        assert upgraded == (
            second_job_id,
            first_generation_hash,
            "legacy-v1",
        )
        assert connection.scalar(
            text(
                """
                SELECT count(*) = 1
                FROM pg_indexes
                WHERE schemaname = :schema_name
                  AND indexname =
                      'uq_regulatory_indexing_job_active_user_file'
                """
            ),
            {"schema_name": schema_name},
        )
