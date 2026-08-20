"""add chunk generation identity

Revision ID: d2a9c7e4b1f6
Revises: c8f1a6d4e2b7
Create Date: 2026-08-20
"""

import sqlalchemy as sa
from alembic import op

revision = "d2a9c7e4b1f6"
down_revision = "c8f1a6d4e2b7"
branch_labels = None
depends_on = None

_CURRENT_CHUNK_GENERATION_HASH = (
    "c8e1ab454f0ac79eea2db7e0c1a54979d55fa97232da08130ee8fa4b8b324e04"
)
_UNRESOLVED_INPUT_HASH_VERSION = "legacy-or-canonical"


def upgrade() -> None:
    op.add_column(
        "regulatory_indexing_job",
        sa.Column("chunk_generation_hash", sa.String(length=64), nullable=True),
    )
    op.execute(
        sa.text(
            """
            UPDATE regulatory_indexing_job
            SET chunk_generation_hash = :generation_hash,
                config_snapshot = jsonb_set(
                    jsonb_set(
                        jsonb_set(
                            config_snapshot,
                            '{input_content_hash}',
                            to_jsonb(content_hash),
                            true
                        ),
                        '{input_hash_version}',
                        to_jsonb(CAST(:input_hash_version AS text)),
                        true
                    ),
                    '{chunk_generation_hash}',
                    to_jsonb(CAST(:generation_hash AS text)),
                    true
                )
            """
        ).bindparams(
            generation_hash=_CURRENT_CHUNK_GENERATION_HASH,
            input_hash_version=_UNRESOLVED_INPUT_HASH_VERSION,
        )
    )
    op.alter_column(
        "regulatory_indexing_job",
        "chunk_generation_hash",
        existing_type=sa.String(length=64),
        nullable=False,
    )
    op.drop_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        [
            "user_file_id",
            "content_hash",
            "search_settings_id",
            "prompt_hash",
            "chunk_generation_hash",
        ],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        type_="unique",
    )
    # Downgrading removes chunk-generation identity from the legacy unique key.
    # Preserve the current/best row deterministically and let the existing
    # ON DELETE CASCADE foreign key remove only its superseded job items.
    op.execute(
        """
        WITH ranked_jobs AS (
            SELECT
                id,
                row_number() OVER (
                    PARTITION BY
                        user_file_id,
                        content_hash,
                        search_settings_id,
                        prompt_hash
                    ORDER BY
                        CASE
                            WHEN status IN (
                                'QUEUED', 'RUNNING', 'RETRY_WAIT', 'CANCELLING'
                            ) THEN 0
                            WHEN status = 'SUCCEEDED' THEN 1
                            WHEN status = 'FAILED' THEN 2
                            ELSE 3
                        END,
                        updated_at DESC,
                        created_at DESC,
                        lease_generation DESC,
                        id DESC
                ) AS row_rank
            FROM regulatory_indexing_job
        )
        DELETE FROM regulatory_indexing_job AS discarded
        USING ranked_jobs
        WHERE discarded.id = ranked_jobs.id
          AND ranked_jobs.row_rank > 1
        """
    )
    op.create_unique_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        ["user_file_id", "content_hash", "search_settings_id", "prompt_hash"],
    )
    op.execute(
        """
        UPDATE regulatory_indexing_job
        SET config_snapshot = config_snapshot
            - 'input_content_hash'
            - 'input_hash_version'
            - 'chunk_generation_hash'
        """
    )
    op.drop_column("regulatory_indexing_job", "chunk_generation_hash")
