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
                        config_snapshot,
                        '{input_content_hash}',
                        to_jsonb(content_hash),
                        true
                    ),
                    '{chunk_generation_hash}',
                    to_jsonb(CAST(:generation_hash AS text)),
                    true
                )
            """
        ).bindparams(generation_hash=_CURRENT_CHUNK_GENERATION_HASH)
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
            - 'chunk_generation_hash'
        """
    )
    op.drop_column("regulatory_indexing_job", "chunk_generation_hash")
