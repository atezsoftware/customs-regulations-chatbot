"""Retain durable indexing successors without overwriting terminal attempts.

Revision ID: 0bc7d32e5fa1
Revises: 2fcae483a930
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0bc7d32e5fa1"
down_revision = "2fcae483a930"
branch_labels = None
depends_on = None

_IDENTITY = (
    "user_file_id",
    "content_hash",
    "search_settings_id",
    "prompt_hash",
    "chunk_generation_hash",
)


def upgrade() -> None:
    op.add_column(
        "regulatory_indexing_job",
        sa.Column("restart_ordinal", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "regulatory_indexing_job",
        sa.Column("restart_of_job_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "regulatory_indexing_job",
        sa.Column("restart_attempt_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_foreign_key(
        "fk_regulatory_index_restart_parent",
        "regulatory_indexing_job",
        "regulatory_indexing_job",
        ["restart_of_job_id"],
        ["id"],
    )
    op.create_unique_constraint(
        "uq_regulatory_index_restart_attempt",
        "regulatory_indexing_job",
        ["restart_attempt_id"],
    )
    op.create_check_constraint(
        "regulatory_indexing_job_restart_identity",
        "regulatory_indexing_job",
        "(restart_ordinal = 0 AND restart_of_job_id IS NULL AND restart_attempt_id IS NULL) OR (restart_ordinal > 0 AND restart_of_job_id IS NOT NULL AND restart_attempt_id IS NOT NULL)",
    )
    op.drop_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        [*_IDENTITY, "restart_ordinal"],
    )


def downgrade() -> None:
    # Do not silently discard retained successor history on downgrade.
    connection = op.get_bind()
    if connection.scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM regulatory_indexing_job WHERE restart_ordinal > 0)"
        )
    ):
        raise RuntimeError("Durable indexing successor history prevents downgrade")
    op.drop_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_regulatory_indexing_job_idempotency",
        "regulatory_indexing_job",
        list(_IDENTITY),
    )
    op.drop_constraint(
        "regulatory_indexing_job_restart_identity",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_constraint(
        "uq_regulatory_index_restart_attempt", "regulatory_indexing_job", type_="unique"
    )
    op.drop_constraint(
        "fk_regulatory_index_restart_parent",
        "regulatory_indexing_job",
        type_="foreignkey",
    )
    op.drop_column("regulatory_indexing_job", "restart_attempt_id")
    op.drop_column("regulatory_indexing_job", "restart_of_job_id")
    op.drop_column("regulatory_indexing_job", "restart_ordinal")
