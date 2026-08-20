"""Merge CHUNKED flow and add durable OpenRouter embedding Batch state.

Revision ID: b7e9d2c4f6a8
Revises: 6df58c46ade5, a6d4c8e2f1b7
Create Date: 2026-08-20
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "b7e9d2c4f6a8"
down_revision = ("6df58c46ade5", "a6d4c8e2f1b7")
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_file",
        sa.Column("regulatory_chunk_generation_hash", sa.String(64), nullable=True),
    )
    for column in (
        sa.Column("remote_openrouter_batch_id", sa.String(256), nullable=True),
        sa.Column("openrouter_submission_key", sa.String(128), nullable=True),
        sa.Column(
            "openrouter_submission_state",
            sa.String(32),
            server_default="NONE",
            nullable=False,
        ),
        sa.Column(
            "openrouter_submission_attempt_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "openrouter_submission_charged",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "openrouter_reconcile_miss_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "openrouter_reconcile_until", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "openrouter_completion_deadline",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "openrouter_active_item_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    ):
        op.add_column("regulatory_indexing_job", column)
    op.add_column(
        "regulatory_indexing_item",
        sa.Column(
            "embedding_attempt_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
    )
    op.create_check_constraint(
        "regulatory_indexing_job_openrouter_submission_state_check",
        "regulatory_indexing_job",
        "openrouter_submission_state IN ('NONE', 'SUBMITTING', "
        "'RECONCILE_REQUIRED', 'RECONCILED_ABSENT', "
        "'MANUAL_RECONCILE_REQUIRED', 'SUBMITTED')",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_openrouter_attempts_check",
        "regulatory_indexing_job",
        "openrouter_submission_attempt_count >= 0 AND "
        "openrouter_reconcile_miss_count >= 0",
    )
    op.create_check_constraint(
        "regulatory_indexing_item_embedding_attempt_count_check",
        "regulatory_indexing_item",
        "embedding_attempt_count >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "regulatory_indexing_item_embedding_attempt_count_check",
        "regulatory_indexing_item",
        type_="check",
    )
    op.drop_constraint(
        "regulatory_indexing_job_openrouter_attempts_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_constraint(
        "regulatory_indexing_job_openrouter_submission_state_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_column("regulatory_indexing_item", "embedding_attempt_count")
    for column_name in (
        "openrouter_active_item_ids",
        "openrouter_completion_deadline",
        "openrouter_reconcile_until",
        "openrouter_reconcile_miss_count",
        "openrouter_submission_charged",
        "openrouter_submission_attempt_count",
        "openrouter_submission_state",
        "openrouter_submission_key",
        "remote_openrouter_batch_id",
    ):
        op.drop_column("regulatory_indexing_job", column_name)
    op.drop_column("user_file", "regulatory_chunk_generation_hash")
