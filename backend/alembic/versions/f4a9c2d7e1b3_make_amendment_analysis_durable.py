"""make amendment analysis durable

Revision ID: f4a9c2d7e1b3
Revises: c2e8f5b1d3a7
Create Date: 2026-08-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f4a9c2d7e1b3"
down_revision = "c2e8f5b1d3a7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("amendment_batch_status_check", "amendment_batch", type_="check")
    op.add_column(
        "amendment_batch",
        sa.Column(
            "user_file_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "amendment_batch",
        sa.Column(
            "segmented_instructions",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "amendment_batch",
        sa.Column(
            "unmatched_instructions",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "amendment_batch",
        sa.Column(
            "stage", sa.Text(), nullable=False, server_default=sa.text("'queued'")
        ),
    )
    op.add_column(
        "amendment_batch",
        sa.Column(
            "instruction_count", sa.Integer(), nullable=False, server_default="0"
        ),
    )
    op.add_column(
        "amendment_batch",
        sa.Column(
            "processed_instruction_count",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "amendment_batch",
        sa.Column("lease_generation", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "amendment_batch",
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "amendment_batch",
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "amendment_batch",
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_check_constraint(
        "amendment_batch_status_check",
        "amendment_batch",
        "status IN ('queued', 'analyzing', 'analyzed', 'failed')",
    )
    op.create_check_constraint(
        "amendment_batch_stage_check",
        "amendment_batch",
        "stage IN ('queued', 'segmenting', 'processing', 'finalizing')",
    )
    op.create_check_constraint(
        "amendment_batch_progress_check",
        "amendment_batch",
        "instruction_count >= 0 AND processed_instruction_count >= 0 "
        "AND processed_instruction_count <= instruction_count",
    )
    op.create_check_constraint(
        "amendment_batch_lease_generation_check",
        "amendment_batch",
        "lease_generation >= 0",
    )
    op.create_unique_constraint(
        "uq_amendment_proposal_batch_instruction",
        "amendment_proposal",
        ["batch_id", "instruction_index"],
    )
    op.create_index(
        "ix_amendment_batch_recovery",
        "amendment_batch",
        ["status", "heartbeat_at", "created_at"],
    )

    # Preserve the exact document-set scope for pre-existing batches too. This
    # is especially important for a formerly in-flight row that is re-queued
    # below and must not accidentally analyze an empty or later-edited scope.
    op.execute(
        "UPDATE amendment_batch AS batch "
        "SET user_file_ids = files.ids "
        "FROM ("
        "  SELECT document_set_id, "
        "         jsonb_agg(user_file_id::text ORDER BY user_file_id::text) AS ids "
        "  FROM document_set__user_file "
        "  GROUP BY document_set_id"
        ") AS files "
        "WHERE files.document_set_id = batch.document_set_id"
    )

    # Rows created by the former synchronous endpoint have no resumable
    # checkpoint. Keep completed/failed history terminal and let any in-flight
    # row be retried from its original raw text.
    op.execute(
        "UPDATE amendment_batch SET stage = CASE "
        "WHEN status = 'analyzing' THEN 'queued' ELSE 'finalizing' END"
    )
    op.execute(
        "UPDATE amendment_batch SET status = 'queued' WHERE status = 'analyzing'"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE amendment_batch SET status = 'failed', "
        "error_message = COALESCE(error_message, "
        "'Analysis interrupted by schema downgrade') "
        "WHERE status = 'queued'"
    )
    op.drop_constraint("amendment_batch_status_check", "amendment_batch", type_="check")
    op.create_check_constraint(
        "amendment_batch_status_check",
        "amendment_batch",
        "status IN ('analyzing', 'analyzed', 'failed')",
    )
    op.drop_constraint(
        "uq_amendment_proposal_batch_instruction",
        "amendment_proposal",
        type_="unique",
    )
    op.drop_index("ix_amendment_batch_recovery", table_name="amendment_batch")
    op.drop_constraint(
        "amendment_batch_lease_generation_check", "amendment_batch", type_="check"
    )
    op.drop_constraint(
        "amendment_batch_progress_check", "amendment_batch", type_="check"
    )
    op.drop_constraint("amendment_batch_stage_check", "amendment_batch", type_="check")
    for column in (
        "completed_at",
        "heartbeat_at",
        "started_at",
        "lease_generation",
        "processed_instruction_count",
        "instruction_count",
        "stage",
        "unmatched_instructions",
        "segmented_instructions",
        "user_file_ids",
    ):
        op.drop_column("amendment_batch", column)
