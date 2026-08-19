"""add regulatory indexing jobs

Revision ID: c8f1a6d4e2b7
Revises: f7b2e4c6a8d1
Create Date: 2026-08-19
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c8f1a6d4e2b7"
down_revision = "f7b2e4c6a8d1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_indexing_job",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("search_settings_id", sa.Integer(), nullable=False),
        sa.Column("prompt_hash", sa.String(length=128), nullable=False),
        sa.Column(
            "config_snapshot",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("stage", sa.String(length=32), nullable=False),
        sa.Column("lease_generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("remote_vertex_job_name", sa.String(length=1024), nullable=True),
        sa.Column("vertex_input_uri", sa.String(length=2048), nullable=True),
        sa.Column("vertex_output_uri", sa.String(length=2048), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.String(length=4000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "attempt_count >= 0",
            name="regulatory_indexing_job_attempt_count_check",
        ),
        sa.CheckConstraint(
            "lease_generation >= 0",
            name="regulatory_indexing_job_lease_generation_check",
        ),
        sa.CheckConstraint(
            "stage IN ('PREPARING', 'CONTEXT_SUBMIT', 'CONTEXT_WAIT', "
            "'CONTEXT_APPLY', 'EMBEDDING', 'INDEX_WRITE', 'VERIFY', 'PUBLISH')",
            name="regulatory_indexing_job_stage_check",
        ),
        sa.CheckConstraint(
            "status IN ('QUEUED', 'RUNNING', 'RETRY_WAIT', 'SUCCEEDED', "
            "'FAILED', 'CANCELLING', 'CANCELLED')",
            name="regulatory_indexing_job_status_check",
        ),
        sa.ForeignKeyConstraint(["user_file_id"], ["user_file.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_file_id",
            "content_hash",
            "search_settings_id",
            "prompt_hash",
            name="uq_regulatory_indexing_job_idempotency",
        ),
    )
    op.create_index(
        "ix_regulatory_indexing_job_recovery",
        "regulatory_indexing_job",
        ["status", "next_retry_at", "heartbeat_at"],
    )
    op.create_table(
        "regulatory_indexing_item",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("regulatory_chunk_id", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("request_hash", sa.String(length=128), nullable=False),
        sa.Column("context", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("vector", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.String(length=4000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('PENDING', 'CONTEXT_READY', 'EMBEDDED', 'FAILED', 'SKIPPED')",
            name="regulatory_indexing_item_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["regulatory_indexing_job.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["regulatory_chunk_id"], ["regulatory_chunk.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id",
            "regulatory_chunk_id",
            name="uq_regulatory_indexing_item_job_chunk",
        ),
    )
    op.create_index(
        "ix_regulatory_indexing_item_job_status",
        "regulatory_indexing_item",
        ["job_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_regulatory_indexing_item_job_status",
        table_name="regulatory_indexing_item",
    )
    op.drop_table("regulatory_indexing_item")
    op.drop_index(
        "ix_regulatory_indexing_job_recovery",
        table_name="regulatory_indexing_job",
    )
    op.drop_table("regulatory_indexing_job")
