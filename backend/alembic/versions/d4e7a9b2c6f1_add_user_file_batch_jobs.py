"""add user file batch jobs

Revision ID: d4e7a9b2c6f1
Revises: c8d1e4f2a7b9
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "d4e7a9b2c6f1"
down_revision = "c8d1e4f2a7b9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "user_file_batch_job",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_set_id", sa.Integer(), nullable=False),
        sa.Column("created_by", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("search_settings_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("contextual_model", sa.Text(), nullable=False),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("output_dimension", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
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
            "status IN ('pending','running','completed','failed','canceled')",
            name="user_file_batch_job_status_check",
        ),
        sa.CheckConstraint(
            "stage IN ('prepare','document_summary','chunk_context','embedding','index_write')",
            name="user_file_batch_job_stage_check",
        ),
        sa.ForeignKeyConstraint(["created_by"], ["user.id"]),
        sa.ForeignKeyConstraint(
            ["document_set_id"], ["document_set.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["search_settings_id"], ["search_settings.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_user_file_batch_job_document_set",
        "user_file_batch_job",
        ["document_set_id", "created_at"],
    )
    op.create_table(
        "user_file_batch_item",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("regulatory_chunk_id", sa.Text(), nullable=True),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("request_key", sa.Text(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("provider_job_name", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("attempts", sa.Integer(), nullable=False),
        sa.Column("output_text", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
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
        sa.CheckConstraint("attempts >= 0", name="user_file_batch_item_attempts_check"),
        sa.CheckConstraint(
            "stage IN ('prepare','document_summary','chunk_context','embedding','index_write')",
            name="user_file_batch_item_stage_check",
        ),
        sa.CheckConstraint(
            "status IN ('pending','submitted','running','completed','failed','canceled')",
            name="user_file_batch_item_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"], ["user_file_batch_job.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["regulatory_chunk_id"], ["regulatory_chunk.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_file_id"], ["user_file.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "job_id", "stage", "request_key", name="uq_user_file_batch_item_request"
        ),
    )
    op.create_index(
        "ix_user_file_batch_item_job_stage_status",
        "user_file_batch_item",
        ["job_id", "stage", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_file_batch_item_job_stage_status", table_name="user_file_batch_item"
    )
    op.drop_table("user_file_batch_item")
    op.drop_index(
        "ix_user_file_batch_job_document_set", table_name="user_file_batch_job"
    )
    op.drop_table("user_file_batch_job")
