"""add durable regulatory chunk labeling

Revision ID: c8b7a6d5e4f3
Revises: 1325beb9ce60
Create Date: 2026-09-13
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "c8b7a6d5e4f3"
down_revision = "1325beb9ce60"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_label_taxonomy",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=200), nullable=False),
        sa.Column("version_hash", sa.String(length=64), nullable=False),
        sa.Column("definition", postgresql.JSONB(), nullable=False),
        sa.Column("label_count", sa.Integer(), nullable=False),
        sa.Column("created_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "label_count > 0", name="regulatory_label_taxonomy_label_count_check"
        ),
        sa.ForeignKeyConstraint(["created_by_id"], ["user.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("version_hash"),
    )
    op.create_table(
        "regulatory_labeling_run",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("document_set_id", sa.Integer(), nullable=False),
        sa.Column("taxonomy_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_configuration_id", sa.Integer(), nullable=True),
        sa.Column("requested_by_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("retry_of_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model", sa.String(length=200), nullable=False),
        sa.Column("provider_binding", postgresql.JSONB(), nullable=False),
        sa.Column("file_ids", postgresql.JSONB(), nullable=False),
        sa.Column("snapshot_hash", sa.String(length=64), nullable=False),
        sa.Column(
            "status", sa.String(length=32), server_default="queued", nullable=False
        ),
        sa.Column(
            "stage", sa.String(length=32), server_default="preparing", nullable=False
        ),
        sa.Column("total_chunks", sa.Integer(), server_default="0", nullable=False),
        sa.Column("completed_chunks", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failed_chunks", sa.Integer(), server_default="0", nullable=False),
        sa.Column("stale_chunks", sa.Integer(), server_default="0", nullable=False),
        sa.Column("derived_chunks", sa.Integer(), server_default="0", nullable=False),
        sa.Column(
            "unresolved_derived_chunks",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "cancel_requested", sa.Boolean(), server_default=sa.false(), nullable=False
        ),
        sa.Column("error", sa.String(length=4000), nullable=True),
        sa.Column("lease_generation", sa.Integer(), server_default="0", nullable=False),
        sa.Column("lease_token", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'completed_with_errors', "
            "'failed', 'cancelled')",
            name="regulatory_labeling_run_status_check",
        ),
        sa.CheckConstraint(
            "stage IN ('preparing', 'submitting', 'waiting', 'applying', "
            "'projecting', 'finished')",
            name="regulatory_labeling_run_stage_check",
        ),
        sa.CheckConstraint(
            "total_chunks >= 0 AND completed_chunks >= 0 AND failed_chunks >= 0 "
            "AND stale_chunks >= 0 AND derived_chunks >= 0 "
            "AND unresolved_derived_chunks >= 0",
            name="regulatory_labeling_run_counts_check",
        ),
        sa.CheckConstraint(
            "lease_generation >= 0",
            name="regulatory_labeling_run_lease_generation_check",
        ),
        sa.ForeignKeyConstraint(
            ["document_set_id"], ["document_set.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["model_configuration_id"],
            ["model_configuration.id"],
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(["requested_by_id"], ["user.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["retry_of_id"], ["regulatory_labeling_run.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["taxonomy_id"], ["regulatory_label_taxonomy.id"], ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "document_set_id",
            "idempotency_key",
            name="uq_regulatory_labeling_run_idempotency",
        ),
    )
    op.create_index(
        "ix_regulatory_labeling_run_recovery",
        "regulatory_labeling_run",
        ["status", "next_retry_at", "lease_expires_at"],
    )
    op.create_index(
        "uq_regulatory_labeling_run_active_document_set",
        "regulatory_labeling_run",
        ["document_set_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )
    op.create_table(
        "regulatory_labeling_shard",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("item_ids", postgresql.JSONB(), nullable=False),
        sa.Column(
            "status", sa.String(length=32), server_default="prepared", nullable=False
        ),
        sa.Column("submission_key", sa.String(length=128), nullable=False),
        sa.Column("attempt_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("failure_count", sa.Integer(), server_default="0", nullable=False),
        sa.Column("remote_job_name", sa.String(length=1024), nullable=True),
        sa.Column("input_uri", sa.String(length=2048), nullable=True),
        sa.Column("output_uri", sa.String(length=2048), nullable=True),
        sa.Column("reconcile_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("next_retry_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.String(length=4000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('prepared', 'submitting', 'reconcile_required', "
            "'submitted', 'succeeded', 'failed', 'cancelled')",
            name="regulatory_labeling_shard_status_check",
        ),
        sa.CheckConstraint(
            "ordinal >= 0 AND attempt_count >= 0 AND failure_count >= 0",
            name="regulatory_labeling_shard_counts_check",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["regulatory_labeling_run.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("submission_key"),
        sa.UniqueConstraint(
            "run_id", "ordinal", name="uq_regulatory_labeling_shard_run_ordinal"
        ),
    )
    op.create_index(
        "ix_regulatory_labeling_shard_due",
        "regulatory_labeling_shard",
        ["run_id", "status", "next_retry_at"],
    )
    op.create_table(
        "regulatory_labeling_item",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("regulatory_chunk_id", sa.Text(), nullable=False),
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("canonical_text_sha256", sa.String(length=64), nullable=True),
        sa.Column("context_sha256", sa.String(length=64), nullable=True),
        sa.Column("text_snapshot", sa.Text(), nullable=False),
        sa.Column("source_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("context_snapshot", sa.Text(), nullable=True),
        sa.Column("request_hash", sa.String(length=64), nullable=True),
        sa.Column("request_payload", postgresql.JSONB(), nullable=True),
        sa.Column("shard_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "status", sa.String(length=32), server_default="pending", nullable=False
        ),
        sa.Column(
            "labels",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "assignments",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("error", sa.String(length=4000), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'submitted', 'completed', 'failed', "
            "'stale', 'cancelled')",
            name="regulatory_labeling_item_status_check",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["regulatory_labeling_run.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["shard_id"], ["regulatory_labeling_shard.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "regulatory_chunk_id",
            name="uq_regulatory_labeling_item_run_chunk",
        ),
    )
    op.create_index(
        "ix_regulatory_labeling_item_run_status",
        "regulatory_labeling_item",
        ["run_id", "status"],
    )
    op.execute(
        "CREATE INDEX ix_regulatory_labeling_item_unprepared_position "
        "ON regulatory_labeling_item "
        "(run_id, user_file_id, ((source_snapshot ->> 'position')::integer), id) "
        "WHERE status = 'pending' AND request_hash IS NULL"
    )
    op.create_table(
        "regulatory_derived_label_projection",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("regulatory_chunk_id", sa.Text(), nullable=False),
        sa.Column("user_file_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("derived_text_sha256", sa.String(length=64), nullable=True),
        sa.Column("text_snapshot", sa.Text(), nullable=False),
        sa.Column("source_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column(
            "labels",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "provenance",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "resolution", sa.String(length=32), server_default="pending", nullable=False
        ),
        sa.Column("unresolved_reason", sa.String(length=256), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "resolution IN ('pending', 'lineage', 'legacy_containment', 'unresolved')",
            name="regulatory_derived_label_projection_resolution_check",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"], ["regulatory_labeling_run.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "run_id",
            "regulatory_chunk_id",
            name="uq_regulatory_derived_label_projection_run_chunk",
        ),
    )
    op.create_index(
        "ix_regulatory_derived_label_projection_pending",
        "regulatory_derived_label_projection",
        ["run_id", "user_file_id", "regulatory_chunk_id"],
        postgresql_where=sa.text("resolution = 'pending'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_regulatory_derived_label_projection_pending",
        table_name="regulatory_derived_label_projection",
    )
    op.drop_table("regulatory_derived_label_projection")
    op.drop_index(
        "ix_regulatory_labeling_item_unprepared_position",
        table_name="regulatory_labeling_item",
    )
    op.drop_index(
        "ix_regulatory_labeling_item_run_status",
        table_name="regulatory_labeling_item",
    )
    op.drop_table("regulatory_labeling_item")
    op.drop_index(
        "ix_regulatory_labeling_shard_due",
        table_name="regulatory_labeling_shard",
    )
    op.drop_table("regulatory_labeling_shard")
    op.drop_index(
        "uq_regulatory_labeling_run_active_document_set",
        table_name="regulatory_labeling_run",
    )
    op.drop_index(
        "ix_regulatory_labeling_run_recovery",
        table_name="regulatory_labeling_run",
    )
    op.drop_table("regulatory_labeling_run")
    op.drop_table("regulatory_label_taxonomy")
