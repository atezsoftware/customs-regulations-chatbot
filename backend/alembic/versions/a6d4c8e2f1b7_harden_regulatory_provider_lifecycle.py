"""harden regulatory provider lifecycle

Revision ID: a6d4c8e2f1b7
Revises: f4e8a2c6d1b3
Create Date: 2026-08-20
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a6d4c8e2f1b7"
down_revision = "f4e8a2c6d1b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "regulatory_indexing_job_cancellation_intent_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_cancellation_intent_check",
        "regulatory_indexing_job",
        "cancellation_intent IN ('NONE', 'USER_CANCEL', 'USER_DELETE', "
        "'SUPERSEDE', 'TERMINAL_FAILURE')",
    )
    job_columns = (
        sa.Column(
            "vertex_submission_attempt_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "vertex_submission_charged",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "vertex_reconcile_miss_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column("vertex_reconcile_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "provider_cleanup_state",
            sa.String(32),
            server_default="NONE",
            nullable=False,
        ),
        sa.Column(
            "provider_cleanup_phase",
            sa.String(32),
            server_default="NONE",
            nullable=False,
        ),
        sa.Column(
            "provider_cleanup_attempt_count",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "provider_cleanup_generation",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "provider_cleanup_token", postgresql.UUID(as_uuid=True), nullable=True
        ),
        sa.Column(
            "provider_cleanup_next_retry_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "provider_cleanup_heartbeat_at", sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            "provider_cleanup_had_failure",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("provider_cleanup_error_code", sa.String(128), nullable=True),
        sa.Column("provider_cleanup_error_message", sa.String(4000), nullable=True),
        sa.Column(
            "provider_cleanup_completed_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    for column in job_columns:
        op.add_column("regulatory_indexing_job", column)
    op.add_column(
        "regulatory_indexing_item",
        sa.Column(
            "context_attempt_count", sa.Integer(), server_default="0", nullable=False
        ),
    )
    op.create_check_constraint(
        "regulatory_indexing_job_vertex_attempts_check",
        "regulatory_indexing_job",
        "vertex_submission_attempt_count >= 0 AND vertex_reconcile_miss_count >= 0",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_provider_cleanup_state_check",
        "regulatory_indexing_job",
        "provider_cleanup_state IN ('NONE', 'PENDING', 'RUNNING', 'RETRY_WAIT', "
        "'SUCCEEDED', 'EXHAUSTED')",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_provider_cleanup_phase_check",
        "regulatory_indexing_job",
        "provider_cleanup_phase IN ('NONE', 'VERTEX_CANCEL', 'VERTEX_RECONCILE', "
        "'VERTEX_DELETE', 'GCS_CLEANUP', 'COMPLETE')",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_provider_cleanup_attempts_check",
        "regulatory_indexing_job",
        "provider_cleanup_attempt_count >= 0 AND provider_cleanup_generation >= 0",
    )
    op.create_check_constraint(
        "regulatory_indexing_item_context_attempt_count_check",
        "regulatory_indexing_item",
        "context_attempt_count >= 0",
    )
    op.create_index(
        "ix_regulatory_indexing_job_provider_cleanup",
        "regulatory_indexing_job",
        [
            "provider_cleanup_state",
            "provider_cleanup_next_retry_at",
            "provider_cleanup_heartbeat_at",
        ],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_regulatory_indexing_job_provider_cleanup",
        table_name="regulatory_indexing_job",
    )
    op.drop_constraint(
        "regulatory_indexing_item_context_attempt_count_check",
        "regulatory_indexing_item",
        type_="check",
    )
    op.drop_constraint(
        "regulatory_indexing_job_provider_cleanup_attempts_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_constraint(
        "regulatory_indexing_job_provider_cleanup_phase_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_constraint(
        "regulatory_indexing_job_provider_cleanup_state_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_constraint(
        "regulatory_indexing_job_vertex_attempts_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_column("regulatory_indexing_item", "context_attempt_count")
    for column_name in (
        "provider_cleanup_completed_at",
        "provider_cleanup_error_message",
        "provider_cleanup_error_code",
        "provider_cleanup_had_failure",
        "provider_cleanup_heartbeat_at",
        "provider_cleanup_next_retry_at",
        "provider_cleanup_token",
        "provider_cleanup_generation",
        "provider_cleanup_attempt_count",
        "provider_cleanup_phase",
        "provider_cleanup_state",
        "vertex_reconcile_until",
        "vertex_reconcile_miss_count",
        "vertex_submission_charged",
        "vertex_submission_attempt_count",
    ):
        op.drop_column("regulatory_indexing_job", column_name)
    op.drop_constraint(
        "regulatory_indexing_job_cancellation_intent_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.create_check_constraint(
        "regulatory_indexing_job_cancellation_intent_check",
        "regulatory_indexing_job",
        "cancellation_intent IN ('NONE', 'USER_CANCEL', 'USER_DELETE', 'SUPERSEDE')",
    )
