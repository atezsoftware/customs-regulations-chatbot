"""add regulatory benchmark tables

Revision ID: 4b8e9f2a1c7d
Revises: 9ec902e08372
Create Date: 2026-07-31
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "4b8e9f2a1c7d"
down_revision = "9ec902e08372"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "benchmark_question",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("reference_answer", sa.Text(), nullable=True),
        sa.Column(
            "expected_facts",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("rubric_notes", sa.Text(), nullable=True),
        sa.Column(
            "tags",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "project_id",
            sa.Integer(),
            sa.ForeignKey("user_project.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column(
            "created_by",
            sa.Uuid(),
            sa.ForeignKey("user.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "updated_by",
            sa.Uuid(),
            sa.ForeignKey("user.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_index("ix_benchmark_question_active", "benchmark_question", ["is_active"])
    op.create_index(
        "ix_benchmark_question_project_id", "benchmark_question", ["project_id"]
    )

    op.create_table(
        "benchmark_run",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("judge_provider", sa.Text(), nullable=False),
        sa.Column("judge_model", sa.Text(), nullable=False),
        sa.Column(
            "deep_research",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
        sa.Column(
            "created_by",
            sa.Uuid(),
            sa.ForeignKey("user.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("total_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failed_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'error', 'cancelled')",
            name="benchmark_run_status_check",
        ),
    )
    op.create_index("ix_benchmark_run_status", "benchmark_run", ["status"])

    op.create_table(
        "benchmark_run_item",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_run.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.Text(), nullable=False),
        sa.Column("model_id", sa.Text(), nullable=False),
        sa.Column(
            "question_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_question.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'pending'"),
        ),
        sa.Column("final_result", sa.Text(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("total_tokens", sa.Integer(), nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("cost_cents", sa.Float(), nullable=True),
        sa.Column(
            "cost_source",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'unavailable'"),
        ),
        sa.Column(
            "cited_chunk_ids",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'running', 'completed', 'error', 'cancelled')",
            name="benchmark_run_item_status_check",
        ),
        sa.CheckConstraint(
            "cost_source IN ('measured', 'unavailable')",
            name="benchmark_run_item_cost_source_check",
        ),
    )
    op.create_index("ix_benchmark_run_item_run_id", "benchmark_run_item", ["run_id"])
    op.create_index(
        "ix_benchmark_run_item_question_id", "benchmark_run_item", ["question_id"]
    )

    op.create_table(
        "benchmark_run_judgment",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "run_item_id",
            sa.Integer(),
            sa.ForeignKey("benchmark_run_item.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("judge_provider", sa.Text(), nullable=False),
        sa.Column("judge_model", sa.Text(), nullable=False),
        sa.Column("correctness_score", sa.Integer(), nullable=False),
        sa.Column("groundedness_score", sa.Integer(), nullable=False),
        sa.Column("completeness_score", sa.Integer(), nullable=False),
        sa.Column("clarity_score", sa.Integer(), nullable=False),
        sa.Column("overall_score", sa.Integer(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "correctness_score BETWEEN 1 AND 5 AND "
            "groundedness_score BETWEEN 1 AND 5 AND "
            "completeness_score BETWEEN 1 AND 5 AND "
            "clarity_score BETWEEN 1 AND 5",
            name="benchmark_judgment_subscores_check",
        ),
        sa.CheckConstraint(
            "overall_score BETWEEN 0 AND 100",
            name="benchmark_judgment_overall_check",
        ),
    )


def downgrade() -> None:
    op.drop_table("benchmark_run_judgment")
    op.drop_table("benchmark_run_item")
    op.drop_table("benchmark_run")
    op.drop_table("benchmark_question")
