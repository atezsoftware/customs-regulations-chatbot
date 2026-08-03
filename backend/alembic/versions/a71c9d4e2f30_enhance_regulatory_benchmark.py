"""enhance regulatory benchmark

Revision ID: a71c9d4e2f30
Revises: 4b8e9f2a1c7d
Create Date: 2026-07-31
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "a71c9d4e2f30"
down_revision = "4b8e9f2a1c7d"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "benchmark_question",
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
    )
    op.add_column(
        "benchmark_question", sa.Column("as_of_date", sa.Date(), nullable=True)
    )
    op.add_column(
        "benchmark_question",
        sa.Column(
            "expected_citations",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )

    op.add_column(
        "benchmark_run", sa.Column("report", postgresql.JSONB(), nullable=True)
    )
    op.add_column("benchmark_run", sa.Column("report_error", sa.Text(), nullable=True))
    op.add_column(
        "benchmark_run", sa.Column("report_input_tokens", sa.Integer(), nullable=True)
    )
    op.add_column(
        "benchmark_run", sa.Column("report_output_tokens", sa.Integer(), nullable=True)
    )
    op.add_column(
        "benchmark_run", sa.Column("report_cost_cents", sa.Float(), nullable=True)
    )

    op.add_column(
        "benchmark_run_item",
        sa.Column(
            "question_snapshot",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column(
            "cited_sources",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column(
            "execution_steps",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column(
            "llm_calls",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "benchmark_run_item", sa.Column("answer_reasoning", sa.Text(), nullable=True)
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column(
            "chat_session_id",
            sa.Uuid(),
            sa.ForeignKey("chat_session.id", ondelete="SET NULL"),
            nullable=True,
        ),
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column("assistant_message_id", sa.Integer(), nullable=True),
    )
    op.add_column(
        "benchmark_run_item", sa.Column("citation_recall", sa.Float(), nullable=True)
    )
    op.add_column(
        "benchmark_run_item",
        sa.Column("citation_precision", sa.Float(), nullable=True),
    )
    op.add_column(
        "benchmark_run_item", sa.Column("judge_error", sa.Text(), nullable=True)
    )

    op.add_column(
        "benchmark_run_judgment",
        sa.Column(
            "report",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.add_column(
        "benchmark_run_judgment",
        sa.Column("input_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "benchmark_run_judgment",
        sa.Column("output_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "benchmark_run_judgment",
        sa.Column("total_tokens", sa.Integer(), nullable=True),
    )
    op.add_column(
        "benchmark_run_judgment",
        sa.Column("cost_cents", sa.Float(), nullable=True),
    )
    op.add_column(
        "benchmark_run_judgment",
        sa.Column(
            "cost_source",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'unavailable'"),
        ),
    )
    op.create_check_constraint(
        "benchmark_judgment_cost_source_check",
        "benchmark_run_judgment",
        "cost_source IN ('measured', 'unavailable')",
    )


def downgrade() -> None:
    op.drop_constraint(
        "benchmark_judgment_cost_source_check",
        "benchmark_run_judgment",
        type_="check",
    )
    for column in (
        "cost_source",
        "cost_cents",
        "total_tokens",
        "output_tokens",
        "input_tokens",
        "report",
    ):
        op.drop_column("benchmark_run_judgment", column)
    for column in (
        "judge_error",
        "citation_precision",
        "citation_recall",
        "assistant_message_id",
        "chat_session_id",
        "answer_reasoning",
        "llm_calls",
        "execution_steps",
        "cited_sources",
        "question_snapshot",
    ):
        op.drop_column("benchmark_run_item", column)
    for column in (
        "report_cost_cents",
        "report_output_tokens",
        "report_input_tokens",
        "report_error",
        "report",
    ):
        op.drop_column("benchmark_run", column)
    for column in ("expected_citations", "as_of_date", "title"):
        op.drop_column("benchmark_question", column)
