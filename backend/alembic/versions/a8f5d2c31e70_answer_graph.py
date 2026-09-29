"""Persist per-answer execution graphs.

Revision ID: a8f5d2c31e70
Revises: f6a1c9d27b40
Create Date: 2026-09-29
"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "a8f5d2c31e70"
down_revision: str | None = "f6a1c9d27b40"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "answer_graph_run",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column(
            "assistant_message_id",
            sa.Integer(),
            sa.ForeignKey("chat_message.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "user_message_id",
            sa.Integer(),
            sa.ForeignKey("chat_message.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "chat_session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_session.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("trace_id", sa.String(), nullable=False, unique=True),
        sa.Column("model_name", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("capture_status", sa.String(), nullable=False),
        sa.Column("capture_error", sa.Text(), nullable=True),
        sa.Column("final_answer_sha256", sa.String(), nullable=True),
        sa.Column(
            "time_created",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("time_finished", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_answer_graph_run_session",
        "answer_graph_run",
        ["chat_session_id", "time_created"],
    )
    op.create_table(
        "answer_graph_node",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(), nullable=False),
        sa.Column("parent_node_id", sa.String(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("operation", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attributes", postgresql.JSONB(), nullable=False),
        sa.Column("has_input", sa.Boolean(), nullable=False),
        sa.Column("has_output", sa.Boolean(), nullable=False),
        sa.Column("has_reasoning", sa.Boolean(), nullable=False),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("capture_status", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["answer_graph_run.id"], ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("run_id", "node_id"),
    )
    op.create_index(
        "ix_answer_graph_node_run_started",
        "answer_graph_node",
        ["run_id", "started_at"],
    )
    op.create_table(
        "answer_graph_payload",
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(), nullable=False),
        sa.Column("part", sa.String(), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id", "node_id"],
            ["answer_graph_node.run_id", "answer_graph_node.node_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("run_id", "node_id", "part"),
    )
    op.create_table(
        "answer_graph_edge",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("from_node_id", sa.String(), nullable=False),
        sa.Column("to_node_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["run_id"], ["answer_graph_run.id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("run_id", "from_node_id", "to_node_id", "kind"),
    )
    op.create_index("ix_answer_graph_edge_run", "answer_graph_edge", ["run_id"])
    op.create_table(
        "answer_graph_access_event",
        sa.Column("id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("node_id", sa.String(), nullable=True),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column(
            "time_created",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_answer_graph_access_event_run",
        "answer_graph_access_event",
        ["run_id", "time_created"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_answer_graph_access_event_run", table_name="answer_graph_access_event"
    )
    op.drop_table("answer_graph_access_event")
    op.drop_index("ix_answer_graph_edge_run", table_name="answer_graph_edge")
    op.drop_table("answer_graph_edge")
    op.drop_table("answer_graph_payload")
    op.drop_index("ix_answer_graph_node_run_started", table_name="answer_graph_node")
    op.drop_table("answer_graph_node")
    op.drop_index("ix_answer_graph_run_session", table_name="answer_graph_run")
    op.drop_table("answer_graph_run")
