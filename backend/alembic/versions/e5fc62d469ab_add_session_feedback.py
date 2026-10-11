"""add session feedback

Revision ID: e5fc62d469ab
Revises: 4ecf8032a916
Create Date: 2026-10-11 21:20:09.320685

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e5fc62d469ab"
down_revision = "4ecf8032a916"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "chat_session_feedback",
        sa.Column("chat_session_id", sa.UUID(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("feedback_text", sa.Text(), nullable=True),
        sa.Column("submitted_by_user_id", sa.UUID(), nullable=False),
        sa.Column("evaluated_through_message_id", sa.Integer(), nullable=True),
        sa.Column("version", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column(
            "time_created",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "time_updated",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "rating BETWEEN 1 AND 4", name="session_feedback_rating_check"
        ),
        sa.CheckConstraint("version >= 1", name="session_feedback_version_check"),
        sa.CheckConstraint(
            "char_length(feedback_text) <= 5000",
            name="session_feedback_text_length_check",
        ),
        sa.ForeignKeyConstraint(
            ["chat_session_id"], ["chat_session.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["submitted_by_user_id"], ["user.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["evaluated_through_message_id"], ["chat_message.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("chat_session_id"),
    )


def downgrade() -> None:
    op.drop_table("chat_session_feedback")
