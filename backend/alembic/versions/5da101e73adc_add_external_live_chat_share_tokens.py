"""add external live chat share tokens

Revision ID: 5da101e73adc
Revises: 6f902740eafa
Create Date: 2026-10-14 15:54:55.762785

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "5da101e73adc"
down_revision = "6f902740eafa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "external_chat_share",
        sa.Column("token_hash", sa.String(64), primary_key=True),
        sa.Column("chat_session_id", sa.UUID(), nullable=False),
        sa.Column("created_by_user_id", sa.UUID(), nullable=False),
        sa.Column("api_key_id", sa.Integer(), nullable=False),
        sa.Column(
            "time_created",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["chat_session_id"], ["chat_session.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["user.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_external_chat_share_chat_session_id",
        "external_chat_share",
        ["chat_session_id"],
    )


def downgrade() -> None:
    op.drop_table("external_chat_share")
