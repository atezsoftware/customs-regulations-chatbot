"""store external chat session user and api key identity

Revision ID: 6f902740eafa
Revises: b7d3e9a1c4f2
Create Date: 2026-10-14 14:03:50.927100

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "6f902740eafa"
down_revision = "b7d3e9a1c4f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "chat_session", sa.Column("external_user_name", sa.Text(), nullable=True)
    )
    op.add_column("chat_session", sa.Column("api_key_id", sa.Integer(), nullable=True))
    op.add_column("chat_session", sa.Column("api_key_name", sa.Text(), nullable=True))
    op.add_column(
        "chat_session", sa.Column("api_key_display", sa.String(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("chat_session", "api_key_display")
    op.drop_column("chat_session", "api_key_name")
    op.drop_column("chat_session", "api_key_id")
    op.drop_column("chat_session", "external_user_name")
