"""Add workflow to chat_message

Records which research workflow produced an assistant answer so it can be shown
under the answer when the chat is reopened.

Revision ID: b7d3e9a1c4f2
Revises: e5fc62d469ab
Create Date: 2026-10-14 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

revision = "b7d3e9a1c4f2"
down_revision = "e5fc62d469ab"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("chat_message", sa.Column("workflow", sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column("chat_message", "workflow")
