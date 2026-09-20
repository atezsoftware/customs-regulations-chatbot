"""add multi chunk amendment proposals

Revision ID: e92f3a1c7b60
Revises: d81c4a2f9b30
Create Date: 2026-09-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e92f3a1c7b60"
down_revision: str | None = "d81c4a2f9b30"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "amendment_proposal",
        sa.Column(
            "chunk_changes",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )
    op.add_column(
        "amendment_proposal",
        sa.Column(
            "applied_new_chunk_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("amendment_proposal", "applied_new_chunk_ids")
    op.drop_column("amendment_proposal", "chunk_changes")
