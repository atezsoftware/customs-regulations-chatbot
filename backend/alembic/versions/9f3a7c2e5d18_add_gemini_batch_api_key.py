"""add Gemini Batch API key

Revision ID: 9f3a7c2e5d18
Revises: 8d19d521d9fa
Create Date: 2026-09-13 23:03:00.000000

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "9f3a7c2e5d18"
down_revision = "8d19d521d9fa"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "llm_provider",
        sa.Column("gemini_batch_api_key", sa.LargeBinary(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("llm_provider", "gemini_batch_api_key")
