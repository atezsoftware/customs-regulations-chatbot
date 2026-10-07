"""Add qualified temporal projection bindings

Revision ID: 3a5ed5495b72
Revises: 853737f76cd0
Create Date: 2026-09-10 04:22:33.257140

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "3a5ed5495b72"
down_revision = "853737f76cd0"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_temporal_projection",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "user_file_id", sa.UUID(), sa.ForeignKey("user_file.id"), nullable=False
        ),
        sa.Column(
            "canonical_chunk_id",
            sa.Text(),
            sa.ForeignKey("regulatory_chunk.id"),
            nullable=False,
        ),
        sa.Column("index_uuid", sa.Text(), nullable=False),
        sa.Column("index_identity_sha256", sa.Text(), nullable=False),
        sa.Column("projection_ordinal", sa.BigInteger(), nullable=False),
        sa.Column("effective_start", sa.Date(), nullable=True),
        sa.Column("effective_end", sa.Date(), nullable=True),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column(
            "published_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "user_file_id",
            "index_uuid",
            "projection_ordinal",
            name="uq_temporal_projection_ordinal",
        ),
        sa.CheckConstraint(
            "effective_end IS NULL OR effective_start IS NULL OR effective_end > effective_start",
            name="temporal_projection_dates_check",
        ),
    )
    op.create_index(
        "ix_temporal_projection_lookup",
        "regulatory_temporal_projection",
        [
            "canonical_chunk_id",
            "index_identity_sha256",
            "effective_start",
            "effective_end",
        ],
    )


def downgrade() -> None:
    op.drop_table("regulatory_temporal_projection")
