"""add immutable regulatory projection ordinal

Revision ID: c9d4e7f1a2b6
Revises: b7e2c5f9a1d3
Create Date: 2026-08-31
"""

import sqlalchemy as sa
from alembic import op


revision = "c9d4e7f1a2b6"
down_revision = "b7e2c5f9a1d3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_chunk",
        sa.Column("projection_ordinal", sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE regulatory_chunk SET projection_ordinal = position "
        "WHERE source = 'indexed'"
    )
    op.execute(
        """
        UPDATE regulatory_chunk AS chunk
        SET projection_ordinal = 1000000000 + proposal.id
        FROM amendment_proposal AS proposal
        WHERE chunk.source = 'amendment'
          AND proposal.applied_new_chunk_id = chunk.id
        """
    )
    op.execute(
        """
        WITH unlinked_amendments AS (
            SELECT
                id,
                row_number() OVER (ORDER BY user_file_id, created_at, id) AS ordinal
            FROM regulatory_chunk
            WHERE source = 'amendment'
              AND projection_ordinal IS NULL
        )
        UPDATE regulatory_chunk AS chunk
        SET projection_ordinal = 2000000000 + unlinked.ordinal
        FROM unlinked_amendments AS unlinked
        WHERE chunk.id = unlinked.id
        """
    )
    op.alter_column(
        "regulatory_chunk",
        "projection_ordinal",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.create_unique_constraint(
        "uq_regulatory_chunk_file_projection_ordinal",
        "regulatory_chunk",
        ["user_file_id", "projection_ordinal"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_regulatory_chunk_file_projection_ordinal",
        "regulatory_chunk",
        type_="unique",
    )
    op.drop_column("regulatory_chunk", "projection_ordinal")
