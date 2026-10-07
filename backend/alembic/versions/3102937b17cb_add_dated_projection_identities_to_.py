"""Add dated projection identities to durable indexing items

Revision ID: 3102937b17cb
Revises: 1fa78a83fa88
Create Date: 2026-09-10 17:27:06.709919

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "3102937b17cb"
down_revision = "1fa78a83fa88"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_indexing_item", sa.Column("projection_id", sa.Uuid(), nullable=True)
    )
    op.add_column(
        "regulatory_indexing_item",
        sa.Column("projection_ordinal", sa.Integer(), nullable=True),
    )
    op.add_column(
        "regulatory_indexing_item",
        sa.Column("effective_start", sa.Date(), nullable=True),
    )
    op.add_column(
        "regulatory_indexing_item", sa.Column("effective_end", sa.Date(), nullable=True)
    )
    op.add_column(
        "regulatory_indexing_item",
        sa.Column("projection_input", postgresql.JSONB(), nullable=True),
    )
    op.drop_constraint(
        "uq_regulatory_indexing_item_job_chunk",
        "regulatory_indexing_item",
        type_="unique",
    )
    op.create_index(
        "uq_regulatory_indexing_item_job_chunk",
        "regulatory_indexing_item",
        ["job_id", "regulatory_chunk_id"],
        unique=True,
        postgresql_where=sa.text("projection_id IS NULL"),
    )
    op.create_unique_constraint(
        "uq_regulatory_indexing_item_job_projection",
        "regulatory_indexing_item",
        ["job_id", "projection_id"],
    )
    op.create_check_constraint(
        "regulatory_indexing_item_projection_identity_check",
        "regulatory_indexing_item",
        "(projection_id IS NULL AND projection_ordinal IS NULL AND projection_input IS NULL) OR "
        "(projection_id IS NOT NULL AND projection_ordinal >= 0 AND projection_input IS NOT NULL)",
    )


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM regulatory_indexing_item WHERE projection_id IS NOT NULL) THEN
      RAISE EXCEPTION 'dated durable items require a data-preserving migration';
    END IF; END $$""")
    op.drop_constraint(
        "regulatory_indexing_item_projection_identity_check",
        "regulatory_indexing_item",
        type_="check",
    )
    op.drop_constraint(
        "uq_regulatory_indexing_item_job_projection",
        "regulatory_indexing_item",
        type_="unique",
    )
    op.drop_index(
        "uq_regulatory_indexing_item_job_chunk", table_name="regulatory_indexing_item"
    )
    op.create_unique_constraint(
        "uq_regulatory_indexing_item_job_chunk",
        "regulatory_indexing_item",
        ["job_id", "regulatory_chunk_id"],
    )
    for column in (
        "projection_input",
        "effective_end",
        "effective_start",
        "projection_ordinal",
        "projection_id",
    ):
        op.drop_column("regulatory_indexing_item", column)
