"""Add fenced file publication ownership.

Revision ID: 853737f76cd0
Revises: fbb3b683c9d8
"""

from alembic import op
import sqlalchemy as sa

revision = "853737f76cd0"
down_revision = "fbb3b683c9d8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_publication_clock",
        sa.Column("scope_key", sa.Text(), primary_key=True),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
    )
    op.create_table(
        "regulatory_file_publication",
        sa.Column("user_file_id", sa.Uuid(), primary_key=True),
        sa.Column("scope_key", sa.Text(), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column("fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("gate_closed", sa.Boolean(), nullable=False),
        sa.Column("epoch", sa.BigInteger(), nullable=False),
        sa.Column("next_ordinal", sa.BigInteger(), nullable=False),
    )
    op.create_table(
        "regulatory_publication_ordinal",
        sa.Column(
            "user_file_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_file_publication.user_file_id"),
            primary_key=True,
        ),
        sa.Column("allocation_key", sa.Text(), primary_key=True),
        sa.Column("ordinal", sa.BigInteger(), nullable=False),
        sa.UniqueConstraint(
            "user_file_id", "ordinal", name="uq_publication_file_ordinal"
        ),
    )


def downgrade() -> None:
    op.drop_table("regulatory_publication_ordinal")
    op.drop_table("regulatory_file_publication")
    op.drop_table("regulatory_publication_clock")
