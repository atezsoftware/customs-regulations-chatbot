"""allow independent annex chunk selections

Revision ID: c6b70d871979
Revises: fbe4b3871f5f
Create Date: 2026-09-15 19:35:50.817559

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "c6b70d871979"
down_revision = "fbe4b3871f5f"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_index("ix_annex_initial_instruction", table_name="annex_change_set")
    op.create_index(
        "ix_annex_initial_instruction",
        "annex_change_set",
        ["batch_id", "instruction_index"],
        unique=True,
        postgresql_where=sa.text(
            "review_revision = 1 AND review_payload->>'selection_parent_id' IS NULL"
        ),
    )
    op.create_index(
        "ix_annex_selection_parent",
        "annex_change_set",
        [sa.text("(review_payload->>'selection_parent_id')")],
        postgresql_where=sa.text("review_payload->>'selection_parent_id' IS NOT NULL"),
    )


def downgrade() -> None:
    if op.get_bind().scalar(
        sa.text(
            "SELECT EXISTS (SELECT 1 FROM annex_change_set WHERE review_payload->>'selection_parent_id' IS NOT NULL)"
        )
    ):
        raise ValueError(
            "Cannot remove selection support while immutable selection history exists"
        )
    op.drop_index("ix_annex_selection_parent", table_name="annex_change_set")
    op.drop_index("ix_annex_initial_instruction", table_name="annex_change_set")
    op.create_index(
        "ix_annex_initial_instruction",
        "annex_change_set",
        ["batch_id", "instruction_index"],
        unique=True,
        postgresql_where=sa.text("review_revision = 1"),
    )
