"""add immutable annex review revisions

Revision ID: fbb3b683c9d8
Revises: bde9f17c588a
Create Date: 2026-09-10 02:05:07.168345

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "fbb3b683c9d8"
down_revision = "bde9f17c588a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for column in ("source_parent_batch_id", "superseded_by_batch_id"):
        op.add_column(
            "amendment_batch",
            sa.Column(
                column, sa.Integer(), sa.ForeignKey("amendment_batch.id"), nullable=True
            ),
        )
    op.execute("ALTER TABLE annex_change_set DISABLE TRIGGER immutable_annex_review")
    op.add_column(
        "annex_change_set", sa.Column("logical_group_id", sa.Uuid(), nullable=True)
    )
    op.add_column(
        "annex_change_set",
        sa.Column("review_revision", sa.Integer(), nullable=False, server_default="1"),
    )
    op.execute("UPDATE annex_change_set SET logical_group_id = id")
    op.alter_column("annex_change_set", "logical_group_id", nullable=False)
    op.drop_constraint(
        "uq_annex_change_instruction", "annex_change_set", type_="unique"
    )
    op.create_unique_constraint(
        "uq_annex_review_revision",
        "annex_change_set",
        ["logical_group_id", "review_revision"],
    )
    op.create_index(
        "ix_annex_initial_instruction",
        "annex_change_set",
        ["batch_id", "instruction_index"],
        unique=True,
        postgresql_where=sa.text("review_revision = 1"),
    )
    op.execute("ALTER TABLE annex_change_set ENABLE TRIGGER immutable_annex_review")

    op.create_table(
        "annex_publication_intent",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "change_set_id",
            sa.Uuid(),
            sa.ForeignKey("annex_change_set.id"),
            nullable=False,
        ),
        sa.Column("logical_group_id", sa.Uuid(), nullable=False),
        sa.Column("review_revision", sa.Integer(), nullable=False),
        sa.Column("review_sha256", sa.Text(), nullable=False),
        sa.Column("publication_generation", sa.Integer(), nullable=False),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("database_identity", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "change_set_id",
            "publication_generation",
            name="uq_annex_publication_generation",
        ),
    )
    op.execute(
        "CREATE TRIGGER immutable_annex_review BEFORE UPDATE ON annex_publication_intent FOR EACH ROW EXECUTE FUNCTION reject_annex_review_update()"
    )


def downgrade() -> None:
    # Preserve immutable revision history: an operator must resolve edits before downgrade.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM annex_change_set WHERE review_revision > 1) THEN RAISE EXCEPTION 'review revisions must be retained'; END IF; END $$"
    )
    op.drop_column("amendment_batch", "superseded_by_batch_id")
    op.drop_column("amendment_batch", "source_parent_batch_id")
    op.drop_table("annex_publication_intent")
    op.drop_index("ix_annex_initial_instruction", table_name="annex_change_set")
    op.drop_constraint("uq_annex_review_revision", "annex_change_set", type_="unique")
    op.create_unique_constraint(
        "uq_annex_change_instruction",
        "annex_change_set",
        ["batch_id", "instruction_index"],
    )
    op.drop_column("annex_change_set", "review_revision")
    op.drop_column("annex_change_set", "logical_group_id")
