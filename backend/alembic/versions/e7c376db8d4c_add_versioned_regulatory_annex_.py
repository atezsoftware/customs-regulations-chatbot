"""Add versioned regulatory annex structures.

Revision ID: e7c376db8d4c
Revises: 8ec758be0556
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "e7c376db8d4c"
down_revision = "8ec758be0556"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_annex",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_file_id",
            sa.Uuid(),
            sa.ForeignKey("user_file.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("label", sa.Text(), nullable=False),
        sa.Column("latest_approved_revision_id", sa.Uuid(), nullable=True),
        sa.UniqueConstraint(
            "document_set_id", "user_file_id", "label", name="uq_regulatory_annex_scope"
        ),
    )
    op.create_table(
        "regulatory_annex_revision",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "annex_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_annex.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "predecessor_revision_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_annex_revision.id"),
            nullable=True,
        ),
        sa.Column(
            "source_asset_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_source_asset.id"),
            nullable=True,
        ),
        sa.Column("baseline_sha256", sa.Text(), nullable=False),
        sa.Column("snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("effective_start", sa.Date(), nullable=True),
        sa.Column("effective_end", sa.Date(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "annex_id", "baseline_sha256", name="uq_annex_revision_baseline"
        ),
        sa.CheckConstraint(
            "effective_end IS NULL OR effective_start IS NULL OR effective_end > effective_start",
            name="annex_revision_dates_check",
        ),
    )
    op.create_index(
        "ix_annex_revision_effective",
        "regulatory_annex_revision",
        ["annex_id", "effective_start", "effective_end"],
    )
    op.create_table(
        "regulatory_annex_element",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "annex_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_annex.id", ondelete="CASCADE"),
            nullable=False,
        ),
    )
    op.create_table(
        "regulatory_annex_revision_element",
        sa.Column(
            "revision_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_annex_revision.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "element_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_annex_element.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.UniqueConstraint(
            "revision_id", "position", name="uq_annex_revision_element_position"
        ),
    )
    op.create_table(
        "regulatory_annex_element_chunk",
        sa.Column("revision_id", sa.Uuid(), primary_key=True),
        sa.Column("element_id", sa.Uuid(), primary_key=True),
        sa.Column(
            "chunk_id",
            sa.Text(),
            sa.ForeignKey("regulatory_chunk.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.ForeignKeyConstraint(
            ["revision_id", "element_id"],
            [
                "regulatory_annex_revision_element.revision_id",
                "regulatory_annex_revision_element.element_id",
            ],
            ondelete="CASCADE",
        ),
    )

    op.execute("""CREATE FUNCTION reject_annex_revision_content_update() RETURNS trigger AS $$
    BEGIN
      IF NEW.snapshot IS DISTINCT FROM OLD.snapshot OR NEW.baseline_sha256 IS DISTINCT FROM OLD.baseline_sha256 OR NEW.annex_id IS DISTINCT FROM OLD.annex_id OR NEW.source_asset_id IS DISTINCT FROM OLD.source_asset_id OR NEW.predecessor_revision_id IS DISTINCT FROM OLD.predecessor_revision_id THEN
        RAISE EXCEPTION 'annex revision content is immutable';
      END IF;
      RETURN NEW;
    END; $$ LANGUAGE plpgsql""")
    op.execute(
        "CREATE TRIGGER immutable_annex_revision BEFORE UPDATE ON regulatory_annex_revision FOR EACH ROW EXECUTE FUNCTION reject_annex_revision_content_update()"
    )
    op.execute("""CREATE FUNCTION reject_annex_element_update() RETURNS trigger AS $$
    BEGIN RAISE EXCEPTION 'annex revision elements are immutable'; END;
    $$ LANGUAGE plpgsql""")
    op.execute(
        "CREATE TRIGGER immutable_annex_element BEFORE UPDATE ON regulatory_annex_revision_element FOR EACH ROW EXECUTE FUNCTION reject_annex_element_update()"
    )


def downgrade() -> None:
    for table in (
        "regulatory_annex_element_chunk",
        "regulatory_annex_revision_element",
        "regulatory_annex_element",
        "regulatory_annex_revision",
        "regulatory_annex",
    ):
        op.drop_table(table)

    op.execute("DROP FUNCTION IF EXISTS reject_annex_revision_content_update()")
    op.execute("DROP FUNCTION IF EXISTS reject_annex_element_update()")
