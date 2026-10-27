"""Retain explicit intake publication and existing-parent review decisions.

Revision ID: b63d8e1f2701
Revises: a42e9f0b2601
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "b63d8e1f2701"
down_revision = "a42e9f0b2601"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_document_intake_review_decision",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "intake_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake.id"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake_file.id"),
            nullable=False,
        ),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column(
            "previous_decision_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake_review_decision.id"),
        ),
        sa.Column("disposition", sa.Text(), nullable=False),
        sa.Column(
            "duplicate_of_document_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake_file.id"),
        ),
        sa.Column("existing_parent_file_id", sa.Uuid()),
        sa.Column("existing_parent_snapshot", postgresql.JSONB(none_as_null=True)),
        sa.Column("source_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("reviewed_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "document_id", "sequence", name="uq_intake_review_sequence"
        ),
        sa.CheckConstraint("sequence > 0", name="intake_review_sequence_positive"),
        sa.CheckConstraint(
            "(disposition = 'source_evidence' AND duplicate_of_document_id IS NOT NULL AND existing_parent_file_id IS NULL) OR (disposition = 'legislation' AND duplicate_of_document_id IS NULL)",
            name="intake_review_disposition_check",
        ),
        sa.CheckConstraint(
            "(existing_parent_file_id IS NULL) = (existing_parent_snapshot IS NULL)",
            name="intake_review_parent_snapshot_check",
        ),
    )
    op.create_index(
        "ix_regulatory_document_intake_review_decision_intake_id",
        "regulatory_document_intake_review_decision",
        ["intake_id"],
    )
    op.create_index(
        "ix_regulatory_document_intake_review_decision_document_id",
        "regulatory_document_intake_review_decision",
        ["document_id"],
    )
    op.execute("""
        CREATE FUNCTION retain_intake_publication_review() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'intake publication review decisions are immutable; append a new decision';
        END $$
    """)
    op.execute("""
        CREATE TRIGGER intake_publication_review_immutable
        BEFORE UPDATE OR DELETE ON regulatory_document_intake_review_decision
        FOR EACH ROW EXECUTE FUNCTION retain_intake_publication_review()
    """)


def downgrade() -> None:
    op.execute("""
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM regulatory_document_intake_review_decision) THEN
                RAISE EXCEPTION 'retained publication decisions prevent destructive downgrade';
            END IF;
        END $$
    """)
    op.execute(
        "DROP TRIGGER intake_publication_review_immutable ON regulatory_document_intake_review_decision"
    )
    op.execute("DROP FUNCTION retain_intake_publication_review()")
    op.drop_table("regulatory_document_intake_review_decision")
