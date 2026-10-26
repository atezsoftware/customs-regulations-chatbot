"""add reviewed document lineage

Revision ID: e21b7c6d2601
Revises: d71e4a9b2601
Create Date: 2026-10-26 13:25:06.356407

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "e21b7c6d2601"
down_revision = "d71e4a9b2601"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_document_lineage_review",
        sa.Column(
            "intake_document_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake_file.id"),
            primary_key=True,
        ),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id"),
            nullable=False,
        ),
        sa.Column("predecessor_user_file_id", sa.Uuid(), nullable=False),
        sa.Column("successor_user_file_id", sa.Uuid(), nullable=False, unique=True),
        sa.Column("predecessor_title", sa.Text(), nullable=False),
        sa.Column("successor_title", sa.Text(), nullable=False),
        sa.Column("predecessor_sha256", sa.String(64), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("effective_date", sa.Date()),
        sa.Column("evidence", sa.Text()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("confirmed_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("frozen_at", sa.DateTime(timezone=True)),
        sa.Column("activated_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint(
            "predecessor_user_file_id <> successor_user_file_id",
            name="document_lineage_distinct_files_check",
        ),
        sa.CheckConstraint(
            "relation IN ('successor', 'repeals')",
            name="document_lineage_relation_check",
        ),
        sa.CheckConstraint(
            "status IN ('draft', 'frozen', 'active')",
            name="document_lineage_status_check",
        ),
        sa.CheckConstraint(
            "relation <> 'repeals' OR (effective_date IS NOT NULL AND evidence IS NOT NULL AND length(trim(evidence)) > 0)",
            name="document_lineage_repeal_evidence_check",
        ),
    )
    op.create_index(
        "ix_regulatory_document_lineage_review_predecessor_user_file_id",
        "regulatory_document_lineage_review",
        ["predecessor_user_file_id"],
    )


def downgrade() -> None:
    op.drop_table("regulatory_document_lineage_review")
