"""scoped intake metadata and background labels

Revision ID: 9ac886f432f4
Revises: 5d34c09ad891
Create Date: 2026-11-10 14:31:50.686326

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "9ac886f432f4"
down_revision = "5d34c09ad891"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_document_intake",
        sa.Column(
            "background_labeling_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column("metadata_extraction_mode", sa.String(32), nullable=True),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column("background_label_status", sa.String(32), nullable=True),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column("background_label_error", sa.Text(), nullable=True),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column("background_label_source_sha256", sa.String(64), nullable=True),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column(
            "background_label_attempts",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "regulatory_document_intake",
        sa.Column(
            "background_label_next_retry_at", sa.DateTime(timezone=True), nullable=True
        ),
    )
    op.create_check_constraint(
        "ck_intake_metadata_extraction_mode",
        "regulatory_document_intake",
        "metadata_extraction_mode IS NULL OR metadata_extraction_mode = 'scoped_v1'",
    )
    op.create_check_constraint(
        "ck_intake_background_label_status",
        "regulatory_document_intake",
        "background_label_status IS NULL OR background_label_status IN ('pending','waiting','running','completed','failed','cancelled')",
    )
    op.create_check_constraint(
        "ck_intake_background_label_attempts",
        "regulatory_document_intake",
        "background_label_attempts BETWEEN 0 AND 3",
    )


def downgrade() -> None:
    for name in (
        "ck_intake_background_label_attempts",
        "ck_intake_background_label_status",
        "ck_intake_metadata_extraction_mode",
    ):
        op.drop_constraint(name, "regulatory_document_intake", type_="check")
    for name in (
        "background_labeling_enabled",
        "background_label_next_retry_at",
        "background_label_attempts",
        "background_label_source_sha256",
        "background_label_error",
        "background_label_status",
        "metadata_extraction_mode",
    ):
        op.drop_column("regulatory_document_intake", name)
