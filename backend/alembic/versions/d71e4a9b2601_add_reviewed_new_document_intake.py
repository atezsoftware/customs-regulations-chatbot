"""Add reviewed new document intake

Revision ID: d71e4a9b2601
Revises: f3b39e5c99db
Create Date: 2026-10-26 09:43:09.039764

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "d71e4a9b2601"
down_revision: str | None = "f3b39e5c99db"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "regulatory_document_intake",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "source_package_id",
            sa.Uuid(),
            sa.ForeignKey("amendment_source_package.id"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id"),
            nullable=False,
        ),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("idempotency_key", sa.Uuid(), nullable=False, unique=True),
        sa.Column(
            "labeling_model_configuration_id",
            sa.Integer(),
            sa.ForeignKey("model_configuration.id"),
            nullable=False,
        ),
        sa.Column(
            "labeling_run_id", sa.Uuid(), sa.ForeignKey("regulatory_labeling_run.id")
        ),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("stage", sa.Text(), nullable=False),
        sa.Column("review_sha256", sa.String(64)),
        sa.Column("error", sa.Text()),
        sa.Column("lease_token", sa.Uuid()),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_table(
        "regulatory_document_intake_file",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "intake_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "source_asset_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_source_asset.id"),
            nullable=False,
        ),
        sa.Column(
            "user_file_id",
            sa.Uuid(),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "parent_document_id",
            sa.Uuid(),
            sa.ForeignKey("regulatory_document_intake_file.id"),
        ),
        sa.Column("role", sa.Text(), nullable=False),
        sa.Column("relationships", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("approval_attempt_id", sa.Uuid()),
        sa.UniqueConstraint(
            "intake_id", "source_asset_id", name="uq_document_intake_asset"
        ),
    )
    op.create_index(
        "ix_regulatory_document_intake_file_intake_id",
        "regulatory_document_intake_file",
        ["intake_id"],
    )


def downgrade() -> None:
    op.drop_table("regulatory_document_intake_file")
    op.drop_table("regulatory_document_intake")
