"""Retain exact PDF renditions separately from canonical publication.

Revision ID: 2fcae483a930
Revises: c3b4ea5ee175
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "2fcae483a930"
down_revision = "c3b4ea5ee175"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_retained_pdf_evidence",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_file.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "raw_file_id",
            sa.String(),
            sa.ForeignKey("file_record.file_id"),
            nullable=False,
            unique=True,
        ),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("byte_count", sa.BigInteger(), nullable=False),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("original_filename", sa.Text(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("canonical_sha256", sa.Text(), nullable=False),
        sa.Column("canonical_generation_hash", sa.Text(), nullable=True),
        sa.Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("request_sha256", sa.Text(), nullable=False),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "user_file_id", "sha256", name="uq_retained_pdf_target_hash"
        ),
        sa.UniqueConstraint(
            "user_file_id",
            "created_by",
            "idempotency_key",
            name="uq_retained_pdf_request",
        ),
        sa.CheckConstraint(
            "byte_count BETWEEN 1 AND 26214400", name="retained_pdf_size"
        ),
        sa.CheckConstraint("mime_type = 'application/pdf'", name="retained_pdf_mime"),
        sa.CheckConstraint(
            "relation IN ('equivalent_rendition', 'historical_rendition')",
            name="retained_pdf_relation",
        ),
    )
    op.create_index(
        "ix_regulatory_retained_pdf_evidence_user_file_id",
        "regulatory_retained_pdf_evidence",
        ["user_file_id"],
    )


def downgrade() -> None:
    op.drop_table("regulatory_retained_pdf_evidence")
