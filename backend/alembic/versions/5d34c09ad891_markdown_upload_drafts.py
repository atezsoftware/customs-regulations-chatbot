"""Store owner-scoped Markdown drafts before explicit preparation.

Revision ID: 5d34c09ad891
Revises: 0bc7d32e5fa1
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "5d34c09ad891"
down_revision = "0bc7d32e5fa1"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_markdown_draft",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user.id"),
            nullable=False,
        ),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("idempotency_key", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("revision", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.Text(), nullable=False, server_default="draft"),
        sa.Column(
            "source_package_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("amendment_source_package.id"),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "created_by",
            "document_set_id",
            "environment",
            "idempotency_key",
            name="uq_markdown_draft_request",
        ),
        sa.CheckConstraint(
            "revision >= 0 AND status IN ('draft', 'prepared')",
            name="markdown_draft_status_check",
        ),
        sa.CheckConstraint(
            "(status = 'prepared') = (source_package_id IS NOT NULL)",
            name="markdown_draft_prepared_check",
        ),
    )
    op.create_index(
        "ix_markdown_draft_owner",
        "regulatory_markdown_draft",
        ["created_by", "document_set_id", "environment"],
    )
    op.create_table(
        "regulatory_markdown_draft_document",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "draft_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("regulatory_markdown_draft.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parent_document_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("file_id", sa.Text(), nullable=True),
        sa.Column("byte_count", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("removed", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "draft_id", "id", name="uq_markdown_draft_document_identity"
        ),
        sa.ForeignKeyConstraint(
            ["draft_id", "parent_document_id"],
            [
                "regulatory_markdown_draft_document.draft_id",
                "regulatory_markdown_draft_document.id",
            ],
        ),
        sa.CheckConstraint(
            "byte_count > 0 AND byte_count <= 2097152",
            name="markdown_draft_document_size_check",
        ),
    )
    op.create_index(
        "ix_markdown_draft_document_draft",
        "regulatory_markdown_draft_document",
        ["draft_id"],
    )
    op.create_table(
        "regulatory_markdown_draft_upload",
        sa.Column(
            "draft_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("regulatory_markdown_draft.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("idempotency_key", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("resulting_revision", sa.Integer(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("regulatory_markdown_draft_upload")
    op.drop_table("regulatory_markdown_draft_document")
    op.drop_table("regulatory_markdown_draft")
