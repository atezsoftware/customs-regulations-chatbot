"""Add immutable amendment source packages.

Revision ID: 8ec758be0556
Revises: e1a7c4b9d2f6
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "8ec758be0556"
down_revision = "e1a7c4b9d2f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "amendment_source_package",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "document_set_id",
            sa.Integer(),
            sa.ForeignKey("document_set.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("environment", sa.Text(), nullable=False),
        sa.Column("idempotency_key", sa.Text(), nullable=False),
        sa.Column("request_hash", sa.Text(), nullable=False),
        sa.Column("input_spec", postgresql.JSONB(), nullable=False),
        sa.Column("input_file_id", sa.Text(), nullable=True),
        sa.Column("status", sa.Text(), nullable=False, server_default="processing"),
        sa.Column("asset_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column(
            "issues",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("manifest_file_id", sa.Text(), nullable=True),
        sa.Column("manifest_sha256", sa.Text(), nullable=True),
        sa.Column("lease_token", sa.Uuid(), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=True),
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
            "document_set_id",
            "environment",
            "idempotency_key",
            name="uq_amendment_source_request",
        ),
        sa.CheckConstraint(
            "status IN ('processing', 'ready', 'partial', 'blocked', 'failed')",
            name="amendment_source_status_check",
        ),
        sa.CheckConstraint(
            "asset_count >= 0 AND asset_count <= 21 AND total_bytes >= 0 AND total_bytes <= 104857600",
            name="amendment_source_limits_check",
        ),
    )
    op.create_index(
        "ix_amendment_source_package_document_set_id",
        "amendment_source_package",
        ["document_set_id"],
    )
    op.create_table(
        "regulatory_source_asset",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "package_id",
            sa.Uuid(),
            sa.ForeignKey("amendment_source_package.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("sha256", sa.Text(), nullable=False),
        sa.Column("file_id", sa.Text(), nullable=False),
        sa.Column("text_file_id", sa.Text(), nullable=True),
        sa.Column("text_sha256", sa.Text(), nullable=True),
        sa.Column("mime_type", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("byte_count", sa.BigInteger(), nullable=False),
        sa.Column("original_url", sa.Text(), nullable=True),
        sa.Column("final_url", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "package_id", "sha256", name="uq_regulatory_source_asset_hash"
        ),
        sa.CheckConstraint(
            "byte_count > 0 AND byte_count <= 26214400",
            name="regulatory_source_asset_size_check",
        ),
    )
    op.execute("""CREATE FUNCTION reject_regulatory_source_asset_update() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN
          RAISE EXCEPTION 'Regulatory source assets are immutable';
        END $$""")
    op.execute("""CREATE TRIGGER regulatory_source_asset_immutable BEFORE UPDATE
        ON regulatory_source_asset FOR EACH ROW EXECUTE FUNCTION reject_regulatory_source_asset_update()""")
    op.add_column(
        "amendment_batch", sa.Column("source_package_id", sa.Uuid(), nullable=True)
    )
    op.create_foreign_key(
        "fk_amendment_batch_source_package",
        "amendment_batch",
        "amendment_source_package",
        ["source_package_id"],
        ["id"],
    )
    op.add_column(
        "amendment_batch", sa.Column("source_text_sha256", sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column("amendment_batch", "source_text_sha256")
    op.drop_constraint(
        "fk_amendment_batch_source_package", "amendment_batch", type_="foreignkey"
    )
    op.drop_column("amendment_batch", "source_package_id")
    op.drop_table("regulatory_source_asset")
    op.execute("DROP FUNCTION reject_regulatory_source_asset_update()")
    op.drop_table("amendment_source_package")
