"""Retain immutable evidence for unpublished intake publication recovery.

Revision ID: a42e9f0b2601
Revises: f31c8d7e2601
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision = "a42e9f0b2601"
down_revision = "f31c8d7e2601"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "regulatory_document_intake_publication_recovery",
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
        sa.Column("user_file_id", sa.Uuid(), nullable=False),
        sa.Column("previous_attempt_id", sa.Uuid(), nullable=False),
        sa.Column("replacement_attempt_id", sa.Uuid(), nullable=False, unique=True),
        sa.Column("approved_review_sha256", sa.String(64), nullable=False),
        sa.Column("canonical_sha256", sa.String(64), nullable=False),
        sa.Column("original_manifest_sha256", sa.String(64), nullable=False),
        sa.Column("original_manifest", postgresql.JSONB(), nullable=False),
        sa.Column("original_fencing_token", sa.BigInteger(), nullable=False),
        sa.Column("replacement_manifest_sha256", sa.String(64)),
        sa.Column("replacement_manifest", postgresql.JSONB(none_as_null=True)),
        sa.Column("status", sa.Text(), nullable=False, server_default="queued"),
        sa.Column("error", sa.Text()),
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
            "user_file_id",
            "previous_attempt_id",
            name="uq_intake_publication_recovery_previous",
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'preparing', 'prepared', 'completed', 'failed')",
            name="intake_publication_recovery_status_check",
        ),
        sa.CheckConstraint(
            "(replacement_manifest IS NULL) = (replacement_manifest_sha256 IS NULL)",
            name="intake_publication_recovery_replacement_pair",
        ),
    )
    op.create_index(
        "ix_intake_publication_recovery_intake_created",
        "regulatory_document_intake_publication_recovery",
        ["intake_id", "created_at"],
    )
    op.execute(
        """
        CREATE FUNCTION enforce_intake_publication_recovery_evidence()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                RAISE EXCEPTION 'intake publication recovery evidence is retained';
            END IF;
            IF (to_jsonb(NEW) - ARRAY[
                'status', 'error', 'updated_at',
                'replacement_manifest', 'replacement_manifest_sha256'
            ]) IS DISTINCT FROM (to_jsonb(OLD) - ARRAY[
                'status', 'error', 'updated_at',
                'replacement_manifest', 'replacement_manifest_sha256'
            ]) THEN
                RAISE EXCEPTION 'intake publication recovery source evidence is immutable';
            END IF;
            IF OLD.replacement_manifest IS NOT NULL AND (
                NEW.replacement_manifest IS DISTINCT FROM OLD.replacement_manifest
                OR NEW.replacement_manifest_sha256 IS DISTINCT FROM OLD.replacement_manifest_sha256
            ) THEN
                RAISE EXCEPTION 'intake publication recovery replacement is immutable';
            END IF;
            RETURN NEW;
        END
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER intake_publication_recovery_evidence_immutable
        BEFORE UPDATE OR DELETE
        ON regulatory_document_intake_publication_recovery
        FOR EACH ROW EXECUTE FUNCTION enforce_intake_publication_recovery_evidence()
        """
    )


def downgrade() -> None:
    op.drop_table("regulatory_document_intake_publication_recovery")
    op.execute("DROP FUNCTION enforce_intake_publication_recovery_evidence()")
