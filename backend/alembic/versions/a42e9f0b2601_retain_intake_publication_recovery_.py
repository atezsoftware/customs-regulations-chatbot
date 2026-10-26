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
    op.execute(
        """
        CREATE OR REPLACE FUNCTION protect_writer_manifest()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF OLD.writer_manifest IS NOT NULL AND NEW.writer_manifest IS NOT NULL
                AND (OLD.writer_manifest IS DISTINCT FROM NEW.writer_manifest
                    OR OLD.writer_manifest_sha256 IS DISTINCT FROM NEW.writer_manifest_sha256)
            THEN
                IF OLD.gate_closed AND NEW.gate_closed
                    AND OLD.user_file_id = NEW.user_file_id
                    AND OLD.scope_key = NEW.scope_key
                    AND OLD.owner_id = NEW.owner_id
                    AND OLD.fencing_token = NEW.fencing_token
                    AND NEW.epoch >= OLD.epoch
                    AND EXISTS (
                        SELECT 1
                        FROM regulatory_document_intake_publication_recovery r
                        JOIN regulatory_document_intake_file m ON m.id = r.document_id
                        JOIN regulatory_document_intake b ON b.id = r.intake_id
                        JOIN user_file_projection_repair p ON p.user_file_id = r.user_file_id
                        JOIN user_file f ON f.id = r.user_file_id
                        WHERE r.user_file_id = NEW.user_file_id
                            AND r.original_manifest = OLD.writer_manifest
                            AND r.original_manifest_sha256 = OLD.writer_manifest_sha256
                            AND r.replacement_manifest = NEW.writer_manifest
                            AND r.replacement_manifest_sha256 = NEW.writer_manifest_sha256
                            AND r.status = 'prepared'
                            AND r.original_fencing_token < NEW.fencing_token
                            AND r.original_manifest->>'kind' = 'reindex'
                            AND r.replacement_manifest->>'kind' = 'reindex'
                            AND r.original_manifest->>'canonical_before_sha256' = r.canonical_sha256
                            AND r.replacement_manifest->>'canonical_before_sha256' = r.canonical_sha256
                            AND m.intake_id = b.id AND m.user_file_id = r.user_file_id
                            AND m.approval_attempt_id = r.replacement_attempt_id
                            AND b.review_sha256 = r.approved_review_sha256
                            AND b.stage = 'indexing' AND b.status IN ('approving', 'failed')
                            AND p.attempt_id = r.replacement_attempt_id
                            AND p.status = 'RUNNING' AND p.initial_index
                            AND f.status = 'CHUNKED'
                            AND NOT EXISTS (
                                SELECT 1 FROM regulatory_temporal_projection t
                                WHERE t.user_file_id = NEW.user_file_id
                            )
                    )
                THEN
                    RETURN NEW;
                END IF;
                RAISE EXCEPTION 'pending writer manifest is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM regulatory_document_intake_publication_recovery)
            THEN RAISE EXCEPTION 'intake recovery evidence requires a data-preserving migration';
            END IF;
        END $$
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION protect_writer_manifest()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            IF OLD.writer_manifest IS NOT NULL AND NEW.writer_manifest IS NOT NULL
                AND (OLD.writer_manifest IS DISTINCT FROM NEW.writer_manifest
                    OR OLD.writer_manifest_sha256 IS DISTINCT FROM NEW.writer_manifest_sha256)
            THEN RAISE EXCEPTION 'pending writer manifest is immutable';
            END IF;
            RETURN NEW;
        END;
        $$
        """
    )
    op.drop_table("regulatory_document_intake_publication_recovery")
    op.execute("DROP FUNCTION enforce_intake_publication_recovery_evidence()")
