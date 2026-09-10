"""add durable annex publication execution

Revision ID: 8c88cdd6317e
Revises: 3a5ed5495b72
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "8c88cdd6317e"
down_revision = "3a5ed5495b72"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_temporal_projection",
        sa.Column("retired_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.drop_constraint(
        "uq_temporal_projection_ordinal",
        "regulatory_temporal_projection",
        type_="unique",
    )
    op.create_index(
        "uq_temporal_projection_ordinal",
        "regulatory_temporal_projection",
        ["user_file_id", "index_uuid", "projection_ordinal"],
        unique=True,
        postgresql_where=sa.text("retired_at IS NULL"),
    )
    op.create_table(
        "annex_publication_manifest",
        sa.Column(
            "change_set_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("annex_change_set.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column(
            "first_intent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("annex_publication_intent.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("payload_sha256", sa.Text(), nullable=False),
        sa.Column("source_history", postgresql.JSONB(), nullable=False),
        sa.Column("operations", postgresql.JSONB(), nullable=True),
        sa.Column("operations_sha256", sa.Text(), nullable=True),
        sa.Column(
            "es_started", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.create_table(
        "annex_publication_embedding",
        sa.Column(
            "change_set_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "annex_publication_manifest.change_set_id", ondelete="CASCADE"
            ),
            primary_key=True,
        ),
        sa.Column("projection_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("request_sha256", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("remote_id", sa.Text(), nullable=True),
        sa.Column("vectors", postgresql.JSONB(), nullable=True),
        sa.Column("vectors_sha256", sa.Text(), nullable=True),
        sa.Column("provider_receipt", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint(
            "status IN ('pending', 'submitting', 'submitted', 'complete', 'indeterminate')",
            name="annex_embedding_status_check",
        ),
    )
    op.execute("""CREATE FUNCTION protect_annex_publication_manifest() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.change_set_id IS DISTINCT FROM OLD.change_set_id OR NEW.first_intent_id IS DISTINCT FROM OLD.first_intent_id
        OR NEW.source_history IS DISTINCT FROM OLD.source_history
        OR NEW.payload IS DISTINCT FROM OLD.payload OR NEW.payload_sha256 IS DISTINCT FROM OLD.payload_sha256
        OR NEW.created_at IS DISTINCT FROM OLD.created_at
        OR (OLD.operations IS NOT NULL AND (NEW.operations IS DISTINCT FROM OLD.operations OR NEW.operations_sha256 IS DISTINCT FROM OLD.operations_sha256))
        OR (OLD.es_started AND NOT NEW.es_started)
        OR (OLD.approved_at IS NOT NULL AND NEW IS DISTINCT FROM OLD)
      THEN RAISE EXCEPTION 'immutable annex publication manifest'; END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER annex_manifest_immutable BEFORE UPDATE ON annex_publication_manifest FOR EACH ROW EXECUTE FUNCTION protect_annex_publication_manifest()"
    )
    op.execute("""CREATE FUNCTION protect_annex_publication_embedding() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF NEW.change_set_id IS DISTINCT FROM OLD.change_set_id OR NEW.projection_id IS DISTINCT FROM OLD.projection_id
        OR NEW.request IS DISTINCT FROM OLD.request OR NEW.request_sha256 IS DISTINCT FROM OLD.request_sha256
        OR (OLD.remote_id IS NOT NULL AND NEW.remote_id IS DISTINCT FROM OLD.remote_id)
        OR (OLD.status = 'complete' AND NEW IS DISTINCT FROM OLD)
      THEN RAISE EXCEPTION 'immutable annex embedding evidence'; END IF;
      RETURN NEW;
    END $$""")
    op.execute(
        "CREATE TRIGGER annex_embedding_immutable BEFORE UPDATE ON annex_publication_embedding FOR EACH ROW EXECUTE FUNCTION protect_annex_publication_embedding()"
    )


def downgrade() -> None:
    # Never discard historical rows to make a downgrade pass after activation.
    op.drop_index(
        "uq_temporal_projection_ordinal", table_name="regulatory_temporal_projection"
    )
    op.create_unique_constraint(
        "uq_temporal_projection_ordinal",
        "regulatory_temporal_projection",
        ["user_file_id", "index_uuid", "projection_ordinal"],
    )
    op.drop_column("regulatory_temporal_projection", "retired_at")
    op.drop_table("annex_publication_embedding")
    op.drop_table("annex_publication_manifest")
    op.execute("DROP FUNCTION protect_annex_publication_embedding()")
    op.execute("DROP FUNCTION protect_annex_publication_manifest()")
