"""add regulatory chunk table

Revision ID: 2010a61d7d88
Revises: 3debc2b55899
Create Date: 2026-07-31 15:36:21.701751

Chunk rows produced by the structure-aware RegulatoryChunker
(backend/onyx/regulatory/chunker.py). Postgres is the source of truth for
chunk text/metadata/validity; the OpenSearch index is a projection of these
rows. Validity dates + supersession links back the amendment (update)
mechanism: an approved amendment inserts a new `source='amendment'` chunk and
closes the old chunk's validity window instead of mutating it.
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA

# revision identifiers, used by Alembic.
revision = "2010a61d7d88"
down_revision = "3debc2b55899"
branch_labels = None
depends_on = None


def upgrade() -> None:
    shared_schema = op.get_bind().dialect.identifier_preparer.quote_schema(
        POSTGRES_DEFAULT_SCHEMA
    )
    # The extension is shared by every tenant schema. Installing it into the
    # current tenant's search_path would make later tenant migrations see the
    # extension but not its operator classes.
    op.execute(f"CREATE EXTENSION IF NOT EXISTS pg_trgm WITH SCHEMA {shared_schema}")

    op.create_table(
        "regulatory_chunk",
        sa.Column("id", sa.Text(), primary_key=True),
        sa.Column(
            "user_file_id",
            sa.Uuid(),
            sa.ForeignKey("user_file.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("chunk_type", sa.Text(), nullable=True),
        sa.Column(
            "heading_path",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "chunk_metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("validity_start_date", sa.Date(), nullable=True),
        sa.Column("validity_end_date", sa.Date(), nullable=True),
        sa.Column(
            "status",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'active'"),
        ),
        sa.Column(
            "source",
            sa.Text(),
            nullable=False,
            server_default=sa.text("'indexed'"),
        ),
        sa.Column(
            "supersedes_chunk_id",
            sa.Text(),
            sa.ForeignKey("regulatory_chunk.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "superseded_by_chunk_id",
            sa.Text(),
            sa.ForeignKey("regulatory_chunk.id", ondelete="SET NULL"),
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
        sa.CheckConstraint(
            "status IN ('active', 'superseded')",
            name="regulatory_chunk_status_check",
        ),
        sa.CheckConstraint(
            "source IN ('indexed', 'amendment')",
            name="regulatory_chunk_source_check",
        ),
    )

    op.create_index(
        "ix_regulatory_chunk_user_file_status",
        "regulatory_chunk",
        ["user_file_id", "status"],
    )
    # Fuzzy matching for the amendment candidate finder — trigram similarity is
    # character-n-gram based, so it works on Turkish text without a
    # language-specific tokenizer.
    op.execute(
        "CREATE INDEX ix_regulatory_chunk_text_trgm ON regulatory_chunk "
        f"USING gin (text {shared_schema}.gin_trgm_ops)"
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION regulatory_chunk_heading_path_text(heading_path jsonb)
        RETURNS text AS $$
          SELECT COALESCE(
            array_to_string(ARRAY(SELECT jsonb_array_elements_text(heading_path)), ' > '),
            ''
          );
        $$ LANGUAGE sql IMMUTABLE
        """
    )
    op.execute(
        "CREATE INDEX ix_regulatory_chunk_heading_path_trgm ON regulatory_chunk "
        "USING gin (regulatory_chunk_heading_path_text(heading_path) "
        f"{shared_schema}.gin_trgm_ops)"
    )
    # Short structured locators for exact-match candidate lookups.
    op.execute(
        "CREATE INDEX ix_regulatory_chunk_article_no ON regulatory_chunk "
        "((chunk_metadata->>'article_no'))"
    )
    op.execute(
        "CREATE INDEX ix_regulatory_chunk_document_number ON regulatory_chunk "
        "((chunk_metadata->>'document_number'))"
    )


def downgrade() -> None:
    op.drop_table("regulatory_chunk")
    op.execute("DROP FUNCTION IF EXISTS regulatory_chunk_heading_path_text(jsonb)")
