"""Persist owned legacy writer publication recovery

Revision ID: 601b08f82b8c
Revises: 3102937b17cb
Create Date: 2026-09-10 17:39:55.020014

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "601b08f82b8c"
down_revision = "3102937b17cb"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_file_publication",
        sa.Column("writer_manifest", postgresql.JSONB(), nullable=True),
    )
    op.add_column(
        "regulatory_file_publication",
        sa.Column("writer_manifest_sha256", sa.Text(), nullable=True),
    )
    op.execute("""CREATE FUNCTION protect_writer_manifest() RETURNS trigger LANGUAGE plpgsql AS $$
    BEGIN
      IF OLD.writer_manifest IS NOT NULL AND NEW.writer_manifest IS NOT NULL AND
        (OLD.writer_manifest IS DISTINCT FROM NEW.writer_manifest OR OLD.writer_manifest_sha256 IS DISTINCT FROM NEW.writer_manifest_sha256) THEN
        RAISE EXCEPTION 'pending writer manifest is immutable';
      END IF;
      RETURN NEW;
    END; $$""")
    op.execute(
        "CREATE TRIGGER immutable_pending_writer_manifest BEFORE UPDATE ON regulatory_file_publication FOR EACH ROW EXECUTE FUNCTION protect_writer_manifest()"
    )


def downgrade() -> None:
    op.execute("""DO $$ BEGIN IF EXISTS (SELECT 1 FROM regulatory_file_publication WHERE writer_manifest IS NOT NULL) THEN
      RAISE EXCEPTION 'pending writer publications require a data-preserving migration';
    END IF; END $$""")
    op.execute(
        "DROP TRIGGER immutable_pending_writer_manifest ON regulatory_file_publication"
    )
    op.execute("DROP FUNCTION protect_writer_manifest()")
    op.drop_column("regulatory_file_publication", "writer_manifest_sha256")
    op.drop_column("regulatory_file_publication", "writer_manifest")
