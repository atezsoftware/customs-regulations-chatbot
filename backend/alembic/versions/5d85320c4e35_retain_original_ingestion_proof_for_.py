"""retain original ingestion proof for current attachments

Revision ID: 5d85320c4e35
Revises: 7b76341c584a
Create Date: 2026-09-11 00:56:25.046401

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "5d85320c4e35"
down_revision = "7b76341c584a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_file_publication",
        sa.Column(
            "original_ingestion_receipt",
            postgresql.JSONB(none_as_null=True),
            nullable=True,
        ),
    )
    op.execute("""
        CREATE FUNCTION reject_original_ingestion_rewrite() RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.original_ingestion_receipt IS NOT NULL
             AND NEW.original_ingestion_receipt IS DISTINCT FROM OLD.original_ingestion_receipt THEN
            RAISE EXCEPTION 'original ingestion evidence is immutable';
          END IF;
          RETURN NEW;
        END $$;
    """)
    op.execute("""
        CREATE TRIGGER immutable_original_ingestion BEFORE UPDATE ON regulatory_file_publication
        FOR EACH ROW EXECUTE FUNCTION reject_original_ingestion_rewrite();
    """)


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER immutable_original_ingestion ON regulatory_file_publication"
    )
    op.execute("DROP FUNCTION reject_original_ingestion_rewrite()")
    op.drop_column("regulatory_file_publication", "original_ingestion_receipt")
