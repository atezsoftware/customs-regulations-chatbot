"""freeze new document table preparation policy

Revision ID: e363f58e0eaf
Revises: b63d8e1f2701
Create Date: 2026-11-02 15:59:03.392176

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "e363f58e0eaf"
down_revision = "b63d8e1f2701"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_document_intake",
        sa.Column(
            "preparation_policy", postgresql.JSONB(none_as_null=True), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("regulatory_document_intake", "preparation_policy")
