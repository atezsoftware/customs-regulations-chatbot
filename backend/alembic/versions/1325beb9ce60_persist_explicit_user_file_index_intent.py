"""persist explicit user file index intent

Revision ID: 1325beb9ce60
Revises: 5d85320c4e35
Create Date: 2026-09-13 16:34:03.751113

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "1325beb9ce60"
down_revision = "5d85320c4e35"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "user_file_projection_repair",
        sa.Column(
            "initial_index", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )

    op.create_index(
        "ix_user_file_index_request_pending",
        "user_file_projection_repair",
        ["updated_at"],
        postgresql_where=sa.text("initial_index AND status = 'PENDING'"),
    )


def downgrade() -> None:
    op.drop_index(
        "ix_user_file_index_request_pending", table_name="user_file_projection_repair"
    )
    op.drop_column("user_file_projection_repair", "initial_index")
