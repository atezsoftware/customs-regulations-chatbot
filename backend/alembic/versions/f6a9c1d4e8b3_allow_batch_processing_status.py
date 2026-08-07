"""allow batch processing user file status

Revision ID: f6a9c1d4e8b3
Revises: e5f8b0c3d7a2
"""

from alembic import op

revision = "f6a9c1d4e8b3"
down_revision = "e5f8b0c3d7a2"
branch_labels = None
depends_on = None

_ORIGINAL_VALUES = (
    "PROCESSING",
    "INDEXING",
    "COMPLETED",
    "SKIPPED",
    "FAILED",
    "CANCELED",
    "DELETING",
)


def _status_check(values: tuple[str, ...]) -> str:
    allowed = ", ".join(f"'{value}'" for value in values)
    return f"status IN ({allowed})"


def upgrade() -> None:
    op.drop_constraint("ck_user_file_status", "user_file", type_="check")
    op.create_check_constraint(
        "ck_user_file_status",
        "user_file",
        _status_check((*_ORIGINAL_VALUES, "BATCH_PROCESSING")),
    )


def downgrade() -> None:
    op.drop_constraint("ck_user_file_status", "user_file", type_="check")
    op.create_check_constraint(
        "ck_user_file_status", "user_file", _status_check(_ORIGINAL_VALUES)
    )
