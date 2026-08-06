"""repair reranking schema compatibility

Revision ID: c8d1e4f2a7b9
Revises: f4a7c2d91e6b
"""

from alembic import op

revision = "c8d1e4f2a7b9"
down_revision = "f4a7c2d91e6b"
branch_labels = None
depends_on = None

_DISABLED_CONSTRAINT = "ck_search_settings_rerank_disabled_configuration"


def upgrade() -> None:
    # Some installations may have applied the parent revision before the
    # configuration-generation field and retain-on-disable contract were added.
    op.execute(
        "ALTER TABLE search_settings "
        "ADD COLUMN IF NOT EXISTS rerank_configuration_generation VARCHAR(32) "
        "DEFAULT md5(random()::text || clock_timestamp()::text) NOT NULL"
    )
    op.execute(
        f"ALTER TABLE search_settings DROP CONSTRAINT IF EXISTS {_DISABLED_CONSTRAINT}"
    )


def downgrade() -> None:
    # The parent revision's current schema already contains this column and
    # intentionally omits the obsolete disabled-state constraint.
    pass
