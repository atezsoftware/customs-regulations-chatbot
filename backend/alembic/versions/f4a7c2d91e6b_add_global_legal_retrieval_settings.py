"""add global legal retrieval settings

Revision ID: f4a7c2d91e6b
Revises: c6a8d9e4f2b1
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "f4a7c2d91e6b"
down_revision = "c6a8d9e4f2b1"
branch_labels = None
depends_on = None

_ENABLED_CONSTRAINT = "ck_search_settings_rerank_enabled_configuration"
_DISABLED_CONSTRAINT = "ck_search_settings_rerank_disabled_configuration"


def upgrade() -> None:
    op.alter_column(
        "search_settings",
        "enable_contextual_rag",
        existing_type=sa.Boolean(),
        server_default=sa.true(),
        existing_nullable=False,
    )
    op.add_column(
        "search_settings",
        sa.Column(
            "rerank_enabled",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )
    op.add_column(
        "search_settings",
        sa.Column("rerank_provider_type", sa.String(), nullable=True),
    )
    op.add_column(
        "search_settings",
        sa.Column("rerank_model_name", sa.String(), nullable=True),
    )
    op.add_column(
        "search_settings",
        sa.Column("rerank_api_key", sa.LargeBinary(), nullable=True),
    )
    op.add_column(
        "search_settings",
        sa.Column(
            "rerank_updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
    )
    op.add_column(
        "search_settings",
        sa.Column(
            "rerank_updated_by_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.create_foreign_key(
        "search_settings_rerank_updated_by_user_id_fkey",
        "search_settings",
        "user",
        ["rerank_updated_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_check_constraint(
        _ENABLED_CONSTRAINT,
        "search_settings",
        "NOT rerank_enabled OR "
        "(rerank_provider_type IS NOT NULL "
        "AND rerank_model_name IS NOT NULL "
        "AND rerank_api_key IS NOT NULL)",
    )
    op.create_check_constraint(
        _DISABLED_CONSTRAINT,
        "search_settings",
        "rerank_enabled OR "
        "(rerank_provider_type IS NULL "
        "AND rerank_model_name IS NULL "
        "AND rerank_api_key IS NULL)",
    )


def downgrade() -> None:
    op.drop_constraint(_DISABLED_CONSTRAINT, "search_settings", type_="check")
    op.drop_constraint(_ENABLED_CONSTRAINT, "search_settings", type_="check")
    op.drop_constraint(
        "search_settings_rerank_updated_by_user_id_fkey",
        "search_settings",
        type_="foreignkey",
    )
    op.drop_column("search_settings", "rerank_updated_by_user_id")
    op.drop_column("search_settings", "rerank_updated_at")
    op.drop_column("search_settings", "rerank_api_key")
    op.drop_column("search_settings", "rerank_model_name")
    op.drop_column("search_settings", "rerank_provider_type")
    op.drop_column("search_settings", "rerank_enabled")
    op.alter_column(
        "search_settings",
        "enable_contextual_rag",
        existing_type=sa.Boolean(),
        server_default=sa.false(),
        existing_nullable=False,
    )
