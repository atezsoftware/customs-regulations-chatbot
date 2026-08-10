"""Show the complete OpenRouter catalog in auto mode.

Revision ID: a8c4e2f1b7d9
Revises: f6a9c1d4e8b3
"""

from alembic import op

revision = "a8c4e2f1b7d9"
down_revision = "f6a9c1d4e8b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE model_configuration AS mc
        SET is_visible = TRUE
        FROM llm_provider AS lp
        WHERE mc.llm_provider_id = lp.id
          AND lp.provider = 'openrouter'
          AND lp.is_auto_mode = TRUE
          AND mc.is_visible = FALSE
        """
    )


def downgrade() -> None:
    # Previous per-model visibility cannot be reconstructed safely.
    pass
