"""Request cohort ledger for resettable usage reporting."""

from alembic import op
import sqlalchemy as sa

revision = "a61e0f9b245c"
down_revision = "898eaa5536e4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "usage_measurement",
        sa.Column("request_id", sa.UUID(), primary_key=True),
        sa.Column("model", sa.String(), primary_key=True),
        sa.Column("flow", sa.String(), primary_key=True),
        sa.Column("provider", sa.String(), primary_key=True),
        sa.Column("epoch", sa.String(), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("user_id", sa.UUID(), sa.ForeignKey("user.id", ondelete="SET NULL")),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("question_id", sa.BigInteger(), nullable=False),
        sa.Column("workflow", sa.String(), nullable=False),
        sa.Column("benchmark", sa.Boolean(), nullable=False),
        sa.Column("input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cache_read_tokens", sa.BigInteger(), nullable=False),
        sa.Column("cost_cents", sa.Numeric(18, 6), nullable=False),
    )
    op.create_index("ix_usage_measurement_epoch", "usage_measurement", ["epoch"])


def downgrade() -> None:
    op.drop_table("usage_measurement")
