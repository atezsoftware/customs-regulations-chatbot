"""Retain observed same-day archival intervals without changing source dates.

Revision ID: f43b8c129e70
Revises: e92f3a1c7b60
Create Date: 2026-09-23
"""

from collections.abc import Sequence

from alembic import op

revision: str = "f43b8c129e70"
down_revision: str | None = "e92f3a1c7b60"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_constraint(
        "temporal_projection_dates_check",
        "regulatory_temporal_projection",
        type_="check",
    )
    op.create_check_constraint(
        "temporal_projection_dates_check",
        "regulatory_temporal_projection",
        "effective_end IS NULL OR effective_start IS NULL OR effective_end > effective_start "
        "OR (effective_end = effective_start AND COALESCE("
        "payload #>> '{projection,evidence_kind}' = 'observed-v1' "
        "AND (payload #>> '{projection,observed_start}')::bigint "
        "= EXTRACT(EPOCH FROM effective_start::timestamp)::bigint "
        "AND (payload #>> '{projection,observed_end}')::bigint "
        "= EXTRACT(EPOCH FROM effective_end::timestamp)::bigint, FALSE))",
    )


def downgrade() -> None:
    op.drop_constraint(
        "temporal_projection_dates_check",
        "regulatory_temporal_projection",
        type_="check",
    )
    op.create_check_constraint(
        "temporal_projection_dates_check",
        "regulatory_temporal_projection",
        "effective_end IS NULL OR effective_start IS NULL OR effective_end > effective_start",
    )
