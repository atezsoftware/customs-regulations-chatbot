"""add grouped amendment proposal coverage

Revision ID: f30912f19544
Revises: f4a9c2d7e1b3
Create Date: 2026-08-27 21:10:43.101972

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "f30912f19544"
down_revision = "f4a9c2d7e1b3"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "amendment_batch",
        sa.Column(
            "processed_instruction_indices",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "amendment_proposal",
        sa.Column(
            "instruction_indices",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.add_column(
        "amendment_proposal",
        sa.Column(
            "instruction_texts",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
    )
    op.execute(
        "UPDATE amendment_batch "
        "SET processed_instruction_indices = CASE "
        "WHEN processed_instruction_count = 0 THEN '[]'::jsonb "
        "ELSE (SELECT jsonb_agg(instruction_index) "
        "FROM generate_series(0, processed_instruction_count - 1) "
        "AS instruction_index) END"
    )
    op.execute(
        "UPDATE amendment_proposal "
        "SET instruction_indices = jsonb_build_array(instruction_index), "
        "instruction_texts = jsonb_build_array(instruction_text)"
    )


def downgrade() -> None:
    op.drop_column("amendment_proposal", "instruction_texts")
    op.drop_column("amendment_proposal", "instruction_indices")
    op.drop_column("amendment_batch", "processed_instruction_indices")
