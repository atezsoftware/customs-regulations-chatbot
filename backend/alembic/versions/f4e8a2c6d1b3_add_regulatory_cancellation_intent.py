"""add regulatory cancellation intent

Revision ID: f4e8a2c6d1b3
Revises: e3b7d5a1c9f2
Create Date: 2026-08-20
"""

import sqlalchemy as sa
from alembic import op

revision = "f4e8a2c6d1b3"
down_revision = "e3b7d5a1c9f2"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "regulatory_indexing_job",
        sa.Column(
            "cancellation_intent",
            sa.String(length=32),
            server_default="NONE",
            nullable=True,
        ),
    )
    op.execute(
        """
        UPDATE regulatory_indexing_job AS job
        SET cancellation_intent = CASE
            WHEN user_file.status = 'DELETING' THEN 'USER_DELETE'
            WHEN user_file.status IN ('PROCESSING', 'INDEXING') THEN 'SUPERSEDE'
            ELSE 'USER_CANCEL'
        END
        FROM user_file
        WHERE job.user_file_id = user_file.id
          AND job.status = 'CANCELLING'
        """
    )
    op.alter_column(
        "regulatory_indexing_job",
        "cancellation_intent",
        existing_type=sa.String(length=32),
        server_default="NONE",
        nullable=False,
    )
    op.create_check_constraint(
        "regulatory_indexing_job_cancellation_intent_check",
        "regulatory_indexing_job",
        "cancellation_intent IN ('NONE', 'USER_CANCEL', 'USER_DELETE', 'SUPERSEDE')",
    )
    # An already-applied e3 draft labelled every missing discriminator legacy.
    # A resolved legacy row cannot remain PREPARING because resolution and stage
    # advance are atomic, so all such PREPARING rows are safely unresolved.
    op.execute(
        """
        UPDATE regulatory_indexing_job
        SET config_snapshot = jsonb_set(
            config_snapshot,
            '{input_hash_version}',
            '"legacy-or-canonical"'::jsonb,
            true
        )
        WHERE stage = 'PREPARING'
          AND config_snapshot ->> 'input_hash_version' = 'legacy-v1'
        """
    )


def downgrade() -> None:
    op.drop_constraint(
        "regulatory_indexing_job_cancellation_intent_check",
        "regulatory_indexing_job",
        type_="check",
    )
    op.drop_column("regulatory_indexing_job", "cancellation_intent")
