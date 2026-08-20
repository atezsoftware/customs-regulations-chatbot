"""enforce one active regulatory job per user file

Revision ID: e3b7d5a1c9f2
Revises: d2a9c7e4b1f6
Create Date: 2026-08-20
"""

import sqlalchemy as sa
from alembic import op

revision = "e3b7d5a1c9f2"
down_revision = "d2a9c7e4b1f6"
branch_labels = None
depends_on = None

_ACTIVE_JOB_PREDICATE = "status IN ('QUEUED', 'RUNNING', 'RETRY_WAIT', 'CANCELLING')"


def upgrade() -> None:
    # Databases upgraded through an earlier draft of d2 did not get the
    # explicit algorithm discriminator. Those rows contain the legacy hash.
    op.execute(
        """
        UPDATE regulatory_indexing_job
        SET config_snapshot = jsonb_set(
            config_snapshot,
            '{input_hash_version}',
            '"legacy-v1"'::jsonb,
            true
        )
        WHERE NOT config_snapshot ? 'input_hash_version'
        """
    )
    # This feature is not shipped. If a development database already contains
    # overlapping active generations, retain the most recently current row and
    # delete older jobs; their items are removed by the job-item cascade.
    op.execute(
        """
        WITH ranked_active_jobs AS (
            SELECT
                id,
                row_number() OVER (
                    PARTITION BY user_file_id
                    ORDER BY
                        updated_at DESC,
                        created_at DESC,
                        lease_generation DESC,
                        id DESC
                ) AS row_rank
            FROM regulatory_indexing_job
            WHERE status IN ('QUEUED', 'RUNNING', 'RETRY_WAIT', 'CANCELLING')
        )
        DELETE FROM regulatory_indexing_job AS discarded
        USING ranked_active_jobs
        WHERE discarded.id = ranked_active_jobs.id
          AND ranked_active_jobs.row_rank > 1
        """
    )
    op.create_index(
        "uq_regulatory_indexing_job_active_user_file",
        "regulatory_indexing_job",
        ["user_file_id"],
        unique=True,
        postgresql_where=sa.text(_ACTIVE_JOB_PREDICATE),
    )


def downgrade() -> None:
    op.drop_index(
        "uq_regulatory_indexing_job_active_user_file",
        table_name="regulatory_indexing_job",
    )
