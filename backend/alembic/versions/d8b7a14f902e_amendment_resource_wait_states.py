"""amendment_resource_wait_states

Revision ID: d8b7a14f902e
Revises: 3ef6b03f58af
Create Date: 2026-09-24 14:00:26.442908

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "d8b7a14f902e"
down_revision = "3ef6b03f58af"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("amendment_batch_status_check", "amendment_batch", type_="check")
    op.drop_constraint("amendment_batch_stage_check", "amendment_batch", type_="check")
    op.create_check_constraint(
        "amendment_batch_status_check",
        "amendment_batch",
        "status IN ('queued', 'analyzing', 'analyzed', 'failed', 'paused')",
    )
    op.create_check_constraint(
        "amendment_batch_stage_check",
        "amendment_batch",
        "stage IN ('queued', 'segmenting', 'processing', 'finalizing', 'waiting_resources')",
    )


def downgrade() -> None:
    op.execute(
        "UPDATE amendment_batch SET status = 'failed', "
        "error_message = 'Analysis paused for resources; saved progress is preserved.' "
        "WHERE status = 'paused'"
    )
    op.execute(
        "UPDATE amendment_batch SET stage = 'queued' WHERE stage = 'waiting_resources'"
    )
    op.drop_constraint("amendment_batch_status_check", "amendment_batch", type_="check")
    op.drop_constraint("amendment_batch_stage_check", "amendment_batch", type_="check")
    op.create_check_constraint(
        "amendment_batch_status_check",
        "amendment_batch",
        "status IN ('queued', 'analyzing', 'analyzed', 'failed')",
    )
    op.create_check_constraint(
        "amendment_batch_stage_check",
        "amendment_batch",
        "stage IN ('queued', 'segmenting', 'processing', 'finalizing')",
    )
