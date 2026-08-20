"""A deployment should only schedule the recurring work it can actually run.

The lightweight image runs a subset of the workers, so an unrestricted beat
would fire tasks onto queues nothing consumes. Restricting the schedule lets it
keep the one loop it does need: re-queueing user files whose processing task
expired before a worker reached it.
"""

import pytest

from onyx.background.celery.tasks import beat_schedule
from onyx.configs.constants import OnyxCeleryTask

USER_FILE_RECOVERY_TASKS = {
    OnyxCeleryTask.CHECK_FOR_USER_FILE_PROCESSING,
    OnyxCeleryTask.CHECK_FOR_USER_FILE_PROJECT_SYNC,
    OnyxCeleryTask.CHECK_FOR_USER_FILE_DELETE,
}


def test_an_empty_allowlist_schedules_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(beat_schedule, "BEAT_TASK_ALLOWLIST", [])

    scheduled = beat_schedule.get_tasks_to_schedule()

    assert len(scheduled) == len(beat_schedule.tasks_to_schedule)


def test_the_allowlist_restricts_the_schedule_to_named_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        beat_schedule, "BEAT_TASK_ALLOWLIST", sorted(USER_FILE_RECOVERY_TASKS)
    )

    scheduled = beat_schedule.get_tasks_to_schedule()

    assert {entry["task"] for entry in scheduled} == USER_FILE_RECOVERY_TASKS


def test_every_allowlisted_task_still_exists_in_the_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A typo in a deployment's allowlist would silently schedule nothing."""

    monkeypatch.setattr(
        beat_schedule, "BEAT_TASK_ALLOWLIST", ["not_a_real_task_name"]
    )

    with pytest.raises(ValueError, match="not_a_real_task_name"):
        beat_schedule.get_tasks_to_schedule()


def test_user_file_recovery_tasks_name_the_queue_a_worker_consumes() -> None:
    """Without an explicit queue these land on the default `celery` queue.

    Only the primary worker consumes that queue, and deployments that run a
    subset of the workers have no primary — so the recovery tasks would sit
    unconsumed until they expired, and the files they were meant to retry would
    stay stuck. They belong on the queue their own worker already serves.
    """

    from onyx.configs.constants import OnyxCeleryQueues

    scheduled_by_task = {
        entry["task"]: entry for entry in beat_schedule.tasks_to_schedule
    }

    for task_name in USER_FILE_RECOVERY_TASKS:
        options = scheduled_by_task[task_name]["options"]
        assert (
            options.get("queue") == OnyxCeleryQueues.USER_FILE_PROCESSING
        ), f"{task_name} -> {options.get('queue')}"
