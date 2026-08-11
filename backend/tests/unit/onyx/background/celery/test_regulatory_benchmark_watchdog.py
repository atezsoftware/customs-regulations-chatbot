from unittest.mock import MagicMock, patch

from onyx.background.celery.tasks.regulatory_benchmark import tasks
from onyx.db.enums import BenchmarkRunStatus


def test_benchmark_items_are_sequential_by_default() -> None:
    assert tasks.REGULATORY_BENCHMARK_PARALLEL_ITEMS == 1


def test_run_lease_expires_promptly_after_worker_loss() -> None:
    assert tasks._RUN_LEASE_SECONDS == 60
    assert tasks._RUN_LEASE_HEARTBEAT_SECONDS <= tasks._RUN_LEASE_SECONDS / 3


def _job(job_id: int, *, done: bool) -> MagicMock:
    job = MagicMock()
    job.id = job_id
    job.process = MagicMock()
    job.done.return_value = done
    job.status = "finished" if done else "running"
    return job


def test_item_watchdog_terminates_only_the_item_that_times_out() -> None:
    timed_out_job = _job(1, done=False)
    client = MagicMock()
    client.submit.return_value = timed_out_job
    heartbeat = MagicMock()

    with (
        patch.object(tasks, "REGULATORY_BENCHMARK_PARALLEL_ITEMS", 1),
        patch.object(tasks, "REGULATORY_BENCHMARK_ITEM_TIMEOUT_SECONDS", 60),
        patch.object(tasks, "SimpleJobClient", return_value=client),
        patch.object(tasks, "_claim_benchmark_item", return_value=True),
        patch.object(
            tasks,
            "_get_benchmark_run_status",
            return_value=BenchmarkRunStatus.RUNNING.value,
        ),
        patch.object(tasks, "_touch_benchmark_run") as touch_run,
        patch.object(tasks, "_record_item_failure") as record_failure,
        patch.object(tasks.time, "monotonic", side_effect=[0.0, 61.0]),
    ):
        timed_out = tasks._run_benchmark_items(
            run_id=12,
            tenant_id="tenant_1",
            item_ids=[34],
            heartbeat=heartbeat,
        )

    assert timed_out is True
    timed_out_job.terminate_and_wait.assert_called_once_with(10)
    record_failure.assert_called_once_with(
        12,
        34,
        "Benchmark item 34 exceeded the 60 second execution deadline",
    )
    assert all(call.args == (12,) for call in touch_run.call_args_list)


def test_benchmark_job_client_preloads_the_runner_in_a_forkserver() -> None:
    with patch.object(tasks, "SimpleJobClient") as client_type:
        tasks._benchmark_job_client(n_workers=5)

    client_type.assert_called_once_with(
        n_workers=5,
        start_method="forkserver",
        preload_modules=("onyx.regulatory.benchmark.runner",),
    )


def test_item_scheduler_stops_active_children_after_run_cancellation() -> None:
    active_job = _job(1, done=False)
    client = MagicMock()
    client.submit.return_value = active_job

    with (
        patch.object(tasks, "REGULATORY_BENCHMARK_PARALLEL_ITEMS", 1),
        patch.object(tasks, "SimpleJobClient", return_value=client),
        patch.object(tasks, "_claim_benchmark_item", return_value=True),
        patch.object(
            tasks,
            "_get_benchmark_run_status",
            side_effect=[
                BenchmarkRunStatus.RUNNING.value,
                BenchmarkRunStatus.CANCELLED.value,
            ],
        ),
        patch.object(tasks, "_touch_benchmark_run"),
        patch.object(tasks.time, "monotonic", side_effect=[0.0, 1.0]),
        patch.object(tasks.time, "sleep"),
    ):
        tasks._run_benchmark_items(
            run_id=12,
            tenant_id="tenant_1",
            item_ids=[34],
            heartbeat=MagicMock(),
        )

    active_job.terminate_and_wait.assert_called_once_with(10)


def test_item_child_uses_a_fresh_session() -> None:
    db_session = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session

    with (
        patch.object(
            tasks,
            "get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            "onyx.regulatory.benchmark.runner.run_claimed_benchmark_item"
        ) as run_claimed_benchmark_item,
    ):
        tasks._execute_benchmark_item(12, 34, "tenant_1")

    run_claimed_benchmark_item.assert_called_once_with(db_session, 12, 34)


def test_active_item_heartbeat_uses_a_fresh_session() -> None:
    db_session = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session

    with (
        patch.object(
            tasks,
            "get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch.object(tasks, "touch_benchmark_run_heartbeat") as touch_run,
        patch.object(tasks, "touch_benchmark_run_items") as touch_items,
    ):
        tasks._touch_benchmark_run(12, [34, 35])

    touch_run.assert_called_once()
    touch_items.assert_called_once()
    assert touch_items.call_args.args[:2] == (db_session, 12)
    assert touch_items.call_args.kwargs["item_ids"] == [34, 35]


def test_report_timeout_is_recorded_without_reopening_terminal_run() -> None:
    finalization_job = _job(9, done=False)

    with (
        patch.object(tasks, "REGULATORY_BENCHMARK_ITEM_TIMEOUT_SECONDS", 60),
        patch.object(
            tasks,
            "_get_benchmark_run_status",
            return_value=BenchmarkRunStatus.COMPLETED.value,
        ),
        patch.object(tasks, "_record_report_failure") as record_report_failure,
        patch.object(tasks.time, "monotonic", side_effect=[0.0, 61.0]),
    ):
        tasks._monitor_finalization_job(
            finalization_job,
            run_id=12,
            heartbeat=MagicMock(),
        )

    finalization_job.terminate_and_wait.assert_called_once_with(10)
    record_report_failure.assert_called_once_with(
        12,
        "Benchmark finalization exceeded the 60 second execution deadline",
    )


def test_task_coordinates_items_then_finalizes_in_a_spawned_process() -> None:
    lock = MagicMock()
    lock.acquire.return_value = True
    lock.owned.return_value = True
    cache = MagicMock()
    cache.lock.return_value = lock
    heartbeat = MagicMock()
    finalization_job = _job(9, done=True)
    client = MagicMock()
    client.submit.return_value = finalization_job

    with (
        patch.object(tasks, "get_cache_backend", return_value=cache),
        patch.object(tasks, "_RunLeaseHeartbeat", return_value=heartbeat),
        patch.object(tasks, "_prepare_benchmark_items", return_value=[34, 35]),
        patch.object(tasks, "_run_benchmark_items", return_value=False) as run_items,
        patch.object(
            tasks,
            "_get_benchmark_run_status",
            return_value=BenchmarkRunStatus.RUNNING.value,
        ),
        patch.object(tasks, "_touch_benchmark_run"),
        patch.object(tasks, "SimpleJobClient", return_value=client),
        patch.object(tasks, "_monitor_finalization_job") as monitor,
    ):
        tasks.run_regulatory_benchmark_task.run(run_id=12, tenant_id="tenant_1")

    run_items.assert_called_once_with(
        run_id=12,
        tenant_id="tenant_1",
        item_ids=[34, 35],
        heartbeat=heartbeat,
    )
    client.submit.assert_called_once_with(
        tasks._execute_benchmark_finalization,
        12,
        "tenant_1",
        False,
    )
    monitor.assert_called_once_with(
        finalization_job,
        run_id=12,
        heartbeat=heartbeat,
    )
    heartbeat.ensure_owned.assert_called_once_with()


def test_item_scheduler_fills_parallel_slots_before_waiting() -> None:
    events: list[str] = []

    class FinishedJob:
        def __init__(self, job_id: int) -> None:
            self.id = job_id
            self.process = MagicMock()
            self.status = "finished"

        def done(self) -> bool:
            return True

    class RecordingClient:
        def __init__(
            self,
            n_workers: int,
            *,
            start_method: str,
            preload_modules: tuple[str, ...],
        ) -> None:
            assert n_workers == 2
            assert start_method == "forkserver"
            assert preload_modules == ("onyx.regulatory.benchmark.runner",)
            self._next_id = 0

        def submit(self, _function: object, *args: object) -> FinishedJob:
            events.append(f"submit:{args[1]}")
            job = FinishedJob(self._next_id)
            self._next_id += 1
            return job

    heartbeat = MagicMock()

    with (
        patch.object(tasks, "REGULATORY_BENCHMARK_PARALLEL_ITEMS", 2),
        patch.object(tasks, "SimpleJobClient", RecordingClient),
        patch.object(
            tasks,
            "_claim_benchmark_item",
            side_effect=lambda _run_id, item_id: (
                events.append(f"claim:{item_id}") or True
            ),
        ),
        patch.object(
            tasks,
            "_get_benchmark_run_status",
            return_value=BenchmarkRunStatus.RUNNING.value,
        ),
        patch.object(tasks, "_touch_benchmark_run"),
        patch.object(
            tasks.time,
            "sleep",
            side_effect=lambda _seconds: events.append("wait"),
        ),
    ):
        timed_out = tasks._run_benchmark_items(
            run_id=12,
            tenant_id="tenant_1",
            item_ids=[101, 102, 103],
            heartbeat=heartbeat,
        )

    assert timed_out is False
    assert events[:5] == [
        "claim:101",
        "submit:101",
        "claim:102",
        "submit:102",
        "wait",
    ]
    assert events[-2:] == ["claim:103", "submit:103"]
    assert heartbeat.ensure_owned.call_count >= 2


def test_item_scheduler_does_not_spawn_when_atomic_claim_is_rejected() -> None:
    client = MagicMock()

    with (
        patch.object(tasks, "SimpleJobClient", return_value=client),
        patch.object(tasks, "_claim_benchmark_item", return_value=False),
        patch.object(
            tasks,
            "_get_benchmark_run_status",
            return_value=BenchmarkRunStatus.RUNNING.value,
        ),
        patch.object(tasks, "_touch_benchmark_run"),
    ):
        timed_out = tasks._run_benchmark_items(
            run_id=12,
            tenant_id="tenant_1",
            item_ids=[34],
            heartbeat=MagicMock(),
        )

    assert timed_out is False
    client.submit.assert_not_called()
