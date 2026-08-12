import datetime
import threading
import time
from collections import deque
from dataclasses import dataclass

from celery import shared_task

from onyx.background.indexing.job_client import SimpleJob, SimpleJobClient
from onyx.cache.factory import get_cache_backend
from onyx.cache.interface import CACHE_TRANSIENT_ERRORS, CacheLock, CacheLockLostError
from onyx.configs.app_configs import (
    REGULATORY_BENCHMARK_DEEP_RESEARCH_ITEM_TIMEOUT_SECONDS,
    REGULATORY_BENCHMARK_ITEM_TIMEOUT_SECONDS,
    REGULATORY_BENCHMARK_PARALLEL_ITEMS,
)
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import BenchmarkRunFailureCode, BenchmarkRunStatus
from onyx.db.regulatory_benchmark import (
    claim_benchmark_run_item,
    get_benchmark_run,
    get_benchmark_run_status,
    mark_benchmark_run_failed,
    mark_benchmark_run_item_failed,
    mark_benchmark_run_report_failed,
    touch_benchmark_run_heartbeat,
    touch_benchmark_run_items,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

_RUN_LEASE_SECONDS = 60
_RUN_LEASE_HEARTBEAT_SECONDS = 15
_RUN_LEASE_UNCERTAINTY_SECONDS = 45
_WATCHDOG_POLL_SECONDS = 1
_WATCHDOG_SIGTERM_GRACE_SECONDS = 10
_BENCHMARK_PRELOAD_MODULES = ("onyx.regulatory.benchmark.runner",)


class BenchmarkExecutionTimeout(RuntimeError):
    pass


def _benchmark_job_client(*, n_workers: int) -> SimpleJobClient:
    """Create isolated workers without importing the full chat stack per item."""
    return SimpleJobClient(
        n_workers=n_workers,
        start_method="forkserver",
        preload_modules=_BENCHMARK_PRELOAD_MODULES,
    )


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


class _RunLeaseHeartbeat:
    """Keep a finite cross-worker lease alive while one run owns execution."""

    def __init__(self, lock: CacheLock, run_id: int) -> None:
        self._lock = lock
        self._run_id = run_id
        self._stop_event = threading.Event()
        self._lost_event = threading.Event()
        self._state_lock = threading.Lock()
        self._last_confirmed_at = time.monotonic()
        self._started = False
        self._thread = threading.Thread(
            target=self._run,
            name=f"benchmark-run-lease-{run_id}",
            daemon=True,
        )

    def start(self) -> None:
        self._started = True
        try:
            self._thread.start()
        except Exception:
            self._started = False
            raise

    def _run(self) -> None:
        while not self._stop_event.wait(_RUN_LEASE_HEARTBEAT_SECONDS):
            if not self._extend_lease():
                return

    def _confirm_ownership(self) -> None:
        with self._state_lock:
            self._last_confirmed_at = time.monotonic()

    def _ownership_uncertainty_seconds(self) -> float:
        with self._state_lock:
            return time.monotonic() - self._last_confirmed_at

    def _extend_lease(self) -> bool:
        try:
            self._lock.extend(_RUN_LEASE_SECONDS)
        except CacheLockLostError:
            self._lost_event.set()
            logger.exception("Benchmark run %s lost its execution lease", self._run_id)
            return False
        except CACHE_TRANSIENT_ERRORS:
            logger.warning(
                "Benchmark run %s could not extend its execution lease; "
                "ownership will be rechecked before the safety window closes",
                self._run_id,
                exc_info=True,
            )
            return True
        self._confirm_ownership()
        return True

    def ensure_owned(self) -> None:
        if self._lost_event.is_set():
            raise RuntimeError(f"Benchmark run {self._run_id} lost its execution lease")
        try:
            owned = self._lock.owned()
        except CACHE_TRANSIENT_ERRORS as error:
            if self._ownership_uncertainty_seconds() < _RUN_LEASE_UNCERTAINTY_SECONDS:
                logger.warning(
                    "Benchmark run %s could not verify its execution lease; "
                    "tolerating the transient cache failure inside the safety window",
                    self._run_id,
                    exc_info=True,
                )
                return
            self._lost_event.set()
            raise RuntimeError(
                f"Benchmark run {self._run_id} could not verify its execution lease"
            ) from error
        if not owned:
            self._lost_event.set()
            raise RuntimeError(f"Benchmark run {self._run_id} lost its execution lease")
        self._confirm_ownership()

    def stop(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=5)


@dataclass(frozen=True)
class _ActiveBenchmarkJob:
    job: SimpleJob
    item_id: int
    started_at: float


def _execute_benchmark_item(
    run_id: int,
    item_id: int,
    tenant_id: str,  # noqa: ARG001
) -> None:
    """Child entrypoint; SimpleJob establishes tenant context and a fresh engine."""
    from onyx.regulatory.benchmark.runner import run_claimed_benchmark_item

    with get_session_with_current_tenant() as db_session:
        run_claimed_benchmark_item(db_session, run_id, item_id)


def _claim_benchmark_item(run_id: int, item_id: int) -> bool:
    with get_session_with_current_tenant() as db_session:
        return claim_benchmark_run_item(
            db_session,
            run_id=run_id,
            item_id=item_id,
            started_at=_utcnow(),
        )


def _execute_benchmark_finalization(
    run_id: int,
    tenant_id: str,  # noqa: ARG001
    had_execution_timeout: bool,
) -> None:
    from onyx.regulatory.benchmark.runner import finalize_benchmark_run

    with get_session_with_current_tenant() as db_session:
        finalize_benchmark_run(
            db_session,
            run_id,
            had_execution_timeout=had_execution_timeout,
        )


def _prepare_benchmark_items(run_id: int) -> list[int]:
    from onyx.regulatory.benchmark.runner import prepare_benchmark_run

    with get_session_with_current_tenant() as db_session:
        return prepare_benchmark_run(db_session, run_id)


def _get_benchmark_run_status(run_id: int) -> str | None:
    with get_session_with_current_tenant() as db_session:
        return get_benchmark_run_status(db_session, run_id)


def _benchmark_run_uses_deep_research(run_id: int) -> bool:
    with get_session_with_current_tenant() as db_session:
        run = get_benchmark_run(db_session, run_id)
        return bool(run and run.deep_research)


def _benchmark_item_timeout_seconds(*, deep_research: bool) -> int:
    return (
        REGULATORY_BENCHMARK_DEEP_RESEARCH_ITEM_TIMEOUT_SECONDS
        if deep_research
        else REGULATORY_BENCHMARK_ITEM_TIMEOUT_SECONDS
    )


def _touch_benchmark_run(run_id: int, item_ids: list[int] | None = None) -> None:
    with get_session_with_current_tenant() as db_session:
        touch_benchmark_run_heartbeat(db_session, run_id, heartbeat_at=_utcnow())
        if item_ids:
            touch_benchmark_run_items(
                db_session,
                run_id,
                item_ids=item_ids,
                heartbeat_at=_utcnow(),
            )


def _record_item_failure(run_id: int, item_id: int, message: str) -> None:
    with get_session_with_current_tenant() as db_session:
        mark_benchmark_run_item_failed(
            db_session,
            run_id=run_id,
            item_id=item_id,
            error_message=message[:4000],
            completed_at=_utcnow(),
        )


def _record_report_failure(run_id: int, message: str) -> None:
    with get_session_with_current_tenant() as db_session:
        mark_benchmark_run_report_failed(db_session, run_id, message)


def _stop_jobs(active_jobs: dict[int, _ActiveBenchmarkJob]) -> None:
    for active in active_jobs.values():
        active.job.terminate_and_wait(_WATCHDOG_SIGTERM_GRACE_SECONDS)


def _reap_job(job: SimpleJob) -> None:
    if job.process is not None:
        job.process.join()


def _run_benchmark_items(
    *,
    run_id: int,
    tenant_id: str,
    item_ids: list[int],
    item_timeout_seconds: int,
    heartbeat: _RunLeaseHeartbeat,
) -> bool:
    """Run isolated item processes with bounded parallelism and per-item deadlines."""
    pending = deque(item_ids)
    client = _benchmark_job_client(n_workers=REGULATORY_BENCHMARK_PARALLEL_ITEMS)
    active_jobs: dict[int, _ActiveBenchmarkJob] = {}
    had_execution_timeout = False

    try:
        while pending or active_jobs:
            heartbeat.ensure_owned()
            if _get_benchmark_run_status(run_id) != BenchmarkRunStatus.RUNNING.value:
                return had_execution_timeout

            while pending and len(active_jobs) < REGULATORY_BENCHMARK_PARALLEL_ITEMS:
                item_id = pending.popleft()
                if not _claim_benchmark_item(run_id, item_id):
                    logger.info(
                        "Benchmark run %s skipped item %s because its claim was rejected",
                        run_id,
                        item_id,
                    )
                    continue
                job = client.submit(
                    _execute_benchmark_item,
                    run_id,
                    item_id,
                    tenant_id,
                )
                if job is None or job.process is None:
                    raise RuntimeError(f"Failed to spawn benchmark item {item_id}")
                active_jobs[job.id] = _ActiveBenchmarkJob(
                    job=job,
                    item_id=item_id,
                    started_at=time.monotonic(),
                )
                logger.info(
                    "Benchmark run %s started item %s in child process %s",
                    run_id,
                    item_id,
                    job.process.pid,
                )

            for job_id, active in list(active_jobs.items()):
                if active.job.done():
                    if active.job.status == "error":
                        _record_item_failure(
                            run_id,
                            active.item_id,
                            f"Benchmark item process failed: {active.job.exception()}",
                        )
                    _reap_job(active.job)
                    del active_jobs[job_id]
                    continue

                elapsed = time.monotonic() - active.started_at
                if elapsed <= item_timeout_seconds:
                    continue
                message = (
                    f"Benchmark item {active.item_id} exceeded the "
                    f"{item_timeout_seconds} second "
                    "execution deadline"
                )
                active.job.terminate_and_wait(_WATCHDOG_SIGTERM_GRACE_SECONDS)
                _record_item_failure(run_id, active.item_id, message)
                del active_jobs[job_id]
                had_execution_timeout = True

            # The run heartbeat proves coordinator liveness. Item heartbeats are
            # written by the item process at real execution boundaries so the UI
            # never mistakes a live coordinator for forward progress.
            _touch_benchmark_run(run_id)
            if pending or active_jobs:
                time.sleep(_WATCHDOG_POLL_SECONDS)
    finally:
        _stop_jobs(active_jobs)

    return had_execution_timeout


def _monitor_finalization_job(
    job: SimpleJob,
    *,
    run_id: int,
    heartbeat: _RunLeaseHeartbeat,
) -> None:
    """Bound report generation, which can itself make an LLM call."""
    started_at = time.monotonic()
    while not job.done():
        heartbeat.ensure_owned()
        status = _get_benchmark_run_status(run_id)
        if status is None or status == BenchmarkRunStatus.CANCELLED.value:
            job.terminate_and_wait(_WATCHDOG_SIGTERM_GRACE_SECONDS)
            return
        if time.monotonic() - started_at > REGULATORY_BENCHMARK_ITEM_TIMEOUT_SECONDS:
            message = (
                "Benchmark finalization exceeded the "
                f"{REGULATORY_BENCHMARK_ITEM_TIMEOUT_SECONDS} second execution deadline"
            )
            job.terminate_and_wait(_WATCHDOG_SIGTERM_GRACE_SECONDS)
            if status in {
                BenchmarkRunStatus.COMPLETED.value,
                BenchmarkRunStatus.ERROR.value,
            }:
                _record_report_failure(run_id, message)
                return
            with get_session_with_current_tenant() as db_session:
                mark_benchmark_run_failed(
                    db_session,
                    run_id,
                    message,
                    failure_code=BenchmarkRunFailureCode.EXECUTION_TIMEOUT,
                )
            raise BenchmarkExecutionTimeout(message)
        _touch_benchmark_run(run_id)
        time.sleep(_WATCHDOG_POLL_SECONDS)

    if job.status == "error":
        message = f"Benchmark finalization process failed: {job.exception()}"
        if _get_benchmark_run_status(run_id) in {
            BenchmarkRunStatus.COMPLETED.value,
            BenchmarkRunStatus.ERROR.value,
        }:
            _record_report_failure(run_id, message)
            _reap_job(job)
            return
        _reap_job(job)
        raise RuntimeError(message)
    _reap_job(job)


@shared_task(
    name=OnyxCeleryTask.REGULATORY_BENCHMARK_RUN,
    ignore_result=True,
    trail=False,
    acks_late=True,
    reject_on_worker_lost=True,
)
def run_regulatory_benchmark_task(
    *,
    run_id: int,
    tenant_id: str,  # noqa: ARG001 - consumed by TenantAwareTask
) -> None:
    lock = get_cache_backend(tenant_id=tenant_id).lock(
        f"regulatory-benchmark-run:{run_id}", timeout=_RUN_LEASE_SECONDS
    )
    if not lock.acquire(blocking=False):
        logger.info("Benchmark run %s is already owned by another worker", run_id)
        return
    heartbeat = _RunLeaseHeartbeat(lock, run_id)
    try:
        heartbeat.start()
        item_ids = _prepare_benchmark_items(run_id)
        item_timeout_seconds = _benchmark_item_timeout_seconds(
            deep_research=_benchmark_run_uses_deep_research(run_id)
        )
        logger.info(
            "Benchmark run %s prepared %s item(s) for execution with a %ss deadline",
            run_id,
            len(item_ids),
            item_timeout_seconds,
        )
        had_execution_timeout = _run_benchmark_items(
            run_id=run_id,
            tenant_id=tenant_id,
            item_ids=item_ids,
            item_timeout_seconds=item_timeout_seconds,
            heartbeat=heartbeat,
        )
        if _get_benchmark_run_status(run_id) != BenchmarkRunStatus.RUNNING.value:
            return

        client = _benchmark_job_client(n_workers=1)
        job = client.submit(
            _execute_benchmark_finalization,
            run_id,
            tenant_id,
            had_execution_timeout,
        )
        if job is None or job.process is None:
            raise RuntimeError(
                f"Failed to spawn benchmark finalization for run {run_id}"
            )
        _monitor_finalization_job(job, run_id=run_id, heartbeat=heartbeat)
        heartbeat.ensure_owned()
    except Exception as error:
        logger.exception("Regulatory benchmark run %s crashed", run_id)
        with get_session_with_current_tenant() as db_session:
            mark_benchmark_run_failed(db_session, run_id, str(error))
        raise
    finally:
        heartbeat.stop()
        try:
            if lock.owned():
                lock.release()
        except Exception:
            logger.exception("Failed to release benchmark run %s lease", run_id)
