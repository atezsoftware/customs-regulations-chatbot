import threading

from celery import shared_task

from onyx.cache.factory import get_cache_backend
from onyx.cache.interface import CacheLock
from onyx.configs.constants import OnyxCeleryTask
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_benchmark import mark_benchmark_run_failed
from onyx.utils.logger import setup_logger

logger = setup_logger()

_RUN_LEASE_SECONDS = 5 * 60
_RUN_LEASE_HEARTBEAT_SECONDS = 60


class _RunLeaseHeartbeat:
    """Keep a finite cross-worker lease alive while one run owns execution."""

    def __init__(self, lock: CacheLock, run_id: int) -> None:
        self._lock = lock
        self._run_id = run_id
        self._stop_event = threading.Event()
        self._lost_event = threading.Event()
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
            try:
                self._lock.extend(_RUN_LEASE_SECONDS)
            except Exception:
                self._lost_event.set()
                logger.exception(
                    "Benchmark run %s lost its execution lease", self._run_id
                )
                return

    def ensure_owned(self) -> None:
        if self._lost_event.is_set() or not self._lock.owned():
            raise RuntimeError(f"Benchmark run {self._run_id} lost its execution lease")

    def stop(self) -> None:
        self._stop_event.set()
        if self._started:
            self._thread.join(timeout=5)


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
        # Keep worker startup limited to registering this task. The benchmark
        # runner deliberately imports the complete production chat pipeline and
        # should only be loaded when a run actually starts.
        from onyx.regulatory.benchmark.runner import run_benchmark

        with get_session_with_current_tenant() as db_session:
            run_benchmark(db_session, run_id)
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
