"""Container-local ownership and memory supervision of an analysis process."""

import fcntl
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass, field
from pathlib import Path
from threading import BoundedSemaphore, Event, Thread

from onyx.regulatory.amendments.memory_budget import (
    MemoryPolicy,
    MemorySample,
    ResourcePressure,
    read_memory,
)

RESOURCE_EXIT = 75
_SLOT = Path(tempfile.gettempdir()) / "onyx-amendment-analysis.lock"


# A stuck DB call cannot accumulate unbounded watchdog threads across retries.
_OWNER_PROBE_SLOT = BoundedSemaphore(1)
_OWNER_TIMEOUT_SECONDS = 2.0


@dataclass
class _OwnershipProbe:
    started_at: float = field(default_factory=time.monotonic)
    finished: Event = field(default_factory=Event)
    allowed: bool = False


def _start_probe(owned: Callable[[], bool]) -> _OwnershipProbe:
    if not _OWNER_PROBE_SLOT.acquire(blocking=False):
        raise ResourcePressure("ownership_probe_busy")
    probe = _OwnershipProbe()

    def run() -> None:
        try:
            probe.allowed = owned() is True
        except Exception:
            probe.allowed = False
        finally:
            probe.finished.set()
            _OWNER_PROBE_SLOT.release()

    context = copy_context()
    Thread(
        target=lambda: context.run(run), daemon=True, name="amendment-ownership"
    ).start()
    return probe


def _stop(child: subprocess.Popen[bytes]) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    except ProcessLookupError:
        child.wait(timeout=5)


def supervise(
    command: list[str],
    *,
    sample: Callable[[], MemorySample | None] = read_memory,
    policy: MemoryPolicy = MemoryPolicy(),
    lock_path: Path = _SLOT,
    owned: Callable[[], bool] = lambda: True,
    deadline_seconds: float = 3600,
) -> None:
    """The child inherits the flock, preventing overlap if its parent disappears."""
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    child: subprocess.Popen[bytes] | None = None
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ResourcePressure("analysis_slot_busy") from error
        initial_probe = _start_probe(owned)
        if (
            not initial_probe.finished.wait(_OWNER_TIMEOUT_SECONDS)
            or not initial_probe.allowed
        ):
            raise RuntimeError("Amendment analysis ownership unavailable")
        # Start one supervised process; its measured usage controls expansion.
        # Reserving estimated work twice here can prevent serial progress.
        policy.check(sample())
        environment = dict(os.environ)
        environment["PYTHONPATH"] = os.pathsep.join(sys.path)
        environment["AMENDMENT_SUPERVISOR_PID"] = str(os.getpid())
        child = subprocess.Popen(
            command, pass_fds=(descriptor,), start_new_session=True, env=environment
        )
        started = time.monotonic()
        next_owner_check = started
        probe: _OwnershipProbe | None = None
        while child.poll() is None:
            policy.check(sample())
            now = time.monotonic()
            if now - started >= deadline_seconds:
                raise ResourcePressure("analysis_time_budget")
            if probe is not None:
                if probe.finished.is_set():
                    if not probe.allowed:
                        raise RuntimeError("Amendment analysis ownership lost")
                    probe = None
                    next_owner_check = now + 2
                elif now - probe.started_at >= _OWNER_TIMEOUT_SECONDS:
                    raise RuntimeError(
                        "Amendment analysis ownership verification timed out"
                    )
            elif now >= next_owner_check:
                probe = _start_probe(owned)
            time.sleep(0.1)
        if child.returncode == RESOURCE_EXIT:
            raise ResourcePressure("analysis_resource_wait")
        if child.returncode != 0:
            raise RuntimeError(
                f"Amendment analysis process exited with code {child.returncode}"
            )
    except ResourcePressure as error:
        error.started = child is not None
        raise
    finally:
        if child is not None:
            _stop(child)
        os.close(descriptor)


def run_supervised_amendment(
    *,
    batch_id: int,
    lease_generation: int,
    tenant_id: str,
    parallel: bool,
) -> None:
    from onyx.db.amendment_resources import owns_analysis, record_analysis_resources
    from onyx.db.engine.sql_engine import get_session_with_current_tenant

    measurements: dict[str, int] = {}

    def tracked_sample() -> MemorySample | None:
        reading = read_memory()
        if reading is not None:
            measurements.update(
                current_bytes=reading.current,
                limit_bytes=reading.limit,
                peak_bytes=max(measurements.get("peak_bytes", 0), reading.current),
            )
        return reading

    def owned() -> bool:
        with get_session_with_current_tenant() as session:
            allowed = owns_analysis(
                session, batch_id=batch_id, lease_generation=lease_generation
            )
            if allowed and measurements:
                try:
                    record_analysis_resources(
                        session,
                        batch_id=batch_id,
                        lease_generation=lease_generation,
                        measurements=dict(measurements),
                    )
                except Exception:
                    session.rollback()
            return allowed

    supervise(
        [
            sys.executable,
            "-m",
            __name__,
            str(batch_id),
            str(lease_generation),
            tenant_id,
            "parallel" if parallel else "serial",
        ],
        owned=owned,
        sample=tracked_sample,
    )


def protect_parent_lifetime() -> None:
    """Stop the analysis if its supervisor dies, including during an LLM call."""
    import ctypes

    parent = int(os.environ["AMENDMENT_SUPERVISOR_PID"])
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Supervised amendment execution requires Linux")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0 or os.getppid() != parent:
        raise RuntimeError("Amendment supervisor unavailable")


def _child_main() -> None:
    protect_parent_lifetime()
    from onyx.utils.variable_functionality import set_is_ee_based_on_env_variable

    # Spawned processes must select the same secret codec as the Celery entrypoint.
    set_is_ee_based_on_env_variable()

    from functools import partial

    from onyx.context.search.retrieval.concurrency import limit_search_concurrency
    from onyx.db.amendment_analysis_settings import get_batch_analysis_model
    from onyx.db.amendment_resources import record_analysis_resources
    from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
    from onyx.regulatory.amendments.analysis_llm import use_analysis_model
    from onyx.regulatory.amendments.job import run_amendment_batch
    from onyx.regulatory.amendments.memory_budget import (
        INITIAL_ANALYSIS_PARALLELISM,
        MAX_ANALYSIS_PARALLELISM,
        bounded_map,
    )
    from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

    batch_id, generation, tenant_id, mode = sys.argv[1:]
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(tenant_id)
    SqlEngine.reset_engine()
    SqlEngine.set_app_name("amendment_analysis_child")
    # Ten instructions share twelve inner search slots. Keep capacity for
    # nested publication reads and checkpoints; burst connections close on return.
    SqlEngine.init_engine(pool_size=24, max_overflow=24, pool_pre_ping=True)
    policy = MemoryPolicy()

    def report(measurements: dict[str, int]) -> None:
        from onyx.utils.logger import setup_logger

        try:
            with get_session_with_current_tenant() as session:
                record_analysis_resources(
                    session,
                    batch_id=int(batch_id),
                    lease_generation=int(generation),
                    measurements=measurements,
                )
        except Exception:
            setup_logger().warning("Amendment resource snapshot could not be stored")

    def check_resources() -> None:
        policy.check(read_memory())

    try:
        selected_model = get_batch_analysis_model(int(batch_id))
        with (
            use_analysis_model(selected_model.value),
            limit_search_concurrency(12, check_resources=check_resources),
        ):
            run_amendment_batch(
                batch_id=int(batch_id),
                lease_generation=int(generation),
                instruction_runner=partial(
                    bounded_map,
                    max_parallel=MAX_ANALYSIS_PARALLELISM
                    if mode == "parallel"
                    else INITIAL_ANALYSIS_PARALLELISM,
                    report=report,
                ),
                check_resources=check_resources,
                before_work=check_resources,
            )
    except ResourcePressure:
        raise SystemExit(RESOURCE_EXIT) from None
    except Exception as error:
        from onyx.db.regulatory_amendments import record_analysis_child_failure
        from onyx.utils.logger import setup_logger

        try:
            with get_session_with_current_tenant() as session:
                record_analysis_child_failure(
                    session,
                    batch_id=int(batch_id),
                    lease_generation=int(generation),
                    failure=error,
                )
        except Exception:
            setup_logger().exception(
                "Amendment child failure receipt could not be stored"
            )
        raise
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


if __name__ == "__main__":
    _child_main()
