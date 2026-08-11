import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from sqlalchemy.orm import Session

from onyx.db.enums import BenchmarkRunStatus
from onyx.db.models import BenchmarkRun
from onyx.db.regulatory_benchmark import (
    cancel_benchmark_run,
    get_benchmark_run,
    get_benchmark_run_for_update,
    mark_benchmark_run_failed,
    reset_benchmark_run_for_retry,
)
from onyx.regulatory.benchmark.runner import run_benchmark
from tests.external_dependency_unit.conftest import create_test_user


def test_cancelled_run_cannot_be_overwritten_by_waiting_failure_transition(
    db_session: Session,
) -> None:
    run = BenchmarkRun(
        status=BenchmarkRunStatus.RUNNING.value,
        judge_provider="test-provider",
        judge_model="test-model",
        total_items=0,
    )
    db_session.add(run)
    db_session.commit()
    run_id = run.id
    engine = db_session.get_bind()

    cancel_session = Session(engine)
    locked_run = get_benchmark_run_for_update(cancel_session, run_id)
    assert locked_run is not None

    def fail_run() -> None:
        with Session(engine) as failure_session:
            mark_benchmark_run_failed(
                failure_session, run_id, "persona selection exploded"
            )

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            failure = executor.submit(fail_run)
            try:
                failure.result(timeout=0.2)
            except TimeoutError:
                pass
            else:
                raise AssertionError("failure transition did not wait for the row lock")

            cancel_benchmark_run(cancel_session, run_id)
            failure.result(timeout=5)

        db_session.expire_all()
        persisted = get_benchmark_run(db_session, run_id)
        assert persisted is not None
        assert persisted.status == BenchmarkRunStatus.CANCELLED.value
        assert persisted.report_error is None
        assert persisted.failure_message is None
    finally:
        cancel_session.close()
        persisted = db_session.get(BenchmarkRun, run_id)
        if persisted is not None:
            db_session.delete(persisted)
            db_session.commit()


def test_retry_dispatch_lock_blocks_worker_until_reset_commits(
    db_session: Session,
) -> None:
    creator = create_test_user(db_session, "benchmark-retry-creator")
    run = BenchmarkRun(
        status=BenchmarkRunStatus.ERROR.value,
        judge_provider="test-provider",
        judge_model="test-model",
        created_by=creator.id,
        total_items=0,
        failure_code="execution_failed",
        failure_message="previous failure",
        completed_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db_session.add(run)
    db_session.commit()
    run_id = run.id
    engine = db_session.get_bind()

    dispatch_session = Session(engine)
    locked_run = get_benchmark_run_for_update(dispatch_session, run_id)
    assert locked_run is not None
    reset_benchmark_run_for_retry(locked_run)
    locked_run.status = BenchmarkRunStatus.QUEUED.value
    locked_run.queued_at = datetime.datetime.now(datetime.timezone.utc)

    def run_worker() -> None:
        with Session(engine) as worker_session:
            run_benchmark(worker_session, run_id)

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            worker = executor.submit(run_worker)
            try:
                worker.result(timeout=0.2)
            except TimeoutError:
                pass
            else:
                raise AssertionError("worker did not wait for the dispatch row lock")

            dispatch_session.commit()
            worker.result(timeout=5)

        db_session.expire_all()
        persisted = get_benchmark_run(db_session, run_id)
        assert persisted is not None
        assert persisted.status == BenchmarkRunStatus.ERROR.value
        assert persisted.completed_at is not None
        assert persisted.report_error is None
        assert persisted.failure_code == "execution_failed"
        assert persisted.failure_message == "One or more benchmark items failed"
    finally:
        dispatch_session.close()
        persisted = db_session.get(BenchmarkRun, run_id)
        if persisted is not None:
            db_session.delete(persisted)
        db_session.delete(creator)
        db_session.commit()
