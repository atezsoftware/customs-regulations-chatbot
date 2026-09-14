from __future__ import annotations

import datetime
import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable
from uuid import UUID

from onyx.db import regulatory_labeling as repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.labeling_configuration import (
    LabelingProviderBinding,
    resolve_labeling_gateway,
)
from onyx.db.models import RegulatoryLabelingItem
from onyx.db.regulatory_labeling_results import LabelingResultSpool
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayError,
    IndexingGatewayIndeterminateSubmissionError,
)
from onyx.regulatory.indexing_jobs.retry import classify_indexing_error
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchState,
    vertex_batch_submission_key,
    vertex_jsonl_line_size,
)
from onyx.regulatory.labeling.domain import shard_by_count_and_bytes
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()

LABELING_LEASE_SECONDS = 300
LABELING_PREPARATION_PAGE = 128
LABELING_SHARD_ITEMS = 64
LABELING_SHARD_BYTES = 8 * 1024 * 1024
LABELING_VERTEX_BATCH_ITEMS = 200_000
LABELING_VERTEX_BATCH_BYTES = 1_000_000_000
LABELING_HEARTBEAT_SECONDS = 30
LABELING_MAX_IN_FLIGHT = 4
LABELING_PROJECTION_PAGE = 128
LABELING_RECONCILE_SECONDS = 10 * 60
LABELING_POLL_SECONDS = 30
LABELING_MAX_PROVIDER_FAILURES = 5


@runtime_checkable
class StreamingLabelingGateway(Protocol):
    def submit_stream(
        self,
        requests: Iterable[VertexBatchRequest],
        *,
        submission_key: str,
        max_jsonl_bytes: int,
        on_progress: Callable[[], None],
    ) -> VertexBatchState: ...


def _lease_heartbeat(lease: repository.RunLease) -> Callable[[], None]:
    last_renewed = 0.0

    def heartbeat() -> None:
        nonlocal last_renewed
        now = time.monotonic()
        if now - last_renewed < LABELING_HEARTBEAT_SECONDS:
            return
        with get_session_with_current_tenant() as session:
            run = repository.load_claimed_run(session, lease)
            if run.cancel_requested:
                raise repository.LabelingCancellationRequested(
                    "The labeling run was cancelled"
                )
            repository.renew_run_lease(
                session, lease, lease_seconds=LABELING_LEASE_SECONDS
            )
            session.commit()
        last_renewed = now

    return heartbeat


def _stream_shard_requests(
    lease: repository.RunLease,
    shard_id: UUID,
    heartbeat: Callable[[], None],
) -> Iterator[VertexBatchRequest]:
    after_id: UUID | None = None
    while True:
        heartbeat()
        with get_session_with_current_tenant() as session:
            items = repository.load_shard_request_page(
                session, lease, shard_id, after_id=after_id
            )
            requests = []
            for item in items:
                request = VertexBatchRequest.model_validate(item.request_payload)
                if request.request_hash != item.request_hash:
                    raise ValueError("The frozen labeling request changed")
                requests.append(request)
            if items:
                after_id = items[-1].id
        if not requests:
            return
        yield from requests


class LabelingStepOutcome(StrEnum):
    NEXT = "next"
    WAIT = "wait"
    SKIPPED = "skipped"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class LabelingStepResult:
    run_id: UUID
    outcome: LabelingStepOutcome
    expected_generation: int | None = None
    countdown_seconds: float = 0


def _provider_error_is_terminal(
    error: IndexingGatewayError, failure_count: int
) -> bool:
    return (
        not classify_indexing_error(error).retryable
        or failure_count + 1 >= LABELING_MAX_PROVIDER_FAILURES
    )


def _next(
    lease: repository.RunLease, *, countdown_seconds: float = 0
) -> LabelingStepResult:
    return LabelingStepResult(
        run_id=lease.run_id,
        outcome=(
            LabelingStepOutcome.WAIT
            if countdown_seconds > 0
            else LabelingStepOutcome.NEXT
        ),
        expected_generation=lease.generation,
        countdown_seconds=countdown_seconds,
    )


def _gateway_for_claimed_run(
    lease: repository.RunLease,
) -> VertexBatchGateway:
    with get_session_with_current_tenant() as session:
        run = repository.load_claimed_run(session, lease)
        if run.model_configuration_id is None:
            raise ValueError("The labeling model configuration was removed")
        binding = LabelingProviderBinding.model_validate(run.provider_binding)
        user = repository.get_claimed_run_requester(session, lease)
        if user is None:
            raise ValueError("The user who requested labeling is unavailable")
        return resolve_labeling_gateway(
            session,
            run.model_configuration_id,
            user=user,
            expected_binding=binding,
        )


def _release(
    lease: repository.RunLease,
    *,
    stage: str | None = None,
    status: str | None = None,
    retry_after_seconds: float | None = None,
    error: str | None = None,
    finished: bool = False,
) -> None:
    with get_session_with_current_tenant() as session:
        repository.release_run(
            session,
            lease,
            stage=stage,
            status=status,
            retry_after_seconds=retry_after_seconds,
            error=error,
            finished=finished,
        )
        session.commit()


def _prepare(lease: repository.RunLease, tenant_id: str) -> LabelingStepResult:
    with get_session_with_current_tenant() as session:
        run = repository.load_claimed_run(session, lease)
        if run.cancel_requested:
            repository.cancel_claimed_run(session, lease)
            repository.release_run(
                session,
                lease,
                status="cancelled",
                finished=True,
            )
            session.commit()
            return LabelingStepResult(
                run_id=lease.run_id, outcome=LabelingStepOutcome.TERMINAL
            )
        taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
        items = repository.prepare_next_item_page(
            session, lease, limit=LABELING_PREPARATION_PAGE
        )
        prepared: list[tuple[str, VertexBatchRequest]] = []
        item_by_string_id: dict[str, RegulatoryLabelingItem] = {}
        failed: dict[UUID, str] = {}
        for item in items:
            item_by_string_id[str(item.id)] = item
            try:
                if item.canonical_text_sha256 is None or item.context_snapshot is None:
                    raise ValueError("The labeling source snapshot is incomplete")
                request = build_labeling_request(
                    chunk_id=item.regulatory_chunk_id,
                    text=item.text_snapshot,
                    context=item.context_snapshot,
                    taxonomy=taxonomy,
                    source_hash=item.canonical_text_sha256,
                )
            except ValueError as error:
                failed[item.id] = str(error)
            else:
                prepared.append((str(item.id), request))
        grouped = shard_by_count_and_bytes(
            prepared,
            item_limit=LABELING_SHARD_ITEMS,
            byte_limit=LABELING_SHARD_BYTES,
            size_of=vertex_jsonl_line_size,
        )
        next_ordinal = repository.next_shard_ordinal(session, lease)
        prepared_requests = [
            repository.PreparedRequest(
                item_id=item_by_string_id[item_id].id,
                request_hash=request.request_hash,
                request_payload=request.model_dump(
                    mode="json", exclude_computed_fields=True
                ),
            )
            for item_id, request in prepared
        ]
        shards: list[repository.PreparedShard] = []
        for offset, group in enumerate(grouped):
            ordinal = next_ordinal + offset
            requests = [request for _, request in group]
            key = vertex_batch_submission_key(
                requests,
                tenant_id=tenant_id,
                job_id=run.id,
                output_prefix=f"regulatory-labeling/{run.id}/{ordinal}",
                submission_attempt=1,
            ).replace("regulatory-context-", "regulatory-labeling-", 1)
            shards.append(
                repository.PreparedShard(
                    ordinal=ordinal,
                    item_ids=tuple(
                        item_by_string_id[item_id].id for item_id, _ in group
                    ),
                    submission_key=key,
                )
            )
        next_stage = repository.store_prepared_shards(
            session,
            lease,
            requests=prepared_requests,
            shards=shards,
            failed_items=failed,
        )
        repository.release_run(session, lease, stage=next_stage)
        session.commit()
    return _next(lease)


def _cancel(
    lease: repository.RunLease, gateway: VertexBatchGateway | None
) -> LabelingStepResult:
    with get_session_with_current_tenant() as session:
        shard = repository.next_cancellable_shard(session, lease)
        if shard is None:
            repository.cancel_claimed_run(session, lease)
            repository.release_run(session, lease, status="cancelled", finished=True)
            session.commit()
            return LabelingStepResult(
                run_id=lease.run_id, outcome=LabelingStepOutcome.TERMINAL
            )
        if shard.status == "submitting":
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard.id,
                status="reconcile_required",
                reconcile_seconds=LABELING_RECONCILE_SECONDS,
            )
        shard_id = shard.id
        remote_job_name = shard.remote_job_name
        submission_key = shard.submission_key
        reconcile_until = shard.reconcile_until
        shard_status = shard.status
        failure_count = shard.failure_count
        session.commit()

    terminal_error: str | None = None
    if gateway is None:
        terminal_error = "Cancelled locally; provider cancellation is unavailable"
    elif remote_job_name is None and shard_status in (
        "submitting",
        "reconcile_required",
    ):
        try:
            state = gateway.reconcile_submission(submission_key)
        except IndexingGatewayError as error:
            within_deadline = (
                reconcile_until is not None
                and datetime.datetime.now(datetime.timezone.utc) < reconcile_until
            )
            if (
                not _provider_error_is_terminal(error, failure_count)
                and within_deadline
            ):
                with get_session_with_current_tenant() as session:
                    repository.record_shard_state(
                        session,
                        lease,
                        shard_id=shard_id,
                        status="reconcile_required",
                        retry_after_seconds=LABELING_POLL_SECONDS,
                        error=str(error),
                        increment_failure=True,
                    )
                    repository.release_run(
                        session, lease, retry_after_seconds=LABELING_POLL_SECONDS
                    )
                    session.commit()
                return _next(lease, countdown_seconds=LABELING_POLL_SECONDS)
            terminal_error = (
                "Cancelled locally; provider submission reconciliation failed"
            )
        else:
            if state is None:
                within_deadline = (
                    reconcile_until is not None
                    and datetime.datetime.now(datetime.timezone.utc) < reconcile_until
                )
                if within_deadline:
                    with get_session_with_current_tenant() as session:
                        repository.record_shard_state(
                            session,
                            lease,
                            shard_id=shard_id,
                            status="reconcile_required",
                            retry_after_seconds=LABELING_POLL_SECONDS,
                        )
                        repository.release_run(
                            session, lease, retry_after_seconds=LABELING_POLL_SECONDS
                        )
                        session.commit()
                    return _next(lease, countdown_seconds=LABELING_POLL_SECONDS)
                terminal_error = (
                    "Cancelled locally; provider submission could not be reconciled"
                )
            else:
                remote_job_name = state.remote_job_name

    if remote_job_name and gateway is not None:
        try:
            gateway.cancel(remote_job_name)
        except IndexingGatewayError as error:
            with get_session_with_current_tenant() as session:
                current = repository.load_claimed_shard(session, lease, shard_id)
                failures = current.failure_count
                if not _provider_error_is_terminal(error, failures):
                    repository.record_shard_state(
                        session,
                        lease,
                        shard_id=shard_id,
                        status="submitted",
                        remote_job_name=remote_job_name,
                        retry_after_seconds=LABELING_POLL_SECONDS,
                        error=str(error),
                        increment_failure=True,
                    )
                    repository.release_run(
                        session, lease, retry_after_seconds=LABELING_POLL_SECONDS
                    )
                    session.commit()
                    return _next(lease, countdown_seconds=LABELING_POLL_SECONDS)
            terminal_error = "Cancelled locally; provider cancellation failed"
    with get_session_with_current_tenant() as session:
        repository.record_shard_state(
            session,
            lease,
            shard_id=shard_id,
            status="cancelled",
            remote_job_name=remote_job_name,
            error=terminal_error or "Cancelled by administrator",
        )
        repository.release_run(session, lease)
        session.commit()
    return _next(lease)


def _submit_or_reconcile(
    lease: repository.RunLease,
    shard_id: UUID,
    gateway: VertexBatchGateway,
    tenant_id: str,
) -> LabelingStepResult:
    with get_session_with_current_tenant() as session:
        run = repository.load_claimed_run(session, lease)
        native = (
            LabelingProviderBinding.model_validate(run.provider_binding).transport
            == "vertex_gcs_v1"
        )
        shard = repository.load_claimed_shard(session, lease, shard_id)
        if shard.status == "prepared":
            if native:
                if not isinstance(gateway, StreamingLabelingGateway):
                    raise ValueError("Native labeling requires streaming Batch support")
                shard = repository.coalesce_prepared_shards(
                    session,
                    lease,
                    tenant_id=tenant_id,
                    anchor_shard_id=shard_id,
                    max_items=LABELING_VERTEX_BATCH_ITEMS,
                    max_jsonl_bytes=LABELING_VERTEX_BATCH_BYTES,
                )
                session.commit()
                session.expire_all()
            shard = repository.mark_shard_submitting(
                session,
                lease,
                shard_id=shard_id,
                reconcile_seconds=LABELING_RECONCILE_SECONDS,
            )
            requests = []
            if not native:
                items = repository.load_shard_requests(session, lease, shard.id)
                requests = [
                    VertexBatchRequest.model_validate(item.request_payload)
                    for item in items
                    if item.request_payload is not None
                ]
            submission_key = shard.submission_key
            session.commit()
            try:
                if native and isinstance(gateway, StreamingLabelingGateway):
                    heartbeat = _lease_heartbeat(lease)
                    state = gateway.submit_stream(
                        _stream_shard_requests(lease, shard_id, heartbeat),
                        submission_key=submission_key,
                        max_jsonl_bytes=LABELING_VERTEX_BATCH_BYTES,
                        on_progress=heartbeat,
                    )
                else:
                    state = gateway.submit(
                        requests,
                        submission_key=submission_key,
                        max_jsonl_bytes=LABELING_SHARD_BYTES,
                    )
            except repository.LabelingCancellationRequested:
                if not native:
                    raise
                # Native transport wraps every post-create callback failure as uncertain.
                with get_session_with_current_tenant() as result_session:
                    repository.record_shard_state(
                        result_session,
                        lease,
                        shard_id=shard_id,
                        status="cancelled",
                        error="Cancelled before provider job creation",
                    )
                    repository.release_run(result_session, lease)
                    result_session.commit()
                return _next(lease)
            except IndexingGatewayIndeterminateSubmissionError:
                with get_session_with_current_tenant() as result_session:
                    repository.record_shard_state(
                        result_session,
                        lease,
                        shard_id=shard_id,
                        status="reconcile_required",
                        reconcile_seconds=LABELING_RECONCILE_SECONDS,
                        retry_after_seconds=LABELING_POLL_SECONDS,
                        error="Provider submission must be reconciled",
                    )
                    repository.release_run(
                        result_session,
                        lease,
                        stage="submitting",
                    )
                    result_session.commit()
                return _next(lease)
            except IndexingGatewayError as error:
                with get_session_with_current_tenant() as result_session:
                    current = repository.load_claimed_shard(
                        result_session, lease, shard_id
                    )
                    failures = current.failure_count
                    terminal = _provider_error_is_terminal(error, failures)
                    repository.record_shard_state(
                        result_session,
                        lease,
                        shard_id=shard_id,
                        status="failed" if terminal else "prepared",
                        retry_after_seconds=None if terminal else LABELING_POLL_SECONDS,
                        error=str(error),
                        increment_failure=True,
                    )
                    repository.release_run(
                        result_session,
                        lease,
                        stage="submitting",
                    )
                    result_session.commit()
                return _next(lease)
            except Exception:
                with get_session_with_current_tenant() as result_session:
                    repository.record_shard_state(
                        result_session,
                        lease,
                        shard_id=shard_id,
                        status="reconcile_required",
                        reconcile_seconds=LABELING_RECONCILE_SECONDS,
                        retry_after_seconds=LABELING_POLL_SECONDS,
                        error="Provider submission outcome is uncertain",
                    )
                    repository.release_run(
                        result_session,
                        lease,
                        stage="submitting",
                    )
                    result_session.commit()
                return _next(lease)
            with get_session_with_current_tenant() as result_session:
                repository.record_shard_state(
                    result_session,
                    lease,
                    shard_id=shard_id,
                    status="submitted",
                    remote_job_name=state.remote_job_name,
                    input_uri=state.input_uri,
                    output_uri=state.output_uri,
                )
                repository.release_run(result_session, lease, stage="submitting")
                result_session.commit()
            return _next(lease)

        if shard.status == "submitting":
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="reconcile_required",
                reconcile_seconds=LABELING_RECONCILE_SECONDS,
            )
        submission_key = shard.submission_key
        reconcile_until = shard.reconcile_until
        session.commit()

    try:
        state = gateway.reconcile_submission(submission_key)
    except IndexingGatewayError as error:
        with get_session_with_current_tenant() as session:
            shard = repository.load_claimed_shard(session, lease, shard_id)
            failures = shard.failure_count
            terminal = (
                _provider_error_is_terminal(error, failures)
                or reconcile_until is None
                or datetime.datetime.now(datetime.timezone.utc) >= reconcile_until
            )
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="failed" if terminal else "reconcile_required",
                retry_after_seconds=None if terminal else LABELING_POLL_SECONDS,
                error=str(error),
                increment_failure=True,
            )
            repository.release_run(
                session,
                lease,
                stage="submitting",
            )
            session.commit()
        return _next(lease)
    if state is None:
        terminal = (
            reconcile_until is None
            or datetime.datetime.now(datetime.timezone.utc) >= reconcile_until
        )
        with get_session_with_current_tenant() as session:
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="failed" if terminal else "reconcile_required",
                retry_after_seconds=None if terminal else LABELING_POLL_SECONDS,
                error=(
                    "Provider submission visibility remained indeterminate"
                    if terminal
                    else None
                ),
            )
            repository.release_run(
                session,
                lease,
                stage="submitting",
            )
            session.commit()
        return _next(lease)
    with get_session_with_current_tenant() as session:
        repository.record_shard_state(
            session,
            lease,
            shard_id=shard_id,
            status="submitted",
            remote_job_name=state.remote_job_name,
            input_uri=state.input_uri,
            output_uri=state.output_uri,
        )
        repository.release_run(session, lease, stage="submitting")
        session.commit()
    return _next(lease)


def _poll_or_apply(
    lease: repository.RunLease,
    shard_id: UUID,
    gateway: VertexBatchGateway,
) -> LabelingStepResult:
    with get_session_with_current_tenant() as session:
        shard = repository.load_claimed_shard(session, lease, shard_id)
        if not shard.remote_job_name:
            raise ValueError("Submitted labeling shard has no provider job")
        remote_job_name = shard.remote_job_name
        session.commit()
    try:
        state = gateway.get(remote_job_name)
    except IndexingGatewayError as error:
        with get_session_with_current_tenant() as session:
            shard = repository.load_claimed_shard(session, lease, shard_id)
            failures = shard.failure_count
            terminal = _provider_error_is_terminal(error, failures)
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="failed" if terminal else "submitted",
                retry_after_seconds=None if terminal else LABELING_POLL_SECONDS,
                error=str(error),
                increment_failure=True,
            )
            repository.release_run(
                session,
                lease,
                stage="waiting",
            )
            session.commit()
        return _next(lease)
    if state.status in (
        VertexBatchJobStatus.PENDING,
        VertexBatchJobStatus.RUNNING,
        VertexBatchJobStatus.CANCELLING,
    ):
        with get_session_with_current_tenant() as session:
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="submitted",
                remote_job_name=state.remote_job_name,
                input_uri=state.input_uri,
                output_uri=state.output_uri,
                retry_after_seconds=LABELING_POLL_SECONDS,
            )
            repository.release_run(
                session,
                lease,
                stage="waiting",
            )
            session.commit()
        return _next(lease)
    if state.status is not VertexBatchJobStatus.SUCCEEDED or not state.output_uri:
        with get_session_with_current_tenant() as session:
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="failed",
                error=f"Provider batch ended with {state.status.value}",
            )
            repository.release_run(session, lease, stage="waiting")
            session.commit()
        return _next(lease)

    try:
        _read_and_apply_results(lease, shard_id, gateway, state.output_uri)
    except IndexingGatewayError as error:
        with get_session_with_current_tenant() as session:
            shard = repository.load_claimed_shard(session, lease, shard_id)
            terminal = _provider_error_is_terminal(error, shard.failure_count)
            repository.record_shard_state(
                session,
                lease,
                shard_id=shard_id,
                status="failed" if terminal else "submitted",
                retry_after_seconds=None if terminal else LABELING_POLL_SECONDS,
                error=str(error),
                increment_failure=True,
            )
            repository.release_run(session, lease, stage="waiting")
            session.commit()
        return _next(lease)
    with get_session_with_current_tenant() as session:
        repository.finalize_shard_results(session, lease, shard_id)
        repository.release_run(session, lease, stage="waiting")
        session.commit()
    return _next(lease)


def _result_pages(
    lease: repository.RunLease,
    shard_id: UUID,
    heartbeat: Callable[[], None],
) -> Iterator[list[RegulatoryLabelingItem]]:
    after_id: UUID | None = None
    while True:
        heartbeat()
        with get_session_with_current_tenant() as session:
            items = repository.load_shard_result_items(
                session, lease, shard_id, after_id=after_id
            )
        if not items:
            return
        after_id = items[-1].id
        yield items


def _read_and_apply_results(
    lease: repository.RunLease,
    shard_id: UUID,
    gateway: VertexBatchGateway,
    output_uri: str,
) -> None:
    heartbeat = _lease_heartbeat(lease)
    heartbeat()
    with get_session_with_current_tenant() as session:
        run = repository.load_claimed_run(session, lease)
        repository.set_claimed_run_stage(session, lease, "applying")
        taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
        session.commit()
    with LabelingResultSpool() as spool:
        for items in _result_pages(lease, shard_id, heartbeat):
            spool.add_expected(
                item.request_hash for item in items if item.request_hash is not None
            )
        batch_error: str | None = None
        try:
            spool.stage(gateway.read_results(output_uri), on_progress=heartbeat)
        except ValueError as error:
            batch_error = str(error)
        # Validate the complete output contract before committing any assignments.
        for items in _result_pages(lease, shard_id, heartbeat):
            hashes = [
                item.request_hash for item in items if item.request_hash is not None
            ]
            parsed = spool.get_many(hashes) if batch_error is None else {}
            outcomes: dict[str, tuple[list[str], list[dict[str, object]]] | str] = {}
            for item in items:
                request_hash = item.request_hash
                if request_hash is None:
                    continue
                if batch_error is not None:
                    outcomes[request_hash] = batch_error
                    continue
                result = parsed.get(request_hash)
                if result is None:
                    continue
                if result.error is not None or result.context is None:
                    outcomes[request_hash] = (
                        f"Provider result was rejected: {result.error or 'empty'}"
                    )
                    continue
                try:
                    outcome = validate_labeling_response(
                        result.context, text=item.text_snapshot, taxonomy=taxonomy
                    )
                except ValueError as error:
                    outcomes[request_hash] = str(error)
                    continue
                if outcome.abstained:
                    outcomes[request_hash] = "Model abstained: insufficient evidence"
                    continue
                assignments = [assignment.model_dump() for assignment in outcome.labels]
                outcomes[request_hash] = (
                    [assignment.label_id for assignment in outcome.labels],
                    assignments,
                )
            heartbeat()
            with get_session_with_current_tenant() as session:
                repository.apply_shard_result_page(
                    session,
                    lease,
                    shard_id=shard_id,
                    item_ids=[item.id for item in items],
                    outcomes=outcomes,
                )
                session.commit()


def _process_provider(lease: repository.RunLease, tenant_id: str) -> LabelingStepResult:
    with get_session_with_current_tenant() as session:
        run = repository.load_claimed_run(session, lease)
        cancel_requested = run.cancel_requested
        session.commit()
    if cancel_requested:
        try:
            gateway = _gateway_for_claimed_run(lease)
        except ValueError:
            gateway = None
        return _cancel(lease, gateway)
    gateway = _gateway_for_claimed_run(lease)
    with get_session_with_current_tenant() as session:
        repository.load_claimed_run(session, lease)
        shard = repository.next_due_shard(
            session, lease, max_in_flight=LABELING_MAX_IN_FLIGHT
        )
        if shard is None:
            if repository.all_shards_terminal(session, lease):
                repository.release_run(session, lease, stage="projecting")
                session.commit()
                return _next(lease)
            repository.release_run(
                session,
                lease,
                stage="waiting",
                retry_after_seconds=LABELING_POLL_SECONDS,
            )
            session.commit()
            return _next(lease, countdown_seconds=LABELING_POLL_SECONDS)
        shard_id = shard.id
        status = shard.status
        session.commit()
    if status in ("prepared", "submitting", "reconcile_required"):
        return _submit_or_reconcile(lease, shard_id, gateway, tenant_id)
    return _poll_or_apply(lease, shard_id, gateway)


def _project_and_finish(lease: repository.RunLease) -> LabelingStepResult:
    with get_session_with_current_tenant() as session:
        run = repository.load_claimed_run(session, lease)
        if run.cancel_requested:
            repository.cancel_claimed_run(session, lease)
            status = "cancelled"
        else:
            repository.project_derived_labels(
                session, lease, limit=LABELING_PROJECTION_PAGE
            )
            if repository.has_pending_derived_projections(session, lease):
                repository.release_run(session, lease, stage="projecting")
                session.commit()
                return _next(lease)
            status = repository.final_status(session, lease)
        repository.release_run(
            session,
            lease,
            status=status,
            finished=True,
        )
        session.commit()
    return LabelingStepResult(run_id=lease.run_id, outcome=LabelingStepOutcome.TERMINAL)


def run_labeling_step(
    run_id: UUID, expected_generation: int, tenant_id: str
) -> LabelingStepResult:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    with get_session_with_current_tenant() as session:
        lease = repository.claim_run(
            session,
            run_id=run_id,
            expected_generation=expected_generation,
            lease_seconds=LABELING_LEASE_SECONDS,
        )
        session.commit()
    if lease is None:
        return LabelingStepResult(run_id=run_id, outcome=LabelingStepOutcome.SKIPPED)
    try:
        with get_session_with_current_tenant() as session:
            run = repository.load_claimed_run(session, lease)
            stage = run.stage
            session.commit()
        if stage == "preparing":
            return _prepare(lease, tenant_id)
        if stage in ("submitting", "waiting", "applying"):
            return _process_provider(lease, tenant_id)
        if stage == "projecting":
            return _project_and_finish(lease)
        _release(
            lease,
            status="failed",
            error=f"Unsupported labeling stage: {stage}",
            finished=True,
        )
        return LabelingStepResult(run_id=run_id, outcome=LabelingStepOutcome.TERMINAL)
    except repository.LabelingCancellationRequested:
        try:
            gateway = _gateway_for_claimed_run(lease)
        except ValueError:
            gateway = None
        return _cancel(lease, gateway)
    except repository.LabelingStateConflictError:
        return LabelingStepResult(run_id=run_id, outcome=LabelingStepOutcome.SKIPPED)
    except Exception as error:
        logger.exception("Durable regulatory labeling step failed")
        try:
            _release(
                lease,
                status="failed",
                error=f"Labeling worker failed: {type(error).__name__}",
                finished=True,
            )
        except repository.LabelingStateConflictError:
            return LabelingStepResult(
                run_id=run_id, outcome=LabelingStepOutcome.SKIPPED
            )
        return LabelingStepResult(run_id=run_id, outcome=LabelingStepOutcome.TERMINAL)
