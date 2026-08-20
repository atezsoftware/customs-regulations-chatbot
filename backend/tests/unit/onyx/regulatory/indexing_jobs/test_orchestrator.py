from __future__ import annotations

import datetime
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import Iterator, cast
from unittest.mock import ANY, MagicMock, patch
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.enums import (
    RegulatoryIndexingCancellationPhase,
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
    UserFileStatus,
)
from onyx.db.models import (
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
    SearchSettings,
    UserFile,
)
from onyx.regulatory.indexing_jobs.contextual import ContextApplySummary
from onyx.regulatory.indexing_jobs.embedding import EmbeddingSummary
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayConnectionError,
    IndexingGatewayIndeterminateSubmissionError,
    RegulatoryIndexingConfigSnapshot,
)
from onyx.regulatory.indexing_jobs.orchestrator import (
    OrchestrationOutcome,
    run_preclaimed_regulatory_indexing_step,
    run_regulatory_indexing_step,
)
from onyx.regulatory.indexing_jobs.publisher import PublishOutcome
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchResult,
    VertexBatchResultError,
    VertexBatchState,
)

_NOW = datetime.datetime(2026, 8, 19, tzinfo=datetime.timezone.utc)


def _snapshot() -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot.model_validate(
        {
            "input_content_hash": "1" * 64,
            "input_hash_version": "canonical-v2",
            "chunk_generation_hash": "2" * 64,
            "search_settings_id": 9,
            "embedding_provider": "openrouter",
            "embedding_model_name": "openai/text-embedding-3-large",
            "model_dimension": 3072,
            "reduced_dimension": 1024,
            "effective_dimension": 1024,
            "index_name": "primary",
            "vertex": {
                "model_configuration_id": 4,
                "model_name": "gemini-3.6-flash",
                "project": "project",
                "location": "europe-west1",
                "authentication_mode": "workload_identity",
                "gcs_uri": "gs://bucket/regulatory-indexing",
            },
            "prompt_version": "contextual-rag-v1",
            "prompt_hash": "a" * 64,
            "max_attempts": 3,
            "retry_base_seconds": 15,
            "retry_max_seconds": 900,
            "poll_seconds": 30,
            "lease_seconds": 120,
            "embedding_request_size": 2,
        }
    )


def _runtime(
    stage: RegulatoryIndexingStage,
    *,
    status: RegulatoryIndexingJobStatus = RegulatoryIndexingJobStatus.RUNNING,
    generation: int = 3,
    attempt_count: int = 0,
    remote_job_name: str | None = None,
    submission_key: str | None = None,
    submission_state: str = "NONE",
    cancellation_phase: str = "NONE",
    user_file_status: UserFileStatus = UserFileStatus.INDEXING,
) -> indexing_job_repository.RegulatoryIndexingRuntime:
    job_id = uuid4()
    user_file_id = uuid4()
    job = SimpleNamespace(
        id=job_id,
        user_file_id=user_file_id,
        stage=stage.value,
        status=status.value,
        lease_generation=generation,
        attempt_count=attempt_count,
        content_hash="1" * 64,
        chunk_generation_hash="2" * 64,
        config_snapshot=_snapshot().model_dump(mode="json"),
        remote_vertex_job_name=remote_job_name,
        vertex_input_uri=None,
        vertex_output_uri=None,
        vertex_submission_key=submission_key,
        vertex_submission_state=submission_state,
        cancellation_phase=cancellation_phase,
    )
    user_file = SimpleNamespace(
        id=user_file_id,
        file_id="file-key",
        name="regulation.md",
        status=user_file_status,
        chunk_count=2,
    )
    return indexing_job_repository.RegulatoryIndexingRuntime(
        job=cast(RegulatoryIndexingJob, job),
        user_file=cast(UserFile, user_file),
        search_settings=cast(SearchSettings, SimpleNamespace(id=9)),
        regulatory_chunks=(),
        indexing_items=(),
    )


@contextmanager
def _session_context(session: Session) -> Iterator[Session]:
    yield session


def _patch_session(session: Session) -> AbstractContextManager[MagicMock]:
    return patch(
        "onyx.regulatory.indexing_jobs.orchestrator.get_session_with_current_tenant",
        return_value=_session_context(session),
    )


@contextmanager
def _external_lease(*_args: object, **_kwargs: object) -> Iterator[object]:
    yield SimpleNamespace(commit=lambda: None)


def test_preparing_performs_one_preparation_operation() -> None:
    runtime = _runtime(RegulatoryIndexingStage.PREPARING)
    documents = [MagicMock()]
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._load_claimed_markdown_documents",
            return_value=(documents, []),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.prepare_claimed_regulatory_indexing_job"
        ) as prepare,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    prepare.assert_called_once_with(
        job_id=runtime.job.id,
        expected_generation=3,
        documents=documents,
        tenant_id="tenant-a",
        db_session=ANY,
    )
    assert result.outcome is OrchestrationOutcome.NEXT_STEP
    assert result.expected_generation == 3


def test_preparing_uses_canonical_user_file_loader_and_reaps_staging() -> None:
    runtime = _runtime(RegulatoryIndexingStage.PREPARING)
    document = Document(
        id=str(runtime.user_file.id),
        source=DocumentSource.USER_FILE,
        semantic_identifier="custom-display-name.md",
        title="Embedded Regulatory Title",
        sections=[TextSection(text="MADDE 1 - Hüküm")],
        metadata={"regulation_number": "2026/1"},
    )
    documents = [document]
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.load_user_file_documents",
            return_value=(documents, ["staged-csv"]),
        ) as load,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.delete_files_best_effort"
        ) as reap,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.prepare_claimed_regulatory_indexing_job"
        ) as prepare,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    load.assert_called_once_with(
        user_file_id=str(runtime.user_file.id),
        file_id=runtime.user_file.file_id,
        file_name=runtime.user_file.name,
        tenant_id="tenant-a",
    )
    reap.assert_called_once_with(
        ["staged-csv"],
        context=f"durable regulatory preparation uf={runtime.user_file.id}",
    )
    recovered_document = prepare.call_args.kwargs["documents"][0]
    assert recovered_document is document
    assert recovered_document.semantic_identifier == "custom-display-name.md"
    assert recovered_document.title == "Embedded Regulatory Title"
    assert recovered_document.metadata == {"regulation_number": "2026/1"}


def test_duplicate_normal_delivery_is_fenced_before_stage_work() -> None:
    runtime = _runtime(RegulatoryIndexingStage.CONTEXT_WAIT)
    session = cast(Session, MagicMock())
    with (
        _patch_session(session),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.get_regulatory_indexing_runtime",
            return_value=runtime,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.claim_regulatory_indexing_job",
            return_value=False,
        ) as claim,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._execute_claimed_step"
        ) as execute,
    ):
        result = run_regulatory_indexing_step(
            runtime.job.id,
            expected_generation=2,
            tenant_id="tenant-a",
        )

    assert result.outcome is OrchestrationOutcome.SKIPPED
    claim.assert_called_once()
    execute.assert_not_called()


def test_preclaimed_recovery_delivery_token_is_consumed_once() -> None:
    runtime = _runtime(RegulatoryIndexingStage.CONTEXT_WAIT)
    recovery_token = uuid4()
    session = cast(Session, MagicMock())
    with (
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.get_session_with_current_tenant",
            side_effect=lambda: _session_context(session),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.get_regulatory_indexing_runtime",
            return_value=runtime,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.claim_regulatory_indexing_job"
        ) as claim,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.consume_preclaimed_regulatory_indexing_delivery",
            side_effect=[True, False],
        ) as consume,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._execute_claimed_step",
            return_value=SimpleNamespace(outcome=OrchestrationOutcome.COMPLETE),
        ) as execute,
    ):
        result = run_preclaimed_regulatory_indexing_step(
            runtime.job.id,
            expected_generation=3,
            recovery_token=recovery_token,
            tenant_id="tenant-a",
        )
        duplicate = run_preclaimed_regulatory_indexing_step(
            runtime.job.id,
            expected_generation=3,
            recovery_token=recovery_token,
            tenant_id="tenant-a",
        )

    assert result.outcome is OrchestrationOutcome.COMPLETE
    assert duplicate.outcome is OrchestrationOutcome.SKIPPED
    claim.assert_not_called()
    execute.assert_called_once()
    assert consume.call_count == 2


def test_preclaimed_recovery_executes_cancellation_phase() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_WAIT,
        status=RegulatoryIndexingJobStatus.CANCELLING,
        cancellation_phase=RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value,
    )
    recovery_token = uuid4()
    session = cast(Session, MagicMock())
    with (
        _patch_session(session),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.consume_preclaimed_regulatory_indexing_delivery",
            return_value=True,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.get_regulatory_indexing_runtime",
            return_value=runtime,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._execute_claimed_step",
            return_value=SimpleNamespace(outcome=OrchestrationOutcome.NEXT_STEP),
        ) as execute,
    ):
        result = run_preclaimed_regulatory_indexing_step(
            runtime.job.id,
            expected_generation=3,
            recovery_token=recovery_token,
            tenant_id="tenant-a",
        )

    execute.assert_called_once()
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


@pytest.mark.parametrize(
    ("stage", "service_name", "next_stage"),
    [
        (
            RegulatoryIndexingStage.INDEX_WRITE,
            "stage_regulatory_job_in_index",
            RegulatoryIndexingStage.VERIFY,
        ),
        (
            RegulatoryIndexingStage.VERIFY,
            "verify_staged_regulatory_job",
            RegulatoryIndexingStage.PUBLISH,
        ),
    ],
)
def test_index_stages_perform_one_bounded_service_operation(
    stage: RegulatoryIndexingStage,
    service_name: str,
    next_stage: RegulatoryIndexingStage,
) -> None:
    runtime = _runtime(stage)
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(f"onyx.regulatory.indexing_jobs.orchestrator.{service_name}") as service,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job",
            return_value=True,
        ) as advance,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    service.assert_called_once()
    advance.assert_called_once_with(
        ANY,
        job_id=runtime.job.id,
        expected_stage=stage,
        expected_generation=3,
        next_stage=next_stage,
        next_status=RegulatoryIndexingJobStatus.QUEUED,
        now=_NOW,
    )
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_publish_terminal_result_does_not_advance_after_atomic_completion() -> None:
    runtime = _runtime(RegulatoryIndexingStage.PUBLISH)
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.publish_regulatory_job",
            return_value=PublishOutcome.COMPLETED,
        ) as publish,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job"
        ) as advance,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    publish.assert_called_once()
    advance.assert_not_called()
    assert result.outcome is OrchestrationOutcome.COMPLETE


def test_context_wait_polls_provider_once_and_schedules_delayed_redelivery() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_WAIT,
        remote_job_name="projects/p/locations/l/batchJobs/1",
    )
    gateway = MagicMock()
    gateway.get.return_value = VertexBatchState(
        remote_job_name=cast(str, runtime.job.remote_vertex_job_name),
        status=VertexBatchJobStatus.RUNNING,
    )
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job",
            return_value=True,
        ) as advance,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    gateway.get.assert_called_once_with(runtime.job.remote_vertex_job_name)
    advance.assert_called_once()
    assert result.outcome is OrchestrationOutcome.NEXT_STEP
    assert result.countdown_seconds == 30


def test_submission_intent_is_committed_before_provider_create() -> None:
    request = VertexBatchRequest(prompt="context request")
    runtime = _runtime(RegulatoryIndexingStage.CONTEXT_SUBMIT)
    events: list[str] = []
    gateway = MagicMock()
    gateway.submit.side_effect = lambda _requests: (
        events.append("provider-create"),
        VertexBatchState(
            remote_job_name="remote-1",
            status=VertexBatchJobStatus.PENDING,
        ),
    )[1]

    @contextmanager
    def submission_lease(*_args: object, **_kwargs: object) -> Iterator[object]:
        events.append("lease-enter")
        yield SimpleNamespace(commit=lambda: events.append("lease-commit"))
        events.append("lease-exit")

    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_contextual_requests",
            return_value=[request],
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.vertex_batch_submission_key",
            return_value="regulatory-context-" + "a" * 64,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.record_vertex_submission_intent",
            side_effect=lambda *_args, **_kwargs: (
                events.append("committed-intent") or True
            ),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.regulatory_indexing_external_mutation_lease",
            side_effect=submission_lease,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.record_vertex_submission",
            side_effect=lambda *_args, **_kwargs: (
                events.append("record-submission") or True
            ),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job",
            return_value=True,
        ),
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    assert events == [
        "committed-intent",
        "lease-enter",
        "provider-create",
        "record-submission",
        "lease-commit",
        "lease-exit",
    ]
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_indeterminate_submit_is_reconciled_before_any_recreate() -> None:
    request = VertexBatchRequest(prompt="context request")
    runtime = _runtime(RegulatoryIndexingStage.CONTEXT_SUBMIT)
    gateway = MagicMock()
    gateway.submit.side_effect = IndexingGatewayIndeterminateSubmissionError(
        "regulatory-context-" + "b" * 64
    )
    session = cast(Session, MagicMock())
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_contextual_requests",
            return_value=[request],
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.vertex_batch_submission_key",
            return_value="regulatory-context-" + "b" * 64,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.record_vertex_submission_intent",
            return_value=True,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.regulatory_indexing_external_mutation_lease",
            side_effect=_external_lease,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.require_vertex_submission_reconciliation",
            return_value=True,
        ) as require_reconciliation,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.schedule_regulatory_indexing_retry",
            return_value=True,
        ),
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        first = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=session,
            now=_NOW,
        )

    assert first.outcome is OrchestrationOutcome.NEXT_STEP
    gateway.submit.assert_called_once_with([request])
    gateway.reconcile_submission.assert_not_called()
    require_reconciliation.assert_called_once()

    recovery_runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_SUBMIT,
        generation=4,
        submission_key="regulatory-context-" + "b" * 64,
        submission_state="RECONCILE_REQUIRED",
    )
    gateway.reset_mock()
    gateway.reconcile_submission.return_value = None
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_contextual_requests",
            return_value=[request],
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.vertex_batch_submission_key",
            return_value="regulatory-context-" + "b" * 64,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.record_vertex_submission_absent",
            return_value=True,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.schedule_regulatory_indexing_retry",
            return_value=True,
        ),
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        second = _execute_claimed_step(
            recovery_runtime,
            tenant_id="tenant-a",
            db_session=session,
            now=_NOW,
        )

    assert second.outcome is OrchestrationOutcome.NEXT_STEP
    gateway.reconcile_submission.assert_called_once_with(
        recovery_runtime.job.vertex_submission_key
    )
    gateway.submit.assert_not_called()

    recreated_runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_SUBMIT,
        generation=5,
        submission_key="regulatory-context-" + "b" * 64,
        submission_state="RECONCILED_ABSENT",
    )
    events: list[str] = []
    gateway.reset_mock()
    gateway.submit.side_effect = lambda _requests: (
        events.append("recreate"),
        VertexBatchState(
            remote_job_name="remote-2",
            status=VertexBatchJobStatus.PENDING,
        ),
    )[1]
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_contextual_requests",
            return_value=[request],
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.vertex_batch_submission_key",
            return_value="regulatory-context-" + "b" * 64,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.record_vertex_submission_intent",
            side_effect=lambda *_args, **_kwargs: (
                events.append("recreate-intent") or True
            ),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.regulatory_indexing_external_mutation_lease",
            side_effect=_external_lease,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.record_vertex_submission",
            return_value=True,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job",
            return_value=True,
        ),
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        third = _execute_claimed_step(
            recreated_runtime,
            tenant_id="tenant-a",
            db_session=session,
            now=_NOW,
        )

    gateway.reconcile_submission.assert_not_called()
    assert events == ["recreate-intent", "recreate"]
    assert third.outcome is OrchestrationOutcome.NEXT_STEP


def test_partial_context_output_requeues_only_still_pending_items() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_APPLY,
        remote_job_name="remote-1",
    )
    runtime.job.vertex_output_uri = "gs://bucket/output"
    pending_request = VertexBatchRequest(prompt="pending")
    ready_request = VertexBatchRequest(prompt="ready")
    runtime = replace(
        runtime,
        indexing_items=cast(
            tuple[RegulatoryIndexingItem, ...],
            (
                SimpleNamespace(
                    request_hash=pending_request.request_hash,
                    status=RegulatoryIndexingItemStatus.PENDING.value,
                ),
                SimpleNamespace(
                    request_hash=ready_request.request_hash,
                    status=RegulatoryIndexingItemStatus.CONTEXT_READY.value,
                ),
            ),
        ),
    )
    gateway = MagicMock()
    gateway.read_results.return_value = "jsonl"
    parsed = {
        pending_request.request_hash: VertexBatchResult(
            request_hash=pending_request.request_hash,
            error=VertexBatchResultError.REMOTE_ERROR,
        ),
        ready_request.request_hash: VertexBatchResult(
            request_hash=ready_request.request_hash,
            context="usable context",
        ),
    }
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.parse_vertex_jsonl_output",
            return_value=parsed,
        ) as parse_output,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.apply_contextual_results",
            return_value=ContextApplySummary(
                context_ready_count=0,
                failed_count=0,
                pending_count=1,
                skipped_count=0,
            ),
        ) as apply_results,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.reset_vertex_submission_for_partial_retry",
            return_value=True,
        ) as reset_submission,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job",
            return_value=True,
        ) as advance,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    gateway.read_results.assert_called_once_with(runtime.job.vertex_output_uri)
    parse_output.assert_called_once_with(
        "jsonl",
        {pending_request.request_hash, ready_request.request_hash},
        require_complete=False,
    )
    assert apply_results.call_args.args[3] == {}
    reset_submission.assert_called_once()
    assert (
        advance.call_args.kwargs["next_stage"] is RegulatoryIndexingStage.CONTEXT_SUBMIT
    )
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_embedding_stage_processes_one_provider_batch_per_delivery() -> None:
    runtime = _runtime(RegulatoryIndexingStage.EMBEDDING)
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.embed_pending_regulatory_items",
            return_value=EmbeddingSummary(
                total_count=5,
                embedded_count=2,
                reused_count=1,
                remaining_count=2,
            ),
        ) as embed,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_job",
            return_value=True,
        ) as advance,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    assert embed.call_args.kwargs["max_batches"] == 1
    assert advance.call_args.kwargs["next_stage"] is RegulatoryIndexingStage.EMBEDDING
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_retryable_failure_persists_backoff_and_terminal_failure_stops() -> None:
    session = cast(Session, MagicMock())
    retry_runtime = _runtime(RegulatoryIndexingStage.CONTEXT_WAIT, attempt_count=0)
    retry_runtime.job.remote_vertex_job_name = "remote-1"
    gateway = MagicMock()
    gateway.get.side_effect = IndexingGatewayConnectionError()
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.schedule_regulatory_indexing_retry",
            return_value=True,
        ) as schedule_retry,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        retry_result = _execute_claimed_step(
            retry_runtime,
            tenant_id="tenant-a",
            db_session=session,
            now=_NOW,
        )

    assert retry_result.outcome is OrchestrationOutcome.NEXT_STEP
    assert retry_result.countdown_seconds is not None
    schedule_retry.assert_called_once()

    terminal_runtime = _runtime(RegulatoryIndexingStage.CONTEXT_WAIT, attempt_count=2)
    terminal_runtime.job.remote_vertex_job_name = "remote-2"
    gateway.get.side_effect = IndexingGatewayConnectionError()
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.fail_regulatory_indexing_job",
            return_value=True,
        ) as fail_job,
    ):
        terminal_result = _execute_claimed_step(
            terminal_runtime,
            tenant_id="tenant-a",
            db_session=session,
            now=_NOW,
        )

    assert terminal_result.outcome is OrchestrationOutcome.COMPLETE
    fail_job.assert_called_once()


def test_any_publication_failure_retries_after_attempt_budget_is_exhausted() -> None:
    runtime = _runtime(RegulatoryIndexingStage.PUBLISH, attempt_count=2)
    with (
        patch("onyx.regulatory.indexing_jobs.orchestrator.validate_snapshot_for_stage"),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.publish_regulatory_job",
            side_effect=RuntimeError("database unavailable"),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.schedule_regulatory_indexing_retry",
            return_value=True,
        ) as schedule_retry,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.fail_regulatory_indexing_job"
        ) as fail_job,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    assert result.outcome is OrchestrationOutcome.NEXT_STEP
    schedule_retry.assert_called_once()
    fail_job.assert_not_called()


def test_deleted_file_requests_durable_cancellation_without_external_work() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_WAIT,
        remote_job_name="remote-1",
        user_file_status=UserFileStatus.DELETING,
    )
    gateway = MagicMock()
    document_index = MagicMock()
    with (
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_elasticsearch_document_index",
            return_value=document_index,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.request_regulatory_indexing_cancellation",
            return_value=True,
        ) as request_cancellation,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    gateway.cancel.assert_not_called()
    gateway.cleanup.assert_not_called()
    document_index.delete.assert_not_called()
    request_cancellation.assert_called_once()
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_cancellation_vertex_phase_performs_only_remote_cancel() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_WAIT,
        status=RegulatoryIndexingJobStatus.CANCELLING,
        remote_job_name="remote-1",
        cancellation_phase=RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value,
    )
    gateway = MagicMock()
    with (
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_cancellation",
            return_value=True,
        ) as advance,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_elasticsearch_document_index"
        ) as build_index,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    gateway.cancel.assert_called_once_with("remote-1")
    gateway.cleanup.assert_not_called()
    build_index.assert_not_called()
    assert advance.call_args.kwargs["next_phase"] is (
        RegulatoryIndexingCancellationPhase.GCS_CLEANUP
    )
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_cancellation_gcs_phase_progresses_when_gateway_config_is_missing() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_WAIT,
        status=RegulatoryIndexingJobStatus.CANCELLING,
        cancellation_phase=RegulatoryIndexingCancellationPhase.GCS_CLEANUP.value,
    )
    with (
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            side_effect=ValueError("Vertex configuration removed"),
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_cancellation",
            return_value=True,
        ) as advance,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_elasticsearch_document_index"
        ) as build_index,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    build_index.assert_not_called()
    assert advance.call_args.kwargs["next_phase"] is (
        RegulatoryIndexingCancellationPhase.INDEX_DELETE
    )
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_cancellation_external_phase_progresses_after_retry_budget() -> None:
    runtime = _runtime(
        RegulatoryIndexingStage.CONTEXT_WAIT,
        status=RegulatoryIndexingJobStatus.CANCELLING,
        attempt_count=2,
        remote_job_name="remote-1",
        cancellation_phase=RegulatoryIndexingCancellationPhase.VERTEX_CANCEL.value,
    )
    gateway = MagicMock()
    gateway.cancel.side_effect = IndexingGatewayConnectionError()
    with (
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator._build_vertex_gateway",
            return_value=gateway,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_cancellation",
            return_value=True,
        ) as advance,
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.schedule_regulatory_indexing_cancellation_retry"
        ) as schedule_retry,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        result = _execute_claimed_step(
            runtime,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    gateway.cancel.assert_called_once_with("remote-1")
    advance.assert_called_once()
    assert advance.call_args.kwargs["next_phase"] is (
        RegulatoryIndexingCancellationPhase.GCS_CLEANUP
    )
    schedule_retry.assert_not_called()
    assert result.outcome is OrchestrationOutcome.NEXT_STEP


def test_cancellation_index_delete_and_finalize_are_separate_deliveries() -> None:
    deleting = _runtime(
        RegulatoryIndexingStage.INDEX_WRITE,
        status=RegulatoryIndexingJobStatus.CANCELLING,
        cancellation_phase=RegulatoryIndexingCancellationPhase.INDEX_DELETE.value,
    )
    document_index = MagicMock()
    with (
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.build_elasticsearch_document_index",
            return_value=document_index,
        ),
        patch(
            "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.advance_regulatory_indexing_cancellation",
            return_value=True,
        ) as advance,
    ):
        from onyx.regulatory.indexing_jobs.orchestrator import _execute_claimed_step

        delete_result = _execute_claimed_step(
            deleting,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    document_index.delete.assert_called_once_with(
        str(deleting.user_file.id), chunk_count=2
    )
    assert advance.call_args.kwargs["next_phase"] is (
        RegulatoryIndexingCancellationPhase.FINALIZE
    )
    assert delete_result.outcome is OrchestrationOutcome.NEXT_STEP

    finalizing = _runtime(
        RegulatoryIndexingStage.INDEX_WRITE,
        status=RegulatoryIndexingJobStatus.CANCELLING,
        cancellation_phase=RegulatoryIndexingCancellationPhase.FINALIZE.value,
    )
    with patch(
        "onyx.regulatory.indexing_jobs.orchestrator.indexing_job_repository.finalize_regulatory_indexing_cancellation",
        return_value=True,
    ) as finalize:
        finalize_result = _execute_claimed_step(
            finalizing,
            tenant_id="tenant-a",
            db_session=cast(Session, MagicMock()),
            now=_NOW,
        )

    finalize.assert_called_once()
    assert finalize_result.outcome is OrchestrationOutcome.COMPLETE
