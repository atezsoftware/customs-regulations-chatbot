from __future__ import annotations

import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from onyx.configs.constants import DocumentSource
from onyx.connectors.models import Document, TextSection
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import (
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingStage,
    RegulatoryIndexingSubmissionState,
    UserFileStatus,
)
from onyx.db.llm import fetch_model_configuration_by_id
from onyx.document_index.factory import build_elasticsearch_document_index
from onyx.file_processing.extract_file_text import file_io_to_text
from onyx.file_store.file_store import get_default_file_store
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import VERTEX_CREDENTIALS_FILE_KWARG
from onyx.natural_language_processing.utils import BaseTokenizer, get_tokenizer
from onyx.regulatory.indexing_jobs.configuration import validate_snapshot_for_stage
from onyx.regulatory.indexing_jobs.contextual import (
    apply_contextual_results,
    build_contextual_requests,
    get_contextual_token_budget_tokenizer,
)
from onyx.regulatory.indexing_jobs.embedding import embed_pending_regulatory_items
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayIndeterminateSubmissionError,
    RegulatoryIndexingConfigSnapshot,
)
from onyx.regulatory.indexing_jobs.preparation import (
    prepare_claimed_regulatory_indexing_job,
)
from onyx.regulatory.indexing_jobs.publisher import (
    publish_regulatory_job,
    stage_regulatory_job_in_index,
    verify_staged_regulatory_job,
)
from onyx.regulatory.indexing_jobs.retry import (
    classify_indexing_error,
    retry_delay_seconds,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    GoogleVertexBatchGateway,
    VertexBatchContractError,
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchResultError,
    parse_vertex_jsonl_output,
    vertex_batch_submission_key,
)


class OrchestrationOutcome(StrEnum):
    SKIPPED = "SKIPPED"
    NEXT_STEP = "NEXT_STEP"
    COMPLETE = "COMPLETE"


class OrchestrationDeliveryKind(StrEnum):
    NORMAL = "NORMAL"
    PRECLAIMED = "PRECLAIMED"


class OrchestrationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: UUID
    outcome: OrchestrationOutcome
    expected_generation: int | None = Field(default=None, ge=0)
    countdown_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_followup(self) -> "OrchestrationResult":
        if self.outcome is OrchestrationOutcome.NEXT_STEP:
            if self.expected_generation is None:
                raise ValueError("next-step result requires an expected generation")
        elif self.expected_generation is not None or self.countdown_seconds is not None:
            raise ValueError("terminal result must not schedule another delivery")
        return self


def _skipped(job_id: UUID) -> OrchestrationResult:
    return OrchestrationResult(job_id=job_id, outcome=OrchestrationOutcome.SKIPPED)


def _complete(job_id: UUID) -> OrchestrationResult:
    return OrchestrationResult(job_id=job_id, outcome=OrchestrationOutcome.COMPLETE)


def _next_step(
    job_id: UUID,
    expected_generation: int,
    *,
    countdown_seconds: float = 0,
) -> OrchestrationResult:
    return OrchestrationResult(
        job_id=job_id,
        outcome=OrchestrationOutcome.NEXT_STEP,
        expected_generation=expected_generation,
        countdown_seconds=countdown_seconds,
    )


def _snapshot(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
) -> RegulatoryIndexingConfigSnapshot:
    return RegulatoryIndexingConfigSnapshot.model_validate(runtime.job.config_snapshot)


def _vertex_object_prefix(tenant_id: str, job_id: UUID) -> str:
    tenant_key = sha256(tenant_id.encode()).hexdigest()
    return f"tenants/{tenant_key}/jobs/{job_id}"


def _build_vertex_gateway(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
) -> VertexBatchGateway:
    snapshot = _snapshot(runtime)
    model_configuration = fetch_model_configuration_by_id(
        db_session,
        snapshot.vertex.model_configuration_id,
    )
    if model_configuration is None:
        raise ValueError("contextual model configuration disappeared")
    raw_credentials = (model_configuration.llm_provider.custom_config or {}).get(
        VERTEX_CREDENTIALS_FILE_KWARG
    )

    def credential_json_provider() -> str | None:
        return raw_credentials

    return GoogleVertexBatchGateway(
        config=snapshot.vertex,
        object_prefix=_vertex_object_prefix(tenant_id, runtime.job.id),
        credential_json_provider=credential_json_provider,
    )


def _advance(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    db_session: Session,
    *,
    next_stage: RegulatoryIndexingStage,
    now: datetime.datetime,
    next_status: RegulatoryIndexingJobStatus = RegulatoryIndexingJobStatus.QUEUED,
    countdown_seconds: float = 0,
) -> OrchestrationResult:
    stage = RegulatoryIndexingStage(runtime.job.stage)
    persisted = indexing_job_repository.advance_regulatory_indexing_job(
        db_session,
        job_id=runtime.job.id,
        expected_stage=stage,
        expected_generation=runtime.job.lease_generation,
        next_stage=next_stage,
        next_status=next_status,
        now=now,
    )
    if not persisted:
        return _skipped(runtime.job.id)
    if next_status in {
        RegulatoryIndexingJobStatus.SUCCEEDED,
        RegulatoryIndexingJobStatus.FAILED,
        RegulatoryIndexingJobStatus.CANCELLED,
    }:
        return _complete(runtime.job.id)
    return _next_step(
        runtime.job.id,
        runtime.job.lease_generation,
        countdown_seconds=countdown_seconds,
    )


def _load_claimed_markdown_documents(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
) -> list[Document]:
    file_name = runtime.user_file.name or ""
    if PurePosixPath(file_name).suffix.lower() not in {".md", ".mdx"}:
        raise ValueError("durable regulatory indexing accepts Markdown only")
    with get_default_file_store().read_file(runtime.user_file.file_id, mode="b") as raw:
        text = file_io_to_text(raw).strip()
    if not text:
        raise ValueError("regulatory Markdown file is empty")
    return [
        Document(
            id=str(runtime.user_file.id),
            source=DocumentSource.USER_FILE,
            semantic_identifier=file_name,
            title=file_name,
            sections=[TextSection(text=text, link=None)],
            metadata={},
        )
    ]


def _contextual_tokenizers(
    snapshot: RegulatoryIndexingConfigSnapshot,
) -> tuple[BaseTokenizer, BaseTokenizer]:
    return (
        get_tokenizer(snapshot.embedding_model_name, snapshot.embedding_provider),
        get_contextual_token_budget_tokenizer(
            model_provider=LlmProviderNames.VERTEX_AI,
            model_name=snapshot.vertex.model_name,
        ),
    )


def _context_submit(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    snapshot = _snapshot(runtime)
    embedding_tokenizer, contextual_tokenizer = _contextual_tokenizers(snapshot)
    requests = build_contextual_requests(
        runtime.job,
        runtime.regulatory_chunks,
        runtime.indexing_items,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
    )
    if not requests:
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.EMBEDDING,
            now=now,
        )

    gateway = _build_vertex_gateway(runtime, tenant_id=tenant_id, db_session=db_session)
    persisted_key = runtime.job.vertex_submission_key
    submission_state = RegulatoryIndexingSubmissionState(
        runtime.job.vertex_submission_state
    )
    expected_key = vertex_batch_submission_key(requests)
    if persisted_key is not None and persisted_key != expected_key:
        raise ValueError("persisted Vertex submission key does not match pending items")

    if submission_state is RegulatoryIndexingSubmissionState.SUBMITTED:
        if not runtime.job.remote_vertex_job_name:
            raise ValueError("submitted Vertex batch has no remote job name")
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.CONTEXT_WAIT,
            now=now,
        )

    if submission_state in {
        RegulatoryIndexingSubmissionState.SUBMITTING,
        RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED,
    }:
        state = gateway.reconcile_submission(expected_key)
        if state is None:
            persisted = indexing_job_repository.record_vertex_submission_absent(
                db_session,
                job_id=runtime.job.id,
                expected_generation=runtime.job.lease_generation,
                submission_key=expected_key,
                now=now,
            )
            if not persisted:
                return _skipped(runtime.job.id)
            return _advance(
                runtime,
                db_session,
                next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
                now=now,
                countdown_seconds=snapshot.poll_seconds,
            )
        persisted = indexing_job_repository.record_vertex_submission(
            db_session,
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            submission_key=expected_key,
            remote_job_name=state.remote_job_name,
            input_uri=state.input_uri,
            output_uri=state.output_uri,
            now=now,
        )
        if not persisted:
            return _skipped(runtime.job.id)
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.CONTEXT_WAIT,
            now=now,
        )

    persisted = indexing_job_repository.record_vertex_submission_intent(
        db_session,
        job_id=runtime.job.id,
        expected_generation=runtime.job.lease_generation,
        submission_key=expected_key,
        now=now,
    )
    if not persisted:
        return _skipped(runtime.job.id)
    try:
        state = gateway.submit(requests)
    except IndexingGatewayIndeterminateSubmissionError as error:
        if error.submission_key != expected_key:
            raise ValueError(
                "Vertex returned an unexpected submission identity"
            ) from error
        reconciliation_persisted = (
            indexing_job_repository.require_vertex_submission_reconciliation(
                db_session,
                job_id=runtime.job.id,
                expected_generation=runtime.job.lease_generation,
                submission_key=expected_key,
                now=now,
            )
        )
        if not reconciliation_persisted:
            return _skipped(runtime.job.id)
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
            now=now,
            countdown_seconds=snapshot.poll_seconds,
        )

    persisted = indexing_job_repository.record_vertex_submission(
        db_session,
        job_id=runtime.job.id,
        expected_generation=runtime.job.lease_generation,
        submission_key=expected_key,
        remote_job_name=state.remote_job_name,
        input_uri=state.input_uri,
        output_uri=state.output_uri,
        now=now,
    )
    if not persisted:
        return _skipped(runtime.job.id)
    return _advance(
        runtime,
        db_session,
        next_stage=RegulatoryIndexingStage.CONTEXT_WAIT,
        now=now,
    )


def _context_wait(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    remote_job_name = runtime.job.remote_vertex_job_name
    if not remote_job_name:
        raise ValueError("Vertex wait stage has no remote job name")
    snapshot = _snapshot(runtime)
    gateway = _build_vertex_gateway(runtime, tenant_id=tenant_id, db_session=db_session)
    state = gateway.get(remote_job_name)
    persisted = indexing_job_repository.persist_vertex_poll_state(
        db_session,
        job_id=runtime.job.id,
        expected_generation=runtime.job.lease_generation,
        remote_job_name=remote_job_name,
        output_uri=state.output_uri,
        now=now,
    )
    if not persisted:
        return _skipped(runtime.job.id)
    if state.status in {
        VertexBatchJobStatus.PENDING,
        VertexBatchJobStatus.RUNNING,
        VertexBatchJobStatus.CANCELLING,
    }:
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.CONTEXT_WAIT,
            now=now,
            countdown_seconds=snapshot.poll_seconds,
        )
    if state.status is VertexBatchJobStatus.SUCCEEDED:
        if not state.output_uri and not runtime.job.vertex_output_uri:
            raise VertexBatchContractError("successful Vertex batch has no output URI")
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.CONTEXT_APPLY,
            now=now,
        )
    raise VertexBatchContractError("Vertex contextual batch terminated unsuccessfully")


def _context_apply(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    output_uri = runtime.job.vertex_output_uri
    if not output_uri:
        raise ValueError("Vertex apply stage has no output URI")
    snapshot = _snapshot(runtime)
    gateway = _build_vertex_gateway(runtime, tenant_id=tenant_id, db_session=db_session)
    raw_output = gateway.read_results(output_uri)
    pending_hashes = {
        item.request_hash
        for item in runtime.indexing_items
        if getattr(item, "status", RegulatoryIndexingItemStatus.PENDING.value)
        == RegulatoryIndexingItemStatus.PENDING.value
    }
    parsed = parse_vertex_jsonl_output(
        raw_output,
        pending_hashes,
        require_complete=False,
    )
    applicable = {
        request_hash: result
        for request_hash, result in parsed.items()
        if result.error is not VertexBatchResultError.REMOTE_ERROR
    }
    embedding_tokenizer = get_tokenizer(
        snapshot.embedding_model_name,
        snapshot.embedding_provider,
    )
    summary = apply_contextual_results(
        runtime.job,
        runtime.regulatory_chunks,
        runtime.indexing_items,
        applicable,
        embedding_tokenizer,
        db_session,
    )
    if summary.failed_count:
        raise ValueError("contextual batch contains terminal item failures")
    if summary.pending_count:
        reset = indexing_job_repository.reset_vertex_submission_for_partial_retry(
            db_session,
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            now=now,
        )
        if not reset:
            return _skipped(runtime.job.id)
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
            now=now,
        )
    return _advance(
        runtime,
        db_session,
        next_stage=RegulatoryIndexingStage.EMBEDDING,
        now=now,
    )


def _cancel_deleted_file(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    snapshot = _snapshot(runtime)
    gateway = _build_vertex_gateway(runtime, tenant_id=tenant_id, db_session=db_session)
    if runtime.job.remote_vertex_job_name:
        gateway.cancel(runtime.job.remote_vertex_job_name)
    cleanup_uri = (
        f"{snapshot.vertex.gcs_uri.rstrip('/')}/"
        f"{_vertex_object_prefix(tenant_id, runtime.job.id)}"
    )
    try:
        gateway.cleanup(cleanup_uri)
    except Exception:
        # Bucket lifecycle remains the final safety net; cancellation correctness
        # must not be reversed by cleanup failure.
        pass
    if runtime.search_settings is not None:
        document_index = build_elasticsearch_document_index(runtime.search_settings)
        document_index.delete(
            str(runtime.user_file.id),
            chunk_count=runtime.user_file.chunk_count,
        )
    cancelled = indexing_job_repository.cancel_regulatory_indexing_job(
        db_session,
        job_id=runtime.job.id,
        expected_stage=RegulatoryIndexingStage(runtime.job.stage),
        expected_generation=runtime.job.lease_generation,
        now=now,
    )
    return _complete(runtime.job.id) if cancelled else _skipped(runtime.job.id)


def _execute_claimed_step_impl(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    if runtime.user_file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}:
        return _cancel_deleted_file(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
    stage = RegulatoryIndexingStage(runtime.job.stage)
    snapshot = _snapshot(runtime)
    validate_snapshot_for_stage(db_session, snapshot, stage)

    if stage is RegulatoryIndexingStage.PREPARING:
        prepare_claimed_regulatory_indexing_job(
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            documents=_load_claimed_markdown_documents(runtime),
            tenant_id=tenant_id,
            db_session=db_session,
        )
        return _next_step(runtime.job.id, runtime.job.lease_generation)
    if stage is RegulatoryIndexingStage.CONTEXT_SUBMIT:
        return _context_submit(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
    if stage is RegulatoryIndexingStage.CONTEXT_WAIT:
        return _context_wait(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
    if stage is RegulatoryIndexingStage.CONTEXT_APPLY:
        return _context_apply(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
    if stage is RegulatoryIndexingStage.EMBEDDING:
        if runtime.search_settings is None:
            raise ValueError("regulatory indexing SearchSettings disappeared")
        summary = embed_pending_regulatory_items(
            job=runtime.job,
            rows=runtime.regulatory_chunks,
            items=runtime.indexing_items,
            search_settings=runtime.search_settings,
            tenant_id=tenant_id,
            db_session=db_session,
            max_batches=1,
        )
        return _advance(
            runtime,
            db_session,
            next_stage=(
                RegulatoryIndexingStage.EMBEDDING
                if summary.remaining_count
                else RegulatoryIndexingStage.INDEX_WRITE
            ),
            now=now,
        )
    if stage is RegulatoryIndexingStage.INDEX_WRITE:
        if runtime.search_settings is None:
            raise ValueError("regulatory indexing SearchSettings disappeared")
        stage_regulatory_job_in_index(
            job=runtime.job,
            user_file=runtime.user_file,
            rows=runtime.regulatory_chunks,
            items=runtime.indexing_items,
            search_settings=runtime.search_settings,
            tenant_id=tenant_id,
            db_session=db_session,
        )
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.VERIFY,
            now=now,
        )
    if stage is RegulatoryIndexingStage.VERIFY:
        verify_staged_regulatory_job(
            job=runtime.job,
            user_file=runtime.user_file,
            rows=runtime.regulatory_chunks,
            items=runtime.indexing_items,
            search_settings=runtime.search_settings,
            db_session=db_session,
        )
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.PUBLISH,
            now=now,
        )
    publish_regulatory_job(
        job=runtime.job,
        user_file=runtime.user_file,
        rows=runtime.regulatory_chunks,
        items=runtime.indexing_items,
        search_settings=runtime.search_settings,
        db_session=db_session,
    )
    return _advance(
        runtime,
        db_session,
        next_stage=RegulatoryIndexingStage.PUBLISH,
        next_status=RegulatoryIndexingJobStatus.SUCCEEDED,
        now=now,
    )


def _execute_claimed_step(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    try:
        return _execute_claimed_step_impl(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
    except Exception as error:
        snapshot = _snapshot(runtime)
        decision = classify_indexing_error(error)
        next_attempt = runtime.job.attempt_count + 1
        if decision.retryable and next_attempt < snapshot.max_attempts:
            delay = retry_delay_seconds(
                runtime.job.id,
                RegulatoryIndexingStage(runtime.job.stage),
                next_attempt,
                snapshot.retry_base_seconds,
                snapshot.retry_max_seconds,
            )
            scheduled = indexing_job_repository.schedule_regulatory_indexing_retry(
                db_session,
                job_id=runtime.job.id,
                expected_stage=RegulatoryIndexingStage(runtime.job.stage),
                expected_generation=runtime.job.lease_generation,
                next_retry_at=now + datetime.timedelta(seconds=delay),
                error_code=decision.error_code,
                error_message=error.__class__.__name__,
            )
            return (
                _next_step(
                    runtime.job.id,
                    runtime.job.lease_generation,
                    countdown_seconds=delay,
                )
                if scheduled
                else _skipped(runtime.job.id)
            )
        failed = indexing_job_repository.fail_regulatory_indexing_job(
            db_session,
            job_id=runtime.job.id,
            expected_stage=RegulatoryIndexingStage(runtime.job.stage),
            expected_generation=runtime.job.lease_generation,
            error_code=decision.error_code,
            error_message=error.__class__.__name__,
            now=now,
        )
        return _complete(runtime.job.id) if failed else _skipped(runtime.job.id)


def run_regulatory_indexing_step(
    job_id: UUID,
    expected_generation: int,
    tenant_id: str,
) -> OrchestrationResult:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    now = datetime.datetime.now(datetime.timezone.utc)
    with get_session_with_current_tenant() as db_session:
        runtime = indexing_job_repository.get_regulatory_indexing_runtime(
            db_session, job_id
        )
        if runtime is None:
            return _skipped(job_id)
        claimed = indexing_job_repository.claim_regulatory_indexing_job(
            db_session,
            job_id=job_id,
            expected_stage=RegulatoryIndexingStage(runtime.job.stage),
            expected_generation=expected_generation,
            now=now,
        )
        if not claimed:
            return _skipped(job_id)
        claimed_runtime = indexing_job_repository.get_regulatory_indexing_runtime(
            db_session, job_id
        )
        if claimed_runtime is None:
            return _skipped(job_id)
        return _execute_claimed_step(
            claimed_runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )


def run_preclaimed_regulatory_indexing_step(
    job_id: UUID,
    expected_generation: int,
    tenant_id: str,
) -> OrchestrationResult:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    with get_session_with_current_tenant() as db_session:
        runtime = indexing_job_repository.get_regulatory_indexing_runtime(
            db_session, job_id
        )
        if (
            runtime is None
            or runtime.job.status != RegulatoryIndexingJobStatus.RUNNING.value
            or runtime.job.lease_generation != expected_generation
        ):
            return _skipped(job_id)
        return _execute_claimed_step(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=datetime.datetime.now(datetime.timezone.utc),
        )
