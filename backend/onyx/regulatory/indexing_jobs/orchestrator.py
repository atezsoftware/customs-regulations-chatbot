from __future__ import annotations

import datetime
from enum import StrEnum
from hashlib import sha256
from pathlib import PurePosixPath
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy.orm import Session

from onyx.configs import app_configs
from onyx.connectors.models import Document
from onyx.db import regulatory_indexing_jobs as indexing_job_repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import (
    RegulatoryIndexingCancellationIntent,
    RegulatoryIndexingCancellationPhase,
    RegulatoryIndexingItemStatus,
    RegulatoryIndexingJobStatus,
    RegulatoryIndexingProviderCleanupPhase,
    RegulatoryIndexingProviderCleanupState,
    RegulatoryIndexingStage,
    RegulatoryIndexingSubmissionState,
    UserFileStatus,
)
from onyx.db.llm import fetch_model_configuration_by_id
from onyx.document_index.factory import build_elasticsearch_document_index
from onyx.file_processing.user_file_loader import load_user_file_documents
from onyx.file_store.staging import delete_files_best_effort
from onyx.llm.constants import LlmProviderNames
from onyx.llm.well_known_providers.constants import VERTEX_CREDENTIALS_FILE_KWARG
from onyx.natural_language_processing.utils import BaseTokenizer, get_tokenizer
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
    validate_snapshot_for_stage,
)
from onyx.regulatory.indexing_jobs.contextual import (
    apply_contextual_results,
    build_contextual_requests,
    get_contextual_token_budget_tokenizer,
)
from onyx.regulatory.indexing_jobs.embedding import (
    build_openrouter_embedding_batch,
    embed_pending_regulatory_items,
)
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayError,
    IndexingGatewayHTTPError,
    IndexingGatewayIndeterminateSubmissionError,
    RegulatoryIndexingConfigSnapshot,
    RegulatoryInputHashVersion,
    RetryReason,
)
from onyx.regulatory.indexing_jobs.openrouter_batch import (
    HttpxOpenRouterBatchGateway,
    OpenRouterBatchContractError,
    OpenRouterBatchJobStatus,
    openrouter_batch_submission_key,
    parse_openrouter_embedding_results,
)
from onyx.regulatory.indexing_jobs.preparation import (
    prepare_claimed_regulatory_indexing_job,
    prepare_claimed_regulatory_indexing_job_from_chunks,
)
from onyx.regulatory.indexing_jobs.publisher import (
    PublishOutcome,
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
    PROVIDER_CLEANUP = "PROVIDER_CLEANUP"


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
    snapshot = RegulatoryIndexingConfigSnapshot.model_validate(
        runtime.job.config_snapshot
    )
    if snapshot.input_content_hash != runtime.job.content_hash:
        raise ValueError("regulatory job input hash does not match its snapshot")
    if snapshot.chunk_generation_hash != runtime.job.chunk_generation_hash:
        raise ValueError("regulatory job generation hash does not match its snapshot")
    return snapshot


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


def _build_openrouter_gateway(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
) -> HttpxOpenRouterBatchGateway:
    snapshot = _snapshot(runtime)
    config = snapshot.openrouter_batch
    if config is None:
        raise ValueError("regulatory job has no OpenRouter Batch snapshot")
    if runtime.search_settings is None:
        raise ValueError("regulatory indexing SearchSettings disappeared")
    search_settings = runtime.search_settings

    def api_key_provider() -> str:
        return search_settings.api_key or ""

    return HttpxOpenRouterBatchGateway(
        config=config,
        api_key_provider=api_key_provider,
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
    tenant_id: str,
) -> tuple[list[Document], list[str]]:
    file_name = runtime.user_file.name or ""
    if PurePosixPath(file_name).suffix.lower() not in {".md", ".mdx"}:
        raise ValueError("durable regulatory indexing accepts Markdown only")
    return load_user_file_documents(
        user_file_id=str(runtime.user_file.id),
        file_id=runtime.user_file.file_id,
        file_name=runtime.user_file.name,
        tenant_id=tenant_id,
    )


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
    submission_state = RegulatoryIndexingSubmissionState(
        runtime.job.vertex_submission_state
    )
    embedding_tokenizer, contextual_tokenizer = _contextual_tokenizers(snapshot)
    requests = build_contextual_requests(
        runtime.job,
        runtime.regulatory_chunks,
        runtime.indexing_items,
        embedding_tokenizer=embedding_tokenizer,
        contextual_tokenizer=contextual_tokenizer,
        max_requests=snapshot.context_request_size,
        max_jsonl_bytes=snapshot.context_jsonl_max_bytes,
        # A charged, indeterminate submission must remain reconcilable even when
        # it consumed the final new-submission attempt.
        max_attempts=(
            snapshot.max_attempts
            if submission_state is RegulatoryIndexingSubmissionState.NONE
            else None
        ),
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
    persisted_attempt = getattr(runtime.job, "vertex_submission_attempt_count", 0)
    submission_attempt = (
        persisted_attempt + 1
        if submission_state is RegulatoryIndexingSubmissionState.NONE
        else persisted_attempt
    )
    expected_key = vertex_batch_submission_key(
        requests,
        tenant_id=tenant_id,
        job_id=runtime.job.id,
        output_prefix=_vertex_object_prefix(tenant_id, runtime.job.id),
        submission_attempt=submission_attempt,
    )
    request_hashes = tuple(request.request_hash for request in requests)
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

    if submission_state is RegulatoryIndexingSubmissionState.SUBMITTING:
        persisted = indexing_job_repository.require_vertex_submission_reconciliation(
            db_session,
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            submission_key=expected_key,
            request_hashes=request_hashes,
            reconcile_until=now
            + datetime.timedelta(seconds=snapshot.submission_reconcile_seconds),
            now=now,
        )
        return (
            _advance(
                runtime,
                db_session,
                next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
                now=now,
                countdown_seconds=snapshot.poll_seconds,
            )
            if persisted
            else _skipped(runtime.job.id)
        )

    if submission_state is RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED:
        reconcile_until = getattr(runtime.job, "vertex_reconcile_until", None)
        if reconcile_until is None or now >= reconcile_until:
            raise VertexBatchContractError(
                "Vertex submission visibility remained indeterminate"
            )
        state = gateway.reconcile_submission(expected_key)
        if state is None:
            persisted = indexing_job_repository.record_vertex_reconciliation_miss(
                db_session,
                job_id=runtime.job.id,
                expected_generation=runtime.job.lease_generation,
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
            request_hashes=request_hashes,
            charge_items=not getattr(runtime.job, "vertex_submission_charged", True),
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
        submission_attempt=submission_attempt,
        now=now,
    )
    if not persisted:
        return _skipped(runtime.job.id)
    with indexing_job_repository.regulatory_indexing_external_mutation_lease(
        db_session,
        job_id=runtime.job.id,
        expected_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
        expected_generation=runtime.job.lease_generation,
    ) as submission_lease:
        if submission_lease is None:
            return _skipped(runtime.job.id)
        try:
            state = gateway.submit(
                requests,
                submission_key=expected_key,
                max_jsonl_bytes=snapshot.context_jsonl_max_bytes,
            )
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
                    request_hashes=request_hashes,
                    reconcile_until=now
                    + datetime.timedelta(seconds=snapshot.submission_reconcile_seconds),
                    now=now,
                    commit=False,
                )
            )
            if not reconciliation_persisted:
                return _skipped(runtime.job.id)
            submission_lease.commit()
            return _advance(
                runtime,
                db_session,
                next_stage=RegulatoryIndexingStage.CONTEXT_SUBMIT,
                now=now,
                countdown_seconds=snapshot.poll_seconds,
            )
        except IndexingGatewayError:
            definitely_not_sent = (
                indexing_job_repository.record_vertex_submission_not_sent(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    submission_key=expected_key,
                    now=now,
                    commit=False,
                )
            )
            if definitely_not_sent:
                submission_lease.commit()
            raise
        except Exception:
            # Once control entered the provider submission boundary, an
            # unrecognized failure cannot prove that create was rejected. Keep
            # the persisted identity and reconcile instead of risking a second
            # billable create.
            reconciliation_persisted = (
                indexing_job_repository.require_vertex_submission_reconciliation(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    submission_key=expected_key,
                    request_hashes=request_hashes,
                    reconcile_until=now
                    + datetime.timedelta(seconds=snapshot.submission_reconcile_seconds),
                    now=now,
                    commit=False,
                )
            )
            if not reconciliation_persisted:
                return _skipped(runtime.job.id)
            submission_lease.commit()
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
            request_hashes=request_hashes,
            charge_items=True,
            remote_job_name=state.remote_job_name,
            input_uri=state.input_uri,
            output_uri=state.output_uri,
            now=now,
            commit=False,
        )
        if not persisted:
            return _skipped(runtime.job.id)
        submission_lease.commit()
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
    all_hashes = {item.request_hash for item in runtime.indexing_items}
    pending_hashes = {
        item.request_hash
        for item in runtime.indexing_items
        if getattr(item, "status", RegulatoryIndexingItemStatus.PENDING.value)
        == RegulatoryIndexingItemStatus.PENDING.value
    }
    parsed = parse_vertex_jsonl_output(
        raw_output,
        all_hashes,
        require_complete=False,
    )
    applicable = {
        request_hash: result
        for request_hash, result in parsed.items()
        if request_hash in pending_hashes
        and result.error is not VertexBatchResultError.REMOTE_ERROR
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


def _openrouter_embedding_batch(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    snapshot = _snapshot(runtime)
    config = snapshot.openrouter_batch
    if config is None:
        raise ValueError("regulatory job has no OpenRouter Batch snapshot")
    plan = build_openrouter_embedding_batch(
        job=runtime.job,
        rows=runtime.regulatory_chunks,
        items=runtime.indexing_items,
        config=config,
        max_attempts=snapshot.max_attempts,
    )
    if not plan.requests:
        return _advance(
            runtime,
            db_session,
            next_stage=RegulatoryIndexingStage.INDEX_WRITE,
            now=now,
        )
    submission_state = RegulatoryIndexingSubmissionState(
        runtime.job.openrouter_submission_state
    )
    persisted_attempt = runtime.job.openrouter_submission_attempt_count
    submission_attempt = (
        persisted_attempt + 1
        if submission_state is RegulatoryIndexingSubmissionState.NONE
        else persisted_attempt
    )
    expected_key = openrouter_batch_submission_key(
        plan.requests,
        tenant_id=tenant_id,
        job_id=runtime.job.id,
        submission_attempt=submission_attempt,
    )
    persisted_key = runtime.job.openrouter_submission_key
    if persisted_key is not None and persisted_key != expected_key:
        raise ValueError(
            "persisted OpenRouter submission key does not match pending items"
        )
    active_item_ids = tuple(
        item_id
        for request in plan.requests
        for item_id in plan.item_ids_by_custom_id[request.custom_id]
    )
    if (
        runtime.job.openrouter_active_item_ids
        and runtime.job.openrouter_active_item_ids
        != [str(item_id) for item_id in active_item_ids]
    ):
        raise ValueError("persisted OpenRouter active items do not match pending items")
    gateway = _build_openrouter_gateway(runtime)

    if submission_state is RegulatoryIndexingSubmissionState.SUBMITTED:
        remote_batch_id = runtime.job.remote_openrouter_batch_id
        if not remote_batch_id:
            raise ValueError("submitted OpenRouter Batch has no remote id")
        completion_deadline = runtime.job.openrouter_completion_deadline
        if completion_deadline is None or now >= completion_deadline:
            raise OpenRouterBatchContractError(
                "OpenRouter embedding Batch exceeded its completion horizon"
            )
        state = gateway.get(remote_batch_id)
        if state.status in {
            OpenRouterBatchJobStatus.PENDING,
            OpenRouterBatchJobStatus.RUNNING,
            OpenRouterBatchJobStatus.CANCELLING,
        }:
            return _advance(
                runtime,
                db_session,
                next_stage=RegulatoryIndexingStage.EMBEDDING,
                now=now,
                countdown_seconds=snapshot.poll_seconds,
            )
        if state.status is not OpenRouterBatchJobStatus.SUCCEEDED:
            raise OpenRouterBatchContractError(
                "OpenRouter embedding Batch terminated unsuccessfully"
            )
        parsed = parse_openrouter_embedding_results(
            state.results or [],
            expected_custom_ids={request.custom_id for request in plan.requests},
            expected_model=config.model_name,
            expected_dimension=config.effective_dimension,
        )
        item_vectors: list[tuple[UUID, list[float]]] = []
        failed_item_ids: list[UUID] = []
        for request in plan.requests:
            item_ids = plan.item_ids_by_custom_id[request.custom_id]
            result = parsed.get(request.custom_id)
            if (
                result is None
                or result.error_code is not None
                or result.vectors is None
                or len(result.vectors) != len(item_ids)
            ):
                failed_item_ids.extend(item_ids)
                continue
            item_vectors.extend(zip(item_ids, result.vectors, strict=True))
        persisted = indexing_job_repository.apply_openrouter_embedding_batch(
            db_session,
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            remote_batch_id=remote_batch_id,
            item_vectors=item_vectors,
            failed_item_ids=failed_item_ids,
            now=now,
        )
        if not persisted:
            return _skipped(runtime.job.id)
        return _advance(
            runtime,
            db_session,
            next_stage=(
                RegulatoryIndexingStage.EMBEDDING
                if plan.remaining_item_count or failed_item_ids
                else RegulatoryIndexingStage.INDEX_WRITE
            ),
            now=now,
        )

    if submission_state in {
        RegulatoryIndexingSubmissionState.SUBMITTING,
        RegulatoryIndexingSubmissionState.RECONCILE_REQUIRED,
        RegulatoryIndexingSubmissionState.RECONCILED_ABSENT,
    }:
        persisted = indexing_job_repository.record_openrouter_submission_ambiguous(
            db_session,
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            submission_key=expected_key,
            now=now,
        )
        if not persisted:
            return _skipped(runtime.job.id)
        raise OpenRouterBatchContractError(
            "OpenRouter Batch creation outcome requires manual reconciliation"
        )
    if submission_state is RegulatoryIndexingSubmissionState.MANUAL_RECONCILE_REQUIRED:
        raise OpenRouterBatchContractError(
            "OpenRouter Batch creation outcome requires manual reconciliation"
        )

    if submission_state is not RegulatoryIndexingSubmissionState.NONE:
        raise ValueError("OpenRouter submission state is unsupported")
    persisted = indexing_job_repository.record_openrouter_submission_intent(
        db_session,
        job_id=runtime.job.id,
        expected_generation=runtime.job.lease_generation,
        submission_key=expected_key,
        submission_attempt=submission_attempt,
        active_item_ids=active_item_ids,
        now=now,
    )
    if not persisted:
        return _skipped(runtime.job.id)
    with indexing_job_repository.regulatory_indexing_external_mutation_lease(
        db_session,
        job_id=runtime.job.id,
        expected_stage=RegulatoryIndexingStage.EMBEDDING,
        expected_generation=runtime.job.lease_generation,
    ) as submission_lease:
        if submission_lease is None:
            return _skipped(runtime.job.id)
        try:
            state = gateway.submit(plan.requests, submission_key=expected_key)
        except IndexingGatewayIndeterminateSubmissionError as error:
            if error.submission_key != expected_key:
                raise ValueError(
                    "OpenRouter returned an unexpected submission identity"
                ) from error
            reconciliation_persisted = (
                indexing_job_repository.record_openrouter_submission_ambiguous(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    submission_key=expected_key,
                    now=now,
                    commit=False,
                )
            )
            if not reconciliation_persisted:
                return _skipped(runtime.job.id)
            submission_lease.commit()
            raise OpenRouterBatchContractError(
                "OpenRouter Batch creation outcome requires manual reconciliation"
            ) from error
        except OpenRouterBatchContractError:
            definitely_not_sent = (
                indexing_job_repository.record_openrouter_submission_not_sent(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    submission_key=expected_key,
                    now=now,
                    commit=False,
                )
            )
            if definitely_not_sent:
                submission_lease.commit()
            raise
        except IndexingGatewayError:
            definitely_not_sent = (
                indexing_job_repository.record_openrouter_submission_not_sent(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    submission_key=expected_key,
                    now=now,
                    commit=False,
                )
            )
            if definitely_not_sent:
                submission_lease.commit()
            raise
        except Exception:
            reconciliation_persisted = (
                indexing_job_repository.record_openrouter_submission_ambiguous(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    submission_key=expected_key,
                    now=now,
                    commit=False,
                )
            )
            if not reconciliation_persisted:
                return _skipped(runtime.job.id)
            submission_lease.commit()
            raise OpenRouterBatchContractError(
                "OpenRouter Batch creation outcome requires manual reconciliation"
            )
        else:
            recorded = indexing_job_repository.record_openrouter_submission(
                db_session,
                job_id=runtime.job.id,
                expected_generation=runtime.job.lease_generation,
                submission_key=expected_key,
                remote_batch_id=state.remote_batch_id,
                completion_deadline=now
                + datetime.timedelta(seconds=config.completion_horizon_seconds),
                charge_items=True,
                now=now,
                commit=False,
            )
            if not recorded:
                return _skipped(runtime.job.id)
            submission_lease.commit()
    return _advance(
        runtime,
        db_session,
        next_stage=RegulatoryIndexingStage.EMBEDDING,
        now=now,
        countdown_seconds=snapshot.poll_seconds,
    )


def _request_cancellation(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    requested = indexing_job_repository.request_regulatory_indexing_cancellation(
        db_session,
        job_id=runtime.job.id,
        expected_stage=RegulatoryIndexingStage(runtime.job.stage),
        expected_generation=runtime.job.lease_generation,
        cancellation_intent=(
            RegulatoryIndexingCancellationIntent.USER_DELETE
            if runtime.user_file.status is UserFileStatus.DELETING
            else RegulatoryIndexingCancellationIntent.USER_CANCEL
        ),
        now=now,
    )
    return (
        _next_step(runtime.job.id, runtime.job.lease_generation)
        if requested
        else _skipped(runtime.job.id)
    )


def _advance_cancellation(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    db_session: Session,
    *,
    expected_phase: RegulatoryIndexingCancellationPhase,
    next_phase: RegulatoryIndexingCancellationPhase,
    now: datetime.datetime,
) -> OrchestrationResult:
    advanced = indexing_job_repository.advance_regulatory_indexing_cancellation(
        db_session,
        job_id=runtime.job.id,
        expected_generation=runtime.job.lease_generation,
        expected_phase=expected_phase,
        next_phase=next_phase,
        now=now,
    )
    return (
        _next_step(runtime.job.id, runtime.job.lease_generation)
        if advanced
        else _skipped(runtime.job.id)
    )


def _next_cancellation_phase(
    phase: RegulatoryIndexingCancellationPhase,
) -> RegulatoryIndexingCancellationPhase | None:
    return {
        RegulatoryIndexingCancellationPhase.VERTEX_CANCEL: (
            RegulatoryIndexingCancellationPhase.GCS_CLEANUP
        ),
        RegulatoryIndexingCancellationPhase.GCS_CLEANUP: (
            RegulatoryIndexingCancellationPhase.INDEX_DELETE
        ),
        RegulatoryIndexingCancellationPhase.INDEX_DELETE: (
            RegulatoryIndexingCancellationPhase.FINALIZE
        ),
    }.get(phase)


def _execute_cancellation_phase(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    phase = RegulatoryIndexingCancellationPhase(runtime.job.cancellation_phase)
    if phase is RegulatoryIndexingCancellationPhase.VERTEX_CANCEL:
        remote_job_name = runtime.job.remote_vertex_job_name
        remote_openrouter_batch_id = getattr(
            runtime.job, "remote_openrouter_batch_id", None
        )
        if remote_openrouter_batch_id:
            try:
                _build_openrouter_gateway(runtime).cancel(remote_openrouter_batch_id)
            except IndexingGatewayHTTPError as error:
                if error.status_code != 404:
                    raise
        if remote_job_name:
            gateway = _build_vertex_gateway(
                runtime, tenant_id=tenant_id, db_session=db_session
            )
            gateway.cancel(remote_job_name)
        if not remote_job_name and not remote_openrouter_batch_id:
            raise ValueError("provider cancellation phase has no remote job")
        return _advance_cancellation(
            runtime,
            db_session,
            expected_phase=phase,
            next_phase=RegulatoryIndexingCancellationPhase.GCS_CLEANUP,
            now=now,
        )
    if phase is RegulatoryIndexingCancellationPhase.GCS_CLEANUP:
        snapshot = _snapshot(runtime)
        cleanup_uri = (
            f"{snapshot.vertex.gcs_uri.rstrip('/')}/"
            f"{_vertex_object_prefix(tenant_id, runtime.job.id)}"
        )
        gateway = _build_vertex_gateway(
            runtime, tenant_id=tenant_id, db_session=db_session
        )
        gateway.cleanup(cleanup_uri)
        return _advance_cancellation(
            runtime,
            db_session,
            expected_phase=phase,
            next_phase=RegulatoryIndexingCancellationPhase.INDEX_DELETE,
            now=now,
        )
    if phase is RegulatoryIndexingCancellationPhase.INDEX_DELETE:
        if runtime.search_settings is not None:
            document_index = build_elasticsearch_document_index(runtime.search_settings)
            document_index.delete(
                str(runtime.user_file.id),
                chunk_count=runtime.user_file.chunk_count,
                refresh=True,
            )
        return _advance_cancellation(
            runtime,
            db_session,
            expected_phase=phase,
            next_phase=RegulatoryIndexingCancellationPhase.FINALIZE,
            now=now,
        )
    if phase is RegulatoryIndexingCancellationPhase.FINALIZE:
        finalized = indexing_job_repository.finalize_regulatory_indexing_cancellation(
            db_session,
            job_id=runtime.job.id,
            expected_generation=runtime.job.lease_generation,
            now=now,
        )
        return _complete(runtime.job.id) if finalized else _skipped(runtime.job.id)
    raise ValueError("cancelling regulatory job has no cancellation phase")


def _execute_claimed_step_impl(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    if runtime.job.status == RegulatoryIndexingJobStatus.CANCELLING.value:
        return _execute_cancellation_phase(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
    if runtime.user_file.status in {UserFileStatus.CANCELED, UserFileStatus.DELETING}:
        return _request_cancellation(
            runtime,
            db_session=db_session,
            now=now,
        )
    stage = RegulatoryIndexingStage(runtime.job.stage)
    snapshot = _snapshot(runtime)
    unresolved_preparing_snapshot = (
        stage is RegulatoryIndexingStage.PREPARING
        and snapshot.input_hash_version
        is RegulatoryInputHashVersion.LEGACY_OR_CANONICAL
    )
    if not unresolved_preparing_snapshot:
        current_chunk_generation_hash = compute_regulatory_chunk_generation_hash(
            embedding_provider=snapshot.embedding_provider,
            embedding_model_name=snapshot.embedding_model_name,
        )
        if runtime.job.chunk_generation_hash != current_chunk_generation_hash:
            delivery = indexing_job_repository.supersede_regulatory_indexing_job_for_generation_drift(
                db_session,
                job_id=runtime.job.id,
                expected_stage=stage,
                expected_generation=runtime.job.lease_generation,
                current_chunk_generation_hash=current_chunk_generation_hash,
                now=now,
            )
            return (
                _next_step(runtime.job.id, delivery.expected_generation)
                if delivery is not None
                else _skipped(runtime.job.id)
            )
    validate_snapshot_for_stage(db_session, snapshot, stage)
    if (
        snapshot.input_hash_version is RegulatoryInputHashVersion.CHUNK_ROWS_V3
        and runtime.user_file.regulatory_chunk_generation_hash
        != snapshot.chunk_generation_hash
    ):
        raise ValueError(
            "CHUNKED user file generation identity is absent or mismatched"
        )

    if stage is RegulatoryIndexingStage.PREPARING:
        if snapshot.input_hash_version is RegulatoryInputHashVersion.CHUNK_ROWS_V3:
            prepare_claimed_regulatory_indexing_job_from_chunks(
                job_id=runtime.job.id,
                expected_generation=runtime.job.lease_generation,
                tenant_id=tenant_id,
                db_session=db_session,
            )
        else:
            documents, staged_csv_ids = _load_claimed_markdown_documents(
                runtime,
                tenant_id,
            )
            try:
                prepare_claimed_regulatory_indexing_job(
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    documents=documents,
                    tenant_id=tenant_id,
                    db_session=db_session,
                )
            finally:
                delete_files_best_effort(
                    staged_csv_ids,
                    context=(
                        f"durable regulatory preparation uf={runtime.user_file.id}"
                    ),
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
        if snapshot.openrouter_batch is not None:
            return _openrouter_embedding_batch(
                runtime,
                tenant_id=tenant_id,
                db_session=db_session,
                now=now,
            )
        if not app_configs.REGULATORY_INDEXING_ALLOW_ONLINE_EMBEDDING_FALLBACK:
            raise ValueError(
                "online regulatory embedding fallback is disabled for this job"
            )
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
    publish_outcome = publish_regulatory_job(
        job=runtime.job,
        user_file=runtime.user_file,
        rows=runtime.regulatory_chunks,
        items=runtime.indexing_items,
        search_settings=runtime.search_settings,
        db_session=db_session,
    )
    if publish_outcome is not PublishOutcome.COMPLETED:
        raise RuntimeError("regulatory publication returned an unknown outcome")
    return _complete(runtime.job.id)


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
        if runtime.job.status == RegulatoryIndexingJobStatus.CANCELLING.value:
            phase = RegulatoryIndexingCancellationPhase(runtime.job.cancellation_phase)
            cancellation_intent = RegulatoryIndexingCancellationIntent(
                runtime.job.cancellation_intent
            )
            required_index_cleanup = (
                phase is RegulatoryIndexingCancellationPhase.INDEX_DELETE
                and cancellation_intent
                in {
                    RegulatoryIndexingCancellationIntent.SUPERSEDE,
                    RegulatoryIndexingCancellationIntent.USER_DELETE,
                }
            )
            retry_cancellation = (
                required_index_cleanup
                or phase is RegulatoryIndexingCancellationPhase.FINALIZE
                or decision.retryable
                and next_attempt < snapshot.max_attempts
            )
            if not retry_cancellation:
                next_phase = _next_cancellation_phase(phase)
                if next_phase is None:
                    retry_cancellation = True
                else:
                    return _advance_cancellation(
                        runtime,
                        db_session,
                        expected_phase=phase,
                        next_phase=next_phase,
                        now=now,
                    )
            delay = retry_delay_seconds(
                runtime.job.id,
                RegulatoryIndexingStage(runtime.job.stage),
                next_attempt,
                snapshot.retry_base_seconds,
                snapshot.retry_max_seconds,
            )
            scheduled = (
                indexing_job_repository.schedule_regulatory_indexing_cancellation_retry(
                    db_session,
                    job_id=runtime.job.id,
                    expected_generation=runtime.job.lease_generation,
                    expected_phase=phase,
                    next_retry_at=now + datetime.timedelta(seconds=delay),
                    error_code=decision.error_code,
                    error_message=error.__class__.__name__,
                )
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
        must_reconcile_publication = (
            decision.reason is RetryReason.PUBLICATION_INDETERMINATE
        )
        if must_reconcile_publication or (
            decision.retryable and next_attempt < snapshot.max_attempts
        ):
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
        cleanup_requested = indexing_job_repository.request_regulatory_indexing_terminal_failure_cleanup(
            db_session,
            job_id=runtime.job.id,
            expected_stage=RegulatoryIndexingStage(runtime.job.stage),
            expected_generation=runtime.job.lease_generation,
            error_code=decision.error_code,
            error_message=error.__class__.__name__,
            now=now,
        )
        return (
            _next_step(runtime.job.id, runtime.job.lease_generation)
            if cleanup_requested
            else _skipped(runtime.job.id)
        )


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


def _run_claimed_regulatory_provider_cleanup(
    runtime: indexing_job_repository.RegulatoryIndexingRuntime,
    *,
    cleanup_generation: int,
    tenant_id: str,
    db_session: Session,
    now: datetime.datetime,
) -> OrchestrationResult:
    job = runtime.job
    if (
        job.status
        not in {
            RegulatoryIndexingJobStatus.SUCCEEDED.value,
            RegulatoryIndexingJobStatus.FAILED.value,
            RegulatoryIndexingJobStatus.CANCELLED.value,
        }
        or job.provider_cleanup_state
        != RegulatoryIndexingProviderCleanupState.RUNNING.value
        or job.provider_cleanup_generation != cleanup_generation
        or job.provider_cleanup_token is not None
    ):
        return _skipped(job.id)
    snapshot = _snapshot(runtime)
    phase = RegulatoryIndexingProviderCleanupPhase(job.provider_cleanup_phase)
    next_phase: RegulatoryIndexingProviderCleanupPhase | None = None
    try:
        if phase is RegulatoryIndexingProviderCleanupPhase.VERTEX_RECONCILE:
            submission_key = job.vertex_submission_key
            if not submission_key:
                next_phase = RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP
            else:
                state = _build_vertex_gateway(
                    runtime, tenant_id=tenant_id, db_session=db_session
                ).reconcile_submission(submission_key)
                if state is None:
                    next_attempt = job.provider_cleanup_attempt_count + 1
                    exhausted = next_attempt >= snapshot.max_attempts
                    delay = (
                        snapshot.retry_max_seconds
                        if exhausted
                        else retry_delay_seconds(
                            job.id,
                            RegulatoryIndexingStage(job.stage),
                            next_attempt,
                            snapshot.retry_base_seconds,
                            snapshot.retry_max_seconds,
                        )
                    )
                    scheduled = indexing_job_repository.schedule_regulatory_provider_cleanup_retry(
                        db_session,
                        job_id=job.id,
                        cleanup_generation=cleanup_generation,
                        next_retry_at=now + datetime.timedelta(seconds=delay),
                        error_code="VERTEX_VISIBILITY_PENDING",
                        error_message="VertexSubmissionNotVisible",
                        exhausted=exhausted,
                    )
                    return _complete(job.id) if scheduled else _skipped(job.id)
                recorded = indexing_job_repository.record_reconciled_provider_cleanup_vertex_job(
                    db_session,
                    job_id=job.id,
                    cleanup_generation=cleanup_generation,
                    submission_key=submission_key,
                    remote_job_name=state.remote_job_name,
                    input_uri=state.input_uri,
                    output_uri=state.output_uri,
                    now=now,
                )
                return _complete(job.id) if recorded else _skipped(job.id)
        elif phase is RegulatoryIndexingProviderCleanupPhase.VERTEX_CANCEL:
            remote_openrouter_batch_id = getattr(
                job, "remote_openrouter_batch_id", None
            )
            if remote_openrouter_batch_id:
                try:
                    _build_openrouter_gateway(runtime).cancel(
                        remote_openrouter_batch_id
                    )
                except IndexingGatewayHTTPError as error:
                    if error.status_code != 404:
                        raise
            if job.remote_vertex_job_name:
                _build_vertex_gateway(
                    runtime, tenant_id=tenant_id, db_session=db_session
                ).cancel(job.remote_vertex_job_name)
            next_phase = RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE
        elif phase is RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE:
            if job.remote_vertex_job_name:
                _build_vertex_gateway(
                    runtime, tenant_id=tenant_id, db_session=db_session
                ).delete(job.remote_vertex_job_name)
            next_phase = RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP
        elif phase is RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP:
            cleanup_uri = (
                f"{snapshot.vertex.gcs_uri.rstrip('/')}/"
                f"{_vertex_object_prefix(tenant_id, job.id)}"
            )
            _build_vertex_gateway(
                runtime, tenant_id=tenant_id, db_session=db_session
            ).cleanup(cleanup_uri)
            next_phase = RegulatoryIndexingProviderCleanupPhase.COMPLETE
        elif phase is RegulatoryIndexingProviderCleanupPhase.COMPLETE:
            completed = indexing_job_repository.complete_regulatory_provider_cleanup(
                db_session,
                job_id=job.id,
                cleanup_generation=cleanup_generation,
                now=now,
            )
            return _complete(job.id) if completed else _skipped(job.id)
        else:
            raise ValueError("terminal regulatory job has no provider cleanup phase")
    except Exception as error:
        if isinstance(error, IndexingGatewayHTTPError) and error.status_code == 404:
            if phase is RegulatoryIndexingProviderCleanupPhase.VERTEX_CANCEL:
                next_phase = RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE
            elif phase is RegulatoryIndexingProviderCleanupPhase.VERTEX_DELETE:
                next_phase = RegulatoryIndexingProviderCleanupPhase.GCS_CLEANUP
            else:
                raise
        else:
            decision = classify_indexing_error(error)
            next_attempt = job.provider_cleanup_attempt_count + 1
            exhausted = not decision.retryable or next_attempt >= snapshot.max_attempts
            delay = (
                snapshot.retry_max_seconds
                if exhausted
                else retry_delay_seconds(
                    job.id,
                    RegulatoryIndexingStage(job.stage),
                    next_attempt,
                    snapshot.retry_base_seconds,
                    snapshot.retry_max_seconds,
                )
            )
            scheduled = (
                indexing_job_repository.schedule_regulatory_provider_cleanup_retry(
                    db_session,
                    job_id=job.id,
                    cleanup_generation=cleanup_generation,
                    next_retry_at=now + datetime.timedelta(seconds=delay),
                    error_code=decision.error_code,
                    error_message=error.__class__.__name__,
                    exhausted=exhausted,
                )
            )
            return _complete(job.id) if scheduled else _skipped(job.id)

    if next_phase is None:
        raise RuntimeError("provider cleanup did not select a next phase")
    advanced = indexing_job_repository.advance_regulatory_provider_cleanup(
        db_session,
        job_id=job.id,
        cleanup_generation=cleanup_generation,
        expected_phase=phase,
        next_phase=next_phase,
        now=now,
    )
    return _complete(job.id) if advanced else _skipped(job.id)


def run_preclaimed_regulatory_provider_cleanup(
    job_id: UUID,
    cleanup_generation: int,
    cleanup_token: UUID,
    tenant_id: str,
) -> OrchestrationResult:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    with get_session_with_current_tenant() as db_session:
        now = datetime.datetime.now(datetime.timezone.utc)
        if not indexing_job_repository.consume_regulatory_provider_cleanup_delivery(
            db_session,
            job_id=job_id,
            cleanup_generation=cleanup_generation,
            cleanup_token=cleanup_token,
            consumed_at=now,
        ):
            return _skipped(job_id)
        runtime = indexing_job_repository.get_regulatory_indexing_runtime(
            db_session, job_id
        )
        if runtime is None:
            return _skipped(job_id)
        return _run_claimed_regulatory_provider_cleanup(
            runtime,
            cleanup_generation=cleanup_generation,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )


def run_preclaimed_regulatory_indexing_step(
    job_id: UUID,
    expected_generation: int,
    recovery_token: UUID,
    tenant_id: str,
) -> OrchestrationResult:
    if not tenant_id.strip():
        raise ValueError("tenant_id must not be empty")
    with get_session_with_current_tenant() as db_session:
        now = datetime.datetime.now(datetime.timezone.utc)
        if not indexing_job_repository.consume_preclaimed_regulatory_indexing_delivery(
            db_session,
            job_id=job_id,
            expected_generation=expected_generation,
            recovery_token=recovery_token,
            consumed_at=now,
        ):
            return _skipped(job_id)
        runtime = indexing_job_repository.get_regulatory_indexing_runtime(
            db_session, job_id
        )
        if (
            runtime is None
            or runtime.job.status
            not in {
                RegulatoryIndexingJobStatus.RUNNING.value,
                RegulatoryIndexingJobStatus.CANCELLING.value,
            }
            or runtime.job.lease_generation != expected_generation
        ):
            return _skipped(job_id)
        return _execute_claimed_step(
            runtime,
            tenant_id=tenant_id,
            db_session=db_session,
            now=now,
        )
