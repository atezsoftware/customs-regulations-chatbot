import datetime
import hashlib
import threading
from typing import Any, Literal, cast
from urllib.parse import urlparse

import httpx
from cachetools import TTLCache
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from onyx.auth.permissions import require_permission
from onyx.background.celery.versioned_apps.client import app as celery_app
from onyx.configs.app_configs import (
    REGULATORY_BENCHMARK_MAX_QUESTIONS,
    REGULATORY_BENCHMARK_MAX_RUN_ITEMS,
    REGULATORY_BENCHMARK_RECOVERY_LEASE_SECONDS,
)
from onyx.configs.constants import (
    PUBLIC_API_TAGS,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
)
from onyx.db.document_set import get_document_set_by_id_for_user
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import (
    BenchmarkRunFailureCode,
    BenchmarkRunStatus,
    LLMModelFlowType,
    Permission,
)
from onyx.db.llm import (
    fetch_existing_llm_provider_by_id,
    fetch_existing_llm_providers,
    fetch_llm_provider_for_legacy_selection,
)
from onyx.db.models import (
    BenchmarkQuestion,
    DocumentSet,
    LLMProvider,
    RegulatoryChunk,
    User,
)
from onyx.db.regulatory_benchmark import (
    cancel_benchmark_run,
    claim_stale_benchmark_runs_for_recovery,
    create_benchmark_run,
    get_benchmark_question,
    get_benchmark_run,
    get_benchmark_run_for_update,
    list_benchmark_questions,
    list_benchmark_runs,
    question_has_run_items,
    reset_benchmark_run_for_retry,
)
from onyx.db.regulatory_chunks import get_chunk_by_id
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.regulatory.benchmark.models import BenchmarkExpectedCitationInput
from onyx.server.features.regulatory.benchmark_models import (
    BenchmarkAvailableModel,
    BenchmarkCitationOption,
    BenchmarkModelSelection,
    BenchmarkQuestionCreate,
    BenchmarkQuestionSnapshot,
    BenchmarkQuestionUpdate,
    BenchmarkRunCreate,
    BenchmarkRunSnapshot,
    benchmark_run_snapshot,
)
from onyx.utils.logger import setup_logger
from shared_configs.contextvars import get_current_tenant_id

router = APIRouter(prefix="/regulatory/benchmark")
logger = setup_logger()

_OPENROUTER_ACCOUNT_MODELS_CACHE: TTLCache[tuple[int, str, str], frozenset[str]] = (
    TTLCache(maxsize=32, ttl=5 * 60)
)
_OPENROUTER_ACCOUNT_MODELS_CACHE_LOCK = threading.Lock()


def _account_available_openrouter_model_ids(
    provider: LLMProvider,
) -> frozenset[str] | None:
    """Return models allowed by the OpenRouter account's active policies.

    Custom OpenRouter-compatible gateways keep their existing DB-backed catalog;
    only the official OpenRouter API exposes the account-filtered endpoint.
    A transient catalog failure must not take chat or benchmark administration
    offline, so callers fall back to the persisted visible-model list.
    """
    api_base = (getattr(provider, "api_base", None) or "").strip().rstrip("/")
    if not api_base:
        api_base = "https://openrouter.ai/api/v1"
    hostname = (urlparse(api_base).hostname or "").lower()
    if hostname != "openrouter.ai" and not hostname.endswith(".openrouter.ai"):
        return None

    sensitive_api_key = getattr(provider, "api_key", None)
    if sensitive_api_key is None:
        return None
    api_key = sensitive_api_key.get_value(apply_mask=False)
    if not api_key:
        return None

    key_fingerprint = hashlib.sha256(api_key.encode()).hexdigest()
    cache_key = (provider.id, api_base, key_fingerprint)
    with _OPENROUTER_ACCOUNT_MODELS_CACHE_LOCK:
        cached = _OPENROUTER_ACCOUNT_MODELS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        response = httpx.get(
            f"{api_base}/models/user",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("OpenRouter account model response is missing data")
        available_ids: set[str] = set()
        for item in data:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if isinstance(model_id, str) and model_id:
                available_ids.add(model_id)
        available = frozenset(available_ids)
    except Exception:
        logger.warning(
            "Could not refresh account-filtered OpenRouter models for provider %s",
            provider.id,
            exc_info=True,
        )
        return None

    with _OPENROUTER_ACCOUNT_MODELS_CACHE_LOCK:
        _OPENROUTER_ACCOUNT_MODELS_CACHE[cache_key] = available
    return available


def _get_editable_document_set(
    db_session: Session, document_set_id: int, user: User
) -> DocumentSet:
    document_set = get_document_set_by_id_for_user(
        db_session=db_session,
        document_set_id=document_set_id,
        user=user,
        get_editable=True,
    )
    if document_set is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Document set not found")
    return document_set


def _enqueue_benchmark_run(run_id: int) -> None:
    tenant_id = get_current_tenant_id()
    errors: list[Exception] = []
    for countdown in (None, 5):
        options: dict[str, Any] = {
            "kwargs": {"run_id": run_id, "tenant_id": tenant_id},
            "queue": OnyxCeleryQueues.REGULATORY_BENCHMARK,
            "priority": OnyxCeleryPriority.MEDIUM,
            "expires": 24 * 60 * 60,
        }
        if countdown is not None:
            options["countdown"] = countdown
        try:
            celery_app.send_task(OnyxCeleryTask.REGULATORY_BENCHMARK_RUN, **options)
        except Exception as error:
            errors.append(error)
            logger.warning(
                "Benchmark run %s dispatch attempt failed (countdown=%s)",
                run_id,
                countdown,
                exc_info=True,
            )
    if len(errors) == 2:
        raise RuntimeError("All benchmark dispatch attempts failed") from errors[0]


def _recover_stale_runs(db_session: Session) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    run_ids = claim_stale_benchmark_runs_for_recovery(
        db_session,
        stale_before=now
        - datetime.timedelta(seconds=REGULATORY_BENCHMARK_RECOVERY_LEASE_SECONDS),
        claimed_at=now,
    )
    for run_id in run_ids:
        try:
            _enqueue_benchmark_run(run_id)
        except Exception:
            # The renewed lease bounds retries and prevents an API polling storm.
            # A later poll will make another recovery attempt.
            logger.exception("Failed to requeue stale benchmark run %s", run_id)


def _validate_model(
    db_session: Session, selection: BenchmarkModelSelection
) -> LLMProvider:
    if selection.provider_id is not None:
        provider = fetch_existing_llm_provider_by_id(selection.provider_id, db_session)
        expected_provider = (
            provider.name or provider.provider if provider is not None else None
        )
        if expected_provider != selection.provider:
            provider = None
    else:
        provider = fetch_llm_provider_for_legacy_selection(
            selection.provider, db_session
        )
    if provider is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"LLM provider '{selection.provider}' is not configured",
        )
    available_models = _visible_chat_model_names(provider)
    if selection.model_id not in available_models:
        provider_label = (
            "OpenRouter"
            if provider.provider == "openrouter"
            else provider.name or provider.provider
        )
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Model '{selection.model_id}' is not available through {provider_label}",
        )
    account_models = (
        _account_available_openrouter_model_ids(provider)
        if provider.provider == "openrouter"
        else None
    )
    if account_models is not None and selection.model_id not in account_models:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Model '{selection.model_id}' is blocked by the OpenRouter account's "
            "privacy or guardrail policy",
        )
    return provider


def _visible_chat_model_names(provider: LLMProvider) -> set[str]:
    return {
        configuration.name
        for configuration in provider.model_configurations
        if configuration.is_visible
        and LLMModelFlowType.CHAT in configuration.llm_model_flow_types
    }


def _expected_citation_snapshots(
    db_session: Session,
    *,
    document_set: DocumentSet,
    citations: list[BenchmarkExpectedCitationInput | dict[str, object]],
) -> list[dict[str, object]]:
    document_set_files = {
        user_file.id: user_file for user_file in document_set.user_files
    }
    snapshots: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw_citation in citations:
        citation = (
            raw_citation.model_dump()
            if isinstance(raw_citation, BenchmarkExpectedCitationInput)
            else raw_citation
        )
        chunk_id = str(citation["chunk_id"])
        if chunk_id in seen:
            continue
        chunk = get_chunk_by_id(db_session, chunk_id)
        if chunk is None or chunk.user_file_id not in document_set_files:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                f"Expected citation '{chunk_id}' is not in the selected document set",
            )
        user_file = document_set_files[chunk.user_file_id]
        snapshots.append(
            {
                "chunk_id": chunk.id,
                "requirement": citation.get("requirement", "required"),
                "notes": citation.get("notes"),
                "file_name": user_file.name,
                "heading_path": list(chunk.heading_path),
                "text_excerpt": chunk.text[:1000],
            }
        )
        seen.add(chunk_id)
    return snapshots


@router.get("/models", tags=PUBLIC_API_TAGS)
def list_models(
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> list[BenchmarkAvailableModel]:
    models: list[BenchmarkAvailableModel] = []
    providers = fetch_existing_llm_providers(
        db_session, flow_type_filter=[LLMModelFlowType.CHAT]
    )
    for provider in providers:
        provider_selector = provider.name or provider.provider
        account_models = (
            _account_available_openrouter_model_ids(provider)
            if provider.provider == "openrouter"
            else None
        )
        for configuration in provider.model_configurations:
            if (
                not configuration.is_visible
                or LLMModelFlowType.CHAT not in configuration.llm_model_flow_types
            ):
                continue
            if account_models is not None and configuration.name not in account_models:
                continue
            models.append(
                BenchmarkAvailableModel(
                    provider=provider_selector,
                    provider_id=provider.id,
                    model_id=configuration.name,
                    display_name=(
                        configuration.custom_display_name
                        or configuration.display_name
                        or configuration.name
                    ),
                    max_input_tokens=configuration.max_input_tokens,
                    is_visible=configuration.is_visible,
                )
            )
    return sorted(models, key=lambda model: (model.provider, model.model_id.lower()))


@router.get("/document-sets/{document_set_id}/citation-options", tags=PUBLIC_API_TAGS)
def list_citation_options(
    document_set_id: int,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> list[BenchmarkCitationOption]:
    document_set = _get_editable_document_set(db_session, document_set_id, user)
    options: list[BenchmarkCitationOption] = []
    for user_file in document_set.user_files:
        chunks = (
            db_session.query(RegulatoryChunk)
            .filter(RegulatoryChunk.user_file_id == user_file.id)
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            .all()
        )
        options.extend(
            BenchmarkCitationOption(
                chunk_id=chunk.id,
                user_file_id=str(user_file.id),
                file_name=user_file.name,
                heading_path=list(chunk.heading_path),
                text_excerpt=chunk.text[:1000],
                status=cast(Literal["active", "superseded"], chunk.status),
                validity_start_date=chunk.validity_start_date,
                validity_end_date=chunk.validity_end_date,
            )
            for chunk in chunks
        )
    return options


@router.get("/questions", tags=PUBLIC_API_TAGS)
def list_questions(
    active_only: bool = False,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> list[BenchmarkQuestionSnapshot]:
    return [
        BenchmarkQuestionSnapshot.from_model(question)
        for question in list_benchmark_questions(db_session, active_only=active_only)
    ]


@router.post("/questions", tags=PUBLIC_API_TAGS)
def create_question(
    request: BenchmarkQuestionCreate,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> BenchmarkQuestionSnapshot:
    document_set = _get_editable_document_set(db_session, request.document_set_id, user)
    values = request.model_dump(exclude={"expected_citations"})
    values["title"] = request.title.strip()
    values["prompt"] = request.prompt.strip()
    values["reference_answer"] = (
        request.reference_answer.strip() if request.reference_answer else None
    )
    values["expected_facts"] = [
        fact.strip() for fact in request.expected_facts if fact.strip()
    ]
    values["tags"] = [tag.strip() for tag in request.tags if tag.strip()]
    values["expected_citations"] = _expected_citation_snapshots(
        db_session,
        document_set=document_set,
        citations=list(request.expected_citations),
    )
    question = BenchmarkQuestion(**values, created_by=user.id, updated_by=user.id)
    db_session.add(question)
    db_session.commit()
    db_session.refresh(question)
    return BenchmarkQuestionSnapshot.from_model(question)


@router.patch("/questions/{question_id}", tags=PUBLIC_API_TAGS)
def update_question(
    question_id: int,
    request: BenchmarkQuestionUpdate,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> BenchmarkQuestionSnapshot:
    question = get_benchmark_question(db_session, question_id)
    if question is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark question not found")
    updates = request.model_dump(exclude_unset=True)
    target_document_set = _get_editable_document_set(
        db_session,
        int(updates.get("document_set_id", question.document_set_id)),
        user,
    )
    citation_values = updates.pop("expected_citations", question.expected_citations)
    if "document_set_id" in updates or "expected_citations" in request.model_fields_set:
        updates["expected_citations"] = _expected_citation_snapshots(
            db_session,
            document_set=target_document_set,
            citations=list(citation_values or []),
        )
    for text_field in ("title", "prompt", "reference_answer", "rubric_notes"):
        if isinstance(updates.get(text_field), str):
            updates[text_field] = updates[text_field].strip() or None
    if "expected_facts" in updates and updates["expected_facts"] is not None:
        updates["expected_facts"] = [
            value.strip() for value in updates["expected_facts"] if value.strip()
        ]
    if "tags" in updates and updates["tags"] is not None:
        updates["tags"] = [value.strip() for value in updates["tags"] if value.strip()]
    for field, value in updates.items():
        setattr(question, field, value)
    question.updated_by = user.id
    db_session.commit()
    db_session.refresh(question)
    return BenchmarkQuestionSnapshot.from_model(question)


@router.delete("/questions/{question_id}", tags=PUBLIC_API_TAGS)
def delete_question(
    question_id: int,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> None:
    question = get_benchmark_question(db_session, question_id)
    if question is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark question not found")
    if question_has_run_items(db_session, question_id):
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "This question has benchmark history; deactivate it instead.",
        )
    db_session.delete(question)
    db_session.commit()


@router.post("/runs", tags=PUBLIC_API_TAGS)
def create_run(
    request: BenchmarkRunCreate,
    user: User = Depends(require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)),
    db_session: Session = Depends(get_session),
) -> BenchmarkRunSnapshot:
    candidate_providers = [
        _validate_model(db_session, selection) for selection in request.candidates
    ]
    judge_provider = _validate_model(db_session, request.judge)
    questions = list(list_benchmark_questions(db_session, active_only=True))
    if request.question_ids is not None:
        requested_ids = set(request.question_ids)
        questions = [question for question in questions if question.id in requested_ids]
        if len(questions) != len(requested_ids):
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                "One or more benchmark questions are missing or inactive",
            )
    if not questions:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, "No active questions selected")
    if len(questions) > REGULATORY_BENCHMARK_MAX_QUESTIONS:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Too many benchmark questions selected; choose a smaller explicit set",
        )
    for question in questions:
        _get_editable_document_set(db_session, question.document_set_id, user)
    unique_candidates = list(
        dict.fromkeys(
            (
                provider.name or provider.provider,
                provider.id,
                candidate.model_id,
            )
            for candidate, provider in zip(
                request.candidates, candidate_providers, strict=True
            )
        )
    )
    item_count = len(questions) * len(unique_candidates)
    if item_count > REGULATORY_BENCHMARK_MAX_RUN_ITEMS:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Benchmark run is too large: "
            f"{item_count} items exceeds the configured limit of "
            f"{REGULATORY_BENCHMARK_MAX_RUN_ITEMS}",
        )
    run = create_benchmark_run(
        db_session,
        label=request.label.strip() if request.label else None,
        judge_provider=judge_provider.name or judge_provider.provider,
        judge_provider_id=judge_provider.id,
        judge_model=request.judge.model_id,
        deep_research=request.deep_research,
        search_mode=request.search_mode,
        created_by=user.id,
        questions=questions,
        candidates=unique_candidates,
    )
    return benchmark_run_snapshot(run)


@router.post("/runs/{run_id}/start", tags=PUBLIC_API_TAGS)
def start_run(
    run_id: int,
    rerun_completed: bool = False,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> BenchmarkRunSnapshot:
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark run not found")
    if run.status not in {
        BenchmarkRunStatus.PENDING.value,
        BenchmarkRunStatus.ERROR.value,
        BenchmarkRunStatus.CANCELLED.value,
    }:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, "Run is not pending or retryable")
    if run.status in {
        BenchmarkRunStatus.ERROR.value,
        BenchmarkRunStatus.CANCELLED.value,
    }:
        reset_benchmark_run_for_retry(run, rerun_completed=rerun_completed)
    run.status = BenchmarkRunStatus.QUEUED.value
    run.queued_at = datetime.datetime.now(datetime.timezone.utc)
    run.started_at = None
    run.heartbeat_at = None
    run.failure_code = None
    run.failure_message = None
    # Persist the dispatch-visible state before publishing. A fast worker can
    # otherwise consume the task while the row is still ``pending`` and exit as
    # an idempotent no-op, leaving the run queued until stale-run recovery.
    db_session.commit()
    try:
        _enqueue_benchmark_run(run.id)
    except Exception as error:
        # A broker client can fail after the broker accepted the task. Keep the
        # committed queued state so either that delivery or stale-run recovery can
        # execute it safely under the run lease.
        run.failure_code = BenchmarkRunFailureCode.DISPATCH_FAILED.value
        run.failure_message = "Failed to queue benchmark run"
        db_session.commit()
        raise OnyxError(
            OnyxErrorCode.INTERNAL_ERROR, "Failed to queue benchmark run"
        ) from error
    return benchmark_run_snapshot(run)


@router.post("/runs/{run_id}/cancel", tags=PUBLIC_API_TAGS)
def cancel_run(
    run_id: int,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> BenchmarkRunSnapshot:
    run = cancel_benchmark_run(db_session, run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark run not found")
    return benchmark_run_snapshot(run)


@router.get("/runs", tags=PUBLIC_API_TAGS)
def list_runs(
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> list[BenchmarkRunSnapshot]:
    _recover_stale_runs(db_session)
    return [benchmark_run_snapshot(run) for run in list_benchmark_runs(db_session)]


@router.get("/runs/{run_id}", tags=PUBLIC_API_TAGS)
def get_run(
    run_id: int,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> BenchmarkRunSnapshot:
    _recover_stale_runs(db_session)
    run = get_benchmark_run(db_session, run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark run not found")
    return benchmark_run_snapshot(run)
