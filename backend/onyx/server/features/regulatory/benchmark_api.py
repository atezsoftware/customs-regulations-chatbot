import datetime
from typing import Literal, cast

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
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import BenchmarkRunStatus, Permission
from onyx.db.llm import fetch_existing_llm_provider, fetch_existing_llm_providers
from onyx.db.models import BenchmarkQuestion, RegulatoryChunk, User, UserProject
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


def _get_project(db_session: Session, project_id: int) -> UserProject:
    project = db_session.get(UserProject, project_id)
    if project is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Directory not found")
    return project


def _enqueue_benchmark_run(run_id: int) -> None:
    celery_app.send_task(
        OnyxCeleryTask.REGULATORY_BENCHMARK_RUN,
        kwargs={"run_id": run_id, "tenant_id": get_current_tenant_id()},
        queue=OnyxCeleryQueues.REGULATORY_BENCHMARK,
        priority=OnyxCeleryPriority.MEDIUM,
        expires=24 * 60 * 60,
    )


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


def _validate_model(db_session: Session, selection: BenchmarkModelSelection) -> None:
    provider = fetch_existing_llm_provider(selection.provider, db_session)
    if provider is None:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"LLM provider '{selection.provider}' is not configured",
        )
    if provider.provider != "openrouter":
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            "Benchmark models must use a configured OpenRouter provider",
        )
    available_models = {
        configuration.name for configuration in provider.model_configurations
    }
    if selection.model_id not in available_models:
        raise OnyxError(
            OnyxErrorCode.INVALID_INPUT,
            f"Model '{selection.model_id}' is not available through OpenRouter",
        )


def _expected_citation_snapshots(
    db_session: Session,
    *,
    project: UserProject,
    citations: list[BenchmarkExpectedCitationInput | dict[str, object]],
) -> list[dict[str, object]]:
    project_files = {user_file.id: user_file for user_file in project.user_files}
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
        if chunk is None or chunk.user_file_id not in project_files:
            raise OnyxError(
                OnyxErrorCode.INVALID_INPUT,
                f"Expected citation '{chunk_id}' is not in the selected directory",
            )
        user_file = project_files[chunk.user_file_id]
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
    providers = fetch_existing_llm_providers(db_session, flow_type_filter=[])
    for provider in providers:
        if provider.provider != "openrouter" or not provider.name:
            continue
        for configuration in provider.model_configurations:
            models.append(
                BenchmarkAvailableModel(
                    provider=provider.name,
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


@router.get("/projects/{project_id}/citation-options", tags=PUBLIC_API_TAGS)
def list_citation_options(
    project_id: int,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> list[BenchmarkCitationOption]:
    project = _get_project(db_session, project_id)
    options: list[BenchmarkCitationOption] = []
    for user_file in project.user_files:
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
    project = _get_project(db_session, request.project_id)
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
        db_session, project=project, citations=list(request.expected_citations)
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
    target_project = _get_project(
        db_session, int(updates.get("project_id", question.project_id))
    )
    citation_values = updates.pop("expected_citations", question.expected_citations)
    if "project_id" in updates or "expected_citations" in request.model_fields_set:
        updates["expected_citations"] = _expected_citation_snapshots(
            db_session, project=target_project, citations=list(citation_values or [])
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
    for selection in [*request.candidates, request.judge]:
        _validate_model(db_session, selection)
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
        _get_project(db_session, question.project_id)
    unique_candidates = list(
        dict.fromkeys(
            (candidate.provider, candidate.model_id) for candidate in request.candidates
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
        judge_provider=request.judge.provider,
        judge_model=request.judge.model_id,
        deep_research=request.deep_research,
        created_by=user.id,
        questions=questions,
        candidates=unique_candidates,
    )
    return benchmark_run_snapshot(run)


@router.post("/runs/{run_id}/start", tags=PUBLIC_API_TAGS)
def start_run(
    run_id: int,
    user: User = Depends(  # noqa: ARG001
        require_permission(Permission.FULL_ADMIN_PANEL_ACCESS)
    ),
    db_session: Session = Depends(get_session),
) -> BenchmarkRunSnapshot:
    run = get_benchmark_run_for_update(db_session, run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark run not found")
    if run.status != BenchmarkRunStatus.PENDING.value:
        raise OnyxError(OnyxErrorCode.INVALID_INPUT, "Run is not pending")
    run.status = BenchmarkRunStatus.RUNNING.value
    run.started_at = datetime.datetime.now(datetime.timezone.utc)
    db_session.commit()
    try:
        _enqueue_benchmark_run(run.id)
    except Exception as error:
        # Publishing is not transactional: a broker may accept the message and
        # still drop the client connection before acknowledging it. Keep the run
        # recoverable instead of recording a false terminal failure. The stale-run
        # lease makes the next UI poll publish an idempotent recovery probe.
        run.started_at = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(seconds=REGULATORY_BENCHMARK_RECOVERY_LEASE_SECONDS)
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
    run = get_benchmark_run(db_session, run_id)
    if run is None:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Benchmark run not found")
    cancel_benchmark_run(db_session, run)
    updated_run = get_benchmark_run(db_session, run_id)
    assert updated_run is not None
    return benchmark_run_snapshot(updated_run)


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
