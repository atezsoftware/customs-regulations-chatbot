"""Enrich only newly chunked intake files using cached direct Vertex requests."""

import hashlib
import json
from collections.abc import Callable, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from threading import Event, Lock
from uuid import UUID

from pydantic import ValidationError

from onyx.db import document_intake_metadata as repository
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.labeling_configuration import (
    resolve_intake_labeling_provider_binding,
    resolve_labeling_gateway,
)
from onyx.db.models import User
from onyx.regulatory.document_intake.metadata import (
    MAX_METADATA_REQUEST_BYTES,
    MAX_METADATA_RESPONSE_BYTES,
    METADATA_VERSION,
    TARGETS_PER_REQUEST,
    LegalMetadataResult,
    MetadataChunk,
    build_metadata_body_scope_repair_request,
    build_metadata_repair_request,
    build_metadata_request,
    metadata_input_hash,
    validate_metadata_transport_response,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchRequest,
    VertexBatchResultError,
    parse_vertex_jsonl_output,
)
from onyx.tracing.flows import LLMFlow

METADATA_CONCURRENT_WINDOWS = 2
MetadataResults = list[tuple[str, LegalMetadataResult]]


class _MetadataWindowStopped(ValueError):
    pass


def _metadata_response(
    gateway: VertexBatchGateway, request: VertexBatchRequest, output_uri: str
) -> tuple[str | None, bool]:
    lines = [line for line in gateway.read_results(output_uri) if line.strip()]
    parsed = parse_vertex_jsonl_output(iter(lines), [request.request_hash])[
        request.request_hash
    ]
    if len(lines) != 1:
        raise ValueError("New-document metadata output must contain one response")
    envelope = json.loads(lines[0])
    if envelope.get("request") != request.to_generate_content_request():
        raise ValueError("New-document metadata output does not match its request")
    status = envelope.get("status", "")
    if envelope.get("error") is not None or not isinstance(status, str) or status:
        raise ValueError("New-document metadata output has a provider error")
    response = envelope.get("response")
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not isinstance(candidates, list) or len(candidates) != 1:
        raise ValueError("New-document metadata output must contain one candidate")
    candidate = candidates[0]
    finish = candidate.get("finishReason") if isinstance(candidate, dict) else None
    content = candidate.get("content") if isinstance(candidate, dict) else None
    no_visible_body = content is None or (
        isinstance(content, dict) and content.get("parts") in (None, [])
    )
    # Only a correlated, otherwise valid provider envelope can authorize a split.
    if finish == "MAX_TOKENS" and (
        parsed.error in {None, VertexBatchResultError.EMPTY}
        or (parsed.error == VertexBatchResultError.MALFORMED and no_visible_body)
    ):
        return None, True
    if finish != "STOP" or parsed.error is not None or parsed.context is None:
        raise ValueError(
            "New-document metadata extraction returned incomplete or invalid output"
        )
    return parsed.context, False


def _extract_metadata_targets(
    *,
    gateway: VertexBatchGateway,
    chunks: Sequence[MetadataChunk],
    targets: Sequence[str],
    identity_prefix: str,
    heartbeat: Callable[[], None],
    stopped: Event,
) -> MetadataResults:
    request = build_metadata_request(chunks, targets)
    context, exhausted = _run_metadata_request(
        gateway=gateway,
        request=request,
        identity_prefix=identity_prefix,
        heartbeat=heartbeat,
        stopped=stopped,
    )
    if not exhausted:
        assert context is not None
        try:
            result = validate_metadata_transport_response(
                context, chunks=chunks, target_ids=targets
            )
            return [(request.request_hash, result)]
        except ValueError as error:
            repair = build_metadata_repair_request(
                request,
                context,
                validation_category=(
                    "invalid_json_or_schema"
                    if isinstance(error, ValidationError)
                    else "value_not_covered_by_evidence"
                    if str(error)
                    == "Metadata value contradicts its quoted source wording"
                    else "invalid_coverage_or_evidence"
                ),
            )
            if str(error) in {
                "Metadata field requires body evidence",
                "Metadata publication reference must use body evidence only for relative dates",
            }:
                cached = _cached_metadata_request(
                    gateway=gateway,
                    request=repair,
                    identity_prefix=identity_prefix,
                    heartbeat=heartbeat,
                    stopped=stopped,
                )
                if cached is None:
                    scoped_repair = build_metadata_body_scope_repair_request(
                        request, context
                    )
                    if scoped_repair is not None:
                        repair = scoped_repair
                else:
                    cached_context, cached_exhausted = cached
                    if cached_exhausted:
                        exhausted = True
                    else:
                        assert cached_context is not None
                        try:
                            result = validate_metadata_transport_response(
                                cached_context, chunks=chunks, target_ids=targets
                            )
                            return [(repair.request_hash, result)]
                        except ValueError:
                            scoped_repair = build_metadata_body_scope_repair_request(
                                request,
                                context,
                                cached_repair_response=cached_context,
                            )
                            if scoped_repair is None:
                                raise ValueError(
                                    "New-document metadata failed strict validation after its single repair attempt"
                                ) from None
                            repair = scoped_repair
        if exhausted:
            context = None
        else:
            context, exhausted = _run_metadata_request(
                gateway=gateway,
                request=repair,
                identity_prefix=identity_prefix,
                heartbeat=heartbeat,
                stopped=stopped,
            )
        if not exhausted:
            assert context is not None
            try:
                result = validate_metadata_transport_response(
                    context, chunks=chunks, target_ids=targets
                )
            except ValueError:
                raise ValueError(
                    "New-document metadata failed strict validation after its single repair attempt"
                ) from None
            return [(repair.request_hash, result)]
    if len(targets) == 1:
        raise ValueError(
            "New-document metadata exceeded the output token budget for one target"
        )
    midpoint = len(targets) // 2
    results: MetadataResults = []
    # Children stay sequential inside this window's concurrency slot.
    for child in (targets[:midpoint], targets[midpoint:]):
        results.extend(
            _extract_metadata_targets(
                gateway=gateway,
                chunks=chunks,
                targets=child,
                identity_prefix=identity_prefix,
                heartbeat=heartbeat,
                stopped=stopped,
            )
        )
    return results


def _run_metadata_request(
    *,
    gateway: VertexBatchGateway,
    request: VertexBatchRequest,
    identity_prefix: str,
    heartbeat: Callable[[], None],
    stopped: Event,
) -> tuple[str | None, bool]:
    if stopped.is_set():
        raise _MetadataWindowStopped(
            "New-document metadata stopped after another window failed"
        )
    submission_key = _metadata_submission_key(identity_prefix, request)
    heartbeat()
    state = gateway.submit(
        [request],
        submission_key=submission_key,
        max_jsonl_bytes=MAX_METADATA_REQUEST_BYTES,
    )
    if state.status != VertexBatchJobStatus.SUCCEEDED or not state.output_uri:
        raise ValueError("New-document metadata extraction did not complete")
    context, exhausted = _metadata_response(gateway, request, state.output_uri)
    heartbeat()
    return context, exhausted


def _metadata_submission_key(identity_prefix: str, request: VertexBatchRequest) -> str:
    return (
        "regulatory-labeling-"
        + hashlib.sha256(
            f"{identity_prefix}:{request.request_hash}".encode()
        ).hexdigest()
    )


def _cached_metadata_request(
    *,
    gateway: VertexBatchGateway,
    request: VertexBatchRequest,
    identity_prefix: str,
    heartbeat: Callable[[], None],
    stopped: Event,
) -> tuple[str | None, bool] | None:
    if stopped.is_set():
        raise _MetadataWindowStopped(
            "New-document metadata stopped after another window failed"
        )
    heartbeat()
    state = gateway.reconcile_submission(
        _metadata_submission_key(identity_prefix, request)
    )
    if state is None:
        return None
    if state.status != VertexBatchJobStatus.SUCCEEDED or not state.output_uri:
        raise ValueError("New-document metadata cached repair did not complete")
    result = _metadata_response(gateway, request, state.output_uri)
    heartbeat()
    return result


def _extract_metadata_document(
    *,
    gateway: VertexBatchGateway,
    chunks: Sequence[MetadataChunk],
    identity_prefix: str,
    heartbeat: Callable[[], None],
) -> MetadataResults:
    windows = [
        [chunk.chunk_id for chunk in chunks[offset : offset + TARGETS_PER_REQUEST]]
        for offset in range(0, len(chunks), TARGETS_PER_REQUEST)
    ]
    stopped = Event()
    failure_lock = Lock()
    failures: list[Exception] = []

    def extract(targets: Sequence[str]) -> MetadataResults:
        try:
            return _extract_metadata_targets(
                gateway=gateway,
                chunks=chunks,
                targets=targets,
                identity_prefix=identity_prefix,
                heartbeat=heartbeat,
                stopped=stopped,
            )
        except Exception as error:
            if not isinstance(error, _MetadataWindowStopped):
                with failure_lock:
                    if not failures:
                        failures.append(error)
            stopped.set()
            raise

    results: dict[int, MetadataResults] = {}
    with ThreadPoolExecutor(max_workers=METADATA_CONCURRENT_WINDOWS) as executor:
        pending: dict[Future[MetadataResults], int] = {}
        next_window = 0

        def submit() -> None:
            nonlocal next_window
            context = copy_context()
            targets = windows[next_window]

            def run_in_context() -> MetadataResults:
                return context.run(extract, targets)

            pending[executor.submit(run_in_context)] = next_window
            next_window += 1

        while next_window < min(METADATA_CONCURRENT_WINDOWS, len(windows)):
            submit()
        try:
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: pending[item]):
                    results[pending.pop(future)] = future.result()
                while (
                    not stopped.is_set()
                    and next_window < len(windows)
                    and len(pending) < METADATA_CONCURRENT_WINDOWS
                ):
                    submit()
        except Exception:
            stopped.set()
            for future in pending:
                future.cancel()
            # Executor shutdown waits for active requests to retain their receipts.
            if failures:
                raise failures[0] from None
            raise
    return [result for index in range(len(windows)) for result in results[index]]


def run_intake_metadata(*, intake_id: UUID, lease_token: UUID, tenant_id: str) -> None:
    with get_session_with_current_tenant() as session:
        batch, work = repository.load_metadata_work(session, intake_id, lease_token)
        if not work:
            session.commit()
            return
        owner = session.get_one(User, batch.created_by)
        binding = resolve_intake_labeling_provider_binding(
            session, batch.labeling_model_configuration_id, user=owner
        )
        gateway = resolve_labeling_gateway(
            session,
            batch.labeling_model_configuration_id,
            user=owner,
            expected_binding=binding,
            direct_flow=LLMFlow.REGULATORY_INTAKE_METADATA,
            direct_max_request_bytes=MAX_METADATA_REQUEST_BYTES,
            direct_max_response_bytes=MAX_METADATA_RESPONSE_BYTES,
            direct_request_timeout_seconds=240,
        )
        session.commit()
    for document_id, chunks in work:

        def heartbeat() -> None:
            with get_session_with_current_tenant() as session:
                repository.heartbeat_metadata(session, intake_id, lease_token)
                session.commit()

        results = _extract_metadata_document(
            gateway=gateway,
            chunks=chunks,
            identity_prefix=f"metadata-v{METADATA_VERSION}:{tenant_id}:{intake_id}:{document_id}:{binding.fingerprint}",
            heartbeat=heartbeat,
        )
        with get_session_with_current_tenant() as session:
            repository.apply_metadata_results(
                session,
                intake_id=intake_id,
                lease_token=lease_token,
                document_id=document_id,
                expected_input_sha256=metadata_input_hash(chunks),
                results=results,
            )
            session.commit()
