"""Durable sparse extraction with strict local expansion and bounded fallback."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from threading import Event

from onyx.regulatory.document_intake import metadata
from onyx.regulatory.document_intake.metadata_job import (
    MetadataResults,
    _run_metadata_request,
)
from onyx.regulatory.document_intake.metadata_scopes import (
    build_scoped_metadata_request,
    expand_scoped_metadata_response,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchRequest,
)


def extract_scoped_metadata_document(
    *,
    gateway: VertexBatchGateway,
    chunks: Sequence[metadata.MetadataChunk],
    identity_prefix: str,
    heartbeat: Callable[[], None],
    target_indices: Sequence[int] | None = None,
    previous_attempts: Sequence[tuple[str, str]] = (),
    previous_target_indices: Mapping[str, Sequence[int]] | None = None,
) -> MetadataResults:
    targets = (
        list(range(len(chunks))) if target_indices is None else list(target_indices)
    )
    base = build_scoped_metadata_request(chunks, targets)
    request = base
    previous: list[tuple[str, str]] = list(previous_attempts)
    prior_targets = dict(previous_target_indices or {})
    stopped = Event()
    last_error: ValueError = ValueError(
        "Scoped metadata output exhausted its bounded budget"
    )
    for attempt in range(3):
        response, exhausted = _run_metadata_request(
            gateway=gateway,
            request=request,
            identity_prefix=identity_prefix,
            heartbeat=heartbeat,
            stopped=stopped,
        )
        if exhausted:
            break
        assert response is not None
        try:
            results = expand_scoped_metadata_response(
                response,
                chunks=chunks,
                target_indices=targets,
                previous_attempts=previous,
                previous_target_indices=prior_targets,
            )
            return [(request.request_hash, result) for result in results]
        except ValueError as error:
            last_error = error
            previous.append((request.request_hash, response))
            prior_targets[request.request_hash] = targets
            if attempt == 2:
                break
            payload = json.loads(base.prompt)
            payload["validation_feedback"] = {
                "attempt": attempt + 1,
                "error": str(error)[:4000],
                "previous_request_hash": request.request_hash,
                "previous_response_sha256": hashlib.sha256(
                    response.encode()
                ).hexdigest(),
                "previous_response": response,
                "instruction": "Repair the rejected scope/evidence using the unchanged complete source and target set. Preserve every supported fact. Coverage must be exact; no conflicting field assignments or heading evidence for body-only fields. Never broaden a date's legal scope to avoid a validation error.",
            }
            request = VertexBatchRequest(
                prompt=json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                ),
                system_instruction=base.system_instruction,
                generation_config=base.generation_config,
            )
            try:
                metadata._check_metadata_request_budget(request)
            except ValueError as budget_error:
                last_error = budget_error
                break
    if len(targets) > metadata.TARGETS_PER_REQUEST:
        midpoint = len(targets) // 2
        return [
            result
            for group in (targets[:midpoint], targets[midpoint:])
            for result in extract_scoped_metadata_document(
                gateway=gateway,
                chunks=chunks,
                identity_prefix=identity_prefix,
                heartbeat=heartbeat,
                target_indices=group,
                previous_attempts=previous,
                previous_target_indices=prior_targets,
            )
        ]
    raise ValueError(
        f"Scoped legal metadata failed after bounded repair: {last_error}"
    ) from last_error
