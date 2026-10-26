"""Enrich only newly chunked intake files using cached direct Vertex requests."""

import hashlib
from uuid import UUID

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
    TARGETS_PER_REQUEST,
    LegalMetadataResult,
    build_metadata_request,
    metadata_input_hash,
    validate_metadata_response,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchJobStatus,
    parse_vertex_jsonl_output,
)
from onyx.tracing.flows import LLMFlow


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
        results: list[tuple[str, LegalMetadataResult]] = []
        for offset in range(0, len(chunks), TARGETS_PER_REQUEST):
            targets = [
                chunk.chunk_id
                for chunk in chunks[offset : offset + TARGETS_PER_REQUEST]
            ]
            request = build_metadata_request(chunks, targets)
            identity = f"metadata-v1:{tenant_id}:{intake_id}:{document_id}:{binding.fingerprint}:{request.request_hash}"
            submission_key = (
                "regulatory-labeling-" + hashlib.sha256(identity.encode()).hexdigest()
            )
            with get_session_with_current_tenant() as session:
                repository.heartbeat_metadata(session, intake_id, lease_token)
                session.commit()
            state = gateway.submit(
                [request],
                submission_key=submission_key,
                max_jsonl_bytes=MAX_METADATA_REQUEST_BYTES,
            )
            if state.status != VertexBatchJobStatus.SUCCEEDED or not state.output_uri:
                raise ValueError("New-document metadata extraction did not complete")
            parsed = parse_vertex_jsonl_output(
                gateway.read_results(state.output_uri), [request.request_hash]
            )[request.request_hash]
            if parsed.error is not None or parsed.context is None:
                raise ValueError(
                    "New-document metadata extraction returned incomplete or invalid output"
                )
            result = validate_metadata_response(
                parsed.context, chunks=chunks, target_ids=targets
            )
            results.append((request.request_hash, result))
            with get_session_with_current_tenant() as session:
                repository.heartbeat_metadata(session, intake_id, lease_token)
                session.commit()
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
