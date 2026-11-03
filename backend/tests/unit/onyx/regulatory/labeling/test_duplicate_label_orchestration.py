"""The new selection path sends one request and preserves transport/source fences."""

import json
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as repository
from onyx.db import regulatory_labeling_duplicate_evidence as evidence_db
from onyx.db import regulatory_labeling_duplicate_repair as repair_db
from onyx.db.document_intake_label_repair import IntakeLabelRepairSourceChangedError
from onyx.regulatory.indexing_jobs.models import IndexingGatewayConnectionError
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchJobStatus,
    VertexBatchState,
)
from onyx.regulatory.labeling import orchestrator
from onyx.regulatory.labeling.duplicate_ack_evidence import DuplicateAckEvidence
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from tests.unit.onyx.regulatory.labeling.test_intake_duplicate_label_repair import (
    ack_line,
    repair_for,
    selections,
    source_binding,
)


@pytest.fixture
def frozen() -> dict[str, Any]:
    return json.loads(
        (
            Path(__file__).parent / "fixtures/legacy_duplicate_label_context.json"
        ).read_text()
    )


@pytest.mark.parametrize(
    "mode",
    [
        "normal",
        "streaming",
        "missing-group",
        "duplicate-group",
        "new-quote",
        "abstained",
        "source-before",
        "source-after",
        "source-acceptance",
        "base-bytes",
        "header",
        "transport",
    ],
)
def test_one_genuine_selection_attempt_or_fail_closed_before_acceptance(
    frozen: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    repair = repair_for(frozen)
    base_raw = frozen["actual_base_cache_blob"].encode()
    stored = json.loads(base_raw)
    line = stored["line"]
    base_ack = DuplicateAckEvidence(
        file_id=f"regulatory-labeling-direct/{frozen['actual_binding_fingerprint']}/{frozen['actual_base_submission_key'].removeprefix('regulatory-labeling-')}/{repair.base_request.request_hash}.json",
        binding_fingerprint=frozen["actual_binding_fingerprint"],
        submission_key=frozen["actual_base_submission_key"],
        request_hash=repair.base_request.request_hash,
        blob_sha256=sha256(base_raw).hexdigest(),
        blob_byte_count=len(base_raw),
        line_sha256=sha256(line.encode()).hexdigest(),
        line_byte_count=len(line.encode()),
        response_sha256=sha256(repair.rejected_response.encode()).hexdigest(),
    )
    authority = repair_db.DuplicateRepairAuthority(
        repair.base_request,
        source_binding(frozen),
        repair.provider_binding,
        base_ack.binding_fingerprint,
        base_ack.submission_key,
    )
    load_authority = Mock(return_value=authority)
    if mode.startswith("source-"):
        changed = repair_db.DuplicateRepairAuthority(
            authority.base_request,
            authority.source_binding.model_copy(update={"raw_sha256": "f" * 64}),
            authority.provider_binding,
            authority.binding_fingerprint,
            authority.base_submission_key,
        )
        count = {"source-before": 1, "source-after": 2, "source-acceptance": 3}[mode]
        load_authority.side_effect = [authority] * count + [changed]
    monkeypatch.setattr(
        repair_db, "load_current_duplicate_repair_authority", load_authority
    )
    header = Mock()
    if mode == "header":
        header.side_effect = ValueError("changed cached header")
    monkeypatch.setattr(evidence_db, "validate_duplicate_ack_header", header)
    selected = json.loads(selections(repair))
    if mode == "missing-group":
        selected["labels"] = []
    elif mode == "duplicate-group":
        selected["labels"].append(selected["labels"][0])
    elif mode == "new-quote":
        selected["labels"][0]["evidence_quote"] = frozen["target_text"]
    elif mode == "abstained":
        selected["abstained"] = True
    response = json.dumps(selected)
    output_line = ack_line(repair.request, response)
    cache_calls = []

    def cached(_session: Session, **kwargs: Any) -> DuplicateAckEvidence:
        cache_calls.append(kwargs)
        if kwargs["request"] == repair.base_request:
            if kwargs["expected_line"] != line:
                raise ValueError("changed original raw bytes")
            return base_ack
        assert (
            kwargs["request"] == repair.request
            and kwargs["expected_line"] == output_line
        )
        submission = kwargs["submission_key"]
        return DuplicateAckEvidence(
            file_id=f"regulatory-labeling-direct/{authority.binding_fingerprint}/{submission.removeprefix('regulatory-labeling-')}/{repair.request.request_hash}.json",
            binding_fingerprint=authority.binding_fingerprint,
            submission_key=submission,
            request_hash=repair.request.request_hash,
            blob_sha256=sha256(output_line.encode()).hexdigest(),
            blob_byte_count=len(output_line.encode()),
            line_sha256=sha256(output_line.encode()).hexdigest(),
            line_byte_count=len(output_line.encode()),
            response_sha256=sha256(response.encode()).hexdigest(),
        )

    monkeypatch.setattr(evidence_db, "load_duplicate_ack", cached)
    session = Mock(spec=Session)

    @contextmanager
    def current_session():
        yield session

    monkeypatch.setattr(
        orchestrator, "get_session_with_current_tenant", current_session
    )
    monkeypatch.setattr(orchestrator, "get_current_tenant_id", lambda: "public")
    gateway = Mock(spec=VertexBatchGateway)
    state = VertexBatchState(
        remote_job_name="fixture-only",
        status=VertexBatchJobStatus.SUCCEEDED,
        output_uri="fixture://selection",
    )
    gateway.submit.return_value = state
    gateway.read_results.return_value = iter([output_line])
    if mode == "streaming":
        gateway.submit_stream = Mock(return_value=state)
    if mode == "transport":
        gateway.submit.side_effect = IndexingGatewayConnectionError()
    item = SimpleNamespace(
        id=uuid4(),
        regulatory_chunk_id=frozen["target_chunk_id"],
        text_snapshot=frozen["target_text"],
        canonical_text_sha256=frozen["target_source_hash"],
    )
    lease = repository.RunLease(uuid4(), 1, uuid4())

    def invoke():
        return orchestrator._repair_intake_duplicates(
            lease,
            uuid4(),
            cast(Any, item),
            cast(VertexBatchGateway, gateway),
            frozen["base_response"],
            line + "changed" if mode == "base-bytes" else line,
            TaxonomyDefinition.model_validate(frozen["taxonomy"]),
            lambda: None,
        )

    if mode == "transport":
        with pytest.raises(IndexingGatewayConnectionError):
            invoke()
        gateway.submit.assert_called_once()
        gateway.read_results.assert_not_called()
        return
    if mode.startswith("source-"):
        with pytest.raises(IntakeLabelRepairSourceChangedError):
            invoke()
    elif mode in {
        "missing-group",
        "duplicate-group",
        "new-quote",
        "abstained",
        "base-bytes",
        "header",
    }:
        with pytest.raises(ValueError):
            invoke()
    else:
        accepted = invoke()
        assert accepted.audit.attempt == 1 and not accepted.outcome.abstained
        assert len(accepted.outcome.labels) == 4
        assert all(
            next(
                a.evidence_quote for a in accepted.outcome.labels if a.label_id == label
            )
            == quote
            for label, quote in repair.retained_quotes.items()
        )
        assert cache_calls[0]["expected_line"] == line
    if mode in {"source-before", "base-bytes", "header"}:
        gateway.submit.assert_not_called()
        gateway.read_results.assert_not_called()
    elif mode == "streaming":
        gateway.submit.assert_not_called()
        gateway.submit_stream.assert_called_once()
        assert list(gateway.submit_stream.call_args.args[0]) == [repair.request]
    else:
        gateway.submit.assert_called_once()
        assert gateway.submit.call_args.args[0] == [repair.request]
        assert (
            "/duplicate-label-repair/"
            not in gateway.submit.call_args.kwargs["submission_key"]
        )
