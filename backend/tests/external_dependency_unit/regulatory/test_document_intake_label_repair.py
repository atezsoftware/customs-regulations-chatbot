"""Intake quote repair reads only the currently leased frozen source."""

import datetime
import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from google.auth.credentials import Credentials
from google.genai import types as genai_types
from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_label_repair import (
    IntakeLabelRepairSourceChangedError,
    load_current_intake_label_repair_request,
)
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    User,
)
from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.models import (
    IndexingGatewayError,
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling import orchestrator, vertex_direct
from onyx.regulatory.labeling.provider import (
    LabelDefinition,
    TaxonomyDefinition,
    build_labeling_request,
    validate_labeling_response,
)
from onyx.tracing.flows import LLMFlow
from tests.external_dependency_unit.regulatory.test_document_intake import _admitted
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _Store


def _submitted(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    RegulatoryDocumentIntake,
    RegulatoryLabelingRun,
    labeling.RunLease,
    RegulatoryLabelingShard,
    list[RegulatoryLabelingItem],
]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    root = next(member for member in members if member.role == "root")
    main = session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == root.user_file_id)
    )
    assert main is not None
    main.text = "İthalatçı, ürünün teknik\nbelgesini saklar."
    session.add(
        RegulatoryChunk(
            id=f"rc_repair_neighbor_{uuid4().hex}",
            user_file_id=main.user_file_id,
            text="Komşu madde başka bir yükümlülük açıklar.",
            position=1,
            projection_ordinal=1,
            heading_path=["MADDE 2"],
            chunk_type="article",
            chunk_metadata={"chunk_variant": "atomic"},
            status="active",
            source="indexed",
        )
    )
    taxonomy = labeling.create_taxonomy(
        session,
        created_by_id=user.id,
        taxonomy=TaxonomyDefinition(
            name="Repair labels",
            labels=[LabelDefinition(id="customs", name="Gümrük", description="Gümrük")],
        ),
    )
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="intake-repair-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        labeling, "get_label_settings", lambda *_a: SimpleNamespace(taxonomy=taxonomy)
    )
    token = intake.claim_intake(session, batch.id)
    assert token is not None
    for member in members:
        intake.mark_intake_file_prepared(session, batch.id, token, member.id)
    run_id = intake.start_intake_labeling(session, batch.id, token)
    lease = labeling.claim_run(
        session, run_id=run_id, expected_generation=0, lease_seconds=300
    )
    assert lease is not None
    run = session.get_one(RegulatoryLabelingRun, run_id)
    items = labeling.prepare_next_item_page(session, lease, limit=128)
    prepared = []
    for item in items:
        assert item.context_snapshot is not None
        assert item.canonical_text_sha256 is not None
        request = build_labeling_request(
            chunk_id=item.regulatory_chunk_id,
            text=item.text_snapshot,
            context=item.context_snapshot,
            taxonomy=TaxonomyDefinition.model_validate(taxonomy.definition),
            source_hash=item.canonical_text_sha256,
        )
        prepared.append(
            labeling.PreparedRequest(
                item_id=item.id,
                request_hash=request.request_hash,
                request_payload=request.model_dump(
                    mode="json", exclude_computed_fields=True
                ),
            )
        )
    labeling.store_prepared_shards(
        session,
        lease,
        requests=prepared,
        shards=[
            labeling.PreparedShard(
                ordinal=0,
                item_ids=tuple(item.id for item in items),
                submission_key="regulatory-labeling-"
                + sha256(str(run.id).encode()).hexdigest(),
            )
        ],
        failed_items={},
    )
    shard = session.scalar(
        select(RegulatoryLabelingShard).where(RegulatoryLabelingShard.run_id == run.id)
    )
    assert shard is not None
    labeling.mark_shard_submitting(
        session, lease, shard_id=shard.id, reconcile_seconds=30
    )
    labeling.record_shard_state(session, lease, shard_id=shard.id, status="submitted")
    session.flush()
    for item in items:
        session.refresh(item)
    return batch, run, lease, shard, items


def test_repair_request_is_frozen_complete_and_read_only(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _batch, run, lease, shard, items = _submitted(
        intake_session, intake_owner, monkeypatch
    )
    item = items[0]
    before = deepcopy(item.request_payload), item.request_hash, item.context_sha256
    lease_before = run.lease_token, run.lease_generation, run.lease_expires_at
    request = load_current_intake_label_repair_request(
        intake_session, lease, shard.id, item.id
    )
    assert request.request_hash == item.request_hash
    assert (
        request.model_dump(mode="json", exclude_computed_fields=True)
        == item.request_payload
    )
    assert (item.request_payload, item.request_hash, item.context_sha256) == before
    assert (run.lease_token, run.lease_generation, run.lease_expires_at) == lease_before
    assert (
        not intake_session.new
        and not intake_session.dirty
        and not intake_session.deleted
    )


@pytest.mark.parametrize(
    "change",
    [
        "ordinary",
        "broadened",
        "cancel",
        "expired",
        "token",
        "terminal",
        "shard",
        "membership",
    ],
)
def test_repair_refuses_stale_authority_before_another_request(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    batch, run, lease, shard, items = _submitted(
        intake_session, intake_owner, monkeypatch
    )
    item = items[0]
    if change == "ordinary":
        intake_session.execute(
            update(RegulatoryDocumentIntake)
            .where(RegulatoryDocumentIntake.id == batch.id)
            .values(labeling_run_id=None)
            .execution_options(synchronize_session=False)
        )
    elif change == "broadened":
        intake_session.execute(
            update(RegulatoryLabelingRun)
            .where(RegulatoryLabelingRun.id == run.id)
            .values(file_ids=[*run.file_ids, str(uuid4())])
            .execution_options(synchronize_session=False)
        )
    elif change == "cancel":
        intake_session.execute(
            update(RegulatoryLabelingRun)
            .where(RegulatoryLabelingRun.id == run.id)
            .values(cancel_requested=True)
            .execution_options(synchronize_session=False)
        )
    elif change in {"expired", "token"}:
        values = (
            {
                "lease_expires_at": datetime.datetime.now(datetime.timezone.utc)
                - datetime.timedelta(seconds=1)
            }
            if change == "expired"
            else {"lease_token": uuid4()}
        )
        intake_session.execute(
            update(RegulatoryLabelingRun)
            .where(RegulatoryLabelingRun.id == run.id)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
    elif change == "terminal":
        item.status = "completed"
    elif change == "shard":
        shard.item_ids = []
    else:
        intake_session.execute(
            delete(DocumentSet__UserFile).where(
                DocumentSet__UserFile.user_file_id == item.user_file_id,
                DocumentSet__UserFile.document_set_id == run.document_set_id,
            )
        )
    intake_session.flush()
    with pytest.raises(labeling.LabelingStateConflictError):
        load_current_intake_label_repair_request(
            intake_session, lease, shard.id, item.id
        )


@pytest.mark.parametrize(
    "change", ["text", "heading", "date", "neighbor", "payload", "context"]
)
def test_repair_rechecks_source_and_frozen_payload_even_if_orm_was_cached(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    _batch, _run, lease, shard, items = _submitted(
        intake_session, intake_owner, monkeypatch
    )
    item = next(item for item in items if "teknik" in item.text_snapshot)
    original_request = deepcopy(item.request_payload)
    if change in {"payload", "context"}:
        values = (
            {"request_payload": {"prompt": "different"}}
            if change == "payload"
            else {"context_snapshot": "different"}
        )
        intake_session.execute(
            update(RegulatoryLabelingItem)
            .where(RegulatoryLabelingItem.id == item.id)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
    else:
        chunk_id = item.regulatory_chunk_id
        if change == "neighbor":
            chunk_id = next(
                other.regulatory_chunk_id
                for other in items
                if other.user_file_id == item.user_file_id and other.id != item.id
            )
        values = (
            {"heading_path": ["Changed"]}
            if change == "heading"
            else {"validity_start_date": datetime.date(2026, 1, 1)}
            if change == "date"
            else {"text": "Changed canonical text"}
        )
        intake_session.execute(
            update(RegulatoryChunk)
            .where(RegulatoryChunk.id == chunk_id)
            .values(**values)
            .execution_options(synchronize_session=False)
        )
    with pytest.raises(IntakeLabelRepairSourceChangedError):
        load_current_intake_label_repair_request(
            intake_session, lease, shard.id, item.id
        )
    if change != "payload":
        assert item.request_payload == original_request


@pytest.mark.parametrize(
    "case",
    [
        "repair",
        "ordinary",
        "invalid-repair",
        "cached-resume",
        "cancel-after-ack",
        "lease-after-ack",
    ],
)
def test_selector_repair_preserves_base_receipts_and_resumes_acknowledged_request(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    batch, run, lease, shard, items = _submitted(
        intake_session, intake_owner, monkeypatch
    )
    target = next(item for item in items if "teknik" in item.text_snapshot)
    if case == "ordinary":
        batch.labeling_run_id = None
        intake_session.flush()
    original_requests = {
        item.id: (
            item.request_hash,
            deepcopy(item.request_payload),
            item.context_sha256,
        )
        for item in items
    }
    canonical_before = {
        row.id: (row.text, deepcopy(row.chunk_metadata), list(row.heading_path))
        for row in intake_session.scalars(select(RegulatoryChunk))
    }
    store, sdk = _Store(), MagicMock()
    repair_calls: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        prompt = json.loads(kwargs["contents"][0].parts[0].text)
        repair = prompt.get("quote_repair")
        if repair is not None:
            repair_calls.append(prompt)
            assert repair["original_request_hash"] == target.request_hash
            assert repair["attempt"] == 1
            assert repair["repair_label_ids"] == ["customs"]
            assert prompt["target"]["text"] == target.text_snapshot
            evidence_id = repair["evidence_catalog"][0]["evidence_id"]
            response: dict[str, Any] = {
                "labels": [
                    {
                        "label_id": "customs",
                        "evidence_id": "missing"
                        if case == "invalid-repair"
                        else evidence_id,
                    }
                ],
                "abstained": False,
            }
        else:
            response = {
                "labels": [
                    {
                        "label_id": "customs",
                        "evidence_quote": "ürünün mali belgesini"
                        if prompt["target"]["chunk_id"] == target.regulatory_chunk_id
                        else prompt["target"]["text"],
                    }
                ],
                "abstained": False,
            }
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": json.dumps(response, ensure_ascii=False)}
                            ],
                        },
                    }
                ]
            }
        )

    sdk.models.generate_content.side_effect = generate
    binding = LabelingProviderBinding.model_validate(run.provider_binding)
    gateway = vertex_direct.LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=binding.model_configuration_id,
            model_name=binding.model,
            project=binding.project,
            location=binding.location,
            authentication_mode=binding.authentication_mode,
        ),
        binding_sha256=binding.fingerprint,
        credential_identity=binding.credential_identity,
        credential_json_provider=lambda: None,
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=lambda _credentials: sdk,
        store=cast(FileStore, store),
        flow=LLMFlow.REGULATORY_LABELING_BATCH,
    )

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield intake_session

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    monkeypatch.setattr(orchestrator, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    base = gateway.submit(
        [VertexBatchRequest.model_validate(item.request_payload) for item in items],
        submission_key=shard.submission_key,
        max_jsonl_bytes=orchestrator.LABELING_SHARD_BYTES,
    )
    assert base.output_uri is not None
    base_receipts = deepcopy(store.files)
    assert sdk.models.generate_content.call_count == len(items)
    submit_stream = gateway.submit_stream

    def submit_then_revoke(*args: Any, **kwargs: Any) -> Any:
        state = submit_stream(*args, **kwargs)
        if case in {"cancel-after-ack", "lease-after-ack"}:
            values = (
                {"cancel_requested": True}
                if case == "cancel-after-ack"
                else {
                    "lease_expires_at": datetime.datetime.now(datetime.timezone.utc)
                    - datetime.timedelta(seconds=1)
                }
            )
            intake_session.execute(
                update(RegulatoryLabelingRun)
                .where(RegulatoryLabelingRun.id == run.id)
                .values(**values)
                .execution_options(synchronize_session=False)
            )
        return state

    monkeypatch.setattr(gateway, "submit_stream", submit_then_revoke)
    if case in {"cancel-after-ack", "lease-after-ack"}:
        with pytest.raises(labeling.LabelingStateConflictError):
            orchestrator._read_and_apply_results(
                lease, shard.id, gateway, base.output_uri
            )
        assert len(repair_calls) == 1
        assert all(
            item.status == "submitted" and not item.assignments for item in items
        )
        assert all(store.files[key] == value for key, value in base_receipts.items())
        assert {
            item.id: (item.request_hash, item.request_payload, item.context_sha256)
            for item in items
        } == original_requests
        return
    apply = labeling.apply_shard_result_page
    interruptions = 0

    def apply_after_ack(*args: Any, **kwargs: Any) -> None:
        nonlocal interruptions
        if case == "cached-resume" and interruptions == 0:
            assert len(repair_calls) == 1
            interruptions += 1
            raise IndexingGatewayError(
                "Simulated interruption after repair acknowledgement"
            )
        apply(*args, **kwargs)

    monkeypatch.setattr(labeling, "apply_shard_result_page", apply_after_ack)
    acknowledged = deepcopy(store.files)
    if case == "cached-resume":
        with pytest.raises(IndexingGatewayError, match="Simulated interruption"):
            orchestrator._read_and_apply_results(
                lease, shard.id, gateway, base.output_uri
            )
        assert all(item.status == "submitted" for item in items)
        acknowledged = deepcopy(store.files)
    orchestrator._read_and_apply_results(lease, shard.id, gateway, base.output_uri)
    assert len(repair_calls) == (0 if case == "ordinary" else 1), [
        (item.status, item.error) for item in items
    ]
    assert sdk.models.generate_content.call_count == len(items) + (
        0 if case == "ordinary" else 1
    )
    if case == "cached-resume":
        assert store.files == acknowledged
    assert all(store.files[key] == value for key, value in base_receipts.items())
    assert target.status == (
        "completed" if case in {"repair", "cached-resume"} else "failed"
    )
    assert all(item.status == "completed" for item in items if item.id != target.id)
    if target.status == "completed":
        taxonomy = TaxonomyDefinition.model_validate(run.taxonomy.definition)
        validated = validate_labeling_response(
            json.dumps({"labels": target.assignments, "abstained": False}),
            text=target.text_snapshot,
            taxonomy=taxonomy,
        )
        assert [assignment.label_id for assignment in validated.labels] == ["customs"]
        assert all(
            set(assignment) == {"label_id", "evidence_quote"}
            for assignment in target.assignments
        )
        calls_before_replay = sdk.models.generate_content.call_count
        assignments_before_replay = deepcopy(target.assignments)
        orchestrator._read_and_apply_results(lease, shard.id, gateway, base.output_uri)
        assert sdk.models.generate_content.call_count == calls_before_replay
        assert target.assignments == assignments_before_replay
    assert {
        item.id: (item.request_hash, item.request_payload, item.context_sha256)
        for item in items
    } == original_requests
    assert {
        row.id: (row.text, row.chunk_metadata, row.heading_path)
        for row in intake_session.scalars(select(RegulatoryChunk))
    } == canonical_before
