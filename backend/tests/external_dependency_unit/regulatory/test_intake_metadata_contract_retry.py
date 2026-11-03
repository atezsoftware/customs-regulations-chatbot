"""Real PostgreSQL owner Retry keeps old ACKs and changes only declared clause facts."""

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as metadata_db
from onyx.db import document_intake_metadata_contract_retry as repair
from onyx.db import labeling_configuration
from onyx.db.document_intake_metadata_checkpoint import (
    require_current_metadata_checkpoint,
)
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    User,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.document_intake import metadata
from onyx.regulatory.document_intake.metadata_checkpoint import MetadataCheckpoint
from onyx.regulatory.document_intake.metadata_contract_retry import (
    build_contract_retry_request,
    metadata_ack_file_id,
    validate_contract_retry_response,
)
from onyx.regulatory.document_intake.metadata_job import _metadata_submission_key
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from shared_configs.contextvars import get_current_tenant_id
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _completed_labels,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    _ready_metadata,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake_metadata import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_intake_metadata_checkpoint import (
    LabelingProviderBinding,
)
from tests.unit.onyx.regulatory.test_intake_metadata_budget import _envelope


def _save_ack(
    session: Session,
    store: PostgresBackedFileStore,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    binding: LabelingProviderBinding,
    request: VertexBatchRequest,
    response: str,
) -> str:
    prefix = f"metadata-v2:{get_current_tenant_id()}:{batch.id}:{member.id}:{binding.fingerprint}"
    line = json.dumps(_envelope(request, "STOP", response), ensure_ascii=False)
    raw = json.dumps(
        {
            "binding_sha256": binding.fingerprint,
            "submission_key": _metadata_submission_key(prefix, request),
            "request_hash": request.request_hash,
            "request_payload": request.model_dump(
                mode="json", exclude={"request_hash"}
            ),
            "line_sha256": sha256(line.encode()).hexdigest(),
            "line": line,
        },
        ensure_ascii=False,
    ).encode()
    cache = metadata_ack_file_id(
        tenant_id=get_current_tenant_id(),
        intake_id=batch.id,
        document_id=member.id,
        binding_sha256=binding.fingerprint,
        request_hash=request.request_hash,
    )
    store.save_file(
        BytesIO(raw),
        "metadata ACK",
        FileOrigin.CONNECTOR,
        "application/json",
        file_metadata={"sha256": sha256(raw).hexdigest()},
        file_id=cache,
        db_session=session,
    )
    return cache


def _legacy(
    session: Session,
    owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    RegulatoryDocumentIntake,
    list[RegulatoryDocumentIntakeFile],
    RegulatoryChunk,
    PostgresBackedFileStore,
    LabelingProviderBinding,
]:
    batch, members, _token, target, other = _ready_metadata(session, owner)
    target.text = "b) Bakanlığın gerekçeli talebi yerine getirilir."
    target.heading_path = ["MADDE 13", target.text]
    target.chunk_type = "clause"
    target.chunk_metadata = {
        **target.chunk_metadata,
        "clause_label": "b",
        "article_no": None,
    }
    other.text = "a) Belgeler korunur."
    other.heading_path = ["MADDE 13", other.text]
    other.chunk_type = "clause"
    model = owner[2]
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider_id,
        project="metadata-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="workload_identity",
        model=metadata.METADATA_MODEL,
        transport="vertex_direct_v1",
    )
    monkeypatch.setattr(
        repair, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        intake, "resolve_intake_labeling_provider_binding", lambda *_a, **_k: binding
    )
    monkeypatch.setattr(
        labeling_configuration,
        "resolve_intake_labeling_provider_binding",
        lambda *_a, **_k: binding,
    )
    assert batch.lease_token is not None
    checkpoint = require_current_metadata_checkpoint(
        session,
        batch.id,
        batch.lease_token,
        provider_binding=binding,
        recipe_sha256="a" * 64,
        capture=True,
    )
    store = PostgresBackedFileStore()
    monkeypatch.setattr(repair, "get_default_file_store", lambda: store)
    for member in members:
        rows = repair._rows(session, member.user_file_id)
        chunks = metadata_db.metadata_chunks(rows)
        request = metadata.build_metadata_request(chunks, [c.chunk_id for c in chunks])
        assert request.request_hash in checkpoint.request_hashes[str(member.id)]
        fields = []
        if target.user_file_id == member.user_file_id:
            evidence = next(
                e
                for e in metadata.build_metadata_evidence_catalog(chunks)
                if e.source_chunk_id == other.id
                and e.scope == "heading"
                and e.text == other.text
            )
            fields = [
                {
                    "field": "clause_label",
                    "value": "b",
                    "basis": "explicit",
                    "evidence_id": evidence.evidence_id,
                    "reference_evidence_id": "",
                }
            ]
            _save_ack(
                session,
                store,
                batch,
                member,
                binding,
                request,
                json.dumps(
                    {
                        "chunks": [
                            {
                                "chunk_id": row.id,
                                "fields": fields if row.id == target.id else [],
                            }
                            for row in rows
                        ],
                        "warnings": [],
                    }
                ),
            )
        for row in rows:
            audit_fields = {}
            if row.id == target.id:
                audit_fields["clause_label"] = {
                    "source": "llm",
                    "request_hash": request.request_hash,
                    "basis": "explicit",
                    "evidence_quote": other.text,
                    "source_chunk_id": other.id,
                    "evidence_scope": "heading",
                    "reference_quote": None,
                    "reference_chunk_id": None,
                    "stale": False,
                }
            else:
                audit_fields["paragraph_no"] = {
                    "source": "manual",
                    "basis": "user_review",
                    "source_chunk_id": row.id,
                    "evidence_quote": None,
                    "stale": False,
                }
            row.chunk_metadata = {
                **row.chunk_metadata,
                metadata.METADATA_AUDIT_KEY: {
                    "version": metadata.METADATA_VERSION,
                    "model": metadata.METADATA_MODEL,
                    "request_hash": request.request_hash,
                    "input_sha256": metadata.metadata_input_hash(chunks),
                    "input_stale": False,
                    "warnings": [],
                    "fields": audit_fields,
                },
            }
    run = _completed_labels(session, batch, members)
    run.provider_binding = binding.model_dump(mode="json")
    run.model = binding.model
    run.requested_by_id = batch.created_by
    run.status = "completed_with_errors"
    batch.status, batch.stage, batch.error = (
        "partial",
        "labeling",
        "Valid labeling abstention",
    )
    batch.lease_token, batch.lease_expires_at = None, None
    session.flush()
    return batch, members, target, store, binding


@pytest.mark.parametrize("omitted", [False, True])
@pytest.mark.parametrize(
    "post_apply_change",
    ["none", "old-ack", "changed-result", "descriptor", "manual", "source"],
)
def test_idle_retry_frozen_hints_actual_ack_apply_and_duplicate_identity(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    omitted: bool,
    post_apply_change: str,
) -> None:
    session = intake_session
    batch, members, target, store, binding = _legacy(session, intake_owner, monkeypatch)
    before_cp = deepcopy(batch.metadata_checkpoint)
    source_before = {
        str(m.id): repair.canonical_metadata_source_digest(
            repair._rows(session, m.user_file_id)
        )
        for m in members
    }
    assert intake.retry_intake(session, batch) == []
    intent, pending = repair.require_current_contract(
        session, batch, require_complete=False
    )
    assert len(pending) == 1 and batch.stage == "metadata"
    assert intent.previous_checkpoint.model_dump(mode="json") == before_cp
    assert batch.metadata_checkpoint is not None and before_cp is not None
    assert batch.metadata_checkpoint["request_hashes"] == before_cp["request_hashes"]
    document_id, target_id = pending[0]
    document = intent.documents[str(document_id)]
    assert (
        next(
            c for c in document.chunks if c.chunk_id == target_id
        ).structural_metadata.get("clause_label")
        is None
    )
    request = build_contract_retry_request(intent.id, document.chunks, target_id)
    assert request.request_hash != document.targets[target_id].previous_request_hash
    cp_after = deepcopy(batch.metadata_checkpoint)
    assert (
        intake.retry_intake(session, batch) == []
        and batch.metadata_checkpoint == cp_after
    )
    batch.lease_token = token = uuid4()
    batch.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    evidence = next(
        e
        for e in metadata.build_metadata_evidence_catalog(document.chunks)
        if e.source_chunk_id == target_id
        and e.scope == "body"
        and e.text == target.text
    )
    response = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": target_id,
                    "fields": []
                    if omitted
                    else [
                        {
                            "field": "clause_label",
                            "value": "b",
                            "basis": "explicit",
                            "evidence_id": evidence.evidence_id,
                            "reference_evidence_id": "",
                        }
                    ],
                }
            ],
            "warnings": [],
        }
    )
    result = validate_contract_retry_response(
        response, chunks=document.chunks, target_id=target_id
    )
    member = next(m for m in members if m.id == document_id)
    _save_ack(session, store, batch, member, binding, request, response)
    repair.apply_contract_result(
        session,
        intake_id=batch.id,
        lease_token=token,
        document_id=document_id,
        target_id=target_id,
        request=request,
        result=result,
    )
    assert (
        repair.require_current_contract(session, batch, require_complete=True)[1] == []
    )
    saved = deepcopy(target.chunk_metadata)
    repair.apply_contract_result(
        session,
        intake_id=batch.id,
        lease_token=token,
        document_id=document_id,
        target_id=target_id,
        request=request,
        result=result,
    )
    assert target.chunk_metadata == saved
    assert ("clause_label" not in saved) is omitted
    proof = repair.contract_retry_snapshot(session, batch)
    assert proof is not None and proof["complete"] is True
    assert {
        str(m.id): repair.canonical_metadata_source_digest(
            repair._rows(session, m.user_file_id)
        )
        for m in members
    } == source_before
    assert batch.metadata_checkpoint == cp_after
    if post_apply_change != "none":
        from onyx.regulatory.document_intake.metadata_checkpoint import metadata_digest

        value = deepcopy(target.chunk_metadata)
        stamp = value[metadata.METADATA_AUDIT_KEY]["metadata_contract_retry"]
        if post_apply_change == "old-ack":
            prior = document.targets[target_id]
            stamp["accepted_request_hash"] = prior.previous_request_hash
            stamp["raw_ack_sha256"] = prior.previous_raw_ack_sha256
            target.chunk_metadata = value
        elif post_apply_change == "descriptor":
            stamp["validation_feedback"] = {
                "repair_attempt": 1,
                "original_request_hash": "0" * 64,
            }
            target.chunk_metadata = value
        elif post_apply_change == "changed-result":
            if omitted:
                value["clause_label"] = "b"
                value[metadata.METADATA_AUDIT_KEY]["fields"]["clause_label"] = {
                    "source": "llm",
                    "request_hash": request.request_hash,
                    "basis": "explicit",
                    "evidence_quote": target.text,
                    "source_chunk_id": target.id,
                    "evidence_scope": "body",
                    "reference_chunk_id": None,
                    "reference_quote": None,
                    "stale": False,
                }
                forged = metadata.LegalMetadataResult(
                    chunks=[
                        metadata.ChunkLegalMetadata(
                            chunk_id=target_id,
                            fields=[
                                metadata.LegalMetadataFact(
                                    field="clause_label",
                                    value="b",
                                    source_chunk_id=target_id,
                                    evidence_quote=target.text,
                                )
                            ],
                        )
                    ]
                )
            else:
                value.pop("clause_label", None)
                value[metadata.METADATA_AUDIT_KEY]["fields"].pop("clause_label", None)
                forged = metadata.LegalMetadataResult(
                    chunks=[metadata.ChunkLegalMetadata(chunk_id=target_id, fields=[])]
                )
            stamp["omitted"] = not omitted
            stamp["result_sha256"] = metadata_digest(
                forged.model_dump(mode="json", exclude={"warnings"})
            )
            target.chunk_metadata = value
        elif post_apply_change == "manual":
            other = next(
                row
                for row in repair._rows(session, members[-1].user_file_id)
                if row.id != target_id
            )
            other.chunk_metadata = {
                **other.chunk_metadata,
                "article_title": "Unreviewed replacement",
            }
        else:
            target.text += " Unreviewed source body."
        session.flush()
        with pytest.raises(intake.IntakeConflictError):
            repair.require_current_contract(
                session, batch, require_complete=True, verify_ack_bytes=True
            )
        if post_apply_change != "changed-result":
            with pytest.raises(intake.IntakeConflictError):
                repair.require_current_contract(session, batch, require_complete=True)
        with pytest.raises(intake.IntakeConflictError):
            intake.start_intake_labeling(session, batch.id, token)
        assert batch.metadata_checkpoint == cp_after
        return
    from onyx.db.models import RegulatoryLabelingRun, RegulatoryLabelTaxonomy
    from onyx.regulatory.labeling.provider import TaxonomyDefinition

    previous = session.get_one(RegulatoryLabelingRun, intent.previous_labeling_run_id)
    taxonomy = TaxonomyDefinition.model_validate(
        {
            "name": "Contract fixture",
            "labels": [{"id": "rule", "name": "Kural", "description": "Kaynak hükmü"}],
        }
    )
    stored_taxonomy = session.get_one(RegulatoryLabelTaxonomy, previous.taxonomy_id)
    stored_taxonomy.definition = taxonomy.model_dump(mode="json")
    stored_taxonomy.version_hash = taxonomy.version_hash
    previous.status = "completed"
    session.flush()
    with pytest.raises(intake.IntakeConflictError, match="fresh labeling"):
        metadata_db.validate_intake_metadata_receipts(session, batch)
    fresh = intake.start_intake_labeling(session, batch.id, token)
    assert fresh is not None and fresh != intent.previous_labeling_run_id
    replacement = session.get_one(RegulatoryLabelingRun, fresh)
    assert replacement.completed_chunks == 0 and replacement.status == "queued"
    assert (
        repair.require_current_contract(
            session, batch, require_complete=True, verify_ack_bytes=True
        )[1]
        == []
    )
    if not omitted:
        _approve_then_published_history(
            session, batch, members, replacement, monkeypatch
        )


def _approve_then_published_history(
    session: Session,
    batch: RegulatoryDocumentIntake,
    members: list[RegulatoryDocumentIntakeFile],
    replacement: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Label readiness is a fixture; contract/Approve/index-intent/history are real."""
    from onyx.db.engine.sql_engine import get_session
    from onyx.db.enums import UserFileStatus
    from onyx.db.models import UserFile
    from onyx.server.features.regulatory import document_intake_api as api

    replacement.status = "completed"
    replacement.completed_chunks = replacement.total_chunks
    batch.status, batch.stage, batch.review_sha256 = "ready", "review", "f" * 64
    for member in members:
        member.status = "ready"
    monkeypatch.setattr(intake, "review_digest", lambda *_args: "f" * 64)
    session.flush()
    assert len(intake.approve_intake(session, batch, "f" * 64)) == len(members)
    assert batch.status == "approving" and batch.stage == "indexing"
    with pytest.raises(intake.IntakeConflictError):
        repair.require_current_contract(session, batch, require_complete=True)
    for member in members:
        session.get_one(UserFile, member.user_file_id).status = UserFileStatus.COMPLETED
        session.get_one(RegulatoryFilePublication, member.user_file_id).epoch = 1
    session.flush()
    owner = session.get_one(User, batch.created_by)
    app = FastAPI()
    app.include_router(api.router, prefix="/api/regulatory")
    app.dependency_overrides[api._admin] = lambda: owner
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_current_tenant_id] = lambda: get_current_tenant_id()
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    response = TestClient(app).get(
        f"/api/regulatory/document-intake/batches/{batch.id}"
    )
    assert response.status_code == 200, response.text
    report = response.json()["metadata_contract_retry"]
    assert report["phase"] == "historical_after_approval"
    assert report["complete"] is None and report["current_strict_postcondition"] is None
    assert report["current_repair_authority"] is False and "error" not in report
    assert len(report["recorded_target_receipts"]) == 1


@pytest.mark.parametrize(
    "drift",
    [
        "active-run",
        "active-lease",
        "foreign-owner",
        "source",
        "raw-source",
        "epoch",
        "index-token",
        "unrelated-fact",
        "ack",
        "provider",
    ],
)
def test_only_exact_idle_owned_disproved_fields_can_mint_intent(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    session = intake_session
    batch, members, target, _store, binding = _legacy(
        session, intake_owner, monkeypatch
    )
    if drift == "active-run":
        from onyx.db.models import RegulatoryLabelingRun

        session.get_one(RegulatoryLabelingRun, batch.labeling_run_id).status = "running"
    elif drift == "active-lease":
        batch.lease_token = uuid4()
    elif drift == "foreign-owner":
        from tests.external_dependency_unit.conftest import create_test_user

        batch.created_by = create_test_user(
            session, "contract-foreign", intake_owner[0].role
        ).id
    elif drift == "source":
        target.text += " Değişti."
    elif drift == "raw-source":
        from onyx.db.models import RegulatorySourceAsset

        session.get_one(RegulatorySourceAsset, members[0].source_asset_id).sha256 = (
            "e" * 64
        )
    elif drift == "epoch":
        session.get_one(RegulatoryFilePublication, target.user_file_id).epoch = 1
    elif drift == "index-token":
        members[0].approval_attempt_id = uuid4()
    elif drift == "unrelated-fact":
        value = deepcopy(target.chunk_metadata)
        value[metadata.METADATA_AUDIT_KEY]["fields"]["document_number"] = {
            "source": "llm",
            "stale": False,
            "basis": "explicit",
            "source_chunk_id": target.id,
            "evidence_quote": target.text,
        }
        value["document_number"] = "Invented"
        target.chunk_metadata = value
    elif drift == "ack":
        monkeypatch.setattr(repair, "_read_ack", lambda *_a: b"{}")
    else:
        monkeypatch.setattr(
            repair,
            "resolve_intake_labeling_provider_binding",
            lambda *_a, **_k: binding.model_copy(update={"project": "foreign"}),
        )
    before = deepcopy(batch.metadata_checkpoint)
    session.flush()
    with (
        session.no_autoflush,
        pytest.raises(intake.IntakeConflictError),
    ):
        repair.request_contract_retry(session, batch)
    assert batch.metadata_checkpoint == before


@pytest.mark.parametrize("malformed_ack", [False, True])
def test_owner_http_get_remains_partial_and_normal_retry_mints_exact_contract(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    malformed_ack: bool,
) -> None:
    from onyx.db.engine.sql_engine import get_session
    from onyx.error_handling.exceptions import register_onyx_exception_handlers
    from onyx.server.features.regulatory import document_intake_api as api

    session = intake_session
    batch, _members, _target, _store, _binding = _legacy(
        session, intake_owner, monkeypatch
    )
    batch.status, batch.stage, batch.review_sha256 = "ready", "review", "f" * 64
    session.flush()
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router, prefix="/api/regulatory")
    app.dependency_overrides[api._admin] = lambda: intake_owner[0]
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_current_tenant_id] = lambda: get_current_tenant_id()
    monkeypatch.setattr(api.config, "REGULATORY_ANNEX_UPDATES_ENABLED", True)
    queued = []
    monkeypatch.setattr(api, "_enqueue_batch", lambda _s, b, _t: queued.append(b.id))
    client = TestClient(app)
    before = client.get(f"/api/regulatory/document-intake/batches/{batch.id}")
    assert before.status_code == 200, before.text
    data = before.json()
    assert data["status"] == "partial" and data["review_sha256"] is None
    assert (
        data["edit_review_sha256"] is None and data["publication_review_sha256"] is None
    )
    assert "use Retry to verify the affected metadata" in data["error"]
    old_checkpoint = deepcopy(batch.metadata_checkpoint)
    if malformed_ack:
        monkeypatch.setattr(repair, "_read_ack", lambda *_args: b"{}")
    response = client.post(f"/api/regulatory/document-intake/batches/{batch.id}/retry")
    if malformed_ack:
        assert response.status_code == 409, response.text
        assert batch.metadata_checkpoint == old_checkpoint
        return
    assert response.status_code == 202, response.text
    assert (
        response.json()["stage"] == "metadata"
        and response.json()["metadata_contract_retry"]["target_count"] == 1
    )
    identity = response.json()["metadata_contract_retry"]["intent_sha256"]
    duplicate = client.post(f"/api/regulatory/document-intake/batches/{batch.id}/retry")
    assert (
        duplicate.status_code == 202
        and duplicate.json()["metadata_contract_retry"]["intent_sha256"] == identity
    )
    assert queued == [batch.id, batch.id]
    current = MetadataCheckpoint.model_validate(batch.metadata_checkpoint)
    intent = repair.contract_retry(batch)
    assert intent is not None
    source = deepcopy(current.source_snapshot)
    source["metadata_contract_retry"]["policy"] = "unvetted_waiver"
    batch.metadata_checkpoint = MetadataCheckpoint.build(
        recipe_sha256=current.recipe_sha256,
        provider_binding=current.provider_binding,
        source_snapshot=source,
        request_hashes=current.request_hashes,
        predecessor=intent.previous_checkpoint,
    ).model_dump(mode="json")
    session.flush()
    corrupted = client.get(f"/api/regulatory/document-intake/batches/{batch.id}")
    assert corrupted.status_code == 409, corrupted.text


def test_ordinary_old_recipe_complete_receipts_need_no_contract_or_rewrite(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = intake_session
    batch, members, target, _store, _binding = _legacy(
        session, intake_owner, monkeypatch
    )
    value = deepcopy(target.chunk_metadata)
    value[metadata.METADATA_AUDIT_KEY]["fields"]["clause_label"] = {
        "source": "manual",
        "basis": "user_review",
        "source_chunk_id": target.id,
        "evidence_quote": None,
        "stale": False,
    }
    target.chunk_metadata = value
    session.flush()
    before = deepcopy(batch.metadata_checkpoint)
    assert repair.request_contract_retry(session, batch) is False
    metadata_db.validate_intake_metadata_receipts(session, batch)
    assert batch.metadata_checkpoint == before
    batch.status, batch.stage, batch.review_sha256 = "completed", "completed", "f" * 64
    session.flush()
    # Completed history is retained rather than routed through metadata checkpoint recovery.
    intake.refresh_intake(session, batch)
    assert batch.status == "completed" and batch.metadata_checkpoint == before


def test_worker_uses_literal_frozen_request_after_pending_hint_transition(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager
    from typing import cast

    from onyx.regulatory.document_intake import metadata_job
    from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
    from tests.unit.onyx.regulatory.test_intake_metadata_budget import _Gateway

    session = intake_session
    batch, members, target, store, binding = _legacy(session, intake_owner, monkeypatch)
    assert repair.request_contract_retry(session, batch)
    intent, pending = repair.require_current_contract(
        session, batch, require_complete=False
    )
    member_id, target_id = pending[0]
    document = intent.documents[str(member_id)]
    member = next(row for row in members if row.id == member_id)
    batch.lease_token = token = uuid4()
    batch.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    session.flush()
    expected = build_contract_retry_request(intent.id, document.chunks, target_id)

    def respond(request: VertexBatchRequest) -> dict[str, Any]:
        assert request == expected
        payload = json.loads(request.prompt)
        own = next(
            c for c in payload["complete_document"] if c["chunk_id"] == target_id
        )
        assert own["structural_metadata"].get("clause_label") is None
        evidence = next(
            e
            for e in payload["evidence_catalog"]
            if e["source_chunk_id"] == target_id
            and e["scope"] == "body"
            and e["text"] == target.text
        )
        response = json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": target_id,
                        "fields": [
                            {
                                "field": "clause_label",
                                "value": "b",
                                "basis": "explicit",
                                "evidence_id": evidence["evidence_id"],
                                "reference_evidence_id": "",
                            }
                        ],
                    }
                ],
                "warnings": [],
            }
        )
        _save_ack(session, store, batch, member, binding, request, response)
        return _envelope(request, "STOP", response)

    gateway = _Gateway(respond)

    @contextmanager
    def sessions() -> Iterator[Session]:
        yield session

    monkeypatch.setattr(metadata_job, "get_session_with_current_tenant", sessions)
    monkeypatch.setattr(
        metadata_job,
        "resolve_intake_labeling_provider_binding",
        lambda *_a, **_k: binding,
    )
    monkeypatch.setattr(
        metadata_job,
        "resolve_labeling_gateway",
        lambda *_a, **_k: cast(VertexBatchGateway, gateway),
    )
    metadata_job.run_intake_metadata_locked(
        intake_id=batch.id, lease_token=token, tenant_id=get_current_tenant_id()
    )
    assert gateway.requests == [expected]
    assert (
        repair.require_current_contract(
            session, batch, require_complete=True, verify_ack_bytes=True
        )[1]
        == []
    )
    assert (
        target.chunk_metadata[metadata.METADATA_AUDIT_KEY]["fields"]["clause_label"][
            "request_hash"
        ]
        == expected.request_hash
    )


@pytest.mark.parametrize("drift", ["frozen-hints", "pending-audit"])
def test_worker_never_rebuilds_frozen_hints_or_accepts_changed_pending_audit(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager
    from unittest.mock import MagicMock

    from onyx.regulatory.document_intake import metadata_job
    from onyx.regulatory.document_intake.metadata_contract_retry import (
        MetadataContractRetry,
    )

    session = intake_session
    batch, _members, target, _store, _binding = _legacy(
        session, intake_owner, monkeypatch
    )
    assert repair.request_contract_retry(session, batch)
    intent, pending = repair.require_current_contract(
        session, batch, require_complete=False
    )
    member_id, target_id = pending[0]
    if drift == "frozen-hints":
        raw = intent.model_dump(mode="json")
        doc = raw["documents"][str(member_id)]
        source = next(row for row in doc["chunks"] if row["chunk_id"] == target_id)
        source["structural_metadata"]["article_title"] = "Undeclared hint change"
        chunks = [metadata.MetadataChunk.model_validate(row) for row in doc["chunks"]]
        doc["request_hashes"] = {
            target: build_contract_retry_request(intent.id, chunks, target).request_hash
            for target in doc["targets"]
        }
        changed = MetadataContractRetry.model_validate(raw)
        batch.metadata_checkpoint = MetadataCheckpoint.build(
            recipe_sha256=changed.recipe_sha256,
            provider_binding=changed.previous_checkpoint.provider_binding,
            source_snapshot={
                **changed.previous_checkpoint.source_snapshot,
                "metadata_contract_retry": changed.model_dump(mode="json"),
            },
            request_hashes=changed.previous_checkpoint.request_hashes,
            predecessor=changed.previous_checkpoint,
        ).model_dump(mode="json")
    else:
        value = deepcopy(target.chunk_metadata)
        value[metadata.METADATA_AUDIT_KEY]["fields"]["clause_label"]["stale"] = False
        target.chunk_metadata = value
    batch.lease_token = token = uuid4()
    batch.lease_expires_at = datetime.now(timezone.utc) + timedelta(minutes=10)
    session.flush()

    @contextmanager
    def sessions() -> Iterator[Session]:
        yield session

    resolver = MagicMock()
    monkeypatch.setattr(metadata_job, "get_session_with_current_tenant", sessions)
    monkeypatch.setattr(metadata_job, "resolve_labeling_gateway", resolver)
    with pytest.raises(
        intake.IntakeConflictError, match="document identity|pending field"
    ):
        metadata_job.run_intake_metadata_locked(
            intake_id=batch.id, lease_token=token, tenant_id=get_current_tenant_id()
        )
    resolver.assert_not_called()
