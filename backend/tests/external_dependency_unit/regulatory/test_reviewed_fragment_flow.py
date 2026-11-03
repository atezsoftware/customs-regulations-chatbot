"""Actual PostgreSQL preserves reviewed fragments and their genuine base ACK."""

import json
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from typing import Any, NamedTuple, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_duplicate_evidence as evidence_db
from onyx.db import regulatory_labeling_reviewed_fragment as reviewed
from onyx.db.enums import UserFileProjectionRepairStatus, UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
    RegulatoryLabelTaxonomy,
    UserFile,
    UserFileProjectionRepair,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_readers import complete_frozen_sources
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import FileStore
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchGateway,
    VertexBatchRequest,
)
from onyx.regulatory.labeling.accepted_result import SourceOnlyReviewedFragment
from onyx.regulatory.labeling.intake_abstention_repair import ABSTENTION_REPAIR_KEY
from onyx.regulatory.labeling.intake_duplicate_label_repair import DUPLICATE_REPAIR_KEY
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.reviewed_fragment import REVIEWED_FRAGMENT_KEY
from onyx.regulatory.labeling.self_repeal import SOURCE_ONLY_KEY
from tests.external_dependency_unit.regulatory.test_intake_semantic_repair_audit import (
    seed_v4_owned_intake_source,
)
from tests.external_dependency_unit.regulatory.test_intake_source_only_self_repeal_flow import (
    _end,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)
from tests.unit.onyx.regulatory.labeling import test_reviewed_figure_fragment as figure
from tests.unit.onyx.regulatory.labeling.test_reviewed_fragment import (
    ABSTAINED,
    RAW,
    sources,
)


class Store:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}

    def read_file(self, file_id: str) -> BytesIO:
        if file_id not in self.blobs:
            raise OSError("Missing local fixture blob")
        return BytesIO(self.blobs[file_id])


class Fixture(NamedTuple):
    data: LabelingData
    run: RegulatoryLabelingRun
    items: list[RegulatoryLabelingItem]
    target: RegulatoryLabelingItem
    store: Store
    taxonomy: TaxonomyDefinition
    binding: LabelingProviderBinding
    raw_id: str
    line: str
    semantic_line: str | None = None


def create(session: Session, fixture: Fixture) -> RegulatoryLabelingRun:
    return labeling.create_labeling_run(
        session,
        document_set_id=fixture.data.document_set_id,
        taxonomy=session.get_one(
            RegulatoryLabelTaxonomy,
            fixture.run.taxonomy_id,
        ),
        model_configuration_id=fixture.data.model_id,
        model=fixture.binding.model,
        provider_binding=fixture.binding.model_dump(mode="json"),
        requested_by_id=fixture.data.user_id,
        idempotency_key=uuid4(),
        target_file_ids=[fixture.data.file_id],
    )[0]


def seed(
    session: Session,
    data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: str = "blank",
    semantic: bool = True,
) -> Fixture:
    source_raw = RAW if profile == "blank" else figure.RAW
    file = session.get_one(UserFile, data.file_id)
    file.file_type, file.status = "application/pdf", UserFileStatus.CHUNKED
    session.add(
        FileRecord(
            file_id=file.file_id,
            file_origin=FileOrigin.OTHER,
            file_type="application/pdf",
            file_metadata={"sha256": sha256(source_raw).hexdigest()},
            bucket_name="local-only",
            object_key=file.file_id,
        )
    )
    seed_v4_owned_intake_source(
        session,
        data,
        raw_sha256=sha256(source_raw).hexdigest(),
        raw_byte_count=len(source_raw),
    )
    for identifier in (
        data.first_id,
        data.second_id,
        data.derived_id,
        data.incomplete_id,
    ):
        session.delete(session.get_one(RegulatoryChunk, identifier))
    session.flush()
    original = sources() if profile == "blank" else figure.sources()
    ids = {s.chunk_id: f"reviewed-{uuid4().hex}" for s in original}
    for source in original:
        metadata = deepcopy(source.snapshot["chunk_metadata"])
        for fact in metadata["intake_legal_metadata"]["fields"].values():
            if fact.get("source_chunk_id") in ids:
                fact["source_chunk_id"] = ids[fact["source_chunk_id"]]
        session.add(
            RegulatoryChunk(
                id=ids[source.chunk_id],
                user_file_id=data.file_id,
                text=source.text,
                position=source.snapshot["position"],
                projection_ordinal=source.snapshot["position"],
                chunk_type=source.snapshot["chunk_type"],
                heading_path=source.snapshot["heading_path"],
                chunk_metadata=metadata,
                status="active",
                source="indexed",
            )
        )
    taxonomy = TaxonomyDefinition.model_validate(
        {
            "name": "Reviewed form",
            "labels": [
                {"id": "form", "name": "Form", "description": "An application form."}
            ],
        }
    )
    taxonomy_row = labeling.create_taxonomy(
        session, taxonomy=taxonomy, created_by_id=data.user_id
    )
    binding = LabelingProviderBinding(
        model_configuration_id=data.model_id,
        provider_id=data.provider_id,
        project="local-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="workload_identity",
        model="gemini-3.8-flash",
        transport="vertex_direct_v1",
    )
    session.flush()
    run = labeling.create_labeling_run(
        session,
        document_set_id=data.document_set_id,
        taxonomy=taxonomy_row,
        model_configuration_id=data.model_id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=data.user_id,
        idempotency_key=uuid4(),
        target_file_ids=[data.file_id],
    )[0]
    batch = session.scalar(
        select(RegulatoryDocumentIntake).where(
            RegulatoryDocumentIntake.created_by == data.user_id
        )
    )
    assert batch is not None
    batch.status, batch.stage, batch.labeling_run_id = "preparing", "labeling", run.id
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=600
    )
    assert lease is not None
    items = labeling.prepare_next_item_page(session, lease, limit=128)
    for item in items:
        request = build_labeling_request(
            chunk_id=item.regulatory_chunk_id,
            text=item.text_snapshot,
            context=item.context_snapshot or "",
            taxonomy=taxonomy,
            source_hash=item.canonical_text_sha256 or "",
        )
        item.request_hash, item.request_payload = (
            request.request_hash,
            request.model_dump(mode="json", exclude_computed_fields=True),
        )
    target = next(
        i
        for i in items
        if i.source_snapshot["position"] == (4 if profile == "blank" else 45)
    )
    request = VertexBatchRequest.model_validate(target.request_payload)
    submission = "regulatory-labeling-" + sha256(str(run.id).encode()).hexdigest()
    shard = RegulatoryLabelingShard(
        run_id=run.id, ordinal=0, item_ids=[str(target.id)], submission_key=submission
    )
    session.add(shard)
    session.flush()
    target.shard_id = shard.id
    line = json.dumps(
        {
            "key": request.request_hash,
            "request": request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": ABSTAINED}]},
                    }
                ]
            },
        }
    )
    ack_id = f"regulatory-labeling-direct/{binding.fingerprint}/{submission.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
    raw = json.dumps(
        {
            "binding_sha256": binding.fingerprint,
            "submission_key": submission,
            "request_hash": request.request_hash,
            "request_payload": request.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "line": line,
            "line_sha256": sha256(line.encode()).hexdigest(),
        }
    ).encode()
    store = Store()
    store.blobs[file.file_id], store.blobs[ack_id] = source_raw, raw
    session.add(
        FileRecord(
            file_id=ack_id,
            file_origin=FileOrigin.OTHER,
            file_type="application/json",
            file_metadata={
                "sha256": sha256(raw).hexdigest(),
                "binding_sha256": binding.fingerprint,
            },
            bucket_name="local-only",
            object_key=ack_id,
        )
    )
    monkeypatch.setattr(
        reviewed, "get_default_file_store", lambda: cast(FileStore, store)
    )
    monkeypatch.setattr(
        evidence_db, "get_default_file_store", lambda: cast(FileStore, store)
    )
    session.flush()
    fixture = Fixture(
        data, run, items, target, store, taxonomy, binding, file.file_id, line
    )
    if profile == "figure" and semantic:
        fixture = fixture._replace(semantic_line=cache_semantic(session, fixture))
    return fixture


def cache_semantic(session: Session, fixture: Fixture) -> str:
    repair, submission = reviewed._semantic_repair(
        request=VertexBatchRequest.model_validate(fixture.target.request_payload),
        response=ABSTAINED,
        item=fixture.target,
        run=fixture.run,
        taxonomy=fixture.taxonomy,
    )
    line = json.dumps(
        {
            "key": repair.request.request_hash,
            "request": repair.request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": ABSTAINED}]},
                    }
                ]
            },
        }
    )
    file_id = f"regulatory-labeling-direct/{fixture.binding.fingerprint}/{submission.removeprefix('regulatory-labeling-')}/{repair.request.request_hash}.json"
    raw = json.dumps(
        {
            "binding_sha256": fixture.binding.fingerprint,
            "submission_key": submission,
            "request_hash": repair.request.request_hash,
            "request_payload": repair.request.model_dump(
                mode="json", exclude_computed_fields=True
            ),
            "line": line,
            "line_sha256": sha256(line.encode()).hexdigest(),
        }
    ).encode()
    fixture.store.blobs[file_id] = raw
    session.add(
        FileRecord(
            file_id=file_id,
            file_origin=FileOrigin.OTHER,
            file_type="application/json",
            file_metadata={
                "sha256": sha256(raw).hexdigest(),
                "binding_sha256": fixture.binding.fingerprint,
            },
            bucket_name="local-only",
            object_key=file_id,
        )
    )
    session.flush()
    return line


def apply(session: Session, fixture: Fixture) -> None:
    receipt = reviewed.build_source_only_reviewed_fragment_receipt(
        session, item=fixture.target, run=fixture.run, original_line=fixture.line
    )
    labeling._apply_frozen_window_results(
        session,
        fixture.run,
        items=fixture.items,
        file_id=fixture.data.file_id,
        lower=0,
        upper=200,
        outcomes={
            i.request_hash or "": SourceOnlyReviewedFragment(receipt)
            if i is fixture.target
            else ([], [])
            for i in fixture.items
        },
    )
    session.flush()


@pytest.mark.parametrize("profile", ["blank", "figure"])
def test_real_apply_read_search_review_api_and_reuse_preserve_all_sources(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch, profile=profile)
        before = publication_digest(
            [
                r.model_dump(mode="json")
                for r in capture_canonical_scope(session, fixture.data.file_id)
            ]
        )
        apply(session, fixture)
        target = fixture.target
        assert (
            target.status == "completed" and target.labels == target.assignments == []
        )
        assert labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=[target]
        ) == {target.regulatory_chunk_id}
        assert target.regulatory_chunk_id in complete_frozen_sources(
            session, fixture.run.id, [target], [target.regulatory_chunk_id]
        )
        authority = labeling_review_authority(session, fixture.run.id)
        assert authority and authority["reviewed_fragment_dispositions"]
        page = labeling.list_items(
            session,
            document_set_id=fixture.data.document_set_id,
            run_id=fixture.run.id,
            offset=0,
            limit=20,
        )
        assert page is not None
        view = next(i for i in page.items if i.chunk_id == target.regulatory_chunk_id)
        assert (
            view.classification_disposition == "source_only_reviewed_fragment"
            and view.provider_abstained
        )
        _end(fixture.run, "completed")
        session.flush()
        replacement = create(session, fixture)
        assert editing.reuse_completed_intake_labels(
            session, previous=fixture.run, replacement=replacement
        ) == (9 if profile == "blank" else 27)
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id
                == target.regulatory_chunk_id,
            )
        )
        assert copied is not None
        reviewed.validate_item_reviewed_fragment_receipt(
            session, item=copied, run=replacement
        )
        assert before == publication_digest(
            [
                r.model_dump(mode="json")
                for r in capture_canonical_scope(session, fixture.data.file_id)
            ]
        )


@pytest.mark.parametrize(
    "change",
    [
        "raw",
        "neighbor",
        "context",
        "request",
        "provider",
        "labels",
        "receipt",
        "collision",
        "duplicate",
        "ack_header",
        "source",
        "semantic",
        "retained_mime",
        "user_mime",
    ],
)
@pytest.mark.parametrize("profile", ["blank", "figure"])
def test_drift_rejects_read_search_review(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
    profile: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch, profile=profile)
        apply(session, fixture)
        target = fixture.target
        if change == "raw":
            fixture.store.blobs[fixture.raw_id] += b"changed"
        elif change == "retained_mime":
            session.get_one(FileRecord, fixture.raw_id).file_type = "text/html"
        elif change == "user_mime":
            session.get_one(UserFile, fixture.data.file_id).file_type = "text/html"
        elif change == "neighbor":
            session.get_one(
                RegulatoryChunk, fixture.items[-1].regulatory_chunk_id
            ).text += "changed"
        elif change == "context":
            target.context_snapshot = (target.context_snapshot or "") + "changed"
        elif change == "request":
            target.request_payload = {
                **(target.request_payload or {}),
                "prompt": "changed",
            }
        elif change == "provider":
            fixture.run.provider_binding = {
                **fixture.run.provider_binding,
                "model": "different",
            }
        elif change == "labels":
            target.labels = ["form"]
        elif change == "source":
            fixture.run.document_set_id = fixture.data.other_document_set_id
        elif change == "ack_header":
            receipt = cast(
                dict[str, Any], target.source_snapshot[REVIEWED_FRAGMENT_KEY]
            )
            record = session.get_one(
                FileRecord, receipt["base_abstention"]["ack"]["file_id"]
            )
            record.file_metadata = {
                **cast(dict[str, Any], record.file_metadata),
                "sha256": "0" * 64,
            }
        else:
            snapshot = cast(dict[str, Any], deepcopy(target.source_snapshot))
            if change == "receipt":
                snapshot[REVIEWED_FRAGMENT_KEY]["proof"]["transcription_sha256"] = (
                    "0" * 64
                )
            elif change == "collision":
                snapshot[SOURCE_ONLY_KEY] = {}
            elif change == "duplicate":
                snapshot[DUPLICATE_REPAIR_KEY] = {}
            else:
                snapshot[ABSTENTION_REPAIR_KEY] = {}
            target.source_snapshot = snapshot
        session.flush()
        with pytest.raises(ValueError):
            reviewed.validate_item_reviewed_fragment_receipt(
                session, item=target, run=fixture.run
            )
        assert not labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=[target]
        )
        with pytest.raises(ValueError):
            labeling_review_authority(session, fixture.run.id)


@pytest.mark.parametrize("ack_present", [True, False])
@pytest.mark.parametrize("profile", ["blank", "figure"])
def test_normal_retry_recovery_requires_actual_base_ack_and_preserves_failed_history(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    ack_present: bool,
    profile: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch, profile=profile)
        target = fixture.target
        employee = (
            next((i for i in fixture.items if i.source_snapshot["position"] == 7), None)
            if profile == "blank"
            else None
        )
        employee_before = None
        if employee is not None:
            employee_assignments: list[dict[str, object]] = [
                {"label_id": "form", "evidence_quote": "1"}
            ]
            employee.status, employee.labels, employee.assignments = (
                "completed",
                ["form"],
                employee_assignments,
            )
            employee_before = deepcopy(
                (employee.labels, employee.assignments, employee.source_snapshot)
            )
        target.status, target.error = (
            "failed",
            "arbitrary text is not admission authority",
        )
        history = reviewed._history(target)
        _end(fixture.run)
        session.flush()
        if not ack_present:
            fixture.store.blobs = {fixture.raw_id: RAW}
        replacement = create(session, fixture)
        count = editing.reuse_completed_intake_labels(
            session, previous=fixture.run, replacement=replacement
        )
        assert count == int(ack_present) + int(employee is not None)
        if employee is not None:
            assert (
                employee.labels,
                employee.assignments,
                employee.source_snapshot,
            ) == employee_before
        assert target.status == "failed" and reviewed._history(target) == history
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id
                == target.regulatory_chunk_id,
            )
        )
        assert copied is not None
        assert copied.status == ("completed" if ack_present else "pending")
        if ack_present:
            reviewed.validate_item_reviewed_fragment_receipt(
                session, item=copied, run=replacement
            )
            target.error = "changed history"
            with pytest.raises(ValueError, match="history"):
                reviewed.validate_item_reviewed_fragment_receipt(
                    session, item=copied, run=replacement
                )


@pytest.mark.parametrize("profile", ["blank", "figure"])
def test_real_worker_uses_base_ack_without_another_provider_request(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, profile: str
) -> None:
    from onyx.regulatory.labeling import orchestrator

    with labeling_data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                fixture = seed(session, labeling_data, monkeypatch, profile=profile)
                batch = session.scalar(
                    select(RegulatoryDocumentIntake).where(
                        RegulatoryDocumentIntake.created_by == fixture.data.user_id
                    )
                )
                member = session.scalar(
                    select(RegulatoryDocumentIntakeFile).where(
                        RegulatoryDocumentIntakeFile.user_file_id
                        == fixture.data.file_id
                    )
                )
                assert batch is not None and member is not None
                batch.status, batch.stage, batch.labeling_run_id = (
                    "preparing",
                    "labeling",
                    fixture.run.id,
                )
                member.status = "labeling"
                shard = session.get_one(
                    RegulatoryLabelingShard, fixture.target.shard_id
                )
                shard.status, fixture.target.status = "submitted", "submitted"
                assert fixture.run.lease_token is not None
                lease = labeling.RunLease(
                    fixture.run.id,
                    fixture.run.lease_generation,
                    fixture.run.lease_token,
                )
                shard_id, target_id = shard.id, fixture.target.id
                session.commit()

            @contextmanager
            def local_session():
                with Session(
                    connection, join_transaction_mode="create_savepoint"
                ) as current:
                    yield current

            monkeypatch.setattr(
                orchestrator, "get_session_with_current_tenant", local_session
            )
            monkeypatch.setattr(
                orchestrator, "_lease_heartbeat", lambda _lease: lambda: None
            )
            gateway = Mock(spec=VertexBatchGateway)
            if profile == "blank":
                gateway.read_results.return_value = iter([fixture.line])
            else:
                from onyx.regulatory.indexing_jobs.vertex_batch import (
                    VertexBatchJobStatus,
                    VertexBatchState,
                )

                gateway.read_results.side_effect = [
                    iter([fixture.line]),
                    iter([fixture.semantic_line]),
                ]
                gateway.submit.return_value = VertexBatchState(
                    remote_job_name="fixture",
                    status=VertexBatchJobStatus.SUCCEEDED,
                    output_uri="fixture://sole-semantic",
                )
            orchestrator._read_and_apply_results(
                lease, shard_id, gateway, "fixture://base-only"
            )
            if profile == "blank":
                gateway.submit.assert_not_called()
            else:
                gateway.submit.assert_called_once()
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                target = session.get_one(RegulatoryLabelingItem, target_id)
                assert target.status == "completed", target.error
                assert REVIEWED_FRAGMENT_KEY in target.source_snapshot
                assert target.labels == target.assignments == []
        finally:
            transaction.rollback()


@pytest.mark.parametrize(
    "change",
    [
        "base_only",
        "semantic_positive",
        "semantic_key",
        "semantic_error",
        "semantic_header",
        "policy",
        "attempt",
        "audit",
    ],
)
def test_figure_needs_the_actual_same_item_single_continued_empty_attempt(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(
            session,
            labeling_data,
            monkeypatch,
            profile="figure",
            semantic=change != "base_only",
        )
        target = fixture.target
        if change in {
            "base_only",
            "semantic_positive",
            "semantic_key",
            "semantic_error",
        }:
            if change != "base_only":
                file_id = next(
                    k
                    for k in fixture.store.blobs
                    if k != fixture.raw_id
                    and json.loads(fixture.store.blobs[k])["line"] != fixture.line
                )
                blob = json.loads(fixture.store.blobs[file_id])
                line = json.loads(blob["line"])
                if change == "semantic_positive":
                    line["response"]["candidates"][0]["content"]["parts"][0]["text"] = (
                        json.dumps(
                            {
                                "labels": [
                                    {"label_id": "form", "evidence_quote": "48 mm"}
                                ],
                                "abstained": False,
                            }
                        )
                    )
                elif change == "semantic_key":
                    line["key"] = "0" * 64
                else:
                    line["error"] = {"message": "failed"}
                blob["line"] = json.dumps(line)
                blob["line_sha256"] = sha256(blob["line"].encode()).hexdigest()
                raw = json.dumps(blob).encode()
                fixture.store.blobs[file_id] = raw
                record = session.get_one(FileRecord, file_id)
                record.file_metadata = {
                    **cast(dict[str, Any], record.file_metadata),
                    "sha256": sha256(raw).hexdigest(),
                }
                session.flush()
            with pytest.raises(ValueError):
                reviewed.build_source_only_reviewed_fragment_receipt(
                    session, item=target, run=fixture.run, original_line=fixture.line
                )
            target.status, target.error = "failed", "not authority"
            _end(fixture.run)
            session.flush()
            replacement = create(session, fixture)
            assert (
                editing.reuse_completed_intake_labels(
                    session, previous=fixture.run, replacement=replacement
                )
                == 0
            )
            return
        apply(session, fixture)
        snapshot = cast(dict[str, Any], deepcopy(target.source_snapshot))
        evidence = snapshot[REVIEWED_FRAGMENT_KEY]["base_abstention"]
        if change == "semantic_header":
            record = session.get_one(FileRecord, evidence["semantic_ack"]["file_id"])
            record.file_metadata = {
                **cast(dict[str, Any], record.file_metadata),
                "sha256": "0" * 64,
            }
        elif change == "policy":
            evidence["policy"] = "genuine_base_abstention"
            evidence.pop("semantic_ack")
            evidence.pop("semantic_audit")
        elif change == "attempt":
            evidence["semantic_ack"] = deepcopy(evidence["ack"])
        else:
            evidence["semantic_audit"]["output_response"] = "changed"
        target.source_snapshot = snapshot
        session.flush()
        with pytest.raises(ValueError):
            reviewed.validate_item_reviewed_fragment_receipt(
                session, item=target, run=fixture.run
            )
        assert not labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=[target]
        )
        with pytest.raises(ValueError):
            labeling_review_authority(session, fixture.run.id)


@pytest.mark.parametrize("profile", ["blank", "figure"])
def test_normal_readers_keep_valid_receipt_after_successful_publication(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch, profile=profile)
        apply(session, fixture)
        _end(fixture.run, "completed")
        member = session.scalar(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.user_file_id == fixture.data.file_id
            )
        )
        assert member is not None
        batch = session.get_one(RegulatoryDocumentIntake, member.intake_id)
        attempt = uuid4()
        member.approval_attempt_id, member.status = attempt, "completed"
        batch.status, batch.stage = "completed", "completed"
        session.get_one(
            UserFile, fixture.data.file_id
        ).status = UserFileStatus.COMPLETED
        session.get_one(RegulatoryFilePublication, fixture.data.file_id).epoch = 1
        session.add(
            UserFileProjectionRepair(
                user_file_id=fixture.data.file_id,
                attempt_id=attempt,
                status=UserFileProjectionRepairStatus.SUCCEEDED,
                updated_at=datetime.now(timezone.utc),
                initial_index=True,
            )
        )
        session.flush()
        reviewed.validate_item_reviewed_fragment_receipt(
            session, item=fixture.target, run=fixture.run
        )
        assert labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=[fixture.target]
        ) == {fixture.target.regulatory_chunk_id}
        assert fixture.target.regulatory_chunk_id in complete_frozen_sources(
            session,
            fixture.run.id,
            [fixture.target],
            [fixture.target.regulatory_chunk_id],
        )
        assert labeling_review_authority(session, fixture.run.id)
        replacement = create(session, fixture)
        assert editing.reuse_completed_intake_labels(
            session, previous=fixture.run, replacement=replacement
        ) == (9 if profile == "blank" else 27)
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id
                == fixture.target.regulatory_chunk_id,
            )
        )
        assert copied is not None
        reviewed.validate_item_reviewed_fragment_receipt(
            session, item=copied, run=replacement
        )
        assert labeling.current_labeling_items_for_search(
            session, run=replacement, items=[copied]
        ) == {copied.regulatory_chunk_id}
        with pytest.raises(ValueError, match="unpublished"):
            reviewed.build_source_only_reviewed_fragment_receipt(
                session,
                item=fixture.target,
                run=fixture.run,
                original_line=fixture.line,
            )


@pytest.mark.parametrize("profile", ["blank", "figure"])
@pytest.mark.parametrize("change", ["epoch", "approval", "projection", "active_lease"])
def test_failed_recovery_cannot_create_a_receipt_after_publication_or_with_active_history(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    profile: str,
    change: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch, profile=profile)
        fixture.target.status, fixture.target.error = "failed", "not authority"
        _end(fixture.run)
        if change == "epoch":
            session.get_one(RegulatoryFilePublication, fixture.data.file_id).epoch = 1
        elif change == "approval":
            member = session.scalar(
                select(RegulatoryDocumentIntakeFile).where(
                    RegulatoryDocumentIntakeFile.user_file_id == fixture.data.file_id
                )
            )
            assert member is not None
            member.approval_attempt_id = uuid4()
        elif change == "projection":
            session.add(
                UserFileProjectionRepair(
                    user_file_id=fixture.data.file_id,
                    attempt_id=uuid4(),
                    status=UserFileProjectionRepairStatus.PENDING,
                    updated_at=datetime.now(timezone.utc),
                    initial_index=True,
                )
            )
        session.flush()
        replacement = create(session, fixture)
        if change == "active_lease":
            fixture.run.lease_token = uuid4()
            fixture.run.lease_expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=5
            )
        session.flush()
        assert (
            reviewed.recover_failed_reviewed_fragment_items(
                session, previous=fixture.run, replacement=replacement
            )
            == 0
        )
        assert fixture.target.status == "failed"
