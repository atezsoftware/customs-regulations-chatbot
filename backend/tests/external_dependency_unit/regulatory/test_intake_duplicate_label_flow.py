"""Isolated PostgreSQL exercises duplicate receipts across real runtime consumers."""

import datetime
import json
from collections.abc import Mapping
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_duplicate_evidence as evidence_db
from onyx.db.enums import UserFileStatus
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    FileRecord,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_duplicate_repair import (
    require_current_duplicate_source,
    validate_item_duplicate_repair_audit,
)
from onyx.db.regulatory_labeling_readers import complete_frozen_sources
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.indexing_jobs.models import VertexAuthenticationMode
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.accepted_result import AcceptedDuplicateLabelRepair
from onyx.regulatory.labeling.duplicate_ack_evidence import DuplicateAckEvidence
from onyx.regulatory.labeling.intake_duplicate_label_repair import (
    DUPLICATE_REPAIR_KEY,
    DuplicateLabelRepairAudit,
    build_duplicate_label_repair,
    build_duplicate_repair_audit,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from tests.external_dependency_unit.regulatory.test_intake_semantic_repair_audit import (
    seed_v4_owned_intake_source,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)
from tests.unit.onyx.regulatory.labeling.test_intake_duplicate_label_repair import (
    ack_line,
    selections,
)


class LocalAckStore:
    def __init__(self) -> None:
        self.blobs: dict[str, bytes] = {}
        self.reads = 0

    def read_file(self, file_id: str) -> BytesIO:
        self.reads += 1
        return BytesIO(self.blobs[file_id])


class Fixture(NamedTuple):
    data: LabelingData
    run: RegulatoryLabelingRun
    items: list[RegulatoryLabelingItem]
    target: RegulatoryLabelingItem
    store: LocalAckStore
    taxonomy: TaxonomyDefinition
    frozen: dict[str, Any]
    binding: LabelingProviderBinding
    asset_id: UUID


def _create(
    session: Session,
    data: LabelingData,
    taxonomy: TaxonomyDefinition,
    binding: LabelingProviderBinding,
) -> RegulatoryLabelingRun:
    taxonomy_row = labeling.create_taxonomy(
        session, taxonomy=taxonomy, created_by_id=data.user_id
    )
    run, _ = labeling.create_labeling_run(
        session,
        document_set_id=data.document_set_id,
        taxonomy=taxonomy_row,
        model_configuration_id=data.model_id,
        model=binding.model,
        provider_binding=binding.model_dump(mode="json"),
        requested_by_id=data.user_id,
        idempotency_key=uuid4(),
        target_file_ids=[data.file_id],
    )
    return run


def seed(
    session: Session, data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> Fixture:
    home = (
        Path(__file__).resolve().parents[2]
        / "unit/onyx/regulatory/labeling/fixtures/legacy_duplicate_label_context.json"
    )
    frozen = json.loads(home.read_text())
    file = session.get_one(UserFile, data.file_id)
    file.file_type = "application/pdf"
    file.status = UserFileStatus.CHUNKED
    asset_id = seed_v4_owned_intake_source(
        session,
        data,
        raw_sha256=frozen["source_pdf_sha256"],
        raw_byte_count=frozen["source_pdf_bytes"],
    )
    for identifier in (
        data.first_id,
        data.second_id,
        data.derived_id,
        data.incomplete_id,
    ):
        session.delete(session.get_one(RegulatoryChunk, identifier))
    session.flush()
    for position, identifier, text in [
        (0, data.first_id, "MAKİNE EMNİYETİ YÖNETMELİĞİ"),
        (1, data.second_id, frozen["target_text"]),
        (2, data.derived_id, "Kaynağın son sınırı."),
    ]:
        headings = (
            ["MAKİNE EMNİYETİ YÖNETMELİĞİ"]
            if position != 1
            else ["MAKİNE EMNİYETİ YÖNETMELİĞİ", "MADDE 11", "(10)"]
        )
        session.add(
            RegulatoryChunk(
                id=identifier,
                user_file_id=data.file_id,
                text=text,
                position=position,
                projection_ordinal=position,
                heading_path=headings,
                chunk_type="paragraph" if position == 1 else "free_text",
                chunk_metadata={
                    "chunk_variant": "atomic",
                    "intake_chunking_version": "2",
                    "heading_path": headings,
                    "intake_legal_metadata": {"input_stale": False, "fields": {}},
                },
                status="active",
                source="indexed",
            )
        )
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
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
    run = _create(session, data, taxonomy, binding)
    batch = session.scalar(
        select(RegulatoryDocumentIntake).where(
            RegulatoryDocumentIntake.created_by == data.user_id
        )
    )
    member = session.scalar(
        select(RegulatoryDocumentIntakeFile).where(
            RegulatoryDocumentIntakeFile.user_file_id == data.file_id
        )
    )
    assert batch is not None and member is not None
    batch.status, batch.stage, batch.labeling_run_id = "preparing", "labeling", run.id
    member.status = "labeling"
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
        item.request_hash = request.request_hash
        item.request_payload = request.model_dump(
            mode="json", exclude_computed_fields=True
        )
    target = next(item for item in items if item.regulatory_chunk_id == data.second_id)
    assert (
        target.request_hash
        != VertexBatchRequest.model_validate(frozen["base_request"]).request_hash
    )
    store = LocalAckStore()
    monkeypatch.setattr(
        evidence_db,
        "read_direct_ack",
        lambda _session, file_id, *, max_bytes: store.read_file(file_id).read(
            max_bytes + 1
        ),
    )
    session.flush()
    return Fixture(data, run, items, target, store, taxonomy, frozen, binding, asset_id)


def cached_ack(
    session: Session,
    fixture: Fixture,
    request: VertexBatchRequest,
    response: str,
    *,
    ordinal: int,
) -> DuplicateAckEvidence:
    submission = (
        "regulatory-labeling-"
        + sha256(f"local-only-{fixture.run.id}-{ordinal}".encode()).hexdigest()
    )
    line = ack_line(request, response)
    file_id = f"regulatory-labeling-direct/{fixture.binding.fingerprint}/{submission.removeprefix('regulatory-labeling-')}/{request.request_hash}.json"
    raw = json.dumps(
        {
            "binding_sha256": fixture.binding.fingerprint,
            "submission_key": submission,
            "request_hash": request.request_hash,
            "request_payload": request.model_dump(
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
    return evidence_db.load_duplicate_ack(
        session,
        binding_fingerprint=fixture.binding.fingerprint,
        submission_key=submission,
        request=request,
        expected_line=line,
    )


def accepted(session: Session, fixture: Fixture) -> AcceptedDuplicateLabelRepair:
    target = fixture.target
    repair = build_duplicate_label_repair(
        base_request=VertexBatchRequest.model_validate(target.request_payload),
        rejected_response=fixture.frozen["base_response"],
        text=target.text_snapshot,
        chunk_id=target.regulatory_chunk_id,
        source_hash=target.canonical_text_sha256 or "",
        taxonomy=fixture.taxonomy,
        provider_binding=fixture.run.provider_binding,
    )
    response = selections(repair)
    audit = build_duplicate_repair_audit(
        repair=repair,
        selection_response=response,
        base_ack=cached_ack(
            session, fixture, repair.base_request, repair.rejected_response, ordinal=0
        ),
        repair_ack=cached_ack(session, fixture, repair.request, response, ordinal=1),
        source_binding=require_current_duplicate_source(
            session, item=target, run=fixture.run
        ),
        text=target.text_snapshot,
        taxonomy=fixture.taxonomy,
    )
    return AcceptedDuplicateLabelRepair(audit.outcome, audit)


def apply(session: Session, fixture: Fixture) -> DuplicateLabelRepairAudit:
    result = accepted(session, fixture)
    labeling._apply_frozen_window_results(
        session,
        fixture.run,
        items=fixture.items,
        file_id=fixture.data.file_id,
        lower=0,
        upper=300,
        outcomes={
            item.request_hash or "": result if item is fixture.target else ([], [])
            for item in fixture.items
        },
    )
    session.flush()
    assert fixture.target.status == "completed"
    return result.audit


def test_durable_apply_search_review_and_reuse_preserve_source_and_raw_ack_bytes(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        before = publication_digest(
            [
                row.model_dump(mode="json")
                for row in capture_canonical_scope(session, fixture.data.file_id)
            ]
        )
        audit = apply(session, fixture)
        raw_before = deepcopy(fixture.store.blobs)
        reads = fixture.store.reads
        session.expire_all()
        validate_item_duplicate_repair_audit(
            session, item=fixture.target, run=fixture.run
        )
        assert labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=fixture.items
        ) == {item.regulatory_chunk_id for item in fixture.items}
        assert fixture.target.regulatory_chunk_id in complete_frozen_sources(
            session,
            fixture.run.id,
            [fixture.target],
            [fixture.target.regulatory_chunk_id],
        )
        authority = labeling_review_authority(session, fixture.run.id)
        assert authority is not None
        assert authority["duplicate_label_selections"] == [
            {
                "chunk_id": fixture.target.regulatory_chunk_id,
                "audit_sha256": audit.audit_sha256,
            }
        ]
        assert fixture.store.reads == reads
        (
            fixture.run.status,
            fixture.run.stage,
            fixture.run.lease_token,
            fixture.run.lease_expires_at,
        ) = "completed", "finished", None, None
        fixture.run.finished_at = datetime.datetime.now(datetime.timezone.utc)
        session.flush()
        replacement = _create(session, fixture.data, fixture.taxonomy, fixture.binding)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=fixture.run, replacement=replacement
            )
            == 3
        )
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id
                == fixture.target.regulatory_chunk_id,
            )
        )
        assert copied is not None
        assert copied.source_snapshot[DUPLICATE_REPAIR_KEY] == audit.model_dump(
            mode="json"
        )
        validate_item_duplicate_repair_audit(session, item=copied, run=replacement)
        assert fixture.store.blobs == raw_before and fixture.store.reads == reads
        assert (
            publication_digest(
                [
                    row.model_dump(mode="json")
                    for row in capture_canonical_scope(session, fixture.data.file_id)
                ]
            )
            == before
        )


@pytest.mark.parametrize(
    "change",
    [
        "context",
        "metadata",
        "frozen-and-current",
        "provider",
        "request",
        "assignment",
        "raw-owner",
        "pointer",
        "receipt",
        "mixed",
        "mixed-reviewed",
        "owner",
    ],
)
def test_invalid_durable_receipts_fail_all_current_consumers(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        audit = apply(session, fixture)
        target = fixture.target
        if change in {"context", "metadata", "frozen-and-current"}:
            neighbor = session.get_one(RegulatoryChunk, fixture.data.first_id)
            if change == "context":
                neighbor.text += " changed"
            else:
                neighbor.chunk_metadata = {
                    **neighbor.chunk_metadata,
                    "paragraph_no": "9",
                }
                if change == "frozen-and-current":
                    saved = next(
                        item
                        for item in fixture.items
                        if item.regulatory_chunk_id == neighbor.id
                    )
                    saved.source_snapshot = {
                        **saved.source_snapshot,
                        "chunk_metadata": deepcopy(neighbor.chunk_metadata),
                    }
        elif change == "provider":
            fixture.run.provider_binding = {
                **fixture.run.provider_binding,
                "model": "other-model",
            }
        elif change == "request":
            target.context_snapshot = (target.context_snapshot or "") + " changed"
        elif change == "assignment":
            invalid_assignments: list[dict[str, object]] = [
                {"label_id": "unknown", "evidence_quote": "unknown"}
            ]
            target.assignments = invalid_assignments
        elif change == "raw-owner":
            old = session.get_one(RegulatorySourceAsset, fixture.asset_id)
            other = RegulatorySourceAsset(
                package_id=old.package_id,
                sha256="f" * 64,
                file_id=old.file_id,
                mime_type=old.mime_type,
                display_name=old.display_name,
                byte_count=old.byte_count,
            )
            session.add(other)
            session.flush()
            member = session.scalar(
                select(RegulatoryDocumentIntakeFile).where(
                    RegulatoryDocumentIntakeFile.user_file_id == fixture.data.file_id
                )
            )
            assert member is not None
            member.source_asset_id = other.id
        elif change == "pointer":
            row = session.get_one(FileRecord, audit.base_ack.file_id)
            row.file_metadata = {
                **cast(Mapping[str, object], row.file_metadata),
                "sha256": "f" * 64,
            }
        elif change == "receipt":
            changed = audit.model_dump(mode="json")
            changed["attempt"] = 2
            target.source_snapshot = {
                **target.source_snapshot,
                DUPLICATE_REPAIR_KEY: changed,
            }
        elif change == "mixed":
            target.source_snapshot = {
                **target.source_snapshot,
                "intake_abstention_repair": {},
            }
        elif change == "mixed-reviewed":
            target.source_snapshot = {
                **target.source_snapshot,
                "intake_source_only_reviewed_fragment": {},
            }
        elif change == "owner":
            changed = audit.model_dump(mode="json")
            changed["source_binding"]["owner_id"] = str(uuid4())
            target.source_snapshot = {
                **target.source_snapshot,
                DUPLICATE_REPAIR_KEY: changed,
            }
        session.flush()
        with pytest.raises(ValueError):
            validate_item_duplicate_repair_audit(session, item=target, run=fixture.run)
        assert (
            target.regulatory_chunk_id
            not in labeling.current_labeling_items_for_search(
                session, run=fixture.run, items=[target]
            )
        )
        assert target.regulatory_chunk_id not in complete_frozen_sources(
            session, fixture.run.id, [target], [target.regulatory_chunk_id]
        )
        with pytest.raises(ValueError):
            labeling_review_authority(session, fixture.run.id)
