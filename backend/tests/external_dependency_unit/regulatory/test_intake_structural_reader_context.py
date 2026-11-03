"""Real PostgreSQL replay keeps canonical sources and ordinary label receipts intact."""

import base64
import datetime
import gzip
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db.document_intake_metadata import metadata_chunks
from onyx.db.document_intake_structural_readers import (
    validate_current_structural_reader_proof,
)
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelTaxonomy,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.document_index.publication_models import publication_digest
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.document_intake.metadata import metadata_input_hash
from onyx.regulatory.document_intake.models import IntakePreparationPolicy
from onyx.regulatory.document_intake.structural_readers import (
    build_native_chapter_reader_proofs,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    LabelingData,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)

MAIN_FILE = UUID("6cb352ab-1331-445d-8dee-7d2c2fbe8a87")
PDF_FILE = UUID("8cbb0ca5-7e37-4880-aef6-72f816af7539")
MAIN_INPUT_SHA = "21b05187e74424039e086570c512e55b0e20abce4c43afb06e383c86857b9ea0"
IBRAZ = "rc_9445ee9075213f582ea39b7124dc6275a32793c6"


def seed_reader_sources(
    session: Session, data: LabelingData
) -> tuple[list[UUID], bytes, UUID]:
    fixture = json.loads(
        gzip.decompress(
            (
                Path(__file__).parent / "fixtures/rg_native_chapters_20251220.json.gz"
            ).read_bytes()
        )
    )
    raw = base64.b64decode(fixture["raw_html_base64"])
    assert sha256(raw).hexdigest() == fixture["raw_sha256"]
    package = AmendmentSourcePackage(
        document_set_id=data.document_set_id,
        created_by=data.user_id,
        environment="reader-test",
        idempotency_key=str(uuid4()),
        request_hash="a" * 64,
        input_spec={},
        status="ready",
    )
    session.add(package)
    session.flush()
    batch = RegulatoryDocumentIntake(
        source_package_id=package.id,
        document_set_id=data.document_set_id,
        created_by=data.user_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=data.model_id,
    )
    session.add(batch)
    session.flush()
    main_asset = None
    for file_id, mime in ((MAIN_FILE, "text/html"), (PDF_FILE, "application/pdf")):
        file = UserFile(
            id=file_id,
            user_id=data.user_id,
            file_id=f"reader-original-{file_id}",
            name="Verified legal source",
            file_type=mime,
            status=UserFileStatus.CHUNKED,
        )
        session.add(file)
        session.flush()
        asset = RegulatorySourceAsset(
            package_id=package.id,
            sha256=fixture["raw_sha256"] if file_id == MAIN_FILE else "d" * 64,
            file_id=file.file_id,
            mime_type=mime,
            display_name=file.name,
            byte_count=len(raw) if file_id == MAIN_FILE else 1,
        )
        session.add(asset)
        session.flush()
        session.add_all(
            [
                DocumentSet__UserFile(
                    document_set_id=data.document_set_id, user_file_id=file_id
                ),
                RegulatoryDocumentIntakeFile(
                    intake_id=batch.id,
                    source_asset_id=asset.id,
                    user_file_id=file_id,
                    role="root" if file_id == MAIN_FILE else "annex",
                    relationships=[],
                    status="chunked",
                ),
                RegulatoryFilePublication(
                    user_file_id=file_id,
                    scope_key="reader-test",
                    owner_id=uuid4(),
                    fencing_token=1,
                    lease_expires_at=datetime.datetime.now(datetime.timezone.utc),
                    epoch=0,
                    original_ingestion_receipt=OriginalIngestionReceipt(
                        file_id=file.file_id,
                        raw_sha256=asset.sha256,
                        documents_sha256="b" * 64,
                        plaintext_sha256="c" * 64,
                        canonical_sha256="e" * 64,
                        generation_hash="f" * 64,
                    ).model_dump(mode="json"),
                ),
            ]
        )
        if file_id == MAIN_FILE:
            main_asset = asset.id
    policy = IntakePreparationPolicy().model_dump(mode="json")
    for chunk in fixture["chunks"]:
        values = deepcopy(chunk)
        values["user_file_id"] = MAIN_FILE
        values["projection_ordinal"] = values["position"]
        values["chunk_metadata"].update(
            intake_chunking_version="4", intake_preparation_policy=policy
        )
        session.add(RegulatoryChunk(**values))
    for position in range(332):
        aggregate = position >= 320
        session.add(
            RegulatoryChunk(
                id=f"reader-pdf-{position}",
                user_file_id=PDF_FILE,
                position=position,
                projection_ordinal=position,
                text=f"Certificate of origin source {position}.",
                heading_path=["Annex", str(position)],
                chunk_type="hierarchical_aggregate" if aggregate else "paragraph",
                chunk_metadata={
                    "intake_chunking_version": "4",
                    "intake_preparation_policy": policy,
                    "chunk_variant": "hierarchical_aggregate"
                    if aggregate
                    else "atomic",
                    **(
                        {
                            "source_regulatory_chunk_ids": [
                                f"reader-pdf-{(position - 320) * 2}",
                                f"reader-pdf-{(position - 320) * 2 + 1}",
                            ]
                        }
                        if aggregate
                        else {}
                    ),
                },
                source="indexed",
                status="active",
            )
        )
    session.flush()
    assert main_asset is not None
    return [MAIN_FILE, PDF_FILE], raw, main_asset


@pytest.mark.parametrize("pending_body", [False, True])
def test_native_reader_retry_reuses_all_ordinary_receipts_without_dispatch(
    labeling_data: LabelingData,
    pending_body: bool,
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        file_ids, raw, asset_id = seed_reader_sources(session, data)
        main_before = capture_canonical_scope(session, MAIN_FILE)
        all_before = {
            identifier: publication_digest(
                [
                    row.model_dump(mode="json")
                    for row in capture_canonical_scope(session, identifier)
                ]
            )
            for identifier in file_ids
        }
        main_rows = list(
            session.scalars(
                select(RegulatoryChunk)
                .where(RegulatoryChunk.user_file_id == MAIN_FILE)
                .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            )
        )
        assert metadata_input_hash(metadata_chunks(main_rows)) == MAIN_INPUT_SHA
        proofs = build_native_chapter_reader_proofs(
            raw_html=raw,
            expected_sha256=sha256(raw).hexdigest(),
            user_file_id=MAIN_FILE,
            source_asset_id=asset_id,
            rows=main_before,
        )
        assert len(proofs) == 11 and len(proofs[IBRAZ].governed_chunk_ids) == 26
        pending_id = proofs[IBRAZ].governed_chunk_ids[0] if pending_body else None
        for proof in proofs.values():
            validate_current_structural_reader_proof(session, proof)
        taxonomy = session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id)
        binding: dict[str, Any] = {
            "transport": "vertex_direct_v1",
            "model_configuration_id": data.model_id,
        }
        parameters: dict[str, Any] = dict(
            document_set_id=data.document_set_id,
            taxonomy=taxonomy,
            model_configuration_id=data.model_id,
            model="gemini-3.8-flash",
            provider_binding=binding,
            requested_by_id=data.user_id,
            target_file_ids=file_ids,
        )
        previous, _ = labeling.create_labeling_run(
            session, idempotency_key=uuid4(), **parameters
        )
        lease = labeling.claim_run(
            session, run_id=previous.id, expected_generation=0, lease_seconds=600
        )
        assert lease is not None
        vocabulary = TaxonomyDefinition.model_validate(taxonomy.definition)
        while page := labeling.prepare_next_item_page(session, lease, limit=128):
            for item in page:
                request = build_labeling_request(
                    chunk_id=item.regulatory_chunk_id,
                    text=item.text_snapshot,
                    context=item.context_snapshot or "",
                    taxonomy=vocabulary,
                    source_hash=item.canonical_text_sha256 or "",
                )
                item.request_hash = request.request_hash
                item.request_payload = request.model_dump(
                    mode="json", exclude={"request_hash"}
                )
                if (
                    item.regulatory_chunk_id in proofs
                    or item.regulatory_chunk_id == pending_id
                ):
                    item.status, item.error = (
                        "failed",
                        "Model abstained: insufficient evidence",
                    )
                else:
                    item.status, item.labels = "completed", ["origin"]
                    assignments: list[dict[str, object]] = [
                        {
                            "label_id": "origin",
                            "evidence_quote": item.text_snapshot[:64],
                        }
                    ]
                    item.assignments = assignments
            session.flush()
        previous.status, previous.stage = "completed_with_errors", "finished"
        previous.lease_token, previous.lease_expires_at = None, None
        session.flush()
        previous_items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == previous.id
                )
            )
        )
        original_receipts = {
            item.regulatory_chunk_id: (
                item.request_hash,
                item.context_sha256,
                deepcopy(item.source_snapshot),
                deepcopy(item.assignments),
            )
            for item in previous_items
            if item.status == "completed"
        }
        assert len(original_receipts) == 651 - int(pending_body)
        pending_original = next(
            (item for item in previous_items if item.regulatory_chunk_id == pending_id),
            None,
        )
        replacement, _ = labeling.create_labeling_run(
            session, idempotency_key=uuid4(), structural_readers=proofs, **parameters
        )
        assert replacement.total_chunks == 651 and replacement.derived_chunks == 46
        assert editing.reuse_completed_intake_labels(
            session, previous=previous, replacement=replacement
        ) == 651 - int(pending_body)
        new_items = list(
            session.scalars(
                select(RegulatoryLabelingItem).where(
                    RegulatoryLabelingItem.run_id == replacement.id
                )
            )
        )
        assert {item.regulatory_chunk_id for item in new_items} == set(
            original_receipts
        ) | ({pending_id} if pending_id else set())
        for item in new_items:
            if item.regulatory_chunk_id == pending_id:
                assert item.status == "pending"
                continue
            assert item.status == "completed"
            assert (
                item.request_hash,
                item.context_sha256,
                item.source_snapshot,
                item.assignments,
            ) == original_receipts[item.regulatory_chunk_id]
        lease = labeling.claim_run(
            session, run_id=replacement.id, expected_generation=0, lease_seconds=600
        )
        assert lease is not None
        pending = labeling.prepare_next_item_page(session, lease, limit=128)
        assert len(pending) == int(pending_body)
        if pending:
            item = pending[0]
            assert pending_original is not None
            rebuilt = build_labeling_request(
                chunk_id=item.regulatory_chunk_id,
                text=item.text_snapshot,
                context=item.context_snapshot or "",
                taxonomy=vocabulary,
                source_hash=item.canonical_text_sha256 or "",
            )
            assert pending_original.request_payload is not None
            assert rebuilt == labeling.VertexBatchRequest.model_validate(
                pending_original.request_payload
            )
            assert item.context_snapshot == pending_original.context_snapshot
            item.request_hash = rebuilt.request_hash
            item.request_payload = rebuilt.model_dump(
                mode="json", exclude_computed_fields=True
            )
            item.status, item.labels = "completed", ["origin"]
            assignments: list[dict[str, object]] = [
                {"label_id": "origin", "evidence_quote": item.text_snapshot[:64]}
            ]
            item.assignments = assignments
            session.flush()
        while labeling.has_pending_derived_projections(session, lease):
            labeling.project_derived_labels(session, lease, limit=128)
        assert replacement.unresolved_derived_chunks == 0
        assert labeling.current_labeling_items_for_search(
            session, run=replacement, items=new_items
        ) == {item.regulatory_chunk_id for item in new_items}
        reader = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == replacement.id,
                RegulatoryDerivedLabelProjection.regulatory_chunk_id == IBRAZ,
            )
        )
        assert reader is not None and reader.resolution == "lineage"
        assert set(
            labeling.current_derived_label_sources_for_search(
                session, run=replacement, target=reader
            )
        ) == set(proofs[IBRAZ].governed_chunk_ids)
        provenance = cast(list[dict[str, Any]], reader.provenance["origin"])
        assert {entry["canonical_chunk_id"] for entry in provenance} == set(
            proofs[IBRAZ].governed_chunk_ids
        )
        assert all(
            session.get_one(RegulatoryLabelingItem, item.id).status == "failed"
            for item in previous_items
            if item.regulatory_chunk_id in proofs
        )
        assert not session.scalars(
            select(labeling.RegulatoryLabelingShard).where(
                labeling.RegulatoryLabelingShard.run_id == replacement.id
            )
        ).all()
        assert {
            identifier: publication_digest(
                [
                    row.model_dump(mode="json")
                    for row in capture_canonical_scope(session, identifier)
                ]
            )
            for identifier in file_ids
        } == all_before
        assert metadata_input_hash(metadata_chunks(main_rows)) == MAIN_INPUT_SHA
