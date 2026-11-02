"""Review authority binds only new intake evidence and keeps partial edits possible."""

from hashlib import sha256
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as labeling
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.db.regulatory_labeling_semantic_repair import SEMANTIC_REPAIR_KEY
from onyx.document_index.publication_models import publication_digest
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.document_intake.structural_readers import (
    build_native_chapter_reader_proofs,
)
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairOutput,
    build_abstention_repair,
    build_abstention_repair_audit,
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

HTML = '<style>p.caption {text-align:center;font-weight:bold}</style><p class="caption">İKİNCİ BÖLÜM</p><p class="caption">Belgelerin sunulması</p><p>MADDE 5- (1) Belge saklanır.</p>'


def _prepared_run(
    session: Session,
    data: LabelingData,
    *,
    reader: bool = False,
    semantic: bool = False,
) -> RegulatoryLabelingRun:
    if reader:
        session.get_one(UserFile, data.file_id).file_type = "text/html"
    asset = seed_v4_owned_intake_source(
        session,
        data,
        raw_sha256=sha256(HTML.encode()).hexdigest() if reader else "a" * 64,
    )
    proofs = {}
    if reader:
        heading = session.get_one(RegulatoryChunk, data.first_id)
        heading.text, heading.chunk_type = "Belgelerin sunulması", "free_text"
        heading.heading_path = ["Regulation", "İKİNCİ BÖLÜM"]
        heading.chunk_metadata = {**heading.chunk_metadata, "article_no": None}
        body = session.get_one(RegulatoryChunk, data.second_id)
        body.text, body.chunk_type = "MADDE 5- (1) Belge saklanır.", "paragraph"
        body.heading_path = [*heading.heading_path, "MADDE 5"]
        body.chunk_metadata = {**body.chunk_metadata, "article_no": "5"}
        session.flush()
        proofs = build_native_chapter_reader_proofs(
            raw_html=HTML.encode(),
            expected_sha256=sha256(HTML.encode()).hexdigest(),
            user_file_id=data.file_id,
            source_asset_id=asset,
            rows=capture_canonical_scope(session, data.file_id),
        )
        assert len(proofs) == 1
    taxonomy_row = session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id)
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    binding: dict[str, object] = {
        "transport": "vertex_direct_v1",
        "model": "gemini-3.8-flash",
        "model_configuration_id": data.model_id,
    }
    run, _ = labeling.create_labeling_run(
        session,
        document_set_id=data.document_set_id,
        taxonomy=taxonomy_row,
        model_configuration_id=data.model_id,
        model="gemini-3.8-flash",
        provider_binding=binding,
        requested_by_id=data.user_id,
        idempotency_key=uuid4(),
        target_file_ids=[data.file_id],
        structural_readers=proofs,
    )
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=600
    )
    assert lease is not None
    for item in labeling.prepare_next_item_page(session, lease, limit=128):
        request = build_labeling_request(
            chunk_id=item.regulatory_chunk_id,
            text=item.text_snapshot,
            context=item.context_snapshot or "",
            taxonomy=taxonomy,
            source_hash=item.canonical_text_sha256 or "",
        )
        item.request_payload = request.model_dump(
            mode="json", exclude_computed_fields=True
        )
        item.request_hash = request.request_hash
        label = (
            "origin"
            if reader or item.regulatory_chunk_id == data.second_id
            else "customs_value"
        )
        quote = (
            "Belge"
            if reader
            else "certificate of origin"
            if label == "origin"
            else "customs value"
        )
        item.labels = [label]
        assignments: list[dict[str, object]] = [
            {"label_id": label, "evidence_quote": quote}
        ]
        item.assignments = assignments
        item.status = "completed"
        if semantic and item.regulatory_chunk_id == data.first_id:
            repair = build_abstention_repair(
                base_request=request,
                rejected_response='{"labels":[],"abstained":true}',
                text=item.text_snapshot,
                chunk_id=item.regulatory_chunk_id,
                source_hash=context_hash(item.text_snapshot),
                taxonomy=taxonomy,
                provider_binding=binding,
            )
            response = '{"labels":[{"label_id":"customs_value","evidence_quote":"customs value"}],"abstained":false}'
            output = AbstentionRepairOutput(
                request_hash=repair.request.request_hash,
                response=response,
                response_sha256=sha256(response.encode()).hexdigest(),
                output_row_sha256="f" * 64,
            )
            audit = build_abstention_repair_audit(
                output=output,
                repair=repair,
                output_uri="gs://labeling-test/repair.jsonl",
                text=item.text_snapshot,
                taxonomy=taxonomy,
            )
            item.source_snapshot = {
                **item.source_snapshot,
                SEMANTIC_REPAIR_KEY: audit.model_dump(mode="json"),
            }
    session.flush()
    if reader:
        while labeling.has_pending_derived_projections(session, lease):
            labeling.project_derived_labels(session, lease, limit=128)
    run.status = "completed"
    session.flush()
    return run


def test_legacy_no_roles_returns_none_and_semantic_hashes_are_stable(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        run = _prepared_run(session, labeling_data)
        assert labeling_review_authority(session, run.id) is None
    with Session(labeling_data.database.engine) as session:
        run = _prepared_run(session, labeling_data, semantic=True)
        before = publication_digest(
            [
                r.model_dump(mode="json")
                for r in capture_canonical_scope(session, labeling_data.file_id)
            ]
        )
        authority = labeling_review_authority(session, run.id)
        assert (
            authority is not None
            and len(cast(list[object], authority["semantic_repairs"])) == 1
        )
        assert authority == labeling_review_authority(session, run.id)
        assert (
            publication_digest(
                [
                    r.model_dump(mode="json")
                    for r in capture_canonical_scope(session, labeling_data.file_id)
                ]
            )
            == before
        )
        item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run.id,
                RegulatoryLabelingItem.regulatory_chunk_id == labeling_data.first_id,
            )
        )
        assert item is not None
        item.source_snapshot = {**item.source_snapshot, SEMANTIC_REPAIR_KEY: None}
        session.flush()
        with pytest.raises(ValueError):
            labeling_review_authority(session, run.id)
