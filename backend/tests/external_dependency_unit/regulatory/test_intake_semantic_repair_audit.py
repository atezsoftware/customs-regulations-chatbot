"""Real PostgreSQL source/audit checks fence accepted intake repair assignments."""

import datetime
from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryFilePublication,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_labeling_semantic_repair import (
    SEMANTIC_REPAIR_KEY,
    require_current_intake_repair_item,
    validate_item_semantic_repair_audit,
)
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.document_intake.models import IntakePreparationPolicy
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    AbstentionRepairOutput,
    build_abstention_repair,
    build_abstention_repair_audit,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)


def seed_v4_owned_intake_source(
    session: Session,
    data: LabelingData,
    *,
    raw_sha256: str = "a" * 64,
    raw_byte_count: int = 1,
) -> UUID:
    """Minimal real owned source for provider-free semantic repair integration."""
    file = session.get_one(UserFile, data.file_id)
    package = AmendmentSourcePackage(
        document_set_id=data.document_set_id,
        created_by=data.user_id,
        environment="repair-test",
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
    asset = RegulatorySourceAsset(
        package_id=package.id,
        sha256=raw_sha256,
        file_id=file.file_id,
        mime_type=file.file_type,
        display_name=file.name,
        byte_count=raw_byte_count,
    )
    session.add(asset)
    session.flush()
    session.add_all(
        [
            RegulatoryDocumentIntakeFile(
                intake_id=batch.id,
                source_asset_id=asset.id,
                user_file_id=file.id,
                role="root",
                relationships=[],
                status="chunked",
            ),
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key="repair-test",
                owner_id=uuid4(),
                fencing_token=1,
                lease_expires_at=datetime.datetime.now(datetime.timezone.utc),
                epoch=0,
                original_ingestion_receipt=OriginalIngestionReceipt(
                    file_id=file.file_id,
                    raw_sha256=asset.sha256,
                    documents_sha256="b" * 64,
                    plaintext_sha256="c" * 64,
                    canonical_sha256="d" * 64,
                    generation_hash="e" * 64,
                ).model_dump(mode="json"),
            ),
        ]
    )
    for identifier in (data.first_id, data.second_id):
        row = session.get_one(RegulatoryChunk, identifier)
        row.chunk_metadata = {
            **row.chunk_metadata,
            "intake_chunking_version": "4",
            "intake_preparation_policy": IntakePreparationPolicy().model_dump(
                mode="json"
            ),
        }
    session.flush()
    return asset.id


def _item_with_audit(
    session: Session, data: LabelingData, *, empty_result: bool = False
) -> tuple[RegulatoryLabelingRun, RegulatoryLabelingItem, AbstentionRepairAudit]:
    seed_v4_owned_intake_source(session, data)
    row = session.get_one(RegulatoryChunk, data.first_id)
    if empty_result:
        row.text = "|---|---|"
        row.chunk_type = "table"
    taxonomy_row = session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id)
    taxonomy = TaxonomyDefinition.model_validate(taxonomy_row.definition)
    binding = {"transport": "vertex_direct_v1", "model": "gemini-3.8-flash"}
    run = RegulatoryLabelingRun(
        document_set_id=data.document_set_id,
        taxonomy_id=data.taxonomy_id,
        model_configuration_id=data.model_id,
        requested_by_id=data.user_id,
        idempotency_key=uuid4(),
        model=binding["model"],
        provider_binding=binding,
        file_ids=[str(data.file_id)],
        snapshot_hash="a" * 64,
    )
    session.add(run)
    session.flush()
    base = build_labeling_request(
        chunk_id=row.id,
        text=row.text,
        context="Exact frozen advisory context",
        taxonomy=taxonomy,
        source_hash=context_hash(row.text),
    )
    item = RegulatoryLabelingItem(
        run_id=run.id,
        regulatory_chunk_id=row.id,
        user_file_id=row.user_file_id,
        canonical_text_sha256=context_hash(row.text),
        context_sha256=context_hash("fingerprint"),
        text_snapshot=row.text,
        source_snapshot={
            "position": row.position,
            "heading_path": row.heading_path,
            "chunk_type": row.chunk_type,
            "chunk_metadata": row.chunk_metadata,
            "validity_start_date": None,
            "validity_end_date": None,
        },
        context_snapshot="Exact frozen advisory context",
        request_hash=base.request_hash,
        request_payload=base.model_dump(mode="json", exclude_computed_fields=True),
        status="completed",
        labels=[] if empty_result else ["customs_value"],
        assignments=[]
        if empty_result
        else [{"label_id": "customs_value", "evidence_quote": "customs value"}],
    )
    repair = build_abstention_repair(
        base_request=base,
        rejected_response='{"labels":[],"abstained":true}',
        text=row.text,
        chunk_id=row.id,
        source_hash=context_hash(row.text),
        taxonomy=taxonomy,
        provider_binding=binding,
    )
    response = '{"labels":[{"label_id":"customs_value","evidence_quote":"customs value"}],"abstained":false}'
    if empty_result:
        response = '{"labels":[],"abstained":false}'
    output = AbstentionRepairOutput(
        request_hash=repair.request.request_hash,
        response=response,
        response_sha256=sha256(response.encode()).hexdigest(),
        output_row_sha256="f" * 64,
    )
    audit = build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri="gs://labeling-test-bucket/repair.jsonl",
        text=row.text,
        taxonomy=taxonomy,
    )
    session.add(item)
    session.flush()
    return run, item, audit


def test_current_audit_and_prospective_empty_outcome_are_guarded_without_mutation(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        run, item, audit = _item_with_audit(session, labeling_data)
        row = session.get_one(RegulatoryChunk, item.regulatory_chunk_id)
        before = dict(row.chunk_metadata)
        validate_item_semantic_repair_audit(
            session,
            item=item,
            run=run,
            audit=audit,
            labels=item.labels,
            assignments=item.assignments,
        )
        assert SEMANTIC_REPAIR_KEY not in item.source_snapshot
        item.source_snapshot = {
            **item.source_snapshot,
            SEMANTIC_REPAIR_KEY: audit.model_dump(mode="json"),
        }
        session.flush()
        validate_item_semantic_repair_audit(session, item=item, run=run)
        assert row.chunk_metadata == before
        with pytest.raises(ValueError):
            validate_item_semantic_repair_audit(
                session, item=item, run=run, labels=[], assignments=[]
            )


@pytest.mark.parametrize(
    "drift",
    [
        "canonical",
        "context",
        "metadata",
        "model",
        "taxonomy",
        "policy",
        "asset",
        "original",
        "labels",
        "audit",
        "legacy",
    ],
)
def test_any_reserved_audit_drift_is_rejected(
    labeling_data: LabelingData, drift: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        run, item, audit = _item_with_audit(session, labeling_data)
        item.source_snapshot = {
            **item.source_snapshot,
            SEMANTIC_REPAIR_KEY: audit.model_dump(mode="json"),
        }
        row = session.get_one(RegulatoryChunk, item.regulatory_chunk_id)
        if drift == "canonical":
            row.text += " Source changed."
        elif drift == "context":
            assert item.context_snapshot is not None
            item.context_snapshot += " changed"
        elif drift == "metadata":
            row.chunk_metadata = {**row.chunk_metadata, "new_field": "changed"}
        elif drift == "model":
            run.model = "different-model"
        elif drift == "taxonomy":
            taxonomy = session.get_one(RegulatoryLabelTaxonomy, run.taxonomy_id)
            taxonomy.definition = {**taxonomy.definition, "name": "Changed taxonomy"}
        elif drift == "policy":
            row.chunk_metadata = {
                **row.chunk_metadata,
                "intake_preparation_policy": {"version": 99},
            }
        elif drift == "asset":
            member = (
                session.query(RegulatoryDocumentIntakeFile)
                .filter_by(user_file_id=item.user_file_id)
                .one()
            )
            original = session.get_one(RegulatorySourceAsset, member.source_asset_id)
            replacement = RegulatorySourceAsset(
                package_id=original.package_id,
                sha256="b" * 64,
                file_id="replacement-original",
                mime_type=original.mime_type,
                display_name=original.display_name,
                byte_count=1,
            )
            session.add(replacement)
            session.flush()
            member.source_asset_id = replacement.id
        elif drift == "original":
            session.get_one(UserFile, item.user_file_id).file_id = "changed-original"
        elif drift == "labels":
            item.labels = []
        elif drift == "audit":
            item.source_snapshot = {**item.source_snapshot, SEMANTIC_REPAIR_KEY: None}
        elif drift == "legacy":
            row.chunk_metadata = {**row.chunk_metadata, "intake_chunking_version": "3"}
        session.flush()
        with pytest.raises(ValueError):
            validate_item_semantic_repair_audit(session, item=item, run=run)


def test_no_audit_preserves_legacy_path_but_repair_eligibility_rejects_unowned_file(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        run, item, _ = _item_with_audit(session, labeling_data)
        validate_item_semantic_repair_audit(session, item=item, run=run)
        row = session.get_one(RegulatoryChunk, item.regulatory_chunk_id)
        row.chunk_metadata = {**row.chunk_metadata, "intake_chunking_version": "3"}
        item.source_snapshot = {
            **item.source_snapshot,
            "chunk_metadata": row.chunk_metadata,
        }
        session.flush()
        with pytest.raises(ValueError, match="structural"):
            require_current_intake_repair_item(session, item=item, run=run)


def test_formatting_only_accepted_empty_result_has_exact_audit(
    labeling_data: LabelingData,
) -> None:
    with Session(labeling_data.database.engine) as session:
        run, item, audit = _item_with_audit(session, labeling_data, empty_result=True)
        assert item.labels == [] and item.assignments == []
        validate_item_semantic_repair_audit(
            session, item=item, run=run, audit=audit, labels=[], assignments=[]
        )
        item.source_snapshot = {
            **item.source_snapshot,
            SEMANTIC_REPAIR_KEY: audit.model_dump(mode="json"),
        }
        session.flush()
        validate_item_semantic_repair_audit(session, item=item, run=run)
        with pytest.raises(ValueError, match="already been recorded"):
            validate_item_semantic_repair_audit(
                session,
                item=item,
                run=run,
                audit=audit.model_copy(
                    update={"output_uri": "gs://replacement/second-attempt.jsonl"}
                ),
            )
