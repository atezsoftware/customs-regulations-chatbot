"""Real PostgreSQL preserves the criterion caption source and strict repair receipt."""

import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntakeFile,
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
from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
)
from onyx.regulatory.labeling.intake_abstention_repair import (
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition
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
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import line


def _frozen(context_size: int) -> dict[str, Any]:
    path = (
        Path(__file__).resolve().parents[2]
        / "unit/onyx/regulatory/labeling/fixtures/legacy_inspection_criteria_context.json"
    )
    frozen = json.loads(path.read_text())
    if context_size == 25:
        current = json.loads(
            path.with_name("legacy_inspection_criteria_current25.json").read_text()
        )
        frozen.update(
            {key: current[key] for key in ("sources", "base_request", "base_response")}
        )
    return frozen


@pytest.mark.parametrize("context_size", [26, 25])
@pytest.mark.parametrize(
    "mode",
    [
        "accepted",
        "continued",
        "context-drift",
        "metadata-drift",
        "raw-owner",
        "wrong-run-file",
        "provider-drift",
        "request-drift",
        "taxonomy-drift",
    ],
)
def test_current_inspection_context_and_durable_receipt_remain_strict(
    labeling_data: LabelingData, mode: str, context_size: int
) -> None:
    data, frozen = labeling_data, _frozen(context_size)
    with data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                session.get_one(UserFile, data.file_id).file_type = "application/pdf"
                asset_id = seed_v4_owned_intake_source(
                    session,
                    data,
                    raw_sha256=frozen["source_pdf_sha256"],
                    raw_byte_count=frozen["source_pdf_bytes"],
                )
                taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
                taxonomy_row = session.scalar(
                    select(RegulatoryLabelTaxonomy).where(
                        RegulatoryLabelTaxonomy.version_hash == taxonomy.version_hash
                    )
                )
                assert taxonomy_row is not None
                run = RegulatoryLabelingRun(
                    document_set_id=data.document_set_id,
                    taxonomy_id=taxonomy_row.id,
                    model_configuration_id=data.model_id,
                    requested_by_id=data.user_id,
                    idempotency_key=uuid4(),
                    model=frozen["provider_binding"]["model"],
                    provider_binding=frozen["provider_binding"],
                    file_ids=[str(data.file_id)],
                    snapshot_hash=context_hash(frozen["target_chunk_id"]),
                )
                session.add(run)
                session.flush()
                target = None
                chunks = []
                for source in frozen["sources"]:
                    current = source["current_snapshot"]
                    chunk = RegulatoryChunk(
                        id=source["chunk_id"],
                        user_file_id=data.file_id,
                        text=source["current_text"],
                        position=current["position"],
                        projection_ordinal=1000 + current["position"],
                        heading_path=current["heading_path"],
                        chunk_type=current["chunk_type"],
                        chunk_metadata=deepcopy(current["chunk_metadata"]),
                        status="active",
                        source="indexed",
                    )
                    session.add(chunk)
                    session.flush()
                    saved = RegulatoryLabelingItem(
                        run_id=run.id,
                        regulatory_chunk_id=chunk.id,
                        user_file_id=data.file_id,
                        text_snapshot=source["text"],
                        source_snapshot=deepcopy(source["snapshot"]),
                        canonical_text_sha256=source["canonical_text_sha256"],
                    )
                    session.add(saved)
                    chunks.append(chunk)
                    if chunk.id == frozen["target_chunk_id"]:
                        target = saved
                session.flush()
                assert target is not None
                require_current_intake_repair_item(session, item=target, run=run)
                base = VertexBatchRequest.model_validate(frozen["base_request"])
                target.request_payload = frozen["base_request"]
                target.request_hash = base.request_hash
                target.context_snapshot = json.loads(base.prompt)[
                    "interpretation_context"
                ]
                target.context_sha256 = context_hash(target.context_snapshot)
                target.labels, target.assignments = [], []
                repair = build_abstention_repair(
                    base_request=base,
                    rejected_response=frozen["base_response"],
                    text=target.text_snapshot,
                    chunk_id=target.regulatory_chunk_id,
                    source_hash=target.canonical_text_sha256 or "",
                    taxonomy=taxonomy,
                    provider_binding=run.provider_binding,
                )
                response = (
                    frozen["base_response"]
                    if mode == "continued"
                    else json.dumps(
                        {
                            "labels": [
                                {
                                    "label_id": "ANX.TECH_SPEC",
                                    "evidence_quote": target.text_snapshot,
                                }
                            ],
                            "abstained": False,
                        }
                    )
                )
                output = parse_abstention_repair_output(
                    [line(repair, response)], repair=repair, on_progress=lambda: None
                )
                audit = build_abstention_repair_audit(
                    output=output,
                    repair=repair,
                    output_uri="fixture://local-only",
                    text=target.text_snapshot,
                    taxonomy=taxonomy,
                )
                if mode != "continued":
                    target.labels = ["ANX.TECH_SPEC"]
                    assignments: list[dict[str, object]] = [
                        {
                            "label_id": "ANX.TECH_SPEC",
                            "evidence_quote": target.text_snapshot,
                        }
                    ]
                    target.assignments = assignments
                target.source_snapshot = {
                    **target.source_snapshot,
                    SEMANTIC_REPAIR_KEY: audit.model_dump(mode="json"),
                }
                session.flush()
                if mode in {"context-drift", "metadata-drift"}:
                    neighbor = next(
                        chunk
                        for chunk in chunks
                        if chunk.id != target.regulatory_chunk_id
                    )
                    if mode == "context-drift":
                        neighbor.text += " changed"
                    else:
                        neighbor.chunk_metadata = {
                            **neighbor.chunk_metadata,
                            "paragraph_no": "9",
                        }
                elif mode == "raw-owner":
                    original = session.get_one(RegulatorySourceAsset, asset_id)
                    replacement = RegulatorySourceAsset(
                        package_id=original.package_id,
                        sha256="0" * 64,
                        file_id=original.file_id,
                        mime_type=original.mime_type,
                        display_name=original.display_name,
                        byte_count=original.byte_count,
                    )
                    session.add(replacement)
                    session.flush()
                    member = session.scalar(
                        select(RegulatoryDocumentIntakeFile).where(
                            RegulatoryDocumentIntakeFile.user_file_id == data.file_id
                        )
                    )
                    assert member is not None
                    member.source_asset_id = replacement.id
                elif mode == "wrong-run-file":
                    run.file_ids = [str(uuid4())]
                elif mode == "provider-drift":
                    run.provider_binding = {
                        **run.provider_binding,
                        "model": "another-model",
                    }
                elif mode == "request-drift":
                    target.request_hash = "f" * 64
                elif mode == "taxonomy-drift":
                    taxonomy_row.version_hash = "f" * 64
                session.flush()
                if mode == "accepted":
                    validate_item_semantic_repair_audit(session, item=target, run=run)
                    session.expire_all()
                    validate_item_semantic_repair_audit(session, item=target, run=run)
                else:
                    with pytest.raises(ValueError):
                        validate_item_semantic_repair_audit(
                            session, item=target, run=run
                        )
        finally:
            transaction.rollback()
