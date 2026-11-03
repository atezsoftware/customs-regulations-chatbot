"""Real PostgreSQL keeps dependent-list reconsideration source-bound and audited."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import regulatory_labeling as labeling
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelingShard,
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
    VertexBatchGateway,
    VertexBatchRequest,
)
from onyx.regulatory.labeling import orchestrator
from onyx.regulatory.labeling.intake_abstention_repair import (
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
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
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import line


def _frozen(position: int) -> dict[str, Any]:
    home = (
        Path(__file__).resolve().parents[2] / "unit/onyx/regulatory/labeling/fixtures"
    )
    if position == 87:
        return json.loads((home / "legacy_manual_handling_context.json").read_text())
    frozen = json.loads((home / "legacy_requirement_lists_context.json").read_text())
    case = next(row for row in frozen["cases"] if row["position"] == position)
    target = next(
        row for row in frozen["sources"] if row["chunk_id"] == case["target_chunk_id"]
    )
    members = set(target["snapshot"]["context_member_ids"])
    return {
        **frozen,
        **case,
        "sources": [row for row in frozen["sources"] if row["chunk_id"] in members],
    }


@pytest.mark.parametrize("position", [87, 362, 364, 369, 371])
@pytest.mark.parametrize(
    "mode", ["accepted", "continued", "context-drift", "raw-owner"]
)
def test_frozen_list_join_and_stored_receipt_remain_strict(
    labeling_data: LabelingData, position: int, mode: str
) -> None:
    data, frozen = labeling_data, _frozen(position)
    with data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                asset_id = seed_v4_owned_intake_source(session, data)
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
                    else '{"labels":[],"abstained":false}'
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
                target.source_snapshot = {
                    **target.source_snapshot,
                    SEMANTIC_REPAIR_KEY: audit.model_dump(mode="json"),
                }
                session.flush()
                if mode == "context-drift":
                    neighbor = next(
                        chunk
                        for chunk in chunks
                        if chunk.id != target.regulatory_chunk_id
                    )
                    neighbor.text += " changed"
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


def test_unsupported_current_list_fails_without_provider_or_false_source_drift(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = labeling_data
    with data.database.engine.connect() as connection:
        transaction = connection.begin()
        try:
            with Session(
                connection, join_transaction_mode="create_savepoint"
            ) as session:
                seed_v4_owned_intake_source(session, data)
                session.get_one(UserFile, data.file_id).status = UserFileStatus.CHUNKED
                source = session.get_one(RegulatoryChunk, data.first_id)
                source.text, source.chunk_type = (
                    "(a) İmalatçının adı ve adresi.",
                    "clause",
                )
                source.heading_path = ["EK-1", "EK 3", "Inherited scope", source.text]
                source.chunk_metadata = {
                    "intake_chunking_version": "2",
                    "chunk_variant": "atomic",
                    "heading_path": source.heading_path,
                    "clause_label": "a",
                }
                taxonomy_row = session.get_one(
                    RegulatoryLabelTaxonomy, data.taxonomy_id
                )
                vocabulary = TaxonomyDefinition.model_validate(taxonomy_row.definition)
                run, _created = labeling.create_labeling_run(
                    session,
                    document_set_id=data.document_set_id,
                    taxonomy=taxonomy_row,
                    model_configuration_id=data.model_id,
                    model="gemini-3.8-flash",
                    provider_binding={
                        "transport": "vertex_direct_v1",
                        "model": "gemini-3.8-flash",
                        "model_configuration_id": data.model_id,
                    },
                    requested_by_id=data.user_id,
                    idempotency_key=uuid4(),
                    target_file_ids=[data.file_id],
                )
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
                batch.status, batch.stage, batch.labeling_run_id = (
                    "preparing",
                    "labeling",
                    run.id,
                )
                member.status = "labeling"
                lease = labeling.claim_run(
                    session, run_id=run.id, expected_generation=0, lease_seconds=600
                )
                assert lease is not None
                items = labeling.prepare_next_item_page(session, lease, limit=128)
                requests = {
                    item.id: build_labeling_request(
                        chunk_id=item.regulatory_chunk_id,
                        text=item.text_snapshot,
                        context=item.context_snapshot or "",
                        taxonomy=vocabulary,
                        source_hash=item.canonical_text_sha256 or "",
                    )
                    for item in items
                }
                labeling.store_prepared_shards(
                    session,
                    lease,
                    requests=[
                        labeling.PreparedRequest(
                            item_id=item.id,
                            request_hash=requests[item.id].request_hash,
                            request_payload=requests[item.id].model_dump(
                                mode="json", exclude_computed_fields=True
                            ),
                        )
                        for item in items
                    ],
                    shards=[
                        labeling.PreparedShard(
                            ordinal=0,
                            item_ids=tuple(item.id for item in items),
                            submission_key="regulatory-labeling-"
                            + context_hash(str(run.id)),
                        )
                    ],
                    failed_items={},
                )
                shard = session.scalar(
                    select(RegulatoryLabelingShard).where(
                        RegulatoryLabelingShard.run_id == run.id
                    )
                )
                assert shard is not None
                labeling.mark_shard_submitting(
                    session, lease, shard_id=shard.id, reconcile_seconds=30
                )
                labeling.record_shard_state(
                    session, lease, shard_id=shard.id, status="submitted"
                )
                session.flush()
                originals = [
                    json.dumps(
                        {
                            "key": requests[item.id].request_hash,
                            "request": requests[item.id].to_generate_content_request(),
                            "response": {
                                "candidates": [
                                    {
                                        "finishReason": "STOP",
                                        "content": {
                                            "parts": [
                                                {
                                                    "text": json.dumps(
                                                        {
                                                            "labels": [],
                                                            "abstained": item.regulatory_chunk_id
                                                            == data.first_id,
                                                        }
                                                    )
                                                }
                                            ]
                                        },
                                    }
                                ]
                            },
                        }
                    )
                    for item in items
                ]
                gateway = Mock(spec=VertexBatchGateway)
                gateway.read_results.return_value = iter(originals)
                gateway.submit.side_effect = AssertionError(
                    "Unsupported list cannot submit a provider repair"
                )

                @contextmanager
                def current_session() -> Iterator[Session]:
                    yield session

                monkeypatch.setattr(
                    orchestrator, "get_session_with_current_tenant", current_session
                )
                monkeypatch.setattr(
                    orchestrator, "_lease_heartbeat", lambda _lease: lambda: None
                )
                orchestrator._read_and_apply_results(
                    lease, shard.id, cast(VertexBatchGateway, gateway), "fixture://base"
                )
                target = next(
                    item for item in items if item.regulatory_chunk_id == data.first_id
                )
                assert target.status == "failed" and run.stale_chunks == 0
                assert (
                    target.error
                    == "Legacy semantic repair requires a verified structural source"
                )
                assert SEMANTIC_REPAIR_KEY not in target.source_snapshot
                assert target.request_hash == requests[target.id].request_hash
                assert target.request_payload == requests[target.id].model_dump(
                    mode="json", exclude_computed_fields=True
                )
                assert source.text == target.text_snapshot
                gateway.submit.assert_not_called()
        finally:
            transaction.rollback()
