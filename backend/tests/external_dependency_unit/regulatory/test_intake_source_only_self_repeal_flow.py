"""Isolated PostgreSQL checks source-only recovery, review and projection authority."""

import json
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_self_repeal as source_only
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryDocumentIntakeFile,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelTaxonomy,
    RegulatorySourceAsset,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import FileStore
from onyx.regulatory.labeling.accepted_result import SourceOnlySelfRepeal
from onyx.regulatory.labeling.intake_abstention_repair import (
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.self_repeal import (
    SOURCE_ONLY_KEY,
    SourceOnlySelfRepealReceipt,
)
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
from tests.unit.onyx.regulatory.labeling.test_source_only_self_repeal import (
    RAW,
    TEXT,
    TITLE,
)

ABSTAINED = '{"labels":[],"abstained":true}'


class _RetainedStore:
    def __init__(self) -> None:
        self.raw = RAW

    def read_file(self, _file_id: str) -> BytesIO:
        return BytesIO(self.raw)


def _create(session: Session, data: LabelingData) -> RegulatoryLabelingRun:
    return labeling.create_labeling_run(
        session,
        document_set_id=data.document_set_id,
        taxonomy=session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id),
        model_configuration_id=data.model_id,
        model="gemini-3.8-flash",
        provider_binding={"transport": "vertex_direct_v1", "model": "gemini-3.8-flash"},
        requested_by_id=data.user_id,
        idempotency_key=uuid4(),
        target_file_ids=[data.file_id],
    )[0]


def _seed(
    session: Session, data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    RegulatoryLabelingRun,
    labeling.RunLease,
    dict[str, RegulatoryLabelingItem],
    _RetainedStore,
    TaxonomyDefinition,
]:
    file = session.get_one(UserFile, data.file_id)
    file.file_type, file.status = "text/html", UserFileStatus.CHUNKED
    seed_v4_owned_intake_source(
        session, data, raw_sha256=sha256(RAW).hexdigest(), raw_byte_count=len(RAW)
    )
    for identifier, number, caption in [
        (data.first_id, "5", "Teşkilat yapısı"),
        (data.second_id, "6", "Genel Müdür ve Genel Müdür Yardımcıları"),
    ]:
        row = session.get_one(RegulatoryChunk, identifier)
        row.text = (
            TEXT if number == "5" else "MADDE 6 - (Mülga:2/7/2018-KHK-703/76 md.)"
        )
        row.heading_path = [TITLE, "İKİNCİ BÖLÜM", f"MADDE {number} - {caption}"]
        row.chunk_metadata = {
            **row.chunk_metadata,
            "article_no": number,
            "document_number": "6001",
            "document_type": "kanun",
            "document_title": TITLE,
        }
    session.delete(session.get_one(RegulatoryChunk, data.incomplete_id))
    aggregate = session.get_one(RegulatoryChunk, data.derived_id)
    aggregate.text = TEXT + "\n" + session.get_one(RegulatoryChunk, data.second_id).text
    store = _RetainedStore()
    monkeypatch.setattr(
        source_only, "get_default_file_store", lambda: cast(FileStore, store)
    )
    run = _create(session, data)
    lease = labeling.claim_run(
        session, run_id=run.id, expected_generation=0, lease_seconds=600
    )
    assert lease is not None
    items = labeling.prepare_next_item_page(session, lease, limit=128)
    taxonomy = TaxonomyDefinition.model_validate(
        session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id).definition
    )
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
    session.flush()
    return (
        run,
        lease,
        {item.regulatory_chunk_id: item for item in items},
        store,
        taxonomy,
    )


def _end(run: RegulatoryLabelingRun, status: str = "completed_with_errors") -> None:
    run.status, run.stage, run.lease_token, run.lease_expires_at = (
        status,
        "finished",
        None,
        None,
    )


def test_continued_abstention_is_source_only_not_provider_success(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        run, lease, items, _, taxonomy = _seed(session, data, monkeypatch)
        first, second = items[data.first_id], items[data.second_id]
        before = publication_digest(
            [
                row.model_dump(mode="json")
                for row in capture_canonical_scope(session, data.file_id)
            ]
        )
        base = build_labeling_request(
            chunk_id=first.regulatory_chunk_id,
            text=first.text_snapshot,
            context=first.context_snapshot or "",
            taxonomy=taxonomy,
            source_hash=first.canonical_text_sha256 or "",
        )
        repair = build_abstention_repair(
            base_request=base,
            rejected_response=ABSTAINED,
            text=first.text_snapshot,
            chunk_id=first.regulatory_chunk_id,
            source_hash=first.canonical_text_sha256 or "",
            taxonomy=taxonomy,
            provider_binding=run.provider_binding,
        )
        raw = json.dumps(
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
        output = parse_abstention_repair_output(
            [raw], repair=repair, on_progress=lambda: None
        )
        audit = build_abstention_repair_audit(
            output=output,
            repair=repair,
            output_uri="fixture://continued-abstention",
            text=first.text_snapshot,
            taxonomy=taxonomy,
        )
        receipt = source_only.build_source_only_self_repeal_receipt(
            session, item=first, run=run, provider_abstention=audit
        )
        labeling._apply_frozen_window_results(
            session,
            run,
            items=[first, second],
            file_id=data.file_id,
            lower=-12,
            upper=13,
            outcomes={
                first.request_hash or "": SourceOnlySelfRepeal(receipt),
                second.request_hash or "": ([], []),
            },
        )
        assert (
            first.status == "completed"
            and first.labels == []
            and first.assignments == []
        )
        stored = SourceOnlySelfRepealReceipt.model_validate(
            first.source_snapshot[SOURCE_ONLY_KEY]
        )
        assert (
            stored.classification == "abstained" and stored.provider_abstention == audit
        )
        assert (
            audit.outcome.abstained
            and audit.rejected_response == ABSTAINED
            and audit.output_response == ABSTAINED
        )
        labeling.project_derived_labels(session, lease)
        assert labeling.final_status(session, lease) == "completed"
        projected = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == run.id
            )
        )
        assert projected is not None and projected.resolution == "lineage"
        assert projected.labels == [] and projected.provenance == {}
        authority = labeling_review_authority(session, run.id)
        assert authority is not None and authority["source_only_dispositions"]
        snapshot = labeling.list_items(
            session,
            document_set_id=data.document_set_id,
            run_id=run.id,
            offset=0,
            limit=10,
        )
        assert snapshot is not None
        result = next(item for item in snapshot.items if item.chunk_id == data.first_id)
        assert (
            result.classification_disposition == "source_only_self_repeal"
            and result.provider_abstained is True
        )
        assert (
            publication_digest(
                [
                    row.model_dump(mode="json")
                    for row in capture_canonical_scope(session, data.file_id)
                ]
            )
            == before
        )


def test_prior_failed_run_is_immutable_and_replacement_reuses_source_only_receipt(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        old_run, _, items, _, _ = _seed(session, data, monkeypatch)
        old = items[data.first_id]
        old.status, old.error = "failed", source_only.CONTINUED_ABSTENTION
        items[data.second_id].status = "completed"
        _end(old_run)
        before = deepcopy(old.source_snapshot), old.status, old.error, old.request_hash
        session.flush()
        replacement = _create(session, data)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=old_run, replacement=replacement
            )
            == 2
        )
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert (
            copied is not None and copied.status == "completed" and copied.labels == []
        )
        receipt = SourceOnlySelfRepealReceipt.model_validate(
            copied.source_snapshot[SOURCE_ONLY_KEY]
        )
        assert receipt.prior_failed_abstention is not None
        assert receipt.prior_failed_abstention.item_id == old.id
        assert (old.source_snapshot, old.status, old.error, old.request_hash) == before
        assert labeling.current_labeling_items_for_search(
            session, run=replacement, items=[copied]
        ) == {data.first_id}
        _end(replacement, "completed")
        session.flush()
        next_run = _create(session, data)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=replacement, replacement=next_run
            )
            == 2
        )
        next_item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == next_run.id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert (
            next_item is not None
            and next_item.source_snapshot[SOURCE_ONLY_KEY]
            == copied.source_snapshot[SOURCE_ONLY_KEY]
        )
        assert labeling_review_authority(session, next_run.id) is not None


@pytest.mark.parametrize(
    "drift",
    [
        "raw",
        "missing_blob",
        "canonical",
        "metadata",
        "context",
        "taxonomy",
        "asset",
        "receipt",
        "labels",
        "prior_error",
    ],
)
def test_drift_blocks_search_review_reuse_and_derived_projection(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    drift: str,
) -> None:
    data = labeling_data
    with Session(data.database.engine) as session:
        old_run, _, items, store, _ = _seed(session, data, monkeypatch)
        old = items[data.first_id]
        old.status, old.error = "failed", source_only.CONTINUED_ABSTENTION
        items[data.second_id].status = "completed"
        _end(old_run)
        session.flush()
        replacement = _create(session, data)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=old_run, replacement=replacement
            )
            == 2
        )
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert copied is not None
        if drift == "raw":
            store.raw += b" changed"
        elif drift == "missing_blob":
            from google.api_core.exceptions import NotFound

            def unavailable(_file_id: str) -> BytesIO:
                raise NotFound("Retained test object is unavailable")

            monkeypatch.setattr(store, "read_file", unavailable)
        elif drift == "canonical":
            session.get_one(RegulatoryChunk, data.first_id).text += " changed"
        elif drift == "metadata":
            row = session.get_one(RegulatoryChunk, data.first_id)
            row.chunk_metadata = {**row.chunk_metadata, "document_number": "6455"}
        elif drift == "context":
            copied.context_snapshot = (copied.context_snapshot or "") + " changed"
        elif drift == "taxonomy":
            taxonomy = session.get_one(RegulatoryLabelTaxonomy, data.taxonomy_id)
            taxonomy.definition = {
                **taxonomy.definition,
                "name": "Changed frozen taxonomy",
            }
        elif drift == "asset":
            member = session.scalar(
                select(RegulatoryDocumentIntakeFile).where(
                    RegulatoryDocumentIntakeFile.user_file_id == data.file_id
                )
            )
            assert member is not None
            asset = session.get_one(RegulatorySourceAsset, member.source_asset_id)
            changed = RegulatorySourceAsset(
                package_id=asset.package_id,
                sha256="f" * 64,
                file_id="different-retained-file",
                mime_type=asset.mime_type,
                display_name=asset.display_name,
                byte_count=asset.byte_count,
            )
            session.add(changed)
            session.flush()
            member.source_asset_id = changed.id
        elif drift == "receipt":
            value = cast(
                dict[str, object], deepcopy(copied.source_snapshot[SOURCE_ONLY_KEY])
            )
            value["taxonomy_sha256"] = "f" * 64
            copied.source_snapshot = {**copied.source_snapshot, SOURCE_ONLY_KEY: value}
        elif drift == "labels":
            copied.labels = ["customs_value"]
        elif drift == "prior_error":
            old.error = "Unverified generic provider failure"
        session.flush()
        assert data.first_id not in labeling.current_labeling_items_for_search(
            session, run=replacement, items=[copied]
        )
        with pytest.raises(ValueError):
            labeling_review_authority(session, replacement.id)
        lease = labeling.claim_run(
            session, run_id=replacement.id, expected_generation=0, lease_seconds=600
        )
        assert lease is not None
        labeling.project_derived_labels(session, lease)
        projection = session.scalar(
            select(RegulatoryDerivedLabelProjection).where(
                RegulatoryDerivedLabelProjection.run_id == replacement.id
            )
        )
        assert (
            projection is not None
            and projection.resolution == "unresolved"
            and projection.labels == []
        )


def test_normal_semantic_reconsideration_retains_both_abstentions(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    from onyx.db.models import RegulatoryDocumentIntake, RegulatoryLabelingShard
    from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchGateway
    from onyx.regulatory.labeling import orchestrator
    from tests.external_dependency_unit.regulatory.test_intake_semantic_repair_flow import (
        _CachedGateway,
    )

    data = labeling_data
    with Session(data.database.engine) as session:
        run, lease, items, _, taxonomy = _seed(session, data, monkeypatch)
        batch = session.scalar(
            select(RegulatoryDocumentIntake).where(
                RegulatoryDocumentIntake.created_by == data.user_id
            )
        )
        assert batch is not None
        member = session.scalar(
            select(RegulatoryDocumentIntakeFile).where(
                RegulatoryDocumentIntakeFile.intake_id == batch.id
            )
        )
        assert member is not None
        member.status = "labeling"
        batch.status, batch.stage, batch.labeling_run_id = (
            "preparing",
            "labeling",
            run.id,
        )
        shard = RegulatoryLabelingShard(
            run_id=run.id,
            ordinal=0,
            item_ids=[str(item.id) for item in items.values()],
            status="submitted",
            submission_key="regulatory-labeling-"
            + sha256(str(run.id).encode()).hexdigest(),
        )
        session.add(shard)
        session.flush()
        for item in items.values():
            item.shard_id, item.status = shard.id, "submitted"
        session.flush()
        gateway = _CachedGateway([], ABSTAINED)

        @contextmanager
        def current_session() -> Iterator[Session]:
            yield session

        monkeypatch.setattr(
            orchestrator, "get_session_with_current_tenant", current_session
        )
        result = orchestrator._repair_intake_abstention(
            lease,
            shard.id,
            items[data.first_id],
            cast(VertexBatchGateway, gateway),
            ABSTAINED,
            taxonomy,
            lambda: None,
        )
        assert gateway.submissions == 1 and isinstance(result, SourceOnlySelfRepeal)
        audit = result.receipt.provider_abstention
        assert (
            audit is not None and audit.outcome.abstained and audit.outcome.labels == []
        )
        assert (
            audit.rejected_response == ABSTAINED and audit.output_response == ABSTAINED
        )
        assert result.receipt.classification == "abstained"
