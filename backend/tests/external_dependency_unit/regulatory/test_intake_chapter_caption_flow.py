"""Real PostgreSQL keeps native captions separate from operative article authority."""

import json
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from typing import Any, NamedTuple, cast

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_chapter_caption as captions
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_readers import complete_frozen_sources
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.db.regulatory_labeling_semantic_repair import (
    require_current_intake_repair_item,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import FileStore
from onyx.regulatory.labeling.accepted_result import SourceOnlyChapterCaption
from onyx.regulatory.labeling.chapter_caption import (
    CHAPTER_CAPTION_KEY,
    SourceOnlyChapterCaptionReceipt,
)
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.self_repeal import SOURCE_ONLY_KEY
from tests.external_dependency_unit.regulatory.test_intake_semantic_repair_audit import (
    seed_v4_owned_intake_source,
)
from tests.external_dependency_unit.regulatory.test_intake_source_only_self_repeal_flow import (
    _create,
    _end,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import LabelingData
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_data as labeling_data,
)
from tests.external_dependency_unit.regulatory.test_labeling_jobs import (
    labeling_database as labeling_database,
)
from tests.unit.onyx.regulatory.labeling.test_chapter_caption import (
    RAW,
    profile_raw,
    sources,
)

ABSTAINED = '{"labels":[],"abstained":true}'


class CaptionStore:
    raw = RAW

    def read_file(self, _file_id: str) -> BytesIO:
        return BytesIO(self.raw)


class Fixture(NamedTuple):
    data: LabelingData
    run: RegulatoryLabelingRun
    items: dict[str, RegulatoryLabelingItem]
    store: CaptionStore
    taxonomy: TaxonomyDefinition


def seed(
    session: Session,
    data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    *,
    profile: int = 1,
) -> Fixture:
    raw = profile_raw(profile)
    file = session.get_one(UserFile, data.file_id)
    file.file_type, file.status = "application/pdf", UserFileStatus.CHUNKED
    seed_v4_owned_intake_source(
        session, data, raw_sha256=sha256(raw).hexdigest(), raw_byte_count=len(raw)
    )
    for identifier in (
        data.first_id,
        data.second_id,
        data.derived_id,
        data.incomplete_id,
    ):
        session.delete(session.get_one(RegulatoryChunk, identifier))
    session.flush()
    ids = {
        "root": data.first_id,
        "caption": data.second_id,
        "body": data.derived_id,
        "title_source": data.incomplete_id,
    }
    for source in sources(profile=profile):
        metadata = deepcopy(source.snapshot["chunk_metadata"])
        for fact in metadata["intake_legal_metadata"]["fields"].values():
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
            "name": "Native service unit",
            "labels": [
                {
                    "id": "duty",
                    "name": "Duty",
                    "description": "An express obligation performed by the responsible department.",
                }
            ],
        }
    )
    taxonomy_row = labeling.create_taxonomy(
        session, taxonomy=taxonomy, created_by_id=data.user_id
    )
    data = data._replace(taxonomy_id=taxonomy_row.id)
    store = CaptionStore()
    store.raw = raw
    monkeypatch.setattr(
        captions, "get_default_file_store", lambda: cast(FileStore, store)
    )
    session.flush()
    run = _create(session, data)
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
    session.flush()
    return Fixture(
        data, run, {i.regulatory_chunk_id: i for i in items}, store, taxonomy
    )


def continued_audit(
    item: RegulatoryLabelingItem, fixture: Fixture
) -> AbstentionRepairAudit:
    base = build_labeling_request(
        chunk_id=item.regulatory_chunk_id,
        text=item.text_snapshot,
        context=item.context_snapshot or "",
        taxonomy=fixture.taxonomy,
        source_hash=item.canonical_text_sha256 or "",
    )
    repair = build_abstention_repair(
        base_request=base,
        rejected_response=ABSTAINED,
        text=item.text_snapshot,
        chunk_id=item.regulatory_chunk_id,
        source_hash=item.canonical_text_sha256 or "",
        taxonomy=fixture.taxonomy,
        provider_binding=fixture.run.provider_binding,
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
    return build_abstention_repair_audit(
        output=output,
        repair=repair,
        output_uri="fixture://continued-abstention",
        text=item.text_snapshot,
        taxonomy=fixture.taxonomy,
    )


def apply_caption(session: Session, fixture: Fixture) -> RegulatoryLabelingItem:
    item = fixture.items[fixture.data.second_id]
    receipt = captions.build_source_only_chapter_caption_receipt(
        session,
        item=item,
        run=fixture.run,
        provider_abstention=continued_audit(item, fixture),
    )
    labeling._apply_frozen_window_results(
        session,
        fixture.run,
        items=list(fixture.items.values()),
        file_id=fixture.data.file_id,
        lower=0,
        upper=300,
        outcomes={
            i.request_hash or "": SourceOnlyChapterCaption(receipt)
            if i is item
            else ([], [])
            for i in fixture.items.values()
        },
    )
    session.flush()
    return item


@pytest.mark.parametrize("profile", [1, 2, 3, 4, 5, 6])
def test_actual_empty_reconsideration_apply_search_review_and_retry_keep_source_unchanged(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    profile: int,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch, profile=profile)
        item = fixture.items[fixture.data.second_id]
        require_current_intake_repair_item(session, item=item, run=fixture.run)
        before = publication_digest(
            [
                r.model_dump(mode="json")
                for r in capture_canonical_scope(session, fixture.data.file_id)
            ]
        )
        item = apply_caption(session, fixture)
        assert item.status == "completed" and item.labels == item.assignments == []
        receipt = SourceOnlyChapterCaptionReceipt.model_validate(
            item.source_snapshot[CHAPTER_CAPTION_KEY]
        )
        assert (
            receipt.classification == "abstained"
            and receipt.provider_abstention.outcome.abstained
        )
        assert labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=[item]
        ) == {item.regulatory_chunk_id}
        assert item.regulatory_chunk_id in complete_frozen_sources(
            session, fixture.run.id, [item], [item.regulatory_chunk_id]
        )
        authority = labeling_review_authority(session, fixture.run.id)
        assert authority is not None and authority["chapter_caption_dispositions"]
        page = labeling.list_items(
            session,
            document_set_id=fixture.data.document_set_id,
            run_id=fixture.run.id,
            offset=0,
            limit=20,
        )
        assert page is not None
        view = next(v for v in page.items if v.chunk_id == item.regulatory_chunk_id)
        assert (
            view.classification_disposition == "source_only_chapter_caption"
            and view.provider_abstained
        )
        _end(fixture.run, "completed")
        session.flush()
        replacement = _create(session, fixture.data)
        assert editing.reuse_completed_intake_labels(
            session, previous=fixture.run, replacement=replacement
        ) == (4 if profile == 4 else 3)
        assert (
            publication_digest(
                [
                    r.model_dump(mode="json")
                    for r in capture_canonical_scope(session, fixture.data.file_id)
                ]
            )
            == before
        )


@pytest.mark.parametrize(
    "change",
    [
        "raw",
        "context",
        "taxonomy",
        "provider",
        "labels",
        "receipt",
        "collision",
        "article",
        "scope",
        "request",
        "fact",
    ],
)
def test_paired_caption_authority_drift_fails_apply_current_search_and_review(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        item = apply_caption(session, fixture)
        if change == "raw":
            fixture.store.raw = RAW + b"changed"
        elif change == "context":
            item.context_snapshot = (item.context_snapshot or "") + "changed"
        elif change == "taxonomy":
            fixture.run.taxonomy_id = labeling_data.taxonomy_id
        elif change == "provider":
            fixture.run.provider_binding = {
                **fixture.run.provider_binding,
                "changed": True,
            }
        elif change == "labels":
            item.labels = ["duty"]
        elif change == "receipt":
            snapshot = deepcopy(item.source_snapshot)
            receipt_values = cast(dict[str, Any], snapshot[CHAPTER_CAPTION_KEY])
            proof_values = cast(dict[str, Any], receipt_values["proof"])
            proof_values["native_region_sha256"] = "0" * 64
            item.source_snapshot = snapshot
        elif change == "collision":
            item.source_snapshot = {**item.source_snapshot, SOURCE_ONLY_KEY: {}}
        elif change == "article":
            own = session.get_one(RegulatoryChunk, fixture.data.derived_id)
            own.text += "changed"
        elif change == "scope":
            fixture.run.document_set_id = fixture.data.other_document_set_id
        elif change == "request":
            item.request_payload = {**(item.request_payload or {}), "prompt": "changed"}
        elif change == "fact":
            own = session.get_one(RegulatoryChunk, fixture.data.derived_id)
            metadata = deepcopy(own.chunk_metadata)
            metadata["intake_legal_metadata"]["fields"]["article_title"][
                "evidence_scope"
            ] = "heading"
            own.chunk_metadata = metadata
        session.flush()
        with pytest.raises(ValueError):
            captions.validate_item_chapter_caption_receipt(
                session, item=item, run=fixture.run
            )
        assert not labeling.current_labeling_items_for_search(
            session, run=fixture.run, items=[item]
        )
        with pytest.raises(ValueError):
            labeling_review_authority(session, fixture.run.id)


def test_plain_base_abstention_or_error_string_is_not_a_source_only_receipt(
    labeling_data: LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed(session, labeling_data, monkeypatch)
        item = fixture.items[fixture.data.second_id]
        item.status, item.error = "failed", "Model abstained: insufficient evidence"
        _end(fixture.run)
        session.flush()
        replacement = _create(session, fixture.data)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=fixture.run, replacement=replacement
            )
            == 0
        )
        copied = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id == item.regulatory_chunk_id,
            )
        )
        assert (
            copied is not None
            and copied.status == "pending"
            and CHAPTER_CAPTION_KEY not in copied.source_snapshot
        )
