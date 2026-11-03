"""Real owned sources keep structural abstention separate from child authority."""

import json
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from typing import NamedTuple, cast
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake_edit as editing
from onyx.db import regulatory_labeling as labeling
from onyx.db import regulatory_labeling_article_lead_in as lead_in
from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryDerivedLabelProjection,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    UserFile,
)
from onyx.db.regulatory_annex_changes import capture_canonical_scope
from onyx.db.regulatory_labeling_review_authority import labeling_review_authority
from onyx.db.regulatory_labeling_source_only import (
    CONTINUED_ABSTENTION,
)
from onyx.document_index.publication_models import publication_digest
from onyx.file_store.file_store import FileStore
from onyx.regulatory.labeling.accepted_result import SourceOnlyArticleLeadIn
from onyx.regulatory.labeling.article_lead_in import (
    ARTICLE_LEAD_IN_KEY,
    SourceOnlyArticleLeadInReceipt,
)
from onyx.regulatory.labeling.intake_abstention_repair import (
    AbstentionRepairAudit,
    build_abstention_repair,
    build_abstention_repair_audit,
    parse_abstention_repair_output,
)
from onyx.regulatory.labeling.provider import TaxonomyDefinition, build_labeling_request
from onyx.regulatory.labeling.self_repeal import (
    SOURCE_ONLY_KEY,
)
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
from tests.unit.onyx.regulatory.labeling.test_native_word_self_repeal import RAW, rows

ABSTAINED = '{"labels":[],"abstained":true}'


class RetainedLeadInStore:
    def __init__(self) -> None:
        self.raw = RAW

    def read_file(self, _file_id: str) -> BytesIO:
        return BytesIO(self.raw)


class ArticleLeadInFixture(NamedTuple):
    data: LabelingData
    run: RegulatoryLabelingRun
    lease: labeling.RunLease
    items: dict[str, RegulatoryLabelingItem]
    store: RetainedLeadInStore
    taxonomy: TaxonomyDefinition
    clause_id: str


def seed_article_lead_in(
    session: Session, data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> ArticleLeadInFixture:
    file = session.get_one(UserFile, data.file_id)
    file.file_type, file.status = "text/html", UserFileStatus.CHUNKED
    seed_v4_owned_intake_source(
        session, data, raw_sha256=sha256(RAW).hexdigest(), raw_byte_count=len(RAW)
    )
    selected = [
        row for row in rows() if row.position in [113, 114, 115, 116, 117, 118, 367]
    ]
    ids = {row.id: str(uuid4()) for row in selected}
    ids[selected[0].id], ids[selected[1].id] = data.first_id, data.second_id
    ids[next(row.id for row in selected if row.position == 367)] = data.derived_id
    for identifier in [
        data.first_id,
        data.second_id,
        data.derived_id,
        data.incomplete_id,
    ]:
        session.delete(session.get_one(RegulatoryChunk, identifier))
    session.flush()
    for row in selected:
        metadata = deepcopy(row.metadata)
        source_ids = metadata.get("source_regulatory_chunk_ids", [])
        assert isinstance(source_ids, list)
        metadata["source_regulatory_chunk_ids"] = [
            ids[str(identifier)] for identifier in source_ids
        ]
        session.add(
            RegulatoryChunk(
                id=ids[row.id],
                user_file_id=data.file_id,
                text=row.text,
                position=row.position,
                projection_ordinal=row.projection_ordinal,
                chunk_type=row.chunk_type,
                heading_path=row.heading_path,
                chunk_metadata=metadata,
                source=row.source,
                status=row.status,
                validity_start_date=row.validity_start_date,
                validity_end_date=row.validity_end_date,
            )
        )
    taxonomy = TaxonomyDefinition.model_validate(
        {
            "name": f"Explicit municipal condition {uuid4().hex}",
            "labels": [
                {
                    "id": "condition",
                    "name": "Condition",
                    "description": "An explicit condition before a shared municipal predicate.",
                }
            ],
        }
    )
    tax = labeling.create_taxonomy(
        session, taxonomy=taxonomy, created_by_id=data.user_id
    )
    scoped = data._replace(taxonomy_id=tax.id)
    store = RetainedLeadInStore()
    monkeypatch.setattr(
        lead_in, "get_default_file_store", lambda: cast(FileStore, store)
    )
    session.flush()
    run = _create(session, scoped)
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
    return ArticleLeadInFixture(
        scoped,
        run,
        lease,
        {item.regulatory_chunk_id: item for item in items},
        store,
        taxonomy,
        data.second_id,
    )


def assign_article_child_labels(fixture: ArticleLeadInFixture) -> None:
    for item in fixture.items.values():
        item.status = "completed"
    child = fixture.items[fixture.clause_id]
    child.labels = ["condition"]
    assignment: dict[str, object] = {
        "label_id": "condition",
        "evidence_quote": "Danıştay tarafından feshi",
    }
    child.assignments = [assignment]


def recover_article_lead_in(
    session: Session, fixture: ArticleLeadInFixture
) -> tuple[RegulatoryLabelingRun, RegulatoryLabelingItem]:
    assign_article_child_labels(fixture)
    old = fixture.items[fixture.data.first_id]
    old.status, old.error = "failed", CONTINUED_ABSTENTION
    _end(fixture.run)
    session.flush()
    replacement = _create(session, fixture.data)
    assert (
        editing.reuse_completed_intake_labels(
            session, previous=fixture.run, replacement=replacement
        )
        == 6
    )
    copied = session.scalar(
        select(RegulatoryLabelingItem).where(
            RegulatoryLabelingItem.run_id == replacement.id,
            RegulatoryLabelingItem.regulatory_chunk_id == fixture.data.first_id,
        )
    )
    assert copied is not None
    return replacement, copied


def build_article_lead_in_audit(
    item: RegulatoryLabelingItem, fixture: ArticleLeadInFixture
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


def test_recovery_preserves_failed_run_source_and_actual_child_provenance(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed_article_lead_in(session, labeling_data, monkeypatch)
        before = publication_digest(
            [
                row.model_dump(mode="json")
                for row in capture_canonical_scope(session, fixture.data.file_id)
            ]
        )
        replacement, copied = recover_article_lead_in(session, fixture)
        old = fixture.items[fixture.data.first_id]
        old_state = (
            deepcopy(old.source_snapshot),
            old.status,
            old.error,
            old.request_hash,
        )
        receipt = SourceOnlyArticleLeadInReceipt.model_validate(
            copied.source_snapshot[ARTICLE_LEAD_IN_KEY]
        )
        assert copied.labels == copied.assignments == []
        assert (
            receipt.classification == "abstained"
            and receipt.prior_failed_abstention is not None
        )
        assert receipt.prior_failed_abstention.item_id == old.id
        authority = labeling_review_authority(session, replacement.id)
        assert authority is not None and authority["article_lead_in_dispositions"]
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
        assert projection is not None and projection.resolution == "lineage"
        assert projection.labels == ["condition"]
        provenance = projection.provenance["condition"]
        assert isinstance(provenance, list) and len(provenance) == 1
        assert isinstance(provenance[0], dict)
        entry = cast(dict[str, object], provenance[0])
        assert entry["canonical_chunk_id"] == fixture.clause_id
        assert fixture.data.first_id not in str(projection.provenance)
        assert labeling.final_status(session, lease) == "completed"
        _end(replacement, "completed")
        session.flush()
        following = _create(session, fixture.data)
        assert (
            editing.reuse_completed_intake_labels(
                session, previous=replacement, replacement=following
            )
            == 6
        )
        assert (
            old.source_snapshot,
            old.status,
            old.error,
            old.request_hash,
        ) == old_state
        assert (
            publication_digest(
                [
                    row.model_dump(mode="json")
                    for row in capture_canonical_scope(session, fixture.data.file_id)
                ]
            )
            == before
        )


def test_actual_provider_abstention_applies_without_claiming_provider_success(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed_article_lead_in(session, labeling_data, monkeypatch)
        intro = fixture.items[fixture.data.first_id]
        audit = build_article_lead_in_audit(intro, fixture)
        snapshot = deepcopy(intro.source_snapshot)
        intro.source_snapshot = {**snapshot, SOURCE_ONLY_KEY: {}}
        with pytest.raises(ValueError):
            lead_in.build_source_only_article_lead_in_receipt(
                session, item=intro, run=fixture.run, provider_abstention=audit
            )
        intro.source_snapshot = snapshot
        receipt = lead_in.build_source_only_article_lead_in_receipt(
            session, item=intro, run=fixture.run, provider_abstention=audit
        )
        labeling._apply_frozen_window_results(
            session,
            fixture.run,
            items=list(fixture.items.values()),
            file_id=fixture.data.file_id,
            lower=101,
            upper=130,
            outcomes={
                item.request_hash or "": SourceOnlyArticleLeadIn(receipt)
                if item is intro
                else ([], [])
                for item in fixture.items.values()
            },
        )
        assert intro.status == "completed" and intro.labels == intro.assignments == []
        stored = SourceOnlyArticleLeadInReceipt.model_validate(
            intro.source_snapshot[ARTICLE_LEAD_IN_KEY]
        )
        assert stored.provider_abstention == audit and audit.outcome.abstained
        view = labeling.list_items(
            session,
            document_set_id=fixture.data.document_set_id,
            run_id=fixture.run.id,
            offset=0,
            limit=20,
        )
        assert view is not None
        result = next(
            item for item in view.items if item.chunk_id == intro.regulatory_chunk_id
        )
        assert (
            result.classification_disposition == "source_only_article_lead_in"
            and result.provider_abstained
        )


@pytest.mark.parametrize("state", ["pending", "failed", "missing"])
def test_lead_in_does_not_resolve_incomplete_children(
    labeling_data: LabelingData, monkeypatch: pytest.MonkeyPatch, state: str
) -> None:
    with Session(labeling_data.database.engine) as session:
        fixture = seed_article_lead_in(session, labeling_data, monkeypatch)
        replacement, copied = recover_article_lead_in(session, fixture)
        child = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == replacement.id,
                RegulatoryLabelingItem.regulatory_chunk_id == fixture.clause_id,
            )
        )
        assert child is not None
        if state == "missing":
            session.delete(child)
        else:
            child.status, child.labels, child.assignments = state, [], []
        session.flush()
        lead_in.validate_item_article_lead_in_receipt(
            session, item=copied, run=replacement
        )
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
        assert labeling.final_status(session, lease) != "completed"
