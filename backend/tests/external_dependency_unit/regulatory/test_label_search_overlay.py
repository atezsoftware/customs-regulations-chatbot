"""Read-only label retrieval against an owned PostgreSQL schema."""

from collections.abc import Generator, Sequence
from datetime import date, datetime, timezone
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.db import regulatory_label_search as search
from onyx.db import regulatory_labeling as labeling
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryLabelingItem,
    RegulatoryLabelingRun,
    RegulatoryLabelSettings,
)
from onyx.regulatory.labeling.search_models import LabelSearchSnapshot
from tests.external_dependency_unit.regulatory import test_labeling_jobs as fixtures

pytest_plugins = ("tests.external_dependency_unit.regulatory.test_labeling_jobs",)


def test_lookup_latency_does_not_starve_context_validation(
    label_search_data: fixtures.LabelingData, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = label_search_data
    run_id = finish(data)
    clock = [0.0]
    original = search._select_items

    def delayed_selection(
        session: Session,
        snapshot: LabelSearchSnapshot,
        chunk_ids: Sequence[str],
        labels: Sequence[str],
        candidate_chunk_ids: Sequence[str] | None = None,
    ) -> list[RegulatoryLabelingItem]:
        result = original(session, snapshot, chunk_ids, labels, candidate_chunk_ids)
        clock[0] += 2.0
        return result

    monkeypatch.setattr(search, "_select_items", delayed_selection)
    monkeypatch.setattr(search, "monotonic", lambda: clock[0])
    monkeypatch.setattr(labeling, "monotonic", lambda: clock[0])
    with Session(data.database.engine) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        overlay = search.load_label_overlay(
            session,
            snapshot=snapshot,
            chunk_ids=(data.first_id,),
            candidate_label_ids=("origin",),
            candidate_chunk_ids=(data.second_id,),
            as_of_date=date.today(),
        )
        assert data.first_id in overlay.evidence_by_chunk
        assert data.second_id in overlay.candidate_ids


@pytest.fixture
def label_search_data(
    labeling_data: fixtures.LabelingData,
) -> Generator[fixtures.LabelingData, None, None]:
    data = labeling_data
    with Session(data.database.engine) as session:
        settings = session.get(RegulatoryLabelSettings, 1)
        assert settings is not None
        original = settings.taxonomy_id
        settings.taxonomy_id = data.taxonomy_id
        session.commit()
    yield data
    with Session(data.database.engine) as session:
        settings = session.get(RegulatoryLabelSettings, 1)
        assert settings is not None
        settings.taxonomy_id = original
        session.commit()


def finish(data: fixtures.LabelingData) -> UUID:
    run_id, _ = fixtures._start(data)
    lease, shard_id = fixtures._prepare(data, run_id)
    with Session(data.database.engine) as session:
        labeling.apply_shard_results(
            session,
            lease,
            shard_id=shard_id,
            outcomes=fixtures._outcomes(session, lease, shard_id),
        )
        labeling.project_derived_labels(session, lease)
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        run.status = "completed_with_errors"
        run.stage = "finished"
        run.finished_at = datetime.now(timezone.utc)
        session.commit()
    return run_id


def test_overlay_validates_context_and_never_writes(
    label_search_data: fixtures.LabelingData,
) -> None:
    data = label_search_data
    run_id = finish(data)
    with Session(data.database.engine) as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        overlay = search.load_label_overlay(
            session,
            snapshot=snapshot,
            chunk_ids=(data.first_id, data.derived_id),
            candidate_label_ids=("origin",),
            as_of_date=date.today(),
        )
        assert overlay.evidence_by_chunk[data.first_id][0].label_id == "customs_value"
        assert data.second_id in overlay.candidate_ids
        assert {
            e.source_chunk_id for e in overlay.evidence_by_chunk[data.derived_id]
        } == {data.first_id, data.second_id}
        assert not session.new and not session.dirty and not session.deleted
    with Session(data.database.engine) as session:
        second = session.get(RegulatoryChunk, data.second_id)
        assert second is not None
        second.text += " Context has changed."
        session.commit()
    with Session(data.database.engine) as session:
        overlay = search.load_label_overlay(
            session,
            snapshot=snapshot,
            chunk_ids=(data.first_id, data.derived_id),
            candidate_label_ids=("origin",),
            as_of_date=date.today(),
        )
        assert overlay.evidence_by_chunk == {}
        assert not overlay.candidate_ids


def test_latest_run_is_selected_before_label_match(
    label_search_data: fixtures.LabelingData,
) -> None:
    data = label_search_data
    old = finish(data)
    new = finish(data)
    with Session(data.database.engine) as session:
        item = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == new,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert item is not None
        item.labels = []
        item.assignments = []
        session.commit()
    with Session(data.database.engine) as session:
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(old, new), mode="hybrid"
        )
        assert snapshot is not None
        overlay = search.load_label_overlay(
            session,
            snapshot=snapshot,
            chunk_ids=(data.first_id,),
            candidate_label_ids=("customs_value",),
            as_of_date=date.today(),
        )
        assert data.first_id not in overlay.candidate_ids
        assert overlay.evidence_by_chunk.get(data.first_id, ()) == ()


def test_unknown_run_and_nonterminal_run_do_not_activate(
    label_search_data: fixtures.LabelingData,
) -> None:
    data = label_search_data
    pending, _ = fixtures._start(data)
    with Session(data.database.engine) as session:
        assert (
            search.load_search_snapshot(
                session,
                tenant_id=data.database.schema,
                run_ids=(pending,),
                mode="hybrid",
            )
            is None
        )
        assert (
            search.load_search_snapshot(
                session,
                tenant_id=data.database.schema,
                run_ids=(uuid4(),),
                mode="hybrid",
            )
            is None
        )


def test_full_baseline_keeps_separate_candidate_budget(
    label_search_data: fixtures.LabelingData,
) -> None:
    data = label_search_data
    run_id = finish(data)
    identifiers = [data.first_id]
    with Session(data.database.engine) as session:
        original = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert original is not None
        for position in range(10, 73):
            identifier = str(uuid4())
            identifiers.append(identifier)
            session.add(
                RegulatoryChunk(
                    id=identifier,
                    user_file_id=data.file_id,
                    text="test source",
                    position=position,
                    projection_ordinal=position,
                    chunk_type="article",
                    source="indexed",
                    status="active",
                )
            )
            values = {
                column.key: getattr(original, column.key)
                for column in RegulatoryLabelingItem.__table__.columns
                if column.key not in {"id", "regulatory_chunk_id"}
            }
            session.add(
                RegulatoryLabelingItem(**values, regulatory_chunk_id=identifier)
            )
        session.commit()
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        rows = search._select_items(session, snapshot, identifiers, ("origin",))
        assert len(identifiers) == 64
        assert {row.regulatory_chunk_id for row in rows} == {
            *identifiers,
            data.second_id,
        }


def test_taxonomy_change_and_cross_tenant_snapshot_fail_closed(
    label_search_data: fixtures.LabelingData,
) -> None:
    data = label_search_data
    run_id = finish(data)
    with Session(data.database.engine) as session:
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        wrong_tenant = snapshot.model_copy(update={"tenant_id": "different-tenant"})
        assert (
            search.load_label_overlay(
                session,
                snapshot=wrong_tenant,
                chunk_ids=(data.first_id,),
                candidate_label_ids=("origin",),
                as_of_date=date.today(),
            ).evidence_by_chunk
            == {}
        )
        run = session.get(RegulatoryLabelingRun, run_id)
        assert run is not None
        run.taxonomy.version_hash = "0" * 64
        session.flush()
        assert (
            search.load_search_snapshot(
                session,
                tenant_id=data.database.schema,
                run_ids=(run_id,),
                mode="hybrid",
            )
            is None
        )
        session.rollback()


def test_matching_baseline_does_not_consume_overlay_additions(
    label_search_data: fixtures.LabelingData,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = label_search_data
    identifiers: list[str] = []
    with Session(data.database.engine) as session:
        for position in range(10, 42):
            identifier = str(uuid4())
            identifiers.append(identifier)
            session.add(
                RegulatoryChunk(
                    id=identifier,
                    user_file_id=data.file_id,
                    text=fixtures._FIRST_TEXT,
                    position=position,
                    projection_ordinal=position,
                    heading_path=["Customs regulation", f"Article {position}"],
                    chunk_type="article",
                    chunk_metadata={"chunk_variant": "atomic"},
                    source="indexed",
                    status="active",
                )
            )
        session.commit()
    prepare = labeling.prepare_next_item_page

    def prepare_all(
        session: Session, lease: labeling.RunLease, *, limit: int
    ) -> list[RegulatoryLabelingItem]:
        return prepare(session, lease, limit=max(limit, 128))

    monkeypatch.setattr(labeling, "prepare_next_item_page", prepare_all)
    run_id = finish(data)
    baseline = [data.first_id, *identifiers[:-1]]
    with Session(data.database.engine) as session:
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        overlay = search.load_label_overlay(
            session,
            snapshot=snapshot,
            chunk_ids=baseline,
            candidate_label_ids=("customs_value",),
            as_of_date=date.today(),
        )
        assert (
            len(
                [
                    identifier
                    for identifier in baseline
                    if overlay.evidence_by_chunk.get(identifier)
                ]
            )
            == 32
        )
        assert overlay.candidate_ids == (identifiers[-1],)


@pytest.mark.parametrize("include_candidate", [True, False])
def test_candidate_discovery_is_intersected_with_current_labels(
    label_search_data: fixtures.LabelingData, include_candidate: bool
) -> None:
    data = label_search_data
    run_id = finish(data)
    with Session(data.database.engine) as session:
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        overlay = search.load_label_overlay(
            session,
            snapshot=snapshot,
            chunk_ids=(data.first_id,),
            candidate_label_ids=("origin",),
            candidate_chunk_ids=(data.second_id,) if include_candidate else (),
            as_of_date=date.today(),
        )
        assert overlay.candidate_ids == ((data.second_id,) if include_candidate else ())
        assert data.first_id in overlay.evidence_by_chunk


def test_relevant_extra_is_validated_before_remaining_baseline_windows(
    label_search_data: fixtures.LabelingData,
) -> None:
    data = label_search_data
    run_id = finish(data)
    with Session(data.database.engine) as session:
        snapshot = search.load_search_snapshot(
            session, tenant_id=data.database.schema, run_ids=(run_id,), mode="hybrid"
        )
        assert snapshot is not None
        # The fixture's two canonical items plus an unrelated completed item model
        # separate windows competing for the same cooperative validation deadline.
        first = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.first_id,
            )
        )
        assert first is not None
        extra = session.scalar(
            select(RegulatoryLabelingItem).where(
                RegulatoryLabelingItem.run_id == run_id,
                RegulatoryLabelingItem.regulatory_chunk_id == data.second_id,
            )
        )
        assert extra is not None
        third = RegulatoryLabelingItem(
            run_id=run_id,
            regulatory_chunk_id="zz-last-baseline",
            user_file_id=first.user_file_id,
            status="completed",
            source_snapshot=first.source_snapshot,
            text_snapshot=first.text_snapshot,
            labels=first.labels,
            assignments=first.assignments,
        )
        session.add(third)
        session.flush()
        items = search._select_items(
            session,
            snapshot,
            (data.first_id, "zz-last-baseline"),
            ("origin",),
            (data.second_id,),
        )
        assert [item.regulatory_chunk_id for item in items] == [
            data.first_id,
            data.second_id,
            "zz-last-baseline",
        ]
        session.rollback()
