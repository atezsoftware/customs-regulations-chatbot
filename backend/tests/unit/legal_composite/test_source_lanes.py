from contextlib import contextmanager
from datetime import date
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusScopeUnavailable, CorpusSource
from onyx.db.legal_composite_sources import (
    SourceKind,
    SourceLaneCatalogue,
    classify_source,
    source_scope_sha256,
)
from onyx.db.models import User
from onyx.document_index.elasticsearch.search import DocumentQuery
from onyx.document_index.interfaces_new import TenantState
from onyx.legal_composite import source_lanes
from onyx.legal_composite.source_lanes import SourceLaneBroker


def fixture_lane() -> tuple[CorpusBroker, SourceLaneCatalogue, CorpusSource]:
    source = CorpusSource(uuid4(), "kanun.md", "file")
    user = cast(User, SimpleNamespace(id=uuid4()))
    filters = IndexFilters(
        access_control_list=["user"],
        tenant_id="public",
        asv3_document_set_id=15,
        document_set=["PC Külliyatı"],
        forced_document_set=["PC Külliyatı"],
        attached_document_ids=[str(uuid4())],
        persona_id_filter=4,
        project_id_filter=7,
        regulatory_chunks_only=True,
        source_type=[DocumentSource.USER_FILE],
        as_of_date=date(2022, 1, 1),
    )
    base = CorpusBroker(user, filters)
    record = classify_source(
        source, ("kanun",), opening_texts=("GÜMRÜK KANUNU\nMADDE 1- Amaç.",)
    )
    catalogue = SourceLaneCatalogue(
        user_id=user.id,
        scope_sha256=source_scope_sha256(user, filters),
        records=(record,),
        complete=True,
    )
    return base, catalogue, source


def test_before_top_k_index_filter_has_only_lane_ids_with_pinned_pc_and_acl() -> None:
    base, catalogue, source = fixture_lane()
    original = base.filters.model_dump()
    broker = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)
    filters = broker.filters
    clauses = DocumentQuery._get_search_filters(
        tenant_state=TenantState(tenant_id="public", multitenant=True),
        include_hidden=False,
        access_control_list=filters.access_control_list,
        source_types=filters.source_type or [],
        tags=[],
        document_sets=filters.document_set or [],
        project_id_filter=filters.project_id_filter,
        persona_id_filter=filters.persona_id_filter,
        created_at_range=None,
        updated_at_range=None,
        min_chunk_index=None,
        max_chunk_index=None,
        as_of_date=filters.as_of_date,
        regulatory_chunks_only=True,
        attached_document_ids=filters.attached_document_ids,
        hierarchy_node_ids=filters.hierarchy_node_ids,
        forced_document_sets=filters.forced_document_set,
    )
    knowledge = next(
        clause["bool"]
        for clause in clauses
        if "bool" in clause
        and any(
            "document_id" in item.get("terms", {})
            for item in clause["bool"].get("should", [])
        )
    )
    assert knowledge == {
        "should": [{"terms": {"document_id": [str(source.id)]}}],
        "minimum_should_match": 1,
    }
    assert any(
        clause.get("terms", {}).get("document_sets") == ["PC Külliyatı"]
        for clause in clauses
    )
    assert base.filters.model_dump() == original
    assert filters.tenant_id == "public" and filters.access_control_list == ["user"]
    assert filters.asv3_document_set_id == 15 and filters.as_of_date == date(2022, 1, 1)


def test_empty_lane_fails_before_any_broad_search() -> None:
    base, catalogue, _source = fixture_lane()
    with pytest.raises(CorpusScopeUnavailable, match="no authorized sources"):
        SourceLaneBroker(base, catalogue, SourceKind.CIRCULAR)


def test_inventory_cannot_be_rebound_to_different_owner_or_scope() -> None:
    base, catalogue, _source = fixture_lane()
    with pytest.raises(PermissionError, match="authorized run scope"):
        SourceLaneBroker(
            base, catalogue.model_copy(update={"user_id": uuid4()}), SourceKind.STATUTE
        )
    base.filters.as_of_date = date(2023, 1, 1)
    with pytest.raises(PermissionError, match="authorized run scope"):
        SourceLaneBroker(base, catalogue, SourceKind.STATUTE)


def test_direct_read_of_another_kind_is_denied_before_db_or_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, catalogue, _source = fixture_lane()
    broker = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)
    session = MagicMock()
    monkeypatch.setattr(source_lanes, "get_session_with_current_tenant", session)
    with pytest.raises(PermissionError, match="outside"):
        broker.source(str(uuid4()), RunContext(timeout_seconds=10))
    session.assert_not_called()


def test_canonical_recheck_keeps_original_project_and_document_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, catalogue, source = fixture_lane()
    broker = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)

    @contextmanager
    def session():
        yield object()

    def revalidate(_session: object, **kwargs: object) -> CorpusSource:
        filters = cast(IndexFilters, kwargs["filters"])
        assert filters.project_id_filter == 7 and filters.persona_id_filter == 4
        assert filters.document_set == ["PC Külliyatı"]
        assert kwargs["recorded"] == catalogue.records[0]
        return source

    monkeypatch.setattr(source_lanes, "get_session_with_current_tenant", session)
    monkeypatch.setattr(source_lanes, "revalidate_source_classification", revalidate)
    assert broker.source(str(source.id), RunContext(timeout_seconds=10)) == source


def test_guarded_search_refuses_scope_mutation_before_invocation() -> None:
    base, catalogue, _source = fixture_lane()
    broker = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)
    adapter = MagicMock(
        return_value=ToolOutcome(status=OutcomeStatus.NOT_FOUND, summary="No matches")
    )
    search = broker.guard_search_adapter(adapter)
    broker.filters.document_set = ["PC Külliyatı"]
    with pytest.raises(PermissionError, match="filters changed"):
        search({"query": "question"}, RunContext(timeout_seconds=10))
    adapter.assert_not_called()


def test_guarded_search_preserves_full_receipt_and_explicit_classification_provenance() -> (
    None
):
    base, catalogue, _source = fixture_lane()
    broker = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)
    outcome = ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Continue",
        data={"next_position": 12, "has_more": True},
    )
    result = broker.guard_search_adapter(lambda _args, _context: outcome)(
        {}, RunContext(timeout_seconds=10)
    )
    assert result.data["next_position"] == 12 and result.data["has_more"] is True
    assert result.data["source_lane"] == broker.lane_provenance()


def test_wrong_chunk_metadata_does_not_reject_the_verified_original_kind() -> None:
    base, catalogue, source = fixture_lane()
    broker = SourceLaneBroker(base, catalogue, SourceKind.STATUTE)
    broker._check_original_kind(str(source.id), {"document_type": "yonetmelik"})
