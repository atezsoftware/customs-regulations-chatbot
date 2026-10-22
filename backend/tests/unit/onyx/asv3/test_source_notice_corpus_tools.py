"""Metadata-only notices stop original acquisition while retaining real candidates."""

from contextlib import contextmanager
from typing import Iterator, cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.models import OutcomeStatus, RunContext, ToolSpec
from onyx.asv3.source_notices import SourceNoticeState
from onyx.asv3.source_tools import build_source_specs
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.models import User
from onyx.regulatory.legistation_metadata import LegistationMetadata
from onyx.regulatory.source_notices import SourceNotice

NOTICE_ID = "3a6bd598-5027-4541-8816-02ae087eb768"
TARGET_ID = "notice:" + NOTICE_ID
SOURCE_ID = UUID("2c8ce7a1-879c-44e5-b93a-bdd1f51c569a")


def notice() -> SourceNotice:
    return SourceNotice(
        id=NOTICE_ID,
        document_set_id=73,
        identity_key="law:2908",
        instrument_type="law",
        instrument_number="2908",
        catalog_title="2908 sayılı Dernekler Kanunu",
        legistation_metadata=LegistationMetadata(
            validity_status="Mülga",
            repeal_note="Mülga listesinde yer alıyor; özgün metin henüz eklenmedi.",
        ),
        provenance={"workbook": "mülga liste 20.10.2026.xlsx", "row": 29},
    )


def setup() -> tuple[CorpusBroker, RunContext, MagicMock]:
    broker = CorpusBroker(cast(User, object()), IndexFilters(access_control_list=[]))
    state = MagicMock(spec=SourceNoticeState)
    state.lookup_queries.return_value = [notice()]
    state.by_id.return_value = notice()
    state.known_notice_id.side_effect = lambda value: value == NOTICE_ID
    context = RunContext(services={"source_notice_state": state})
    return broker, context, state


def tool(broker: CorpusBroker, name: str) -> ToolSpec:
    return next(
        spec
        for spec in [
            *build_corpus_specs(broker, named_provision_reads=True),
            *build_source_specs(broker),
        ]
        if spec.name == name
    )


def assert_notice(outcome: object) -> None:
    from onyx.asv3.models import ToolOutcome

    assert isinstance(outcome, ToolOutcome)
    assert outcome.status is OutcomeStatus.UNAVAILABLE
    assert outcome.evidence == []
    assert outcome.artifacts == []
    assert outcome.data["source_known"] is True
    assert outcome.data["content_available"] is False
    assert outcome.data["original_acquired"] is False
    assert outcome.data["absence_proven"] is False
    assert outcome.data["source_id"] == TARGET_ID
    metadata = outcome.data["legistation_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["validity_status"] == "Mülga"
    assert outcome.data["provenance"] == {
        "workbook": "mülga liste 20.10.2026.xlsx",
        "row": 29,
    }


@pytest.mark.parametrize(
    "name", ["resolve_source", "query_corpus", "read_named_provision"]
)
def test_named_notice_is_not_a_found_original(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    broker, context, state = setup()
    sources = MagicMock(return_value=([], False))
    monkeypatch.setattr(broker, "sources", sources)
    monkeypatch.setattr(
        broker, "page", MagicMock(side_effect=AssertionError("no read"))
    )
    args: dict[str, JsonValue] = (
        {"source_name": "2908 sayılı Dernekler Kanunu", "article": "1"}
        if name == "read_named_provision"
        else {"query": "2908 sayılı Dernekler Kanunu"}
    )
    result = tool(broker, name).handler(args, context)
    assert_notice(result)
    state.lookup_queries.assert_called_once_with(["2908 sayılı Dernekler Kanunu"])
    sources.assert_called_once()


@pytest.mark.parametrize("name", ["resolve_source", "read_named_provision"])
def test_real_source_and_notice_collision_preserve_both_candidates(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    broker, context, _ = setup()
    source = CorpusSource(SOURCE_ID, "Kanunlar/dernekler_kanunu_5253.md", "original")
    monkeypatch.setattr(broker, "sources", MagicMock(return_value=([source], False)))
    original_read = MagicMock(side_effect=AssertionError("ambiguous identity"))
    monkeypatch.setattr(broker, "source", original_read)
    args: dict[str, JsonValue] = (
        {"source_name": "Dernekler Kanunu", "article": "1"}
        if name == "read_named_provision"
        else {"query": "Dernekler Kanunu"}
    )
    result = tool(broker, name).handler(args, context)
    assert result.status is OutcomeStatus.AMBIGUOUS
    candidates = result.data["sources"]
    assert isinstance(candidates, list)
    assert {item["source_id"] for item in candidates if isinstance(item, dict)} == {
        str(SOURCE_ID),
        TARGET_ID,
    }
    assert result.evidence == []
    original_read.assert_not_called()


@pytest.mark.parametrize(
    "name,args",
    [
        ("read_source_range", {}),
        ("read_chunk", {"chunk_id": "nonexistent"}),
        ("read_chunk_context", {"chunk_id": "nonexistent"}),
        ("read_provision", {"article": "1"}),
        ("search_source_text", {"pattern": "x"}),
        ("query_corpus", {"operation": "headings"}),
        ("diagnose_source", {}),
        ("compare_versions", {"old_date": "2020-01-01", "new_date": "2026-01-01"}),
        ("open_source_file", {}),
        ("inspect_source_page", {"page": 1, "vision": True}),
        ("extract_table", {"page": 1}),
    ],
)
def test_notice_read_never_reaches_original_or_native_operations(
    monkeypatch: pytest.MonkeyPatch, name: str, args: dict[str, JsonValue]
) -> None:
    broker, context, state = setup()
    original = MagicMock(side_effect=AssertionError("original must not be acquired"))
    for method in ("source", "page", "chunk", "iter_chunks", "provision_start"):
        monkeypatch.setattr(broker, method, original)
    monkeypatch.setattr("onyx.asv3.source_tools.read_verified_original", original)
    result = tool(broker, name).handler({"source_id": TARGET_ID, **args}, context)
    assert_notice(result)
    state.by_id.assert_called_once_with(TARGET_ID)
    original.assert_not_called()


def test_repeated_notice_read_revalidates_current_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, context, state = setup()
    state.by_id.side_effect = [notice(), None]
    original = MagicMock(side_effect=AssertionError("original must not be acquired"))
    monkeypatch.setattr(broker, "page", original)
    spec = tool(broker, "read_source_range")
    assert_notice(spec.handler({"source_id": TARGET_ID}, context))
    second = spec.handler({"source_id": TARGET_ID}, context)
    assert second.status is OutcomeStatus.UNAVAILABLE
    assert "legistation_metadata" not in second.data
    assert state.by_id.call_count == 2
    original.assert_not_called()


def test_follow_reference_notice_target_keeps_origin_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, context, state = setup()
    source = CorpusSource(SOURCE_ID, "Origin", "original")
    chunk = CorpusChunk(
        "origin",
        SOURCE_ID,
        "2908 sayılı Kanunun 1 inci maddesi uyarınca",
        0,
        0,
        (),
        {},
        None,
        None,
        "active",
    )
    chunk_read = MagicMock(return_value=(source, chunk))
    monkeypatch.setattr(broker, "chunk", chunk_read)
    original = MagicMock(side_effect=AssertionError("target original unavailable"))
    monkeypatch.setattr(broker, "iter_chunks", original)
    result = tool(broker, "follow_reference").handler(
        {
            "source_id": str(SOURCE_ID),
            "chunk_id": "origin",
            "target_source_id": TARGET_ID,
            "article": "1",
        },
        context,
    )
    assert_notice(result)
    assert result.data["origin_chunk_id"] == "origin"
    chunk_read.assert_called_once_with(str(SOURCE_ID), "origin", context)
    state.by_id.assert_called_once_with(TARGET_ID)
    original.assert_not_called()


def test_unobserved_real_source_read_adds_no_notice_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, context, state = setup()
    source = CorpusSource(SOURCE_ID, "Origin", "original")
    page = MagicMock(return_value=(source, [], False))
    monkeypatch.setattr(broker, "page", page)
    result = tool(broker, "read_source_range").handler(
        {"source_id": str(SOURCE_ID)}, context
    )
    assert result.status is OutcomeStatus.NOT_FOUND
    state.by_id.assert_not_called()
    state.lookup_queries.assert_not_called()
    page.assert_called_once()


def test_notice_support_is_off_without_scoped_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, _, state = setup()
    sources = MagicMock(return_value=([], False))
    monkeypatch.setattr(broker, "sources", sources)
    result = tool(broker, "resolve_source").handler(
        {"query": "2908 sayılı Dernekler Kanunu"}, RunContext()
    )
    assert result.status is OutcomeStatus.NOT_FOUND
    state.lookup_queries.assert_not_called()


def test_broker_notice_lookup_reuses_user_filters_and_current_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, context, _ = setup()
    session = object()

    @contextmanager
    def scoped_session() -> Iterator[object]:
        yield session

    lookup = MagicMock(return_value=[notice()])
    monkeypatch.setattr(
        "onyx.asv3.corpus_tools.get_session_with_current_tenant", scoped_session
    )
    monkeypatch.setattr("onyx.asv3.corpus_tools.find_source_notices", lookup)
    assert broker.source_notices(["2908 sayılı Kanun"], None, context) == [notice()]
    lookup.assert_called_once_with(
        session,
        broker.user,
        broker.filters,
        queries=["2908 sayılı Kanun"],
        notice_ids=None,
        limit=100,
    )


@pytest.mark.parametrize("name", ["read_source_range", "open_source_file"])
def test_materialized_notice_routes_to_current_real_source_without_mutating_args(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    broker, context, state = setup()
    state.by_id.return_value = notice().model_copy(
        update={
            "content_state": "materialized",
            "user_file_id": str(SOURCE_ID),
            "revision": 2,
        }
    )
    if name == "read_source_range":
        source = CorpusSource(SOURCE_ID, "Original", "original")
        read = MagicMock(return_value=(source, [], False))
        monkeypatch.setattr(broker, "page", read)
    else:
        read = MagicMock(return_value=(b"Real original text", "text/plain", "Original"))
        monkeypatch.setattr("onyx.asv3.source_tools.read_verified_original", read)
    args: dict[str, JsonValue] = {"source_id": TARGET_ID}
    result = tool(broker, name).handler(args, context)
    assert result.status is (
        OutcomeStatus.NOT_FOUND if name == "read_source_range" else OutcomeStatus.FOUND
    )
    assert args == {"source_id": TARGET_ID}
    assert read.call_args.args[-2:] == (str(SOURCE_ID), context)
    if name == "open_source_file":
        assert result.evidence[0].source_id == str(SOURCE_ID)
        assert result.evidence[0].text == "Real original text"


def test_materialized_source_requires_ordinary_original_authorization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, context, state = setup()
    state.by_id.return_value = notice().model_copy(
        update={
            "content_state": "materialized",
            "user_file_id": str(SOURCE_ID),
            "revision": 2,
        }
    )
    page = MagicMock(side_effect=PermissionError("source access revoked"))
    monkeypatch.setattr(broker, "page", page)
    result = tool(broker, "read_source_range").handler(
        {"source_id": TARGET_ID}, context
    )
    assert result.status is OutcomeStatus.DENIED
    assert result.evidence == []
    assert "legistation_metadata" not in result.data
    page.assert_called_once_with(str(SOURCE_ID), context, start=0, limit=30)


def test_raw_observed_notice_is_stopped_by_shared_read_authority_fence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broker, context, state = setup()
    session = MagicMock(side_effect=AssertionError("not a canonical source"))
    monkeypatch.setattr(
        "onyx.asv3.corpus_tools.get_session_with_current_tenant", session
    )
    assert_notice(broker.shared_read_fence(NOTICE_ID, context))
    state.by_id.assert_called_once_with(NOTICE_ID)
    session.assert_not_called()


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),
        CorpusScopeUnavailable("narrower scope"),
    ],
)
def test_optional_notice_failure_keeps_real_candidates(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    broker, context, state = setup()
    source = CorpusSource(SOURCE_ID, "Original", "original")
    monkeypatch.setattr(broker, "sources", MagicMock(return_value=([source], False)))
    state.lookup_queries.side_effect = error
    result = tool(broker, "resolve_source").handler({"query": "Original"}, context)
    assert result.status is OutcomeStatus.FOUND
    assert result.data["sources"] == [{"source_id": str(SOURCE_ID), "name": "Original"}]
    assert "source_notices" not in result.data


def test_related_navigation_refreshes_notice_status_outside_real_candidate_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3.corpus_tools import evidence_for_chunk

    broker, context, state = setup()
    source = CorpusSource(SOURCE_ID, "Kanunlar/dernekler_kanunu.md", "original")
    chunk = CorpusChunk(
        "origin",
        SOURCE_ID,
        "Actual original",
        0,
        0,
        ("Dernekler Kanunu", "MADDE 1"),
        {"document_type": "kanun", "title": "Dernekler Kanunu"},
        None,
        None,
        "active",
    )
    real_candidates = MagicMock(return_value=([], False))
    monkeypatch.setattr(broker, "related_catalog_sources", real_candidates)
    state.lookup_queries.side_effect = [[notice()], []]
    first = broker.related_sources_for_provision(
        source,
        [evidence_for_chunk(source, chunk)],
        ("1", None),
        context,
    )
    second = broker.related_sources_for_provision(
        source,
        [evidence_for_chunk(source, chunk)],
        ("1", None),
        context,
    )
    assert first is not None and second is not None
    assert "source_notices" in first
    assert "source_notices" not in second
    assert first["candidates"] == second["candidates"] == []
    assert state.lookup_queries.call_count == 2
    real_candidates.assert_called_once()


@pytest.mark.parametrize("named_reads", [False, True])
def test_repeated_empty_instrument_provision_searches_skip_paid_retrieval(
    monkeypatch: pytest.MonkeyPatch, named_reads: bool
) -> None:
    from onyx.asv3.models import ToolOutcome

    broker, context, state = setup()
    monkeypatch.setattr(broker, "sources", MagicMock(return_value=([], False)))
    paid = MagicMock(
        return_value=ToolOutcome(status=OutcomeStatus.FOUND, summary="Indexed")
    )
    broker.search_adapter = paid
    search = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=named_reads)
        if spec.name == "search_corpus"
    )
    for article in ("1", "2", "99"):
        result = search.handler(
            {
                "query": f"2908 sayılı Dernekler Kanunu madde {article} şartları",
                "mode": "hybrid",
                "evidence_target": f"2908 sayılı Dernekler Kanunu madde {article} şartları",
            },
            context,
        )
        assert_notice(result)
        assert result.data["search_performed"] is False
        assert result.data["article"] == article
    paid.assert_not_called()
    assert state.lookup_queries.call_count == 3


@pytest.mark.parametrize("named_reads", [False, True])
@pytest.mark.parametrize(
    "args",
    [
        {
            "query": "Derneklerin hakları ve tüzel kişilik koşulları",
            "evidence_target": "Derneklerin kuruluş koşulları ve güncel uygulama",
        },
        {
            "query": "2908 sayılı Kanunu kaldıran düzenleme",
            "evidence_target": "2908 sayılı Dernekler Kanunu madde 1 şartları",
        },
        {
            "query": "5253 sayılı Dernekler Kanunu şartları",
            "evidence_target": "2908 sayılı Dernekler Kanunu madde 1 şartları",
        },
        {
            "query": "2908 sayılı Kanun m.1 ile ilgili mahkeme kararları",
            "evidence_target": "2908 sayılı Dernekler Kanunu madde 1 şartları",
            "discover_related_sources": True,
        },
    ],
)
def test_empty_notice_preserves_broad_repealing_and_other_instrument_searches(
    monkeypatch: pytest.MonkeyPatch,
    named_reads: bool,
    args: dict[str, JsonValue],
) -> None:
    from onyx.asv3.models import ToolOutcome

    broker, context, state = setup()
    lookup = MagicMock(side_effect=AssertionError("unrelated search must stay open"))
    monkeypatch.setattr(broker, "sources", lookup)
    paid = MagicMock(
        return_value=ToolOutcome(status=OutcomeStatus.FOUND, summary="Indexed")
    )
    broker.search_adapter = paid
    search = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=named_reads)
        if spec.name == "search_corpus"
    )
    actual_args = {"mode": "hybrid", **args}
    result = search.handler(actual_args, context)
    assert result.status is OutcomeStatus.FOUND
    paid.assert_called_once_with(actual_args, context)
    state.lookup_queries.assert_not_called()
    lookup.assert_not_called()


def test_existing_original_does_not_get_hidden_by_unlinked_metadata_notice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3.models import ToolOutcome

    broker, context, _ = setup()
    source = CorpusSource(SOURCE_ID, "Original", "original")
    monkeypatch.setattr(broker, "sources", MagicMock(return_value=([source], False)))
    paid = MagicMock(
        return_value=ToolOutcome(status=OutcomeStatus.FOUND, summary="Indexed")
    )
    broker.search_adapter = paid
    search = next(
        spec for spec in build_corpus_specs(broker) if spec.name == "search_corpus"
    )
    result = search.handler(
        {
            "query": "2908 sayılı Dernekler Kanunu madde 1 şartları",
            "mode": "hybrid",
            "evidence_target": "2908 sayılı Dernekler Kanunu madde 1 şartları",
        },
        context,
    )
    assert result.status is OutcomeStatus.FOUND
    paid.assert_called_once()
