"""Source-identity guidance cannot rewrite lookup intent or acquisition fences."""

import copy
from datetime import date
from typing import cast
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.models import OutcomeStatus, RunContext, ToolSpec
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusScopeUnavailable, CorpusSource
from onyx.db.models import User


@pytest.fixture
def lookup_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[CorpusBroker, MagicMock]:
    broker = CorpusBroker(
        cast(User, object()),
        IndexFilters(
            access_control_list=["user:captured-owner"],
            asv3_document_set_id=73,
            document_set=["Captured source subset"],
            forced_document_set=["PC Külliyatı"],
            project_id_filter=9,
            as_of_date=date(2026, 5, 18),
        ),
        allow_numbered_title_fallback=True,
    )
    lookup = MagicMock(return_value=([], False))
    monkeypatch.setattr(broker, "sources", lookup)
    return broker, lookup


def source(name: str = "Kanunlar/faaliyet_kanunu_2024.md") -> CorpusSource:
    identifier = UUID("12345678-1234-4234-8234-123456789abc")
    return CorpusSource(identifier, name, str(identifier))


def context(profile: str = "experimental", parallel: object = True) -> RunContext:
    return RunContext(
        scope={"source_date": "2026-05-18", "captured_owner": "user:captured-owner"},
        services={"research_profile": profile, "experimental_parallel": parallel},
    )


def tool(broker: CorpusBroker, name: str, *, guidance: bool = True) -> ToolSpec:
    return next(
        item
        for item in build_corpus_specs(broker, source_identity_guidance=guidance)
        if item.name == name
    )


def without_descriptions(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {
            key: without_descriptions(child)
            for key, child in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [without_descriptions(child) for child in value]
    return value


def test_guidance_is_default_off_and_changes_only_lookup_descriptions(
    lookup_broker: tuple[CorpusBroker, MagicMock],
) -> None:
    broker, lookup = lookup_broker
    baseline = {item.name: item.definition() for item in build_corpus_specs(broker)}
    explicit_off = {
        item.name: item.definition()
        for item in build_corpus_specs(broker, source_identity_guidance=False)
    }
    enabled = {
        item.name: item.definition()
        for item in build_corpus_specs(broker, source_identity_guidance=True)
    }
    assert baseline == explicit_off
    assert baseline.keys() == enabled.keys()
    changed = {name for name in baseline if baseline[name] != enabled[name]}
    assert changed == {"resolve_source", "query_corpus"}
    for name in baseline:
        assert without_descriptions(baseline[name]) == without_descriptions(
            enabled[name]
        )
    lookup.assert_not_called()


def test_opted_in_lookup_schemas_keep_vertex_provider_shape(
    lookup_broker: tuple[CorpusBroker, MagicMock],
) -> None:
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    broker, lookup = lookup_broker
    for name in ("resolve_source", "query_corpus"):
        original = _build_vertex_schema(tool(broker, name, guidance=False).parameters)
        guided = _build_vertex_schema(tool(broker, name).parameters)
        assert without_descriptions(original) == without_descriptions(guided)
    lookup.assert_not_called()


@pytest.mark.parametrize("name", ["resolve_source", "query_corpus"])
def test_no_match_keeps_exact_query_scope_status_pagination_and_one_lookup(
    lookup_broker: tuple[CorpusBroker, MagicMock], name: str
) -> None:
    broker, lookup = lookup_broker
    run = context()
    saved_filters, saved_scope = copy.deepcopy(broker.filters), copy.deepcopy(run.scope)
    query = (
        " 8237 sayılı Faaliyet Kanunu ve Başka Yönetmelik m.17\n"
        "işlem koşulu 2024 sürümü güncel Türkiye  "
    )
    args: dict[str, JsonValue] = {"query": query, "offset": 7, "limit": 13}
    if name == "query_corpus":
        args["operation"] = "inventory"
    saved_args = copy.deepcopy(args)
    result = tool(broker, name).handler(args, run)
    lookup.assert_called_once_with(query, run, offset=7, limit=13)
    assert result.status is OutcomeStatus.NOT_FOUND
    assert result.evidence == []
    assert result.data["sources"] == []
    assert result.data["has_more"] is False
    assert result.data["next_offset"] == 20
    assert result.data["absence_proven"] is False
    diagnostic = result.data["lookup_diagnostic"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["code"] == "source_identity_no_match"
    assert diagnostic["query_preserved"] is True
    assert args == saved_args
    assert broker.filters == saved_filters
    assert run.scope == saved_scope


@pytest.mark.parametrize(
    "guidance,profile,parallel,expected_hint",
    [
        (True, "experimental", True, True),
        (False, "experimental", True, False),
        (True, "experimental", False, False),
        (True, "normal", True, False),
        (True, "deep", True, False),
        (True, "experimental", "true", False),
        (True, "experimental", 1, False),
    ],
)
def test_hint_requires_explicit_opt_in_and_both_exact_parallel_profile_flags(
    lookup_broker: tuple[CorpusBroker, MagicMock],
    guidance: bool,
    profile: str,
    parallel: object,
    expected_hint: bool,
) -> None:
    broker, lookup = lookup_broker
    result = tool(broker, "resolve_source", guidance=guidance).handler(
        {"query": "Faaliyet Kanunu"}, context(profile, parallel)
    )
    assert ("lookup_diagnostic" in result.data) is expected_hint
    assert result.status is OutcomeStatus.NOT_FOUND
    lookup.assert_called_once()


@pytest.mark.parametrize(
    "sources,more,expected_status",
    [
        ([source()], False, OutcomeStatus.FOUND),
        ([source(), source("Başka tam başlık")], False, OutcomeStatus.AMBIGUOUS),
        ([source()], True, OutcomeStatus.PARTIAL),
        ([], True, OutcomeStatus.PARTIAL),
    ],
)
def test_candidates_and_partial_pages_are_not_reclassified_or_changed(
    lookup_broker: tuple[CorpusBroker, MagicMock],
    sources: list[CorpusSource],
    more: bool,
    expected_status: OutcomeStatus,
) -> None:
    broker, lookup = lookup_broker
    lookup.return_value = (sources, more)
    query = "Faaliyet Kanunu 8237 m.17 güncel metin ve bütün değişiklikler"
    result = tool(broker, "resolve_source").handler({"query": query}, context())
    lookup.assert_called_once()
    assert lookup.call_args.args[0] == query
    assert result.status is expected_status
    assert result.data["sources"] == [
        {"source_id": str(item.id), "name": item.name} for item in sources
    ]
    assert result.data["has_more"] is more
    assert result.data["next_offset"] == 20
    assert result.data["absence_proven"] is False
    assert "lookup_diagnostic" not in result.data


@pytest.mark.parametrize(
    "error,status",
    [
        (PermissionError("outside captured scope"), OutcomeStatus.DENIED),
        (
            CorpusScopeUnavailable("captured index unavailable"),
            OutcomeStatus.UNAVAILABLE,
        ),
        (ValueError("invalid offset"), OutcomeStatus.INVALID),
    ],
)
def test_existing_acquisition_errors_do_not_become_identity_hints(
    lookup_broker: tuple[CorpusBroker, MagicMock],
    error: Exception,
    status: OutcomeStatus,
) -> None:
    broker, lookup = lookup_broker
    lookup.side_effect = error
    result = tool(broker, "resolve_source").handler(
        {"query": "Faaliyet Kanunu"}, context()
    )
    assert result.status is status
    assert "lookup_diagnostic" not in result.data
    lookup.assert_called_once()


def test_empty_inventory_still_queries_captured_scope_once(
    lookup_broker: tuple[CorpusBroker, MagicMock],
) -> None:
    broker, lookup = lookup_broker
    lookup.return_value = ([source()], False)
    run = context()
    result = tool(broker, "query_corpus").handler({"operation": "inventory"}, run)
    lookup.assert_called_once_with("", run, offset=0, limit=20)
    assert result.status is OutcomeStatus.FOUND
    assert "lookup_diagnostic" not in result.data
