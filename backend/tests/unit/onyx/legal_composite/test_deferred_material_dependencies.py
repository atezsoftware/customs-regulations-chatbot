"""Deferred source work remains a canonical obligation, never completed research."""

from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.dependencies import (
    CompositeDependencyExpander,
    DependencyExpander,
    material_dependency_gaps,
)
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    material_target,
    plan,
    setup_composite_expander,
)

DEFERRED_GAP = "Material dependency acquisition was deferred before examination."


def fixture() -> tuple[
    CompositeDependencyExpander,
    EvidenceLedger,
    list[tuple[str, dict[str, JsonValue]]],
    Mock,
]:
    expander, ledger, calls, broker, *_ = setup_composite_expander()
    return cast(CompositeDependencyExpander, expander), ledger, calls, broker


def test_registration_preserves_material_obligation_without_source_work() -> None:
    expander, ledger, calls, broker = fixture()
    original = ledger.get(1)
    assert original is not None
    before = original.model_dump(mode="json")
    budget_before = expander.context.budget.snapshot()
    edges = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )
    assert len(edges) == 1
    edge = edges[0]
    assert edge.need_ids == ["permit"]
    assert edge.origins[0].citation == 1
    assert edge.origins[0].text_hash == original.text_hash
    assert edge.discovery_gaps == [DEFERRED_GAP]
    assert material_dependency_gaps(edges, ledger, {1})[edge.edge_id] == [
        "governing_original_unread",
        "discovery_gap_unresolved",
    ]
    assert expander._expanded == {} and expander.receipts == []
    assert calls == [] and broker.mock_calls == []
    assert expander.acquirer.search_calls == 0
    assert expander.context.budget.snapshot() == budget_before
    assert ledger.citation_numbers() == (1,)
    retained = ledger.get(1)
    assert retained is not None and retained.model_dump(mode="json") == before


def test_registration_never_treats_retained_governing_original_as_examined() -> None:
    expander, ledger, calls, broker = fixture()
    source, chunk = broker.chunk.return_value
    ledger.add([evidence_for_chunk(source, chunk)], expander.context)
    edges = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )
    edge = edges[0]
    assert edge.governing_citations == [2]
    assert material_dependency_gaps(edges, ledger, {1, 2})[edge.edge_id] == [
        "discovery_gap_unresolved"
    ]
    assert DEFERRED_GAP in edge.discovery_gaps
    assert expander._expanded == {} and calls == [] and broker.mock_calls == []


def test_repeated_registration_retains_prior_gaps_and_unread_candidates() -> None:
    expander, ledger, calls, broker = fixture()
    edge = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )[0]
    edge.discovery_gaps.append("A prior bounded acquisition was incomplete.")
    edge.discovery_limits.append("A retained candidate page has a continuation.")
    expander._candidate_edges["unread-candidate"] = {edge.edge_id}
    edges = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )
    assert len(edges) == 1 and len(edge.origins) == 1
    assert edge.discovery_gaps == [
        DEFERRED_GAP,
        "A prior bounded acquisition was incomplete.",
    ]
    assert edge.incomplete_source_ids == ["unread-candidate"]
    assert edge.discovery_limits == ["A retained candidate page has a continuation."]
    assert (
        "candidate_original_missing@unread-candidate"
        in material_dependency_gaps(edges, ledger, {1})[edge.edge_id]
    )
    assert expander._expanded == {} and calls == [] and broker.mock_calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"instrument_name": "Invented Governing Instrument"},
        {"article": "118"},
        {"need_ids": ["unknown"]},
        {"need_ids": ["permit", "permit"]},
        {"reason": " "},
    ],
)
def test_any_invalid_target_rejects_entire_registration_atomically(
    changes: dict[str, JsonValue],
) -> None:
    expander, ledger, calls, broker = fixture()
    before = ledger.serialize_records(ledger.citation_numbers())
    with pytest.raises(InvalidSourceAction):
        expander.register_material(
            plan(),
            frontier={1},
            material_targets=[material_target(), material_target(**changes)],
        )
    assert expander.edges == {} and expander._expanded == {}
    assert ledger.serialize_records(ledger.citation_numbers()) == before
    assert calls == [] and broker.mock_calls == []


def test_registration_rejects_out_of_frontier_or_mutated_original() -> None:
    expander, ledger, calls, broker = fixture()
    with pytest.raises(InvalidSourceAction):
        expander.register_material(
            plan(), frontier=set(), material_targets=[material_target()]
        )
    ledger._items[1].text += " Altered text without its canonical hash."
    with pytest.raises(InvalidSourceAction):
        expander.register_material(
            plan(), frontier={1}, material_targets=[material_target()]
        )
    assert expander.edges == {} and expander._expanded == {}
    assert calls == [] and broker.mock_calls == []


def test_already_examined_unchanged_edge_does_not_gain_false_deferred_gap() -> None:
    expander, _ledger, calls, broker = fixture()
    edge = expander.expand(plan(), frontier={1}, material_targets=[material_target()])[
        0
    ]
    assert not edge.discovery_gaps
    existing_calls = list(calls)
    broker.reset_mock()
    expanded_before = dict(expander._expanded)
    edges = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )
    assert edges == [edge] and not edge.discovery_gaps
    assert expander._expanded == expanded_before
    assert calls == existing_calls and broker.mock_calls == []


def test_later_expansion_still_acquires_a_previously_deferred_relation() -> None:
    expander, ledger, calls, _broker = fixture()
    edge = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )[0]
    assert DEFERRED_GAP in edge.discovery_gaps and not calls
    edges = expander.expand(plan(), frontier={1}, material_targets=[material_target()])
    assert edges == [edge] and not edge.discovery_gaps
    assert any(name == "search_corpus" for name, _arguments in calls)
    assert edge.governing_citations and edge.candidate_citations
    assert expander._expanded[edge.edge_id] == expander._edge_state(edge)
    assert material_dependency_gaps(edges, ledger, set(ledger.citation_numbers())) == {
        edge.edge_id: []
    }


def test_empty_explicit_targets_do_not_collect_incidental_references() -> None:
    expander, _ledger, calls, broker = fixture()
    assert expander.register_material(plan(), frontier={1}, material_targets=[]) == []
    assert expander.edges == {} and calls == [] and broker.mock_calls == []


def test_registration_api_is_absent_from_legacy_dependency_expander() -> None:
    assert not hasattr(DependencyExpander, "register_material")
