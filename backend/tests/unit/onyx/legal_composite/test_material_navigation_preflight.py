"""Material navigation validation preserves originals and every live dependency state."""

from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.dependencies import (
    CompositeDependencyExpander,
    DependencyExpander,
)
from onyx.legal_composite.models import MaterialDependencyRequest
from tests.unit.onyx.legal_composite.test_authority_dependencies import (
    material_target,
    plan,
    setup_composite_expander,
)

OTHER_REFERENCE = (
    " 9621 sayılı Denetim Kanunu'nun 13. maddesi farklı işlemleri düzenler."
)


def snapshot(expander: CompositeDependencyExpander) -> dict[str, object]:
    return {
        "edges": {
            key: edge.model_dump(mode="json") for key, edge in expander.edges.items()
        },
        "expanded": dict(expander._expanded),
        "candidate_edges": deepcopy(expander._candidate_edges),
        "receipts": deepcopy(expander.receipts),
        "verified_kinds": dict(expander.verified_kinds),
        "source_kinds": dict(expander.source_kinds),
        "ledger": expander.ledger.export(),
        "budget": expander.context.budget.snapshot(),
        "search_calls": expander.acquirer.search_calls,
        "completed": deepcopy(expander.acquirer._completed),
    }


def forbid_io(
    expander: CompositeDependencyExpander, monkeypatch: pytest.MonkeyPatch
) -> None:
    for method in ("_bind_originals", "acquire", "acquire_host_actions"):
        monkeypatch.setattr(
            expander.acquirer,
            method,
            Mock(side_effect=AssertionError("Validation must not acquire or rebind")),
        )
    monkeypatch.setattr(
        expander,
        "synchronize",
        Mock(side_effect=AssertionError("Validation must not synchronize")),
    )


def other_target() -> MaterialDependencyRequest:
    return material_target(
        instrument_name="Denetim Kanunu",
        instrument_number="9621",
        article="13",
        need_ids=["scope"],
    )


def test_multiple_valid_targets_preserve_existing_history_and_governing_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, ledger, calls, broker, *_ = setup_composite_expander(
        additional_reference=OTHER_REFERENCE
    )
    expander = cast(CompositeDependencyExpander, value)
    edge = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )[0]
    edge.discovery_limits.append("A prior page has a continuation.")
    edge.discovery_gaps.append("A previous operation remains incomplete.")
    expander._expanded[edge.edge_id] = "previous-examination-state"
    expander._candidate_edges["retained-candidate"] = {edge.edge_id}
    expander.receipts.append({"status": "previous-incomplete-operation"})
    source, chunk = broker.chunk.return_value
    governing = evidence_for_chunk(source, chunk)
    governing.question_ids = ["scope"]
    ledger.add([governing], expander.context)
    before = snapshot(expander)
    forbid_io(expander, monkeypatch)

    assert (
        expander.validate_material(
            plan(),
            frontier={1},
            material_targets=[material_target(), other_target()],
            need_bindings={1: {"scope"}},
        )
        is None
    )

    assert snapshot(expander) == before
    assert expander.edges[edge.edge_id] is edge
    assert edge.governing_citations == []
    retained = ledger.get(2)
    assert retained is not None and retained.question_ids == ["scope"]
    assert calls == [] and broker.mock_calls == []


@pytest.mark.parametrize(
    "changes",
    [
        {"instrument_name": "Invented Kanunu"},
        {"instrument_number": "9999"},
        {"article": "118"},
        {"need_ids": ["unknown"]},
        {"need_ids": ["permit", "permit"]},
        {"reason": " "},
    ],
)
def test_invalid_second_target_preserves_all_prior_state_and_originals(
    changes: dict[str, JsonValue], monkeypatch: pytest.MonkeyPatch
) -> None:
    value, _ledger, calls, broker, *_ = setup_composite_expander(
        additional_reference=OTHER_REFERENCE
    )
    expander = cast(CompositeDependencyExpander, value)
    expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )
    before = snapshot(expander)
    forbid_io(expander, monkeypatch)

    with pytest.raises(
        InvalidSourceAction,
        match="Material dependency does not match an observed original reference",
    ):
        expander.validate_material(
            plan(),
            frontier={1},
            material_targets=[material_target(), material_target(**changes)],
        )

    assert snapshot(expander) == before
    assert calls == [] and broker.mock_calls == []


@pytest.mark.parametrize("defect", ["out_of_frontier", "unknown_origin"])
def test_undelivered_origin_is_rejected_without_state_mutation(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, _ledger, calls, broker, *_ = setup_composite_expander()
    expander = cast(CompositeDependencyExpander, value)
    before = snapshot(expander)
    forbid_io(expander, monkeypatch)
    target = (
        material_target(origin_citation=999)
        if defect == "unknown_origin"
        else material_target()
    )

    with pytest.raises(
        InvalidSourceAction, match="Material dependency origin was not delivered"
    ):
        expander.validate_material(
            plan(),
            frontier=set() if defect == "out_of_frontier" else {1},
            material_targets=[target],
        )

    assert snapshot(expander) == before
    assert calls == [] and broker.mock_calls == []


@pytest.mark.parametrize(
    "defect",
    [
        "hash",
        "document",
        "chunk",
        "missing_document",
        "derived",
        "external",
        "untrusted",
        "truncated",
        "canonical_truncated",
    ],
)
def test_canonical_integrity_failure_remains_distinct_and_preserves_state(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    value, ledger, calls, broker, *_ = setup_composite_expander()
    expander = cast(CompositeDependencyExpander, value)
    item = ledger._items[1]
    assert item.search_doc is not None
    if defect == "hash":
        item.text += " Stale hash."
    elif defect == "document":
        item.search_doc.document_id = "foreign-document"
    elif defect == "chunk":
        item.search_doc.metadata["regulatory_chunk_id"] = "foreign-chunk"
    elif defect == "missing_document":
        item.search_doc = None
    elif defect == "truncated":
        item.metadata["truncated"] = True
    elif defect == "canonical_truncated":
        item.metadata["canonical_metadata"] = {"truncated": True}
    else:
        item.metadata["canonical_metadata"] = {defect: True}
    before = snapshot(expander)
    forbid_io(expander, monkeypatch)

    with pytest.raises(
        InvalidSourceAction, match="Material dependency origin is not canonical"
    ):
        expander.validate_material(
            plan(), frontier={1}, material_targets=[material_target()]
        )

    assert snapshot(expander) == before
    assert calls == [] and broker.mock_calls == []


def test_empty_targets_do_not_collect_incidental_reference_or_change_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, _ledger, calls, broker, *_ = setup_composite_expander(
        additional_reference=OTHER_REFERENCE
    )
    expander = cast(CompositeDependencyExpander, value)
    before = snapshot(expander)
    forbid_io(expander, monkeypatch)

    expander.validate_material(plan(), frontier={1}, material_targets=[])

    assert snapshot(expander) == before
    assert calls == [] and broker.mock_calls == []


@pytest.mark.parametrize("canonical", [False, True])
def test_real_collection_rejects_raw_truncation_with_same_integrity_reason(
    canonical: bool,
) -> None:
    value, ledger, calls, broker, *_ = setup_composite_expander()
    expander = cast(CompositeDependencyExpander, value)
    if canonical:
        ledger._items[1].metadata["canonical_metadata"] = {"truncated": True}
    else:
        ledger._items[1].metadata["truncated"] = True
    before = snapshot(expander)

    with pytest.raises(
        InvalidSourceAction, match="Material dependency origin is not canonical"
    ):
        expander.register_material(
            plan(), frontier={1}, material_targets=[material_target()]
        )

    assert snapshot(expander) == before
    assert calls == [] and broker.mock_calls == []


def test_preflight_does_not_replace_real_registration_or_expansion() -> None:
    value, _ledger, calls, broker, *_ = setup_composite_expander()
    expander = cast(CompositeDependencyExpander, value)
    expander.validate_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )
    assert expander.edges == {} and calls == [] and broker.mock_calls == []

    edge = expander.register_material(
        plan(), frontier={1}, material_targets=[material_target()]
    )[0]
    assert edge.discovery_gaps == [
        "Material dependency acquisition was deferred before examination."
    ]
    assert calls == []
    expander.expand(plan(), frontier={1}, material_targets=[material_target()])
    assert edge.governing_citations and edge.candidate_citations
    assert not edge.discovery_gaps and calls


def test_validation_api_is_not_added_to_legacy_expander() -> None:
    assert not hasattr(DependencyExpander, "validate_material")
