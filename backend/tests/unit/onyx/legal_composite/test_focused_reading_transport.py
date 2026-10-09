"""Post-draft focused reads retain their fitted exact original selectors."""

from copy import deepcopy
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue

from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    AnswerSection,
    DraftClaim,
    IssueResearchStep,
    SpanSupport,
    StructuredDraftAnswer,
)
from onyx.legal_composite.reading_evidence import reading_witness_manifest
from tests.unit.legal_composite.test_gateway import model as model
from tests.unit.onyx.legal_composite.test_numbered_reading_integration import (
    engine,
    gateway,
    provider_response,
    sent_body,
)
from tests.unit.onyx.legal_composite.test_reading_evidence import (
    OTHER_RULE,
    RULE,
    evidence,
    requirement,
    response,
)


def draft() -> StructuredDraftAnswer:
    return StructuredDraftAnswer(
        answer="",
        unresolved_need_ids=[],
        sections=[
            AnswerSection(
                section_id="section",
                need_ids=["approval"],
                text="Fictional application result",
                claim_ids=["claim"],
            )
        ],
        claims=[
            DraftClaim(
                claim_id="claim",
                section_id="section",
                need_ids=["approval"],
                answer_excerpt="Both fictional conditions must hold [1].",
                supports=[SpanSupport(citation=1, quotation=RULE)],
            )
        ],
    )


@pytest.mark.parametrize("source_phase", [True, False])
def test_focused_repair_reads_through_real_gateway_without_changing_originals(
    model: Mock, source_phase: bool
) -> None:
    ledger, _records, _prior = evidence(RULE, OTHER_RULE)
    actual = gateway(model, ledger)
    instance, acquirer = engine(actual, ledger)
    original_state = deepcopy(ledger.export())
    payload = instance._focused_payload(
        "Apply the supported fictional condition.",
        "",
        draft(),
        [],
        None,
        {"approval"},
        source_phase,
    )
    frozen = deepcopy(payload)
    model.invoke.return_value = provider_response(model, response(requirement()))

    value = instance._complete_source_reading(payload)
    resolved = instance._resolve_source_reading(value)
    assert isinstance(resolved, IssueResearchStep)
    assert instance.plan is not None
    instance.requirements.update(
        resolved.requirements, instance.plan, actual.last_delivered_citations
    )
    assert instance.requirements.records()[0].supports[0].quotation == RULE
    manifest = actual.last_reading_manifest
    assert manifest is not None
    assert manifest.call_id == actual.last_call_id
    assert {w.citation for w in manifest.witnesses} == {1, 2}
    originals = cast(list[dict[str, JsonValue]], sent_body(model)["original_evidence"])
    assert reading_witness_manifest(originals, call_id=manifest.call_id) == manifest
    for row in originals:
        item = ledger.get(cast(int, row["citation"]))
        assert item is not None
        assert (row["source_id"], row["chunk_id"], row["text_hash"], row["text"]) == (
            item.source_id,
            item.chunk_id,
            item.text_hash,
            item.text,
        )
    assert payload == frozen and ledger.export()["records"] == original_state["records"]
    assert actual.budget.snapshot()["model_calls"] == model.invoke.call_count == 1
    acquirer.acquire.assert_not_called()


def test_focused_manifest_contains_only_actual_fitted_delivery(model: Mock) -> None:
    ledger, _records, _prior = evidence(RULE, "X" * 60_000)
    actual = gateway(model, ledger, context_tokens=12_000)
    instance, acquirer = engine(actual, ledger)
    payload = instance._focused_payload(
        "Apply the required rule.", "", draft(), [], None, {"approval"}, True
    )
    model.invoke.return_value = provider_response(model, response(requirement()))
    value = instance._complete_source_reading(payload)
    assert isinstance(instance._resolve_source_reading(value), IssueResearchStep)
    assert actual.last_delivered_citations == {1}
    manifest = actual.last_reading_manifest
    assert manifest is not None and {w.citation for w in manifest.witnesses} == {1}
    assert sent_body(model)["omitted_original_ids"] == [2]
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        instance._resolve_source_reading(response(requirement(citation=2)))
    assert model.invoke.call_count == 1
    acquirer.acquire.assert_not_called()


@pytest.mark.parametrize(
    "corruption", ["missing_number", "wrong_number", "changed_hash"]
)
def test_focused_transport_still_rejects_host_catalogue_corruption(
    model: Mock, corruption: str
) -> None:
    ledger, _records, _prior = evidence(RULE)
    actual = gateway(model, ledger)
    instance, acquirer = engine(actual, ledger)
    payload = instance._focused_payload(
        "Apply the original.", "", draft(), [], None, {"approval"}, True
    )
    rows = cast(list[dict[str, JsonValue]], payload["original_evidence"])
    spans = cast(list[dict[str, JsonValue]], rows[0]["witness_spans"])
    if corruption == "missing_number":
        spans[0].pop("span_number")
    elif corruption == "wrong_number":
        spans[0]["span_number"] = 999
    else:
        rows[0]["text_hash"] = "0" * 64
    model.invoke.return_value = provider_response(model, response(requirement()))
    with pytest.raises(InvalidSourceAction, match="catalogue is not an exact original"):
        instance._complete_source_reading(payload)
    assert actual.last_reading_manifest is None
    assert instance.requirements.records() == []
    assert model.invoke.call_count == 1
    acquirer.acquire.assert_not_called()


def test_default_focused_transport_does_not_enable_numbered_protocol(
    model: Mock,
) -> None:
    ledger, _records, _prior = evidence(RULE)
    instance, _acquirer = engine(gateway(model, ledger), ledger, numbered=False)
    payload = instance._focused_payload(
        "Apply the original.", "", draft(), [], None, {"approval"}, True
    )
    rows = cast(list[dict[str, JsonValue]], payload["original_evidence"])
    spans = cast(list[dict[str, JsonValue]], rows[0]["witness_spans"])
    assert all("span_number" not in span for span in spans)
