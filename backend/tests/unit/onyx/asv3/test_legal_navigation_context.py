"""Related-source leads survive research/answer handoff without becoming legal evidence."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.llm.models import ChatCompletionMessage
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


def navigation() -> list[dict[str, JsonValue]]:
    return [
        {
            "anchor_source_id": "law",
            "instrument_name": "Example Law",
            "article_no": "17",
            "navigation_only": True,
            "absence_proven": False,
            "status": "available",
            "candidates": [
                {
                    "source_id": "decision",
                    "name": "Court decision concerning Example Law article 17",
                    "candidate_role": "judicial_candidate",
                }
            ],
        }
    ]


def governing_original(text: str, *, article: str = "17") -> EvidenceItem:
    item = original(
        "rule", text, source="law", headings=["Example Law", f"MADDE {article}"]
    )
    assert item.search_doc is not None
    item.search_doc.metadata["regulatory_chunk_id"] = "rule"
    return item


@pytest.mark.parametrize("profile,depth", [("normal", 0), ("deep", 0), ("deep", 1)])
def test_verified_original_supplies_related_lead_to_both_decision_models(
    profile: str,
    depth: int,
) -> None:
    selected, cheap, acquire = model(), model(), MagicMock()
    context = RunContext(depth=depth)
    ledger = EvidenceLedger()
    ledger.add([governing_original("A conditional rule.")], context)
    context.services.update(
        evidence=ledger,
        research_profile=profile,
        legal_source_navigation_acquire=acquire,
        legal_source_navigation=navigation,
    )
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)])
    adapter = ResearchModel(
        selected, context, research_llm=cheap, lean_native_mode=True
    )
    cheap.invoke.return_value = ModelResponse(
        id="candidate", created="0", choice=Choice(message=Message(content="Rule [1]."))
    )
    if depth == 0:
        selected.invoke.return_value = native_action(
            "read_provision", {"article": "17"}
        )
        adapter.decide(current)
    selected.invoke.return_value = ModelResponse(
        id="answer", created="0", choice=Choice(message=Message(content="Rule [1]."))
    )
    assert adapter.decide(current).answer == "Rule [1]."
    cheap_payload, selected_payload = last_payload(cheap), last_payload(selected)
    assert (
        cheap_payload["related_source_navigation"]
        == selected_payload["related_source_navigation"]
    )
    lead = selected_payload["related_source_navigation"][0]
    assert lead["navigation_only"] is True and lead["absence_proven"] is False
    assert lead["candidates"][0]["available_original_citations"] == []
    assert selected_payload["original_evidence"] == [full_record(ledger, 1)]
    assert ledger.citation_numbers() == (1,)
    assert cheap.invoke.call_count == 1
    assert selected.invoke.call_count == (2 if depth == 0 else 1)
    recorded_original = ledger.get(1)
    assert recorded_original is not None
    assert all(
        call.args[0].text_hash == recorded_original.text_hash
        for call in acquire.call_args_list
    )


def test_source_text_delivery_does_not_claim_that_operative_holding_was_examined() -> (
    None
):
    context, ledger, selected = RunContext(), EvidenceLedger(), model()
    ledger.add(
        [
            governing_original("A rule."),
            original("referral", "The referring court argues...", source="decision"),
        ],
        context,
    )
    context.services.update(evidence=ledger, legal_source_navigation=navigation)
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    prompt, _, _ = adapter._fit_native_decision(current)
    payload = json.loads(cast(str, prompt[-1].content))
    candidate = payload["related_source_navigation"][0]["candidates"][0]
    assert candidate["available_original_citations"] == [2]
    assert "resolved" not in candidate and "holding" not in candidate
    selected.invoke.assert_not_called()


def test_forged_or_missing_original_does_not_acquire_or_expose_related_leads() -> None:
    context, ledger, selected, acquire = (
        RunContext(),
        EvidenceLedger(),
        model(),
        MagicMock(),
    )
    ledger.add([governing_original("Actual original.")], context)
    context.services.update(
        evidence=ledger,
        legal_source_navigation_acquire=acquire,
        legal_source_navigation=navigation,
    )
    record = full_record(ledger, 1)
    record["text_hash"] = "0" * 64
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    for current in (
        adaptive_tool_view(),
        adaptive_tool_view(original_evidence=[record]),
    ):
        prompt, _, _ = adapter._fit_native_decision(current)
        assert "related_source_navigation" not in json.loads(
            cast(str, prompt[-1].content)
        )
    acquire.assert_not_called()
    selected.invoke.assert_not_called()


def test_cached_lead_for_another_article_of_same_law_is_not_attached() -> None:
    context, ledger, selected = RunContext(), EvidenceLedger(), model()
    ledger.add([governing_original("An unrelated provision.", article="18")], context)
    context.services.update(evidence=ledger, legal_source_navigation=navigation)
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    prompt, _, _ = adapter._fit_native_decision(
        adaptive_tool_view(original_evidence=[full_record(ledger, 1)])
    )
    assert "related_source_navigation" not in json.loads(cast(str, prompt[-1].content))


def test_optional_catalogue_leads_yield_before_originals_at_physical_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, ledger, selected = RunContext(), EvidenceLedger(), model()
    ledger.add(
        [governing_original("The complete original and its conditions.")], context
    )
    context.services.update(evidence=ledger, legal_source_navigation=navigation)
    adapter = ResearchModel(selected, context, lean_native_mode=True)

    def cost(prompt: list[ChatCompletionMessage], _tools: object) -> int:
        payload = json.loads(cast(str, prompt[-1].content))
        return 999999 if "related_source_navigation" in payload else 1

    monkeypatch.setattr(adapter, "_input_cost", cost)
    prompt, _, _ = adapter._fit_native_decision(
        adaptive_tool_view(original_evidence=[full_record(ledger, 1)])
    )
    payload = json.loads(cast(str, prompt[-1].content))
    assert payload["original_evidence"] == [full_record(ledger, 1)]
    assert "related_source_navigation" not in payload
    assert "related_source_navigation_omitted" in payload
    assert "original_evidence_omitted" not in payload
