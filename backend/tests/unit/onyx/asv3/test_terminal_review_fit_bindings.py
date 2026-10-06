"""Source-review enums follow the actual invocation, including fresh source leads."""

import copy
import json
from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.llm.models import ChatCompletionMessage
from tests.unit.onyx.asv3.test_experimental_workflow import (
    experimental_context,
    terminal_registry,
)
from tests.unit.onyx.asv3.test_legal_source_reviews import review, seen
from tests.unit.onyx.asv3.test_native_model_adapter import adaptive_tool_view, model
from tests.unit.onyx.asv3.test_shared_originals import full_record


def setup(
    hosted: bool,
) -> tuple[RunContext, EvidenceLedger, LegalSourceReviews, ResearchModel]:
    context, ledger, reviews = experimental_context()
    if hosted:
        context.services.update(
            experimental_parallel=False,
            serial_session_diagnostics=True,
            serial_original_transport=True,
            task_id="owned-serial-child",
        )
    else:
        context.services.update(
            experimental_parallel=True,
            scenario_request="Can the rule be applied?",
        )
    return (
        context,
        ledger,
        reviews,
        ResearchModel(model(), context, lean_native_mode=True),
    )


def payload(prompt: list[ChatCompletionMessage]) -> dict[str, JsonValue]:
    assert isinstance(prompt[-1].content, str)
    return cast(dict[str, JsonValue], json.loads(prompt[-1].content))


def schema(tools: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and function.get("name") == "submit_answer":
            parameters = function.get("parameters")
            assert isinstance(parameters, dict)
            return parameters
    raise AssertionError("The complete-answer research action is missing")


def terminal(citation: int | None, *, unresolved: bool = False) -> dict[str, JsonValue]:
    assessment = review()
    if unresolved:
        assessment.update(
            status="unresolved",
            source_role="unknown",
            witnesses=[],
            gap="The candidate's operative text is not fully available.",
        )
    else:
        assert citation is not None
        assessment["witnesses"] = [
            {"citation": citation, "start_char": 0, "end_char": 10}
        ]
    return {
        "answer": "The source has a limited effect [1] [2].",
        "basis": "originals",
        "_related_source_reviews": [assessment],
    }


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_fresh_lead_without_its_own_full_original_remains_unresolved(
    hosted: bool, partial: bool
) -> None:
    context, ledger, reviews, adapter = setup(hosted)
    registry = terminal_registry([])
    records = [full_record(ledger, 1), full_record(ledger, 3)]
    if partial:
        fragment = full_record(ledger, 2)
        text = fragment["text"]
        assert isinstance(text, str)
        fragment.update(text=text[:10], end_char=10, truncated=True)
        records.append(fragment)
    current = adaptive_tool_view(original_evidence=records).model_copy(
        update={"tools": registry.definitions(context)}
    )

    _, tools, _ = adapter._fit_native_decision(current)

    validator = jsonschema.Draft202012Validator(schema(tools))
    assert validator.is_valid(terminal(None, unresolved=True))
    assert not validator.is_valid(terminal(2))
    assert not validator.is_valid(terminal(3))
    assert reviews.export()["records"] == []
    assert adapter.last_call_id is None


@pytest.mark.parametrize("hosted", [False, True])
def test_physical_eviction_rebuilds_owned_lead_enum_without_stale_citations(
    monkeypatch: pytest.MonkeyPatch, hosted: bool
) -> None:
    context, ledger, reviews, adapter = setup(hosted)
    seen(context, ledger, reviews)
    before = copy.deepcopy(reviews.export())
    registry = terminal_registry([])
    current = adaptive_tool_view(
        original_evidence=[full_record(ledger, 1), full_record(ledger, 2)],
        required_evidence_numbers=[1],
    ).model_copy(update={"tools": registry.definitions(context)})
    fitted_citations: list[set[int]] = []

    def physical_cost(
        prompt: list[ChatCompletionMessage], _tools: list[dict[str, JsonValue]]
    ) -> int:
        ranges = payload(prompt).get("original_evidence_ranges", [])
        assert isinstance(ranges, list)
        citations: set[int] = set()
        for row in ranges:
            if isinstance(row, dict):
                number = row.get("citation")
                if type(number) is int:
                    citations.add(number)
        fitted_citations.append(citations)
        return 100 if 2 in citations else 1

    monkeypatch.setattr(adapter, "_limits", lambda _: (10, 100))
    monkeypatch.setattr(adapter, "_input_cost", physical_cost)

    prompt, tools, _ = adapter._fit_native_decision(current)

    assert {1, 2} in fitted_citations and fitted_citations[-1] == {1}
    validator = jsonschema.Draft202012Validator(schema(tools))
    assert validator.is_valid(terminal(None, unresolved=True))
    assert not validator.is_valid(terminal(2))
    assert payload(prompt)["original_evidence_ranges"] == [
        {"citation": 1, "start_char": 0, "end_char": 35}
    ]
    assert reviews.export() == before
    assert adapter.last_call_id is None
