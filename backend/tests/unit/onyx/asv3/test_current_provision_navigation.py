"""Only this decision's literal originals may advertise reusable provisions."""

import copy
from typing import Any, cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.shared_originals import delivered_provision_navigation
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    lead_id,
    navigation,
    review,
    seen,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals, parallel
from tests.unit.onyx.asv3.test_native_model_adapter import last_payload, model, view
from tests.unit.onyx.asv3.test_shared_originals import (
    full_record,
    original,
    recorded,
)


def test_navigation_separates_source_version_and_enclosing_provision() -> None:
    items = [
        original("first", "First clause."),
        original("second", "Separate condition."),
        original("other", "Another source.", source="other-source"),
        original("old", "Earlier version.", metadata={"read_as_of_date": "2025-01-01"}),
        original(
            "temporary", "Temporary clause.", headings=["Statute", "GEÇİCİ MADDE 17"]
        ),
        original("annex", "Annex clause.", headings=["Statute", "Annex", "MADDE 17"]),
    ]
    ledger = recorded(items)
    rows = [full_record(ledger, number) for number in range(1, 7)]
    before = copy.deepcopy(rows)
    groups = delivered_provision_navigation([*rows, rows[0]])
    assert [row["available_full_original_citations"] for row in groups] == [
        [1, 2],
        [3],
        [4],
        [5],
        [6],
    ]
    assert all("complete" not in row for row in groups)
    assert rows == before


@pytest.mark.parametrize("protected", [False, True])
def test_actual_model_catalogue_lists_only_fitted_full_originals(
    protected: bool,
) -> None:
    ledger = recorded(
        [
            original("full", "An operative rule."),
            original(
                "partial",
                "A distinct article's continuation.",
                headings=["Statute", "MADDE 18"],
            ),
        ]
    )
    context = RunContext(services={"evidence": ledger})
    if not protected:
        parallel(context)
    else:
        context.services.update(
            research_profile="experimental", experimental_parallel=False
        )
    full, partial = full_record(ledger, 1), full_record(ledger, 2)
    partial.update(
        text=str(partial["text"])[:5], start_char=0, end_char=5, truncated=True
    )
    selected = model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    adapter.decide(view(original_evidence=[full, partial]))
    current = last_payload(selected)
    if protected:
        assert "delivered_provisions" not in current
    else:
        assert [
            row["available_full_original_citations"]
            for row in current["delivered_provisions"]
        ] == [[1]]
        assert {
            row["citation"]
            for row in actual_originals(selected.invoke.call_args.kwargs["prompt"])
        } == {1, 2}
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert selected.invoke.call_count == 1


@pytest.mark.parametrize("recorded_lead", [False, True])
def test_terminal_schema_binds_current_navigation_and_prior_owned_leads(
    recorded_lead: bool,
) -> None:
    context, ledger, reviews = setup_reviews()
    parallel(context)
    context.services.update(
        lean_native_mode=True, legal_source_reviews=reviews, evidence=ledger
    )
    if recorded_lead:
        seen(context, ledger, reviews)
    else:
        context.services["legal_source_navigation"] = navigation
    from onyx.asv3.models import OutcomeStatus, ToolOutcome, ToolSpec

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Submit.",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                handler=lambda _args, _ctx: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="accepted"
                ),
            )
        ]
    )
    tools = registry.definitions(context)
    original_tools = copy.deepcopy(tools)
    adapter = ResearchModel(model(), context, lean_native_mode=True)
    _, selected, _ = adapter._fit_native_decision(
        view(original_evidence=[full_record(ledger, 1)]).model_copy(
            update={"tools": tools}
        )
    )
    schema = cast(dict[str, Any], selected[0])["function"]["parameters"]
    branches = schema["properties"]["_related_source_reviews"]["items"]["anyOf"]
    assert {
        identity
        for branch in branches
        for identity in branch["properties"]["lead_id"]["enum"]
    } == {lead_id()}
    assert all(
        branch["properties"]["status"]["enum"] == ["unresolved"]
        and branch["properties"]["witnesses"]["maxItems"] == 0
        for branch in branches
    )
    # The only physically fitted original is the anchor law, not this source.
    assert not jsonschema.Draft202012Validator(schema).is_valid(
        {"answer": "Text", "_related_source_reviews": [review()]}
    )
    assert jsonschema.Draft202012Validator(schema).is_valid(
        {
            "answer": "Text",
            "_related_source_reviews": [
                review(
                    status="unresolved", witnesses=[], gap="The candidate is unread."
                )
            ],
        }
    )
    assert not jsonschema.Draft202012Validator(schema).is_valid(
        {
            "answer": "Text",
            "_related_source_reviews": [review(lead_id="lead_" + "0" * 64)],
        }
    )
    assert tools == original_tools
    assert reviews.view(context, ledger, {1})["pending_lead_ids"] == (
        [lead_id()] if recorded_lead else []
    )
    fitted_prompt, fitted, _ = adapter._fit_native_decision(
        view(
            original_evidence=[full_record(ledger, 1), full_record(ledger, 2)]
        ).model_copy(update={"tools": tools})
    )
    assert {row["citation"] for row in actual_originals(fitted_prompt)} == {1, 2}
    own_schema = cast(dict[str, Any], fitted[0])["function"]["parameters"]
    own_branches = own_schema["properties"]["_related_source_reviews"]["items"]["anyOf"]
    assert {
        number
        for branch in own_branches
        for number in branch["properties"]["witnesses"]["items"]["properties"][
            "citation"
        ]["enum"]
    } == {2}
    assert jsonschema.Draft202012Validator(own_schema).is_valid(
        {"answer": "Text", "_related_source_reviews": [review()]}
    )
    assert not jsonschema.Draft202012Validator(own_schema).is_valid(
        {
            "answer": "Text",
            "_related_source_reviews": [
                review(witnesses=[{"citation": 1, "end_char": 20}])
            ],
        }
    )
    assert tools == original_tools


def test_schema_error_names_the_unexpected_field_without_answer_values() -> None:
    from tests.unit.onyx.asv3.test_native_model_adapter import native_action

    schema: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "additionalProperties": False,
    }
    tools: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {"name": "submit_answer", "parameters": schema},
        }
    ]
    response = native_action(
        "submit_answer", {"answer": "PRIVATE_BODY", "resolutions": "PRIVATE_VALUE"}
    )
    legacy = ResearchModel._decision(response, tools, return_argument_errors=True)
    detailed = ResearchModel._decision(
        response, tools, return_argument_errors=True, detailed_argument_errors=True
    )
    assert (
        legacy.calls[0].argument_error
        == "Tool arguments violate the exposed schema at  (additionalProperties)"
    )
    assert (
        detailed.calls[0].argument_error
        == legacy.calls[0].argument_error + ": unexpected fields resolutions"
    )
    assert "PRIVATE" not in str(detailed.calls[0].argument_error)
    assert detailed.calls[0].arguments == legacy.calls[0].arguments
