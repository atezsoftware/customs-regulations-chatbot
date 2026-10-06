import copy
import json
from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import RunContext
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.terminal_fact_references import (
    bind_terminal_fact_references,
    normalize_terminal_fact_references,
    terminal_fact_catalogue,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_native_cache_projection import payloads
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response, view
from tests.unit.onyx.asv3.test_terminal_owned_commit import fact_tools


def context() -> RunContext:
    result = RunContext(
        scope={"tenant_id": "owned", "document_set_id": 15},
        services={
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "research_profile": "normal",
        },
    )
    result.services["outcome_map"] = OutcomeMap(
        ["Explain the effect."],
        result,
        factual_context="Permission was requested. It has not been granted.",
    )
    return result


def arguments(fact_id: JsonValue) -> dict[str, JsonValue]:
    return {
        "answer": "The condition remains unresolved [1].",
        "basis": "originals",
        "_outcomes": [
            {
                "outcome_id": "permission",
                "question_ids": ["q0"],
                "detail": "Assess the effect of the missing approval.",
                "decisive_fact_refs": [fact_id],
            }
        ],
    }


def test_tuned_native_decision_expands_owned_facts_without_an_extra_model_call() -> (
    None
):
    ctx = context()
    ctx.services["evidence"] = EvidenceLedger()
    catalogue = terminal_fact_catalogue(ctx)
    emitted = arguments(catalogue[1]["fact_id"])
    selected = model(limit=1000000)
    selected.invoke.return_value = native_response("submit_answer", json.dumps(emitted))
    adapter = ResearchModel(selected, ctx, lean_native_mode=True)
    current = view()
    current.tools = fact_tools()
    before = copy.deepcopy(current.tools)
    result = adapter.decide(current)

    assert selected.invoke.call_count == 1
    assert current.tools == before
    assert len(result.calls) == 1 and result.calls[0].argument_error is None
    actual = result.calls[0].arguments
    assert actual["answer"] == emitted["answer"]
    outcome = cast(list[dict[str, JsonValue]], actual["_outcomes"])[0]
    assert outcome["decisive_facts"] == ["It has not been granted."]
    assert "decisive_fact_refs" not in outcome
    outcomes = cast(OutcomeMap, ctx.services["outcome_map"])
    outcomes.update(
        OutcomeUpdate.model_validate({"outcomes": [outcome]}), EvidenceLedger()
    )
    assert result.assistant_message is not None
    wire = result.assistant_message.tool_calls
    assert wire and json.loads(wire[0].function.arguments) == emitted
    host = payloads(selected.invoke.call_args.kwargs["prompt"])[-1]
    assert host["decisive_fact_catalogue"] == catalogue


@pytest.mark.parametrize("change", ["run", "scope", "context", "task"])
def test_tuned_foreign_or_stale_fact_references_remain_invalid(change: str) -> None:
    ctx = context()
    original = arguments(terminal_fact_catalogue(ctx)[0]["fact_id"])
    if change == "run":
        ctx.run_id = "foreign"
    elif change == "scope":
        ctx.scope = {"tenant_id": "foreign"}
    elif change == "context":
        ctx.services["outcome_map"] = OutcomeMap(
            ["Explain the effect."], ctx, factual_context="Permission was granted."
        )
    else:
        ctx.services["task_id"] = "foreign"
    assert (
        normalize_terminal_fact_references("submit_answer", original, ctx) is original
    )
    assert (
        "decisive_fact_refs"
        in cast(list[dict[str, JsonValue]], original["_outcomes"])[0]
    )


def test_tuned_literal_facts_still_require_exact_scenario_text() -> None:
    ctx = context()
    supplied = arguments(terminal_fact_catalogue(ctx)[0]["fact_id"])
    row = cast(list[dict[str, JsonValue]], supplied["_outcomes"])[0]
    row.pop("decisive_fact_refs")
    row["decisive_facts"] = ["Permission has been obtained."]
    assert (
        normalize_terminal_fact_references("submit_answer", supplied, ctx) is supplied
    )
    outcomes = cast(OutcomeMap, ctx.services["outcome_map"])
    with pytest.raises(
        ValueError, match="Outcome facts must quote supplied scenario text"
    ):
        outcomes.update(
            OutcomeUpdate.model_validate({"outcomes": [row]}), EvidenceLedger()
        )


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_protected_modes_keep_exact_schemas_and_no_fact_catalogue(profile: str) -> None:
    ctx = context()
    ctx.services.update(asv3_workflow_variant="standard", research_profile=profile)
    supplied = fact_tools()
    before = copy.deepcopy(supplied)
    actual, catalogue = bind_terminal_fact_references(supplied, ctx)
    assert actual == before == supplied and not catalogue
    assert not terminal_fact_catalogue(ctx)


def test_tuned_raw_refs_and_expanded_facts_both_match_exposed_schema() -> None:
    ctx = context()
    definitions, catalogue = bind_terminal_fact_references(fact_tools(), ctx)
    supplied = arguments(catalogue[0]["fact_id"])
    before = copy.deepcopy(supplied)
    schema = cast(dict[str, JsonValue], definitions[0]["function"])["parameters"]
    jsonschema.validate(supplied, schema)
    expanded = normalize_terminal_fact_references("submit_answer", supplied, ctx)
    jsonschema.validate(expanded, schema)
    assert supplied == before
    assert expanded["answer"] == supplied["answer"]
