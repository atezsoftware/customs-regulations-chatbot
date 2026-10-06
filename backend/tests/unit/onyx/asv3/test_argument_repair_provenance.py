"""Mechanical terminal repair preserves the physical semantic-source owner."""

import json
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.serial_experimental_session import _publication_gap
from tests.unit.onyx.asv3.test_experimental_argument_repair import tools
from tests.unit.onyx.asv3.test_model_adapter import (
    argument_patch_response,
    scripted_model,
    tool_response,
)
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals, payloads
from tests.unit.onyx.asv3.test_native_model_adapter import view
from tests.unit.onyx.asv3.test_shared_originals import full_record, original

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("representation", ["answer", "reference", "unit_edit"])
@pytest.mark.parametrize("citation", [1, 2])
@pytest.mark.parametrize("hosted", [False, True])
def test_native_patch_keeps_semantic_call_and_missing_source_still_blocks(
    representation: str, citation: int, hosted: bool
) -> None:
    ledger = EvidenceLedger()
    context = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "scenario_request": "Explain the material condition.",
            "evidence": ledger,
        }
    )
    if hosted:
        context.services.update(
            experimental_parallel=False,
            serial_session_diagnostics=True,
            lean_native_mode=True,
            task_id="owned-task",
        )
    ledger.add(
        [
            original("own", "The genuine original."),
            original(
                "unseen",
                "Unseen text.",
                source="other",
                headings=["Other instrument", "Specific clause"],
            ),
        ],
        context,
    )
    body = f"The material condition applies [{citation}]."
    expected_body = body
    selected = scripted_model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    patch_arguments: dict[str, Any] = {}
    semantic_call = ""
    invocation_count = 0

    def invoke(**kwargs: Any) -> Any:
        nonlocal invocation_count, semantic_call, expected_body
        invocation_count += 1
        if invocation_count == 1:
            assert {row["citation"] for row in actual_originals(kwargs["prompt"])} == {
                1
            }
            patch_arguments.update(_language="tr")
            if representation == "answer":
                patch_arguments["answer"] = body
            else:
                current = payloads(kwargs["prompt"])[-1]
                patch_arguments["retained_answer_id"] = current["retained_answer"][
                    "retained_answer_id"
                ]
                if representation == "unit_edit":
                    unit = current["draft_to_repair"]["units"][0]
                    expected_body = (
                        f"The exact qualified condition applies [{citation}]."
                    )
                    patch_arguments["retained_answer_edits"] = [
                        {"unit_id": unit["unit_id"], "replacement": expected_body}
                    ]
            result = tool_response(
                json.dumps({**patch_arguments, "unknown_field": True}), "submit_answer"
            )
            result.choice.finish_reason = "tool_calls"
            return result
        assert invocation_count == 2
        semantic_call = adapter.last_call_id or ""
        assert ledger.completely_delivered(semantic_call) == {1}
        assert actual_originals(kwargs["prompt"]) == []
        patch = dict(patch_arguments)
        if representation == "answer":
            del patch["answer"]
        result = argument_patch_response(json.dumps(patch))
        result.choice.finish_reason = "stop"
        return result

    selected.invoke.side_effect = invoke
    current_view = view(
        original_evidence=[full_record(ledger, 1)],
        draft_to_repair=body if representation != "answer" else None,
        publication_gap={"repair": True} if representation != "answer" else None,
    )
    current_view.tools = tools("submit_answer")
    decision = adapter.decide(current_view)
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments["answer"] == expected_body
    assert (
        adapter.last_call_id == context.services["last_model_call_id"] == semantic_call
    )
    assert adapter.last_finish_reason == "tool_calls"
    assert adapter.last_response_truncated is False
    gap = _publication_gap(
        expected_body,
        semantic_call,
        context,
        ledger,
        AuthorityRequirements(context, "Explain the material condition."),
        LegalSourceReviews(context, "Explain the material condition."),
        requires_sources=True,
    )
    if citation == 1:
        assert gap is None
    else:
        assert gap is not None and gap.data["undelivered_citations"] == [2]
    assert selected.invoke.call_count == 2


@pytest.mark.parametrize("missing", [False, True])
def test_real_hosted_terminal_handler_checks_semantic_delivery_after_patch(
    monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
        declaration,
        outer,
        serial_run,
        session,
    )

    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.selected.reset_mock()
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    own = original("own", "The genuine original.")
    ledger.add(
        [
            own,
            original(
                "unseen", "Unseen text.", source="other", headings=["Other instrument"]
            ),
        ],
        child.context,
    )
    child.harness.evidence_working_set.remember(1, 0, len(own.text))
    body = f"The condition applies [{2 if missing else 1}]."
    semantic_call = ""
    count = 0
    metadata = {"basis": "originals", "_language": "tr", "_outcomes": [declaration()]}

    def invoke(**kwargs: Any) -> Any:
        nonlocal count, semantic_call
        count += 1
        if count == 1:
            assert {row["citation"] for row in actual_originals(kwargs["prompt"])} == {
                1
            }
            return tool_response(
                json.dumps({"answer": body, **metadata, "unknown": True}),
                "submit_answer",
            )
        assert count == 2 and actual_originals(kwargs["prompt"]) == []
        semantic_call = child.model.last_call_id or ""
        return argument_patch_response(json.dumps(metadata))

    run.selected.invoke.side_effect = invoke
    decision = child.model.decide(child.harness.view())
    child._on_decision(decision)
    receipts = child.harness._dispatch(decision.calls)
    assert len(receipts) == 1
    assert (
        child.model.last_call_id
        == child.context.services["last_model_call_id"]
        == semantic_call
    )
    assert ledger.completely_delivered(semantic_call) == {1}
    if missing:
        assert receipts[0].outcome.status == OutcomeStatus.PARTIAL
        assert receipts[0].outcome.data["undelivered_citations"] == [2]
        assert child.context.services.get("submitted_answer") is None
    else:
        assert receipts[0].outcome.status == OutcomeStatus.FOUND
        assert child.context.services["submitted_answer"] == body
        assert child.publication_gap(body, semantic_call, requires_sources=True) is None
    assert run.selected.invoke.call_count == 2


def test_nonterminal_argument_patch_preserves_original_decision_owner() -> None:
    ledger = EvidenceLedger()
    context = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": True,
            "scenario_request": "Read the condition.",
            "evidence": ledger,
        }
    )
    ledger.add([original("own", "The genuine original.")], context)
    selected = scripted_model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    semantic_call = ""
    count = 0
    definitions: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "read_evidence",
                "parameters": {
                    "type": "object",
                    "properties": {"citation": {"type": "integer"}},
                    "required": ["citation"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    def invoke(**_kwargs: Any) -> Any:
        nonlocal count, semantic_call
        count += 1
        if count == 1:
            return tool_response('{"citation":"1"}')
        semantic_call = adapter.last_call_id or ""
        return argument_patch_response('{"citation":1}')

    selected.invoke.side_effect = invoke
    current = view(original_evidence=[full_record(ledger, 1)])
    current.tools = definitions
    decision = adapter.decide(current)
    assert decision.calls[0].argument_error is None
    assert decision.calls[0].arguments == {"citation": 1}
    assert (
        adapter.last_call_id == context.services["last_model_call_id"] == semantic_call
    )
    assert ledger.completely_delivered(semantic_call) == {1}
    deliveries = ledger.export()["deliveries"]
    assert isinstance(deliveries, list)
    assert {row["call_id"] for row in deliveries if isinstance(row, dict)} == {
        semantic_call
    }


@pytest.mark.parametrize("plain", [False, True])
def test_patch_authored_answer_or_plain_profile_keeps_patch_call(plain: bool) -> None:
    ledger = EvidenceLedger()
    context = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": not plain,
            "scenario_request": "Explain the condition.",
            "evidence": ledger,
        }
    )
    ledger.add([original("own", "The genuine original.")], context)
    selected = scripted_model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    semantic_call = ""
    calls = 0

    def invoke(**_kwargs: Any) -> Any:
        nonlocal semantic_call, calls
        calls += 1
        if calls == 1:
            return tool_response(
                json.dumps(
                    {
                        "answer": "Rule [1]." if plain else None,
                        "_language": "tr",
                        "unknown": True,
                    }
                ),
                "submit_answer",
            )
        semantic_call = adapter.last_call_id or ""
        patch = {"_language": "tr"}
        if not plain:
            patch["answer"] = "Patch-authored claim [1]."
        return argument_patch_response(json.dumps(patch))

    selected.invoke.side_effect = invoke
    current_view = view(original_evidence=[full_record(ledger, 1)])
    current_view.tools = tools("submit_answer")
    decision = adapter.decide(current_view)
    assert decision.calls[0].argument_error is None
    assert adapter.last_call_id == context.services["last_model_call_id"]
    assert adapter.last_call_id != semantic_call
    assert ledger.completely_delivered(semantic_call) == {1}
    assert ledger.completely_delivered(adapter.last_call_id or "") == set()
    assert selected.invoke.call_count == 2
