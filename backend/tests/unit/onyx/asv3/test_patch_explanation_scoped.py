"""Diagnostic explanation placement cannot invalidate a safe owned argument patch."""

import json
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import ResearchModel, ToolArgumentPatch
from onyx.asv3.models import RunContext
from onyx.asv3.serial_experimental_session import _publication_gap
from tests.unit.onyx.asv3.test_model_adapter import (
    scripted_model,
    text_response,
    tool_response,
)
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_native_model_adapter import view
from tests.unit.onyx.asv3.test_shared_originals import full_record, original

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("scope", ["parallel", "hosted", "plain"])
@pytest.mark.parametrize("explanation", ["absent", "root"])
def test_real_adapter_patch_explanation_is_optional_only_for_owned_parallel(
    scope: str, explanation: str
) -> None:
    _check_patch(scope, explanation)


@pytest.mark.parametrize(
    "invalid", ["arguments", "call_id", "extra_entry", "extra_root", "unseen_source"]
)
def test_diagnostics_tolerance_does_not_approve_invalid_patch_or_sources(
    invalid: str,
) -> None:
    _check_patch("parallel", "root", invalid=invalid)


def _check_patch(scope: str, explanation: str, *, invalid: str | None = None) -> None:
    ledger = EvidenceLedger()
    context = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": scope == "parallel",
            "serial_session_diagnostics": scope == "hosted",
            "lean_native_mode": True,
            "task_id": "owned-task",
            "evidence": ledger,
            "scenario_request": "Explain the material condition.",
        }
    )
    ledger.add(
        [
            original("own", "The genuine operative original."),
            original(
                "unseen",
                "An unseen original.",
                source="other",
                headings=["Other instrument"],
            ),
        ],
        context,
    )
    body = (
        "The immutable material condition applies [2]."
        if invalid == "unseen_source"
        else "The immutable material condition applies [1]."
    )
    selected = scripted_model()
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    semantic_call = ""
    count = 0
    definitions: list[dict[str, JsonValue]] = [
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {"type": "string", "enum": ["originals"]},
                        "_language": {"type": "string", "enum": ["tr"]},
                    },
                    "required": ["answer", "basis", "_language"],
                    "additionalProperties": False,
                },
            },
        }
    ]

    def invoke(**kwargs: Any) -> Any:
        nonlocal count, semantic_call
        count += 1
        if count == 1:
            assert {row["citation"] for row in actual_originals(kwargs["prompt"])} == {
                1
            }
            return tool_response(
                json.dumps(
                    {"answer": body, "basis": "originals", "_language": "invalid"}
                ),
                "submit_answer",
            )
        assert count == 2
        semantic_call = adapter.last_call_id or ""
        assert actual_originals(kwargs["prompt"]) == []
        corrected = {
            "basis": "originals",
            "_language": "invalid" if invalid == "arguments" else "tr",
        }
        entry: dict[str, JsonValue] = {
            "call_id": "another-call" if invalid == "call_id" else "call-1",
            "arguments_json": json.dumps(corrected),
        }
        if invalid == "extra_entry":
            entry["unknown"] = True
        payload: dict[str, JsonValue] = {"entries": [entry]}
        if explanation == "root":
            payload["explanation"] = "Correct the invalid language enum only."
        if invalid == "extra_root":
            payload["unknown"] = True
        return text_response(payload)

    selected.invoke.side_effect = invoke
    current = view(original_evidence=[full_record(ledger, 1)])
    current.tools = definitions
    decision = adapter.decide(current)
    assert selected.invoke.call_count == 2
    assert decision.calls[0].arguments["answer"] == body
    patch_valid = scope != "plain" and invalid in {None, "unseen_source"}
    if patch_valid:
        assert decision.calls[0].argument_error is None
        assert decision.calls[0].arguments == {
            "answer": body,
            "basis": "originals",
            "_language": "tr",
        }
        assert (
            adapter.last_call_id
            == context.services["last_model_call_id"]
            == semantic_call
        )
        assert ledger.completely_delivered(semantic_call) == {1}
        gap = _publication_gap(
            body,
            semantic_call,
            context,
            ledger,
            AuthorityRequirements(context, "Explain the material condition."),
            LegalSourceReviews(context, "Explain the material condition."),
            requires_sources=True,
        )
        if invalid == "unseen_source":
            assert gap is not None and gap.data["undelivered_citations"] == [2]
        else:
            assert gap is None
    else:
        assert decision.calls[0].argument_error is not None
        assert decision.calls[0].arguments["_language"] == "invalid"
    assert "explanation" not in decision.calls[0].arguments


def test_legacy_patch_contract_still_requires_per_entry_explanation() -> None:
    schema = ToolArgumentPatch.model_json_schema()
    entry_schema = schema["$defs"]["ToolArgumentPatchEntry"]
    assert "explanation" in entry_schema["required"]
    assert "explanation" not in schema["properties"]
    assert schema["additionalProperties"] is False
