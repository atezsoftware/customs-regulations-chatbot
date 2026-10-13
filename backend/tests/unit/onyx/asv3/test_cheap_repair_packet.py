"""Captured substantive edits survive fresh context and the real retained resolver."""

import copy
import gzip
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.cheap_repair_packet import (
    cheap_repair_observations,
    prepare_cheap_repair_view,
)
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    RunContext,
    RunStopped,
)
from onyx.asv3.retained_answer import bind_retained_answer, resolve_retained_answer
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_native_model_adapter import model, native_response, turn
from tests.unit.onyx.asv3.test_shared_originals import full_record, original


class CapturedEvent(TypedDict):
    generation_step: int
    function_name: str
    captured_retained_answer_id: str
    before_sha256: str
    expected_candidate_sha256: str
    recorded_publication_status: str | None
    recorded_prepared_candidate_sha256: str | None
    arguments: dict[str, JsonValue]
    changes: list[dict[str, JsonValue]]
    publication_gap: dict[str, JsonValue]
    recorded_facts: list[str]


class CapturedRun(TypedDict):
    message_id: int
    request: str
    run_id: str
    events: list[CapturedEvent]
    originals: list[dict[str, JsonValue]]
    delivery_verified: bool


class History(TypedDict):
    bodies: dict[str, str]
    known_5077_final_support: list[int]
    runs: list[CapturedRun]


def history() -> History:
    path = Path(__file__).parent / "fixtures/cheap_repair_history.json.gz"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == (
        "f7234675c1faea170ebc09921642b69a4dcd9777b84c0016f42dc64c4c1c2ba7"
    )
    return cast(History, json.loads(gzip.decompress(raw)))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def definitions() -> list[dict[str, JsonValue]]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "parameters": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            },
        }
        for name in ("submit_answer", "submit_partial_answer")
    ]


def captured_view(run: CapturedRun, event: CapturedEvent) -> HarnessView:
    return HarnessView(
        request=run["request"],
        questions=[run["request"]],
        facts=event["recorded_facts"],
        receipts=[],
        evidence=[],
        tools=definitions(),
        turns=[turn("prior-original-writer", run["originals"])],
        draft_to_repair=history()["bodies"][event["before_sha256"]],
        publication_gap=event["publication_gap"],
        research_state={"unknown_fact": "The supplied scenario is not expanded."},
        original_evidence_omitted=[{"reason": "Unproved source closure remains open"}],
    )


@pytest.mark.parametrize(
    "run", history()["runs"], ids=lambda run: str(run["message_id"])
)
def test_captured_substantive_patches_reconstruct_exact_candidates(
    run: CapturedRun,
) -> None:
    for event in run["events"]:
        current = captured_view(run, event)
        saved = current.model_dump(mode="json")
        fresh = prepare_cheap_repair_view(current, run["originals"])
        assert current.model_dump(mode="json") == saved
        if fresh is None:
            # Historical tool records can precede the ledger's later citable upgrade.
            # Preserve their captured flag and use ordinary context rather than invent it.
            by_number = {row["citation"]: row for row in run["originals"]}
            assert any(
                by_number[number].get("citable") is not True
                for number in extract_citation_numbers(current.draft_to_repair or "")
            )
            fresh = current
        else:
            assert fresh.turns == []
            assert fresh.original_evidence == run["originals"]
            assert set(fresh.required_evidence_numbers) == {
                row["citation"] for row in run["originals"]
            }
            for field in HarnessView.model_fields:
                if field not in {
                    "turns",
                    "original_evidence",
                    "required_evidence_numbers",
                }:
                    assert getattr(fresh, field) == getattr(current, field)
        context = RunContext(
            run_id=run["run_id"],
            scope={"fixture_scope": run["message_id"]},
            services={"asv3_workflow_variant": ASV3_TUNED_VARIANT},
        )
        draft = fresh.draft_to_repair
        assert isinstance(draft, str) and sha(draft) == event["before_sha256"]
        _, descriptor = bind_retained_answer(
            definitions(), context, draft, request=fresh.request
        )
        assert descriptor is not None and isinstance(descriptor["units"], list)
        units = cast(list[dict[str, JsonValue]], descriptor["units"])
        edits: list[JsonValue] = []
        for change in event["changes"]:
            unit = next(
                row
                for row in units
                if row["start_char"] == change["start_char"]
                and row["end_char"] == change["end_char"]
            )
            captured_binding = json.dumps(
                [
                    event["captured_retained_answer_id"],
                    unit["start_char"],
                    unit["end_char"],
                    unit["text"],
                ],
                ensure_ascii=False,
            )
            assert change["captured_unit_id"] == "retained_unit_" + sha(
                captured_binding
            )
            edits.append(
                {"unit_id": unit["unit_id"], "replacement": change["replacement"]}
            )
        arguments = copy.deepcopy(event["arguments"])
        if edits:
            arguments["retained_answer_edits"] = edits
        call = CapabilityCall(
            name=event["function_name"],
            call_id="fresh-provider-call",
            arguments=arguments,
        )
        native = AssistantMessage(
            tool_calls=[
                ToolCall(
                    id=call.call_id,
                    function=FunctionCall(
                        name=call.name, arguments=json.dumps(arguments)
                    ),
                )
            ]
        )
        decision = Decision(calls=[call], assistant_message=native)
        resolved = resolve_retained_answer(
            decision, context, draft, request=fresh.request
        )
        answer = resolved.calls[0].arguments.get("answer")
        assert resolved.calls[0].argument_error is None and isinstance(answer, str)
        assert sha(answer) == event["expected_candidate_sha256"]
        assert resolved.assistant_message is native
        if event["recorded_prepared_candidate_sha256"] is not None:
            assert sha(answer) == event["recorded_prepared_candidate_sha256"]
        if run["message_id"] == 5068:
            assert (
                event["recorded_publication_status"] is None
                and not run["delivery_verified"]
            )


def test_5077_keeps_all_known_benefits_including_navigation_misses() -> None:
    recorded = history()
    run = next(row for row in recorded["runs"] if row["message_id"] == 5077)
    assert set(recorded["known_5077_final_support"]) == {
        50,
        110,
        126,
        127,
        149,
        153,
        185,
        259,
        264,
        265,
        280,
        291,
        414,
        417,
        574,
        575,
        597,
        599,
    }
    assert set(recorded["known_5077_final_support"]).issubset(
        {row["citation"] for row in run["originals"]}
    )
    assert len(run["events"][0]["changes"]) == 6
    for row in run["originals"]:
        assert sha(str(row["text"])) == row["text_hash"]


def minimal() -> tuple[
    HarnessView, list[dict[str, JsonValue]], RunContext, EvidenceLedger
]:
    ledger, context = EvidenceLedger(), RunContext()
    ledger.add(
        [
            original("rule", "The outcome has its own rule."),
            original("condition", "Its material uncited condition also survives."),
        ],
        context,
    )
    context.services.update(
        evidence=ledger,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
    )
    records = [full_record(ledger, number) for number in (1, 2)]
    current = HarnessView(
        request="Apply both outcomes and all conditions.",
        questions=["Apply both outcomes and all conditions."],
        facts=["The date remains unknown."],
        receipts=[],
        evidence=[],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "read_provision",
                    "parameters": {"type": "object"},
                },
            }
        ],
        turns=[turn("old-call", records)],
        draft_to_repair="Conditional result [1].",
        publication_gap={"summary": "Reassess material conditions."},
    )
    return current, records, context, ledger


@pytest.mark.parametrize(
    "defect",
    [
        "missing_uncited",
        "missing_cited",
        "text",
        "hash",
        "truncated",
        "source",
        "range",
        "uncitable",
        "conflicting_duplicate",
    ],
)
def test_incomplete_or_conflicting_source_packet_falls_back(defect: str) -> None:
    current, records, _, _ = minimal()
    rows = copy.deepcopy(records)
    if defect == "missing_uncited":
        rows.pop()
    elif defect == "missing_cited":
        rows.pop(0)
    elif defect == "text":
        rows[0]["text"] = "Changed legal condition."
    elif defect == "hash":
        rows[0]["text_hash"] = "0" * 64
    elif defect == "truncated":
        rows[0]["truncated"] = True
    elif defect == "source":
        rows[0]["source_id"] = ""
    elif defect == "range":
        rows[0]["start_char"] = 1
    elif defect == "uncitable":
        rows[0]["citable"] = False
    else:
        rows.append({**rows[0], "metadata": {"changed_version": True}})
    assert prepare_cheap_repair_view(current, rows) is None


def test_actual_fresh_invocation_records_every_source_without_old_call_receipt() -> (
    None
):
    current, records, context, ledger = minimal()
    fresh = prepare_cheap_repair_view(current, records)
    assert fresh is not None
    llm = model(limit=1_000_000)
    llm.invoke.return_value = native_response("read_provision")
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    adapter.decide(fresh)
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2}
    assert ledger.completely_delivered("old-call") == set()
    rows = actual_originals(llm.invoke.call_args.kwargs["prompt"])
    assert {(row["citation"], row["text_hash"], row["text"]) for row in rows} == {
        (row["citation"], row["text_hash"], row["text"]) for row in records
    }


def test_physical_context_cannot_evict_an_uncited_preserved_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current, records, context, ledger = minimal()
    fresh = prepare_cheap_repair_view(current, records)
    assert fresh is not None and fresh.required_evidence_numbers == [1, 2]
    llm = model()
    adapter = ResearchModel(llm, context, lean_native_mode=True, token_counter=len)
    monkeypatch.setattr(adapter, "_limits", lambda _: (1000, 100))
    with pytest.raises(RunStopped, match="required originals"):
        adapter.decide(fresh)
    llm.invoke.assert_not_called()
    assert ledger.export()["deliveries"] == []


def test_observations_keep_failure_scope_and_nonoriginal_result_data_exact() -> None:
    current, records, _, _ = minimal()
    prior = current.turns[0]
    assert prior.assistant.tool_calls is not None
    prior.assistant.content = "Untrusted prior source assessment."
    payload = json.loads(prior.results[0].content)
    payload["outcome"] = {
        "status": "not_found",
        "summary": "A bounded search is not proof of absence.",
        "data": {
            "source_id": "owned",
            "article": "15",
            "searched_range": [0, 1500],
            "continuation_required": True,
            "version_unknown": True,
        },
    }
    prior.results[0].content = json.dumps(payload)
    prior.assistant.tool_calls[
        0
    ].function.arguments = (
        '{"query":"actual scope AND qualification","source_id":"owned"}'
    )
    observations = cheap_repair_observations(current)
    assert observations[0]["result"] == {
        key: value for key, value in payload.items() if key != "original_evidence"
    }
    assert observations[0]["untrusted_assistant_assessment"] == prior.assistant.content
    assert (
        observations[0]["arguments_json"]
        == prior.assistant.tool_calls[0].function.arguments
    )
    assert prepare_cheap_repair_view(current, records) is not None


def test_terminal_observation_removes_only_prior_draft_arguments() -> None:
    current, records, _, _ = minimal()
    prior = current.turns[0]
    assert prior.assistant.tool_calls is not None
    call = prior.assistant.tool_calls[0]
    call.function.name = "submit_retained_partial_answer"
    arguments = {
        "answer": "Prior unapproved answer.",
        "retained_answer_id": "prior-owned-reference",
        "retained_answer_edits": [{"unit_id": "old", "replacement": "Old."}],
        "_coverage": {"unresolved": ["The unknown fact remains open."]},
        "_related_source_reviews": [{"lead_id": "pending", "status": "unavailable"}],
    }
    call.function.arguments = json.dumps(arguments)
    saved = current.model_dump(mode="json")
    observation = cheap_repair_observations(current)[0]
    assert json.loads(str(observation["arguments_json"])) == {
        key: value
        for key, value in arguments.items()
        if key not in {"answer", "retained_answer_id", "retained_answer_edits"}
    }
    assert current.model_dump(mode="json") == saved
    assert prepare_cheap_repair_view(current, records) is not None


@pytest.mark.parametrize("mismatch", ["identity_ref", "metadata_ref", "version"])
def test_changed_prior_identity_or_version_keeps_ordinary_history(
    mismatch: str,
) -> None:
    current, records, _, _ = minimal()
    payload = json.loads(current.turns[0].results[0].content)
    row = payload["original_evidence"][0]
    if mismatch == "identity_ref":
        row["identity_ref"] = 2
    elif mismatch == "metadata_ref":
        row["metadata_ref"] = {"citation": 2, "text_hash": records[1]["text_hash"]}
    else:
        row["metadata"] = {"version_unknown": False, "read_as_of_date": "2025-01-01"}
    current.turns[0].results[0].content = json.dumps(payload)
    assert prepare_cheap_repair_view(current, records) is None


@pytest.mark.parametrize("malformed", ["result", "terminal_arguments"])
def test_unprojectable_history_conservatively_keeps_ordinary_context(
    malformed: str,
) -> None:
    current, records, _, _ = minimal()
    if malformed == "result":
        current.turns[0].results[
            0
        ].content = "Unstructured source scope and unknown fact."
    else:
        calls = current.turns[0].assistant.tool_calls
        assert calls is not None
        calls[0].function.name = "submit_answer"
        calls[0].function.arguments = '{"answer":'
    assert prepare_cheap_repair_view(current, records) is None
