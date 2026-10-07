"""Verify judicial section continuation and exact retained draft transitions."""

import copy
from threading import Barrier
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.judicial_sections import (
    judicial_disposition_missing,
    judicial_witness_section,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.retained_answer import bind_retained_answer, resolve_retained_answer
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import ModelResponse
from tests.unit.onyx.asv3.test_experimental_workflow import terminal_registry
from tests.unit.onyx.asv3.test_legal_source_reviews import deliver, review, seen
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_runtime import response, setup_run, user_payload
from tests.unit.onyx.asv3.test_shared_originals import full_record
from tests.unit.onyx.asv3.test_tuned_focused_reads import judicial_chunk
from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context


def reasoning_originals() -> list[EvidenceItem]:
    first = judicial_chunk("III. ESASIN İNCELENMESİ\nThe court examines the rule.", 0)
    second = judicial_chunk("23. The rule requires a statutory basis.", 1)
    second.metadata["heading_path"] = [
        "Court decision",
        "23. The rule requires a statutory basis.",
    ]
    heading = judicial_chunk("V. HÜKÜM", 2)
    return [first, second, heading]


def test_numbered_reasoning_and_descriptive_metadata_do_not_supply_disposition() -> (
    None
):
    originals = reasoning_originals()
    assert (
        judicial_witness_section(
            originals[1], 0, len(originals[1].text), source_context=originals
        )
        == "reasoning"
    )
    assert judicial_disposition_missing(originals)
    originals.append(
        judicial_chunk(
            "The challenged phrase is annulled with the following limits.", 3
        )
    )
    assert not judicial_disposition_missing(originals)


def test_lettered_party_arguments_return_to_parent_merits_section() -> None:
    originals = [
        judicial_chunk(
            "III. MERITS\nA. Scope of the rule\nThe court examines scope.", 0
        ),
        judicial_chunk(
            "B. Grounds of the application\nThe applicant requests annulment.", 1
        ),
        judicial_chunk(
            "C. Constitutional assessment\n23. The court examines legality.", 2
        ),
    ]
    assert (
        judicial_witness_section(
            originals[1],
            originals[1].text.index("The applicant"),
            len(originals[1].text),
            source_context=originals,
        )
        == "argument_only"
    )
    assert (
        judicial_witness_section(
            originals[2],
            originals[2].text.index("23."),
            len(originals[2].text),
            source_context=originals,
        )
        == "reasoning"
    )
    assert judicial_disposition_missing(originals)


@pytest.mark.parametrize("broken", ["gap", "date", "derived", "source"])
def test_disposition_heading_cannot_leak_across_unverified_context(broken: str) -> None:
    originals = reasoning_originals()
    current = judicial_chunk("The challenged phrase is annulled.", 3)
    if broken == "gap":
        current.metadata["position"] = 4
    elif broken == "date":
        current.metadata["read_as_of_date"] = "2025-01-02"
    elif broken == "derived":
        originals[-1].metadata["derived"] = True
    else:
        originals[-1].source_id = "foreign-decision"
    assert (
        judicial_witness_section(
            current, 0, len(current.text), source_context=originals
        )
        == "unknown"
    )


def test_reasoning_review_stays_open_until_candidate_disposition_body_is_delivered() -> (
    None
):
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(reasoning_originals(), context)
    deliver(ledger, "answer-call", [1, *numbers])
    context.services["last_model_call_id"] = "answer-call"
    original = ledger.get(numbers[1])
    assert original is not None
    witness = {
        "citation": numbers[1],
        "start_char": 0,
        "end_char": len(original.text),
    }
    call = CapabilityCall(
        name="submit_answer",
        arguments={
            "answer": f"The court's outcome changes the rule [{numbers[1]}].",
            "basis": "originals",
            "_related_source_reviews": [review(witnesses=[witness])],
        },
    )
    rejected = terminal_registry([]).dispatch(call, context)
    assert rejected.status == OutcomeStatus.INVALID
    diagnostic = rejected.data["related_source_review_error"]
    assert isinstance(diagnostic, dict)
    assert diagnostic["code"] == "missing_judicial_disposition"
    assert reviews.view(context, ledger, {1, *numbers})["pending_lead_ids"]


def test_pending_reasoning_uses_exact_acquisition_cursor_instead_of_terminal_retry() -> (
    None
):
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(reasoning_originals(), context)
    state = reviews.view(context, ledger, {1, *numbers})
    rows = cast(list[dict[str, JsonValue]], state["reviews"])
    rows[0]["source_range_read"] = {"has_more": True, "next_position": 3}
    actions = ResearchModel._related_source_acquisition_state(state, ledger)
    assert actions == [
        {
            "lead_id": rows[0]["lead_id"],
            "source_id": "decision",
            "reason": "reasoning_without_disposition_body",
            "suggested_acquisition": {
                "name": "read_source_range",
                "arguments": {"source_id": "decision", "start": 3},
            },
        }
    ]
    number = ledger.add([judicial_chunk("The request is granted.", 3)], context)[0]
    rows[0]["available_original_citations"] = [*numbers, number]
    assert ResearchModel._related_source_acquisition_state(state, ledger) == []


def test_saved_reasoning_assessment_does_not_bypass_tuned_disposition_check() -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(reasoning_originals(), context)
    original = ledger.get(numbers[1])
    assert original is not None
    deliver(ledger, "prior-call", [1, *numbers])
    context.services["last_model_call_id"] = "prior-call"
    context.services.pop("asv3_workflow_variant")
    context.services["research_profile"] = "experimental"
    assessment = review(
        witnesses=[
            {
                "citation": numbers[1],
                "start_char": 0,
                "end_char": len(original.text),
            }
        ]
    )
    outcome = terminal_registry([]).dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": f"Rule [{numbers[1]}].",
                "basis": "originals",
                "_related_source_reviews": [assessment],
            },
        ),
        context,
    )
    assert outcome.status == OutcomeStatus.FOUND
    context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    context.services["research_profile"] = "normal"
    state = reviews.view(context, ledger, {1, *numbers})
    assert state["pending_lead_ids"]
    rows = cast(list[dict[str, JsonValue]], state["reviews"])
    assert rows[0]["status"] == "pending" and rows[0]["review"] is None
    assert (
        reviews.publication_gap(f"Rule [{numbers[1]}].", "prior-call", context, ledger)
        is not None
    )


def test_tuned_native_commit_keeps_whole_draft_and_fresh_source_delivery() -> None:
    context, ledger, _reviews = tuned_context()
    draft = "First supported condition [1].\n\nSecond supported exception [1].\n"
    selected = model()
    selected.invoke.return_value = native_action("submit_retained_answer", {})
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)]).model_copy(
        update={
            "tools": terminal_registry([]).definitions(context),
            "draft_to_repair": draft,
            "publication_gap": {"summary": "Correct publication metadata."},
        }
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    decision = adapter.decide(current)
    payload = last_payload(selected)
    assert "retained_answer" in payload
    assert decision.calls[0].name == "submit_answer"
    assert decision.calls[0].arguments["answer"] == draft
    assert decision.calls[0].arguments["basis"] == "originals"
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1}
    assert selected.invoke.call_count == 1


def test_protected_normal_and_plain_experimental_do_not_expose_retained_commits() -> (
    None
):
    definitions = terminal_registry([]).definitions(RunContext())
    for profile in ("normal", "experimental"):
        context = RunContext(
            services={"research_profile": profile, "experimental_parallel": False}
        )
        bound, reference = bind_retained_answer(
            copy.deepcopy(definitions), context, "Some draft [1].", request="Question"
        )
        assert reference is None
        assert bound == definitions


def test_retained_partial_expands_to_actual_canonical_partial_schema() -> None:
    context, _ledger, _reviews = tuned_context()
    registry = terminal_registry([])
    observed: list[dict[str, JsonValue]] = []

    def submit(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observed.append(arguments)
        return ToolOutcome(status=OutcomeStatus.PARTIAL, summary="Partial answer")

    registry.register(
        ToolSpec(
            name="submit_partial_answer",
            description="End with supported parts and a precise gap.",
            parameters={
                "type": "object",
                "properties": {"answer": {"type": "string", "minLength": 1}},
                "required": ["answer"],
                "additionalProperties": False,
            },
            handler=submit,
        )
    )
    draft = "Supported portion [1].\n\nThe exact source interaction remains unresolved."
    resolved = resolve_retained_answer(
        Decision(calls=[CapabilityCall(name="submit_retained_partial_answer")]),
        context,
        draft,
        request="Question",
    )
    assert resolved.calls[0].arguments == {"answer": draft}
    assert registry.dispatch(resolved.calls[0], context).status == OutcomeStatus.PARTIAL
    assert observed == [{"answer": draft}]


@pytest.mark.parametrize(
    "terminal", ["submit_retained_answer", "submit_retained_partial_answer"]
)
def test_invalid_owned_alias_keeps_specific_repair_error(terminal: str) -> None:
    context, _ledger, _reviews = tuned_context()
    registry = terminal_registry([])
    registry.register(
        ToolSpec(
            name="submit_partial_answer",
            description="Submit a partial answer.",
            parameters={"type": "object"},
            handler=lambda _args, _child: ToolOutcome(
                status=OutcomeStatus.PARTIAL, summary="Accepted"
            ),
        )
    )
    decision = resolve_retained_answer(
        Decision(
            calls=[
                CapabilityCall(
                    name=terminal,
                    arguments={
                        "retained_answer_edits": [
                            {"unit_id": "stale-unit", "replacement": "Changed text."}
                        ]
                    },
                )
            ]
        ),
        context,
        "Owned draft [1].",
        request="Question",
    )
    call = decision.calls[0]
    assert call.name in {"submit_answer", "submit_partial_answer"}
    assert (
        call.argument_error is not None
        and "current owned unit_id" in call.argument_error
    )
    outcome = registry.dispatch(call, context)
    assert outcome.status == OutcomeStatus.INVALID
    assert outcome.data["argument_error"] == call.argument_error
    assert outcome.summary != "Unknown capability"


@pytest.mark.parametrize(
    "status", [OutcomeStatus.PARTIAL, OutcomeStatus.INVALID, OutcomeStatus.DENIED]
)
def test_latest_rejected_partial_replaces_previous_draft_and_gap(
    status: OutcomeStatus,
) -> None:
    context, _ledger, _reviews = tuned_context()
    candidate = "A different current answer [1]."
    gap = ToolOutcome(
        status=status,
        summary="Current source interaction is unread.",
        data={"pending_related_source_review": True},
    )
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_partial_answer",
                description="Submit a partial answer.",
                parameters={"type": "object"},
                handler=lambda _args, _child: gap,
            )
        ]
    )
    calls = 0

    def decide(view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        if calls == 1:
            return Decision(
                calls=[
                    CapabilityCall(
                        name="submit_partial_answer", arguments={"answer": candidate}
                    )
                ]
            )
        assert view.draft_to_repair == candidate
        assert view.publication_gap == {"summary": gap.summary, **gap.data}
        return Decision(answer="Finished")

    harness = Harness(
        request="Question", context=context, registry=registry, decide=decide
    )
    harness.last_draft = "An obsolete answer [1]."
    harness.publication_gap = ToolOutcome(
        status=OutcomeStatus.PARTIAL, summary="Obsolete authority gap."
    )
    assert harness.run().answer == "Finished"
    assert calls == 2


@pytest.mark.usefixtures("source_use_review_not_under_test")
def test_tuned_runtime_retains_first_partial_rejection_for_owned_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    broker.barrier = Barrier(1)
    source_id = str(broker.sources[0].id)
    calls = 0
    rejected = "Supported factual condition [999]."
    repaired = "Supported factual condition [1]."

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": source_id, "_language": "tr"})
                ]
            )
        if calls == 2:
            return response(calls=[("submit_partial_answer", {"answer": rejected})])
        assert calls == 3
        footer = user_payload(arguments["prompt"][-1])
        assert footer["publication_gap"]["unknown_citations"] == [999]
        units = footer["draft_to_repair"]["units"]
        assert "".join(unit["text"] for unit in units) == rejected
        return response(
            calls=[
                (
                    "submit_retained_partial_answer",
                    {
                        "retained_answer_edits": [
                            {"unit_id": units[0]["unit_id"], "replacement": repaired}
                        ]
                    },
                )
            ]
        )

    selected.invoke.side_effect = scripted
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    runtime.run_asv3_loop(**kwargs)
    assert calls == 3
    assert checkpoints[-1]["publication_status"] == "partial"
    assert checkpoints[-1]["final_publication_gap"] is None
    assert kwargs["state_container"].answer_tokens == repaired.replace(
        "[1]", "[[1]](https://example.test/law-0)"
    )


def test_tuned_omission_diagnostic_supplies_only_missing_declared_passage() -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    number = ledger.add(
        [
            judicial_chunk(
                "V. HÜKÜM\nThe challenged phrase is annulled within the stated limits.",
                0,
            )
        ],
        context,
    )[0]
    original = ledger.get(number)
    assert original is not None
    deliver(ledger, "current-call", [1, number])
    assessment = review(
        witnesses=[
            {"citation": number, "start_char": 0, "end_char": len(original.text)}
        ]
    )
    answer = f"The rule [1].\n\n{assessment['effect']} [{number}]."
    gap = reviews.publication_gap(
        answer, "current-call", context, ledger, raw_reviews=[assessment]
    )
    assert gap is not None
    omissions = cast(
        list[dict[str, JsonValue]], gap.data["unretained_examined_source_effects"]
    )
    assert omissions[0]["required_answer_passages"] == {
        "limitations": assessment["limitations"]
    }
    complete = f"{answer}\n\n{assessment['limitations']} [{number}]."
    assert (
        reviews.publication_gap(
            complete, "current-call", context, ledger, raw_reviews=[assessment]
        )
        is None
    )


@pytest.mark.parametrize("tuned", [True, False])
def test_selected_disposition_cannot_be_closed_as_a_party_argument(tuned: bool) -> None:
    context, ledger, reviews = tuned_context()
    if not tuned:
        context.services["asv3_workflow_variant"] = "standard"
    seen(context, ledger, reviews)
    number = ledger.add(
        [judicial_chunk("V. HÜKÜM\nThe identified phrase is annulled.", 0)], context
    )[0]
    original = ledger.get(number)
    assert original is not None
    deliver(ledger, "current-call", [1, number])
    assessment = review(
        status="not_material",
        source_role="argument_only",
        witnesses=[
            {
                "citation": number,
                "start_char": original.text.index("The identified"),
                "end_char": len(original.text),
            }
        ],
    )
    if tuned:
        with pytest.raises(ValueError, match="not merely a party argument"):
            reviews.apply([assessment], "current-call", context, ledger)
        assert reviews.view(context, ledger, {1, number})["pending_lead_ids"]
        assessment["source_role"] = "operative_text"
    reviews.apply([assessment], "current-call", context, ledger)
    assert reviews.view(context, ledger, {1, number})["pending_lead_ids"] == []


def test_publication_acquisition_reads_continuation_before_one_semantic_decision() -> (
    None
):
    context, old_ledger, reviews = tuned_context()
    law = old_ledger.get(1)
    assert law is not None
    ledger = EvidenceLedger()
    ledger.add([law], context)
    context.services["evidence"] = ledger
    deliver(ledger, "law-call", [1])
    seen(context, ledger, reviews)
    acquired: list[int] = []
    registry = terminal_registry([])

    def read(arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        assert child.scope == context.scope
        assert arguments["source_id"] == "decision"
        start = arguments.get("start", 0)
        assert type(start) is int
        acquired.append(start)
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL if start == 0 else OutcomeStatus.FOUND,
            summary="Canonical source page",
            data={"has_more": start == 0, "next_position": 3 if start == 0 else 4},
            evidence=reasoning_originals()
            if start == 0
            else [judicial_chunk("The identified phrase is annulled.", 3)],
        )

    registry.register(
        ToolSpec(
            name="read_source_range",
            description="Read a bounded authorized canonical source page.",
            parameters={
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                },
                "required": ["source_id"],
                "additionalProperties": False,
            },
            handler=read,
        )
    )
    assessment = review(witnesses=[{"citation": 5, "start_char": 0, "end_char": 32}])
    answer = f"The rule [1].\n\n{assessment['effect']} {assessment['limitations']} [5]."

    def submit(arguments: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        body = str(arguments["answer"])
        gap = reviews.publication_gap(
            body,
            str(context.services["last_model_call_id"]),
            context,
            ledger,
        )
        assert gap is None
        context.services["submitted_answer"] = body
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Validated answer")

    spec = registry.get("submit_answer")
    read_spec = registry.get("read_source_range")
    assert spec is not None
    assert read_spec is not None
    registry = CapabilityRegistry(
        [read_spec, spec.model_copy(update={"handler": submit})]
    )
    draft = "The ordinary rule certainly applies [1]."
    _, reference = bind_retained_answer(
        registry.definitions(context), context, draft, request="Question"
    )
    assert reference is not None
    units = cast(list[dict[str, JsonValue]], reference["units"])
    selected = model()
    selected.invoke.return_value = native_action(
        "submit_retained_answer",
        {
            "retained_answer_edits": [
                {"unit_id": units[0]["unit_id"], "replacement": answer}
            ],
            "_related_source_reviews": [assessment],
        },
    )
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    harness = Harness(
        request="Question",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=adapter.decide,
    )
    harness.last_draft = draft
    harness.publication_gap = ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="Related original is unread.",
        data={"pending_related_source_review": True},
    )
    result = harness.run()
    assert result.answer == answer and result.status == OutcomeStatus.FOUND, (
        result.stop_reason,
        [(r.call.name, r.outcome.summary, r.outcome.data) for r in result.receipts],
    )
    assert acquired == [0, 3]
    assert selected.invoke.call_count == 1
    assert ledger.completely_delivered(adapter.last_call_id or "") == {1, 2, 3, 4, 5}


@pytest.mark.parametrize("barrier", ["normal", "experimental", "uncited", "failed"])
def test_publication_acquisition_is_scoped_and_never_repeats_failed_read(
    barrier: str,
) -> None:
    context, old_ledger, reviews = tuned_context()
    law = old_ledger.get(1)
    assert law is not None
    ledger = EvidenceLedger()
    ledger.add([law], context)
    context.services["evidence"] = ledger
    deliver(ledger, "law-call", [1])
    seen(context, ledger, reviews)
    current = adaptive_tool_view(original_evidence=[full_record(ledger, 1)]).model_copy(
        update={
            "draft_to_repair": "The rule [1]." if barrier != "uncited" else "Hello.",
            "publication_gap": {"pending_related_source_review": True},
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_source_range",
                        "parameters": {"type": "object"},
                    },
                }
            ],
        }
    )
    if barrier in {"normal", "experimental"}:
        context.services.pop("asv3_workflow_variant")
        context.services["research_profile"] = barrier
    elif barrier == "failed":
        current.receipts.append(
            ToolReceipt(
                call=CapabilityCall(
                    name="read_source_range", arguments={"source_id": "decision"}
                ),
                outcome=ToolOutcome(
                    status=OutcomeStatus.DENIED, summary="Access denied"
                ),
                elapsed_seconds=0,
            )
        )
    assert (
        ResearchModel(model(), context)._publication_source_acquisition(current) is None
    )
