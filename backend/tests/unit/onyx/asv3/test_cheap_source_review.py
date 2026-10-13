"""Independent repair completeness requires immutable originals and actual receipts."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import AssertionWitness, assertion_inventory
from onyx.asv3.cheap_source_review import (
    CheapConditionAudit,
    CheapConditionCheck,
    CheapConditionResolution,
    CheapSourceRequirementInventory,
    CheapSourceReviewPolicy,
    CompactSourceRequirement,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    MaterialSourceOmission,
    ResearchModel,
)
from onyx.asv3.models import RunContext, RunStopped
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.models import UserMessage
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger, selected_model
from tests.unit.onyx.asv3.test_model_adapter import text_response


@pytest.fixture
def policy_model() -> tuple[
    CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
]:
    ledger, context = original_ledger()
    context.services.update(
        asv3_exp_cheap=True,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
    )
    context.services["last_model_call_id"] = "writer"
    model, llm = selected_model(context)
    model.lean_native_mode = True
    llm.config.max_input_tokens = 200000
    model.token_counter = lambda value: max(1, len(value) // 4)
    model._native_output_capacity = 128000
    model.last_call_id = "writer"
    policy = CheapSourceReviewPolicy(
        context, "A requested operation", ["What procedure applies?"]
    )
    return policy, model, llm, ledger


def requirement(ledger: EvidenceLedger) -> MaterialSourceOmission:
    item = ledger.get(3)
    assert item is not None
    return MaterialSourceOmission(
        witness=AssertionWitness(citation=3, source_quote="condition AND exception"),
        determination_ids=["q0:d0"],
        detail="The operation preserves the condition and its exception.",
        applicability="The requested operation depends on this source restriction.",
    )


def inventory(
    policy: CheapSourceReviewPolicy,
    model: ResearchModel,
    llm: MagicMock,
    ledger: EvidenceLedger,
) -> None:
    llm.invoke.return_value = text_response(inventory_output(policy, ledger))
    gap = policy.ensure_inventory(model, ledger)
    assert gap is None, gap.data if gap is not None else None


def inventory_output(
    policy: CheapSourceReviewPolicy,
    ledger: EvidenceLedger,
    *,
    source_quote: str = "condition AND exception",
) -> dict[str, JsonValue]:
    item = requirement(ledger)
    return CheapSourceRequirementInventory(
        source_scope_id=cast(str, policy._payload(ledger)[0]["source_scope_id"]),
        whole_scope_completed=True,
        requirements=[
            CompactSourceRequirement(
                citation=item.witness.citation,
                source_quote=source_quote,
                **item.model_dump(exclude={"witness"}),
            )
        ],
    ).model_dump(mode="json")


def assessment(
    policy: CheapSourceReviewPolicy, answer: str, *, disposition: str = "covered"
) -> dict[str, JsonValue]:
    retained = policy.required_conditions()[0]
    unit = assertion_inventory(answer)[0]
    assert policy._ledger is not None
    return CheapConditionAudit(
        source_scope_id=cast(
            str, policy._payload(policy._ledger)[0]["source_scope_id"]
        ),
        whole_scope_completed=True,
        conditions=[],
        resolutions=[
            CheapConditionResolution.model_validate(
                {
                    "condition_id": retained["condition_id"],
                    "disposition": disposition,
                    "answer_unit_ids": [unit["unit_id"]]
                    if disposition == "covered"
                    else [],
                    "scenario_quotes": [],
                    "missing_details": [
                        {
                            "source_quote": "exception",
                            "detail": "Communicate the original exception.",
                        }
                    ]
                    if disposition in {"omitted", "uncertain"}
                    else [],
                    "affected_answer_unit_ids": [unit["unit_id"]]
                    if disposition in {"omitted", "uncertain"}
                    else [],
                }
            )
        ],
    ).model_dump(mode="json")


def test_untouched_cheap_answer_adds_no_paid_review_stage(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    assert (
        policy.publication_gap("The operation is permitted [1].", model, ledger) is None
    )
    assert llm.invoke.call_count == 0
    assert policy.export()["repair_requested"] is False


def test_answer_blind_inventory_delivers_all_originals_without_changing_writer_receipt(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    assert model.last_call_id == "writer"
    assert model.context.services["last_model_call_id"] == "writer"
    assert len(policy.required_conditions()) == 1
    call = llm.invoke.call_args.kwargs
    message = call["prompt"][-1]
    assert isinstance(message, UserMessage) and isinstance(message.content, str)
    payload = json.loads(message.content)
    assert payload["required_evidence_numbers"] == [1, 2, 3]
    assert "answer_units" not in payload and "candidate_hash" not in payload
    assert "retained_conditions" not in payload
    assert "source_quote" in str(call["prompt"][0])
    receipts = ledger.export()["deliveries"]
    assert isinstance(receipts, list) and len(receipts) == 1
    receipt = receipts[0]
    assert isinstance(receipt, dict) and isinstance(receipt["call_id"], str)
    assert receipt["flow"] == LLMFlow.ASV3_SOURCE_INVENTORY.value
    assert ledger.completely_delivered(receipt["call_id"]) == {1, 2, 3}
    assert call["max_tokens"] == 24000
    assert policy.ensure_inventory(model, ledger) is None
    assert llm.invoke.call_count == 1


@pytest.mark.parametrize(
    "disposition,citation,blocked",
    [
        ("covered", 3, False),
        ("covered", 1, True),
        ("omitted", 3, True),
        ("uncertain", 3, True),
    ],
)
def test_current_answer_needs_the_condition_and_its_inline_original(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
    disposition: str,
    citation: int,
    blocked: bool,
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    answer = f"Preserve the condition and exception [{citation}]."
    llm.invoke.return_value = text_response(
        assessment(policy, answer, disposition=disposition)
    )
    gap = policy.publication_gap(answer, model, ledger)
    assert (gap is not None) is blocked
    assert llm.invoke.call_count == 2
    assert (
        model.last_call_id == "writer"
        and model.context.services["last_model_call_id"] == "writer"
    )
    assert len(policy.required_conditions()) == 1


def test_a_broad_replacement_cannot_silently_drop_retained_restriction(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    before = policy.required_conditions()
    llm.invoke.return_value = text_response(
        {
            "source_scope_id": policy._payload(ledger)[0]["source_scope_id"],
            "whole_scope_completed": True,
            "conditions": [],
            "resolutions": [],
        }
    )
    gap = policy.publication_gap(
        "The general rule permits the operation [1].", model, ledger
    )
    assert gap is not None and "not reassessed" in str(gap.data)
    assert policy.required_conditions() == before


def test_only_the_exact_current_candidate_can_reuse_a_real_review_receipt(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    answer = "Preserve condition AND exception [3]."
    llm.invoke.return_value = text_response(assessment(policy, answer))
    assert policy.publication_gap(answer, model, ledger) is None
    assert policy.publication_gap(answer, model, ledger) is None
    assert llm.invoke.call_count == 2
    changed = "Preserve condition AND exception [3]. "
    llm.invoke.return_value = text_response(assessment(policy, changed))
    assert policy.publication_gap(changed, model, ledger) is None
    assert llm.invoke.call_count == 3


def test_inventory_rejects_a_paraphrased_witness_or_unexamined_original(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    llm.invoke.return_value = text_response(
        inventory_output(policy, ledger, source_quote="condition or exception")
    )
    assert policy.ensure_inventory(model, ledger) is not None
    assert not policy.required_conditions()


def test_inventory_rejects_missing_actual_delivery_receipt(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, model, llm, ledger = policy_model
    monkeypatch.setattr(ledger, "record_delivery", lambda *_args, **_kwargs: None)
    llm.invoke.return_value = text_response(inventory_output(policy, ledger))
    assert policy.ensure_inventory(model, ledger) is not None
    assert not policy.required_conditions()


def test_checkpoint_preserves_requirements_without_inheriting_review_approval(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    answer = "Preserve condition AND exception [3]."
    llm.invoke.return_value = text_response(assessment(policy, answer))
    assert policy.publication_gap(answer, model, ledger) is None
    snapshot = policy.export()
    assert "receipts" not in str(snapshot) and "reviews" not in snapshot
    restored = CheapSourceReviewPolicy(
        model.context, policy.scenario, list(policy.questions)
    )
    restored.restore(snapshot, ledger)
    assert restored.required_conditions() == policy.required_conditions()
    inventory(restored, model, llm, ledger)
    llm.invoke.return_value = text_response(assessment(restored, answer))
    assert restored.publication_gap(answer, model, ledger) is None
    assert llm.invoke.call_count == 4
    foreign = CheapSourceReviewPolicy(
        model.context, "Different factual scenario", list(policy.questions)
    )
    with pytest.raises(ValueError, match="scenario"):
        foreign.restore(snapshot, ledger)


def test_scope_change_and_cancellation_cannot_approve_repaired_answer(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    policy.mark_repair_requested()
    model.context.scope["document_sets"] = [999]
    assert policy.publication_gap("A rule [3].", model, ledger) is not None
    assert llm.invoke.call_count == 0
    model.context.scope.clear()
    model.context.cancel()
    with pytest.raises(RunStopped, match="cancelled"):
        policy.publication_gap("A rule [3].", model, ledger)


def test_restore_rejects_changed_canonical_source_binding(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    snapshot = policy.export()
    bindings = snapshot["source_bindings"]
    assert isinstance(bindings, list) and isinstance(bindings[0], dict)
    bindings[0]["source_id"] = "foreign-identical-text-source"
    restored = CheapSourceReviewPolicy(
        model.context, policy.scenario, list(policy.questions)
    )
    with pytest.raises(ValueError, match="canonical source identity"):
        restored.restore(snapshot, ledger)


def test_truncated_inventory_cannot_retain_or_approve_requirements(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    truncated = text_response(inventory_output(policy, ledger))
    truncated.choice.finish_reason = "max_output_tokens"
    llm.invoke.return_value = truncated
    assert policy.ensure_inventory(model, ledger) is not None
    assert not policy.required_conditions()


def test_review_context_retains_selected_model_capacity_budget_and_cancel(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, _llm, _ledger = policy_model
    reviewer = policy._reviewer(model)
    assert reviewer.llm is model.llm
    assert reviewer.context is not model.context
    assert reviewer.context.services is not model.context.services
    assert reviewer.context.budget is model.context.budget
    assert reviewer.context.scope is model.context.scope
    assert reviewer.context.deadline == model.context.deadline
    assert reviewer._native_output_capacity == model._native_output_capacity
    reviewer.context.services["last_model_call_id"] = "review"
    assert model.context.services["last_model_call_id"] == "writer"
    model.context.cancel()
    with pytest.raises(RunStopped):
        reviewer.context.check_active()


def test_noncheap_workflow_cannot_construct_the_added_review() -> None:
    with pytest.raises(ValueError, match="root workflow"):
        CheapSourceReviewPolicy(RunContext(), "facts", ["Question?"])


@pytest.mark.parametrize("phase", ["inventory", "review"])
@pytest.mark.parametrize("invalid", ["incomplete", "different_scope"])
def test_receipt_alone_never_substitutes_for_model_whole_scope_declaration(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
    phase: str,
    invalid: str,
) -> None:
    policy, model, llm, ledger = policy_model
    answer = "Preserve condition AND exception [3]."
    if phase == "inventory":
        output = inventory_output(policy, ledger)
    else:
        inventory(policy, model, llm, ledger)
        output = assessment(policy, answer)
    before = policy.required_conditions()
    output[
        "whole_scope_completed" if invalid == "incomplete" else "source_scope_id"
    ] = False if invalid == "incomplete" else "a" * 64
    llm.invoke.return_value = text_response(output)
    gap = (
        policy.ensure_inventory(model, ledger)
        if phase == "inventory"
        else policy.publication_gap(answer, model, ledger)
    )
    assert gap is not None and "exact whole source scope" in str(gap.data)
    assert policy.required_conditions() == before
    assert not policy._reviews


def test_review_scope_declaration_requires_its_own_actual_full_delivery(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    answer = "Preserve condition AND exception [3]."
    llm.invoke.return_value = text_response(assessment(policy, answer))
    monkeypatch.setattr(ledger, "record_delivery", lambda *_args, **_kwargs: None)
    gap = policy.publication_gap(answer, model, ledger)
    assert gap is not None and "actual complete condition-review receipt" in str(
        gap.data
    )
    assert not policy._reviews


@pytest.mark.parametrize(
    "invalid",
    [
        "no_delta",
        "paraphrase",
        "outside_witness",
        "stale_unit",
        "duplicate_unit",
        "closed_delta",
    ],
)
def test_repair_deltas_require_literal_witness_and_exact_current_units(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
    invalid: str,
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    before = policy.required_conditions()
    answer = "Preserve the condition [3]."
    output = assessment(policy, answer, disposition="omitted")
    resolution = cast(list[dict[str, JsonValue]], output["resolutions"])[0]
    details = cast(list[dict[str, JsonValue]], resolution["missing_details"])
    if invalid == "no_delta":
        resolution["missing_details"] = []
    elif invalid == "paraphrase":
        details[0]["source_quote"] = "another exception"
    elif invalid == "outside_witness":
        details[0]["source_quote"] = "original tail"
    elif invalid == "stale_unit":
        resolution["affected_answer_unit_ids"] = ["au-stale"]
    elif invalid == "duplicate_unit":
        ids = cast(list[JsonValue], resolution["affected_answer_unit_ids"])
        resolution["affected_answer_unit_ids"] = ids + ids
    else:
        resolution["disposition"] = "covered"
        resolution["answer_unit_ids"] = resolution["affected_answer_unit_ids"]
    llm.invoke.return_value = text_response(output)
    assert policy.publication_gap(answer, model, ledger) is not None
    assert policy.required_conditions() == before
    assert not policy._reviews


def test_precise_gap_preserves_immutable_requirement_and_all_citing_unit_navigation(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    before = policy.required_conditions()
    answer = (
        "Preserve the condition [3].\n\nThe applicable operation requires proof [3]."
    )
    llm.invoke.return_value = text_response(
        assessment(policy, answer, disposition="omitted")
    )
    gap = policy.publication_gap(answer, model, ledger)
    assert gap is not None
    deltas = cast(list[dict[str, JsonValue]], gap.data["repair_deltas"])
    assert deltas[0]["condition_id"] == before[0]["condition_id"]
    assert deltas[0]["affected_answer_unit_ids"] == [
        assertion_inventory(answer)[0]["unit_id"]
    ]
    assert deltas[0]["missing_details"] == [
        {"source_quote": "exception", "detail": "Communicate the original exception."}
    ]
    assert policy.required_conditions() == before
    expected = [
        unit["unit_id"]
        for unit in assertion_inventory(answer)
        if not unit["presentation_only"] and 3 in unit["evidence_numbers"]
    ]
    assert len(expected) == 2
    assert policy._navigation(answer)[0]["inline_answer_unit_ids"] == expected


def test_repeated_condition_cannot_change_immutable_applicability(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    inventory(policy, model, llm, ledger)
    before = policy.required_conditions()
    answer = "Preserve condition AND exception [3]."
    item = requirement(ledger).model_dump()
    item["applicability"] = "A different, broader operation."
    output = assessment(policy, answer)
    output["resolutions"] = []
    output["conditions"] = [
        CheapConditionCheck(
            **item,
            disposition="covered",
            answer_unit_ids=[assertion_inventory(answer)[0]["unit_id"]],
        ).model_dump(mode="json")
    ]
    llm.invoke.return_value = text_response(output)
    gap = policy.publication_gap(answer, model, ledger)
    assert gap is not None and "immutable" in str(gap.data)
    assert policy.required_conditions() == before


def test_newly_retained_covered_condition_does_not_repeat_exact_same_review(
    policy_model: tuple[
        CheapSourceReviewPolicy, ResearchModel, MagicMock, EvidenceLedger
    ],
) -> None:
    policy, model, llm, ledger = policy_model
    output = inventory_output(policy, ledger)
    output["requirements"] = []
    llm.invoke.return_value = text_response(output)
    assert policy.ensure_inventory(model, ledger) is None
    answer = "Preserve condition AND exception [3]."
    audit = CheapConditionAudit(
        source_scope_id=cast(str, policy._payload(ledger)[0]["source_scope_id"]),
        whole_scope_completed=True,
        conditions=[
            CheapConditionCheck(
                **requirement(ledger).model_dump(),
                disposition="covered",
                answer_unit_ids=[assertion_inventory(answer)[0]["unit_id"]],
            )
        ],
    )
    llm.invoke.return_value = text_response(audit.model_dump(mode="json"))
    assert policy.publication_gap(answer, model, ledger) is None
    assert len(policy.required_conditions()) == 1
    assert policy.publication_gap(answer, model, ledger) is None
    assert policy.publication_gap(answer, model, ledger) is None
    assert llm.invoke.call_count == 2
