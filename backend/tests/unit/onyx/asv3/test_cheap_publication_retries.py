"""Only identical no-progress Cheap publication retries are bounded."""

import copy
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate, RequestedOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import ResearchNeed, ResearchState, ResearchUpdate
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from tests.unit.onyx.asv3.test_native_authority import original

CANDIDATE = "Kaynak destekli olduğu iddia edilen sonuç [1]."
GAP = ToolOutcome(
    status=OutcomeStatus.PARTIAL,
    summary="A governing original is missing.",
    data={"retained_authority_requirements": [{"article": "15"}]},
)


def setup(cheap: bool = True) -> tuple[RunContext, EvidenceLedger, CapabilityRegistry]:
    context = RunContext(
        services={
            "asv3_exp_cheap": cheap,
            "asv3_workflow_variant": ASV3_TUNED_VARIANT,
            "research_profile": "normal",
        },
        budget=SharedBudget(unlimited_execution=True),
        timeout_seconds=float("inf"),
    )
    evidence = EvidenceLedger()
    evidence.add([original("Faaliyet Kanunu", "14")], context)
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Submit a source-bound answer.",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                handler=lambda _args, _context: GAP.model_copy(deep=True),
            ),
            ToolSpec(
                name="research",
                description="Useful independent research.",
                parameters={"type": "object"},
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="Research complete."
                ),
            ),
        ]
    )
    return context, evidence, registry


def rejected(candidate: str, tools: bool) -> Decision:
    return (
        Decision(
            calls=[
                CapabilityCall(name="submit_answer", arguments={"answer": candidate})
            ]
        )
        if tools
        else Decision(answer=candidate)
    )


@pytest.mark.parametrize("tools", [True, False])
def test_identical_candidate_gap_and_state_stops_after_two_fresh_and_one_broader_retry(
    tools: bool,
) -> None:
    context, evidence, registry = setup()
    calls = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        assert calls <= 4
        return rejected(CANDIDATE, tools)

    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        evidence=evidence,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
    )
    result = harness.run()
    assert calls == 4
    assert result.status == OutcomeStatus.PARTIAL
    assert result.stop_reason == "repeated_publication_gap"
    assert result.answer is None
    assert harness.last_draft == CANDIDATE
    assert result.publication_gap == GAP
    assert context.budget.unlimited_execution
    assert context.deadline == float("inf")
    retry = harness.snapshot()["cheap_publication_retry"]
    assert isinstance(retry, dict) and retry["attempts"] == 4


@pytest.mark.parametrize("tools", [True, False])
def test_experimental_raw_stop_and_tool_retries_are_unchanged(tools: bool) -> None:
    context, evidence, registry = setup(cheap=False)
    calls = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        assert calls <= 5
        return Decision(answer="Verified") if calls == 5 else rejected(CANDIDATE, tools)

    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        evidence=evidence,
        draft_guard=lambda answer: (
            None if answer == "Verified" else GAP.model_copy(deep=True)
        ),
    )
    result = harness.run()
    assert result.answer == ("Verified" if tools else None)
    assert calls == (5 if tools else 3)
    assert result.status == (OutcomeStatus.FOUND if tools else OutcomeStatus.PARTIAL)
    assert "cheap_publication_retry" not in harness.snapshot()


@pytest.mark.parametrize(
    "change", ["candidate", "fact", "question", "evidence", "metadata", "outcome"]
)
@pytest.mark.parametrize("tools", [True, False])
def test_new_semantic_state_resets_duplicate_rejection_count(
    change: str, tools: bool
) -> None:
    context, evidence, registry = setup()
    outcomes = OutcomeMap(["Research"], context)
    context.services["outcome_map"] = outcomes
    calls = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        assert calls <= 7
        candidate = CANDIDATE
        if calls >= 4 and change == "candidate":
            candidate += " Yeni koşul açıklaması."
        if calls == 4:
            if change == "fact":
                return rejected(candidate, tools).model_copy(
                    update={"facts": ["New material fact"]}
                )
            if change == "question":
                return rejected(candidate, tools).model_copy(
                    update={"questions": ["New requested issue"]}
                )
            if change == "evidence":
                evidence.add([original("Faaliyet Kanunu", "15")], context)
            if change == "metadata":
                # A refreshed canonical version is substantive, even with identical literal text.
                item = evidence.get(1)
                assert item is not None
                item.metadata["publication_revision"] = "new-version"
                saved = evidence.export()
                rows = cast(list[dict[str, JsonValue]], saved["records"])
                rows[0]["item"] = item.model_dump(mode="json")
                evidence.restore(saved, context)
            if change == "outcome":
                outcomes.update(
                    OutcomeUpdate(
                        outcomes=[
                            RequestedOutcome(
                                outcome_id="condition",
                                question_ids=["q0"],
                                detail="New decisive outcome",
                            )
                        ]
                    ),
                    evidence,
                )
        return rejected(candidate, tools)

    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        evidence=evidence,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
    )
    assert harness.run().stop_reason == "repeated_publication_gap"
    assert calls == 7


def test_revision_only_outcome_change_does_not_reset_and_research_actions_are_not_capped() -> (
    None
):
    context, evidence, registry = setup()
    outcomes = OutcomeMap(["Research"], context)
    context.services["outcome_map"] = outcomes
    calls = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        if calls <= 70:
            return Decision(calls=[CapabilityCall(name="research")])
        assert calls <= 74
        outcomes.revision += 1
        return rejected(CANDIDATE, tools=False)

    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        evidence=evidence,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
    )
    assert harness.run().stop_reason == "repeated_publication_gap"
    assert calls == 74
    assert context.budget.snapshot()["tools"] == 70


def test_checkpoint_continues_duplicate_count_and_rejects_tampering() -> None:
    context, evidence, registry = setup()
    checkpoints: list[dict[str, JsonValue]] = []
    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=lambda _view: rejected(CANDIDATE, tools=False),
        evidence=evidence,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
        checkpoint=checkpoints.append,
    )
    harness.run()
    after_two = next(
        saved
        for saved in checkpoints
        if cast(dict, saved["cheap_publication_retry"])["attempts"] == 2
    )
    calls = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        assert calls <= 2
        return rejected(CANDIDATE, tools=False)

    restored = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
    )
    restored.restore(after_two)
    assert restored.run().stop_reason == "repeated_publication_gap"
    assert calls == 2
    for key, value in (
        ("version", True),
        ("attempts", True),
        ("attempts", 5),
        ("fingerprint", "not-a-hash"),
        ("extra", 1),
    ):
        changed = copy.deepcopy(after_two)
        retries = changed["cheap_publication_retry"]
        assert isinstance(retries, dict)
        retries[key] = value
        with pytest.raises(ValueError):
            restored.restore(changed)


def test_new_lookup_observation_is_used_before_rejection_counter_restarts() -> None:
    context, evidence, registry = setup()
    calls = 0

    def decide(view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        assert calls <= 8
        if calls == 4:
            return Decision(calls=[CapabilityCall(name="research")])
        if calls == 5:
            assert view.receipts[-1].call.name == "research"
            assert view.receipts[-1].outcome.summary == "Research complete."
        return rejected(CANDIDATE, tools=False)

    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        evidence=evidence,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
    )
    assert harness.run().stop_reason == "repeated_publication_gap"
    assert calls == 8


@pytest.mark.parametrize("change", ["need", "revision_only"])
def test_research_semantics_reset_but_revision_only_bookkeeping_does_not(
    change: str,
) -> None:
    context, evidence, registry = setup()
    research = ResearchState(["Research"], context)
    context.services["research_state"] = research
    calls = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal calls
        calls += 1
        assert calls <= 7
        if change == "need" and calls == 4:
            research.update(
                ResearchUpdate(
                    needs=[
                        ResearchNeed(
                            need_id="operative",
                            question_ids=["q0"],
                            purpose="New operative uncertainty",
                            completion_test="Read the continuation",
                        )
                    ]
                ),
                evidence,
            )
        elif change == "revision_only":
            research.revision += 1
        return rejected(CANDIDATE, tools=False)

    harness = Harness(
        request="Research",
        context=context,
        registry=registry,
        decide=decide,
        evidence=evidence,
        draft_guard=lambda _answer: GAP.model_copy(deep=True),
    )
    assert harness.run().stop_reason == "repeated_publication_gap"
    assert calls == (7 if change == "need" else 4)
