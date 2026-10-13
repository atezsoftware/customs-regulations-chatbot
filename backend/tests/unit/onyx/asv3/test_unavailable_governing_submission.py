from collections.abc import Callable

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.workflow_variant import (
    ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    ASV3_GUARDRAILS_V2_VARIANT,
    ASV3_GUARDRAILS_V3_VARIANT,
    ASV3_STANDARD_VARIANT,
    ASV3_TUNED_VARIANT,
)


def _gap() -> ToolOutcome:
    return ToolOutcome(
        status=OutcomeStatus.PARTIAL,
        summary="The governing original remains unavailable",
        data={"unavailable_governing_originals": ["6455:11"]},
    )


def _harness(
    decide: Callable[[HarnessView], Decision],
    *,
    context: RunContext | None = None,
    native: bool = True,
) -> Harness:
    context = context or RunContext(
        budget=SharedBudget(unlimited_execution=True), timeout_seconds=float("inf")
    )
    context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    context.services["tuned_unavailable_governing_keys"] = lambda _gap: {"6455:11"}
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Submit candidate",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                handler=lambda _args, _context: _gap(),
            )
        ]
    )
    return Harness(
        request="Check the missing governing provision",
        context=context,
        registry=registry,
        decide=decide,
        draft_guard=None if native else lambda _answer: _gap(),
    )


@pytest.mark.parametrize("native", [True, False])
def test_changed_answer_text_cannot_restart_unavailable_source_loop(
    native: bool,
) -> None:
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        assert attempts <= 3
        answer = f"Changing candidate and unit identity {attempts}."
        return (
            Decision(
                calls=[
                    CapabilityCall(name="submit_answer", arguments={"answer": answer})
                ]
            )
            if native
            else Decision(answer=answer)
        )

    harness = _harness(decide, native=native)
    result = harness.run()
    assert attempts == 3
    assert result.status == OutcomeStatus.PARTIAL
    assert result.answer is None
    assert result.stop_reason == "repeated_unavailable_governing_original"
    assert harness.last_draft == "Changing candidate and unit identity 3."
    assert harness.publication_gap is not None
    assert len(harness.receipts) == 3
    assert "submitted_answer" not in harness.context.services


def test_submission_counter_survives_checkpoint_without_restarting_episode() -> None:
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RunStopped("Interrupted before next provider invocation")
        return Decision(
            calls=[
                CapabilityCall(name="submit_answer", arguments={"answer": "Candidate"})
            ]
        )

    harness = _harness(decide)
    assert harness.run().status == OutcomeStatus.TRUNCATED
    snapshot = harness.snapshot()
    assert snapshot["tuned_unavailable_governing_attempts"] == {"6455:11": 2}
    restored = _harness(
        lambda _view: Decision(
            calls=[
                CapabilityCall(
                    name="submit_answer", arguments={"answer": "Changed candidate"}
                )
            ]
        ),
        context=harness.context,
    )
    restored.restore(snapshot)
    result = restored.run()
    assert result.stop_reason == "repeated_unavailable_governing_original"
    assert len(restored.receipts) == 3


@pytest.mark.parametrize(
    "variant",
    [
        None,
        ASV3_STANDARD_VARIANT,
        ASV3_GUARDED_EXPERIMENTAL_VARIANT,
        ASV3_GUARDRAILS_V2_VARIANT,
        ASV3_GUARDRAILS_V3_VARIANT,
    ],
)
def test_other_modes_do_not_apply_unavailable_submission_gate(
    variant: str | None,
) -> None:
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        if attempts <= 4:
            return Decision(
                calls=[
                    CapabilityCall(
                        name="submit_answer", arguments={"answer": "Candidate"}
                    )
                ]
            )
        return Decision(answer="Supported answer")

    harness = _harness(decide)
    if variant is None:
        harness.context.services.pop("asv3_workflow_variant")
    else:
        harness.context.services["asv3_workflow_variant"] = variant
    assert harness.run().answer == "Supported answer"
    assert attempts == 5
    assert "tuned_unavailable_governing_attempts" not in harness.snapshot()


def test_older_tuned_checkpoint_without_new_counter_remains_compatible() -> None:
    harness = _harness(lambda _view: Decision(answer="Supported answer"))
    snapshot = harness.snapshot()
    snapshot.pop("tuned_unavailable_governing_attempts")
    harness.restore(snapshot)
    assert harness.snapshot()["tuned_unavailable_governing_attempts"] == {}


@pytest.mark.parametrize("attempts", [{"key": -1}, {"key": True}, {"key": "3"}, []])
def test_invalid_counter_checkpoint_fails_explicitly(attempts: JsonValue) -> None:
    harness = _harness(lambda _view: Decision(answer="Supported answer"))
    snapshot = harness.snapshot()
    snapshot["tuned_unavailable_governing_attempts"] = attempts
    with pytest.raises(ValueError, match="submission checkpoint"):
        harness.restore(snapshot)
