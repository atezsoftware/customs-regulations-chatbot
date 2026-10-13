from collections.abc import Callable

import pytest
from pydantic import JsonValue

from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessView,
    OutcomeStatus,
    RunStopped,
)
from onyx.asv3.workflow_variant import ASV3_STANDARD_VARIANT
from tests.unit.onyx.asv3.test_governing_acquisition_admission import (
    AdmissionRun,
    target,
)


def retry(number: int, *, composed: bool = False) -> CapabilityCall:
    arguments: dict[str, JsonValue] = {
        "query": f"6455 sayılı Kanun m.11 özgün hüküm denemesi {number}"
    }
    return (
        CapabilityCall(
            name="compose_tool_calls",
            arguments={
                "steps": [
                    {
                        "id": f"missing-{number}",
                        "tool": "search_corpus",
                        "arguments": arguments,
                    }
                ]
            },
        )
        if composed
        else CapabilityCall(name="search_corpus", arguments=arguments)
    )


def harness(run: AdmissionRun, decide: Callable[[HarnessView], Decision]) -> Harness:
    return Harness(
        request="Research the missing provision and independent issues",
        context=run.context,
        registry=run.registry,
        evidence=run.ledger,
        decide=decide,
    )


@pytest.mark.parametrize("composed", [False, True])
def test_denied_research_rounds_stop_without_reexecuting_missing_target(
    composed: bool,
) -> None:
    run = AdmissionRun()
    missing = run.unavailable()
    before = run.counts.copy()
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        assert attempts <= 3
        return Decision(calls=[retry(attempts, composed=composed)])

    agent = harness(run, decide)
    result = agent.run()
    assert attempts == 3
    assert run.counts == before
    assert result.answer is None
    assert result.status == OutcomeStatus.PARTIAL
    assert result.stop_reason == "repeated_unavailable_governing_original"
    assert agent.snapshot()["tuned_unavailable_governing_attempts"] == {missing.key: 3}


def test_denied_research_counter_survives_checkpoint() -> None:
    run = AdmissionRun()
    run.unavailable()
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        if attempts == 3:
            raise RunStopped("Interrupted before another model call")
        return Decision(calls=[retry(attempts)])

    first = harness(run, decide)
    assert first.run().status == OutcomeStatus.TRUNCATED
    restored = harness(run, lambda _view: Decision(calls=[retry(3)]))
    restored.restore(first.snapshot())
    assert restored.run().stop_reason == "repeated_unavailable_governing_original"
    assert len(restored.receipts) == 3
    assert run.counts["search_corpus"] == 0


@pytest.mark.parametrize("composed", [False, True])
def test_mixed_rounds_do_not_stop_independent_research(composed: bool) -> None:
    run = AdmissionRun()
    run.unavailable()
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        if attempts == 5:
            return Decision(answer="Answer supported by independent evidence")
        calls = [
            retry(attempts),
            CapabilityCall(
                name="search_corpus",
                arguments={"query": f"5326 sayılı Kanun m.{attempts}"},
            ),
        ]
        if composed:
            calls = [
                CapabilityCall(
                    name="compose_tool_calls",
                    arguments={
                        "steps": [
                            {
                                "id": str(index),
                                "tool": call.name,
                                "arguments": call.arguments,
                            }
                            for index, call in enumerate(calls)
                        ]
                    },
                )
            ]
        return Decision(calls=calls)

    agent = harness(run, decide)
    assert agent.run().answer == "Answer supported by independent evidence"
    assert run.counts["search_corpus"] == 4
    assert agent.snapshot()["tuned_unavailable_governing_attempts"] == {}


def test_pending_owner_is_not_treated_as_completed_failure() -> None:
    run = AdmissionRun()
    missing = target()
    call = run.state.next_call(missing, run.context, run.ledger)
    assert call is not None
    assert run.state.reserve(missing, call, run.context)
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        return (
            Decision(calls=[retry(attempts)])
            if attempts < 5
            else Decision(answer="Other work completed")
        )

    agent = harness(run, decide)
    assert agent.run().answer == "Other work completed"
    assert agent.snapshot()["tuned_unavailable_governing_attempts"] == {}


def test_other_workflow_does_not_apply_research_loop_guard() -> None:
    run = AdmissionRun(ASV3_STANDARD_VARIANT)
    run.unavailable()
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        return (
            Decision(calls=[retry(attempts)])
            if attempts < 5
            else Decision(answer="Standard answer")
        )

    assert harness(run, decide).run().answer == "Standard answer"
    assert run.counts["search_corpus"] == 4


@pytest.mark.parametrize("independent_step", [False, True])
def test_only_cancellations_caused_by_blocked_targets_count_as_no_progress(
    independent_step: bool,
) -> None:
    run = AdmissionRun()
    run.unavailable()
    attempts = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal attempts
        attempts += 1
        if attempts == 4:
            assert independent_step
            return Decision(answer="Independent work completed")
        steps: list[JsonValue] = [
            {
                "id": "missing",
                "tool": "search_corpus",
                "arguments": retry(attempts).arguments,
            },
            {
                "id": "dependent",
                "tool": "read_source_range",
                "arguments": {"source_id": "unexecuted"},
                "depends_on": ["missing"],
            },
            {
                "id": "transitive",
                "tool": "read_source_range",
                "arguments": {"source_id": "also-unexecuted"},
                "depends_on": ["dependent"],
            },
        ]
        if independent_step:
            steps.append(
                {
                    "id": "other",
                    "tool": "search_corpus",
                    "arguments": {"query": f"5326 sayılı Kanun m.{attempts}"},
                }
            )
        return Decision(
            calls=[
                CapabilityCall(name="compose_tool_calls", arguments={"steps": steps})
            ]
        )

    agent = harness(run, decide)
    result = agent.run()
    assert run.counts["read_source_range"] == 0
    assert run.counts["search_corpus"] == (3 if independent_step else 0)
    if independent_step:
        assert result.answer == "Independent work completed"
        assert agent.snapshot()["tuned_unavailable_governing_attempts"] == {}
    else:
        assert result.stop_reason == "repeated_unavailable_governing_original"
        assert result.answer is None
        assert attempts == 3
