"""Cropped closes broad I/O while preserving actual cached originals and workers."""

import json
from collections.abc import Callable

import pytest
from pydantic import JsonValue

from onyx.asv3.cropped_cost_budget import CroppedResearchBudgetReached
from onyx.asv3.cropped_research_limit import CroppedResearchLimit, CroppedResearchPolicy
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry


def setup(
    name: str,
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
    *,
    cropped: bool = True,
    legacy_closure: bool = False,
) -> tuple[Harness, CroppedResearchLimit]:
    context = RunContext(run_id="run", scope={"owner": "onur", "document_set": 15})
    limit = CroppedResearchLimit(
        context,
        "Actual question",
        policy=CroppedResearchPolicy(enforce_observed_limits=legacy_closure),
    )
    context.services.update(
        asv3_exp_cropped=cropped,
        cropped_research_controller=limit,
        cropped_research_limit=limit.finish_request,
    )
    harness = Harness(
        request="Actual question",
        context=context,
        registry=CapabilityRegistry(
            [
                ToolSpec(
                    name=name,
                    description="Actual capability",
                    parameters={"type": "object"},
                    handler=handler,
                )
            ]
        ),
        decide=lambda _view: Decision(answer="Complete"),
    )
    return harness, limit


def force_tail(limit: CroppedResearchLimit) -> None:
    limit.record_search("one")
    limit.record_search("two")
    limit.record_search("three")
    for _ in range(8):
        limit.reserve_model("root")
    limit.force_finish("Explicit closure test boundary")


@pytest.mark.parametrize(
    "assistant_message_id,calls,searches,originals",
    [
        (5079, 11, 3, 161),
        (4914, 9, 1, 261),
        (4917, 9, 2, 251),
        (4908, 8, 2, 229),
        (4896, 12, 3, 134),
        (4890, 10, 3, 138),
    ],
)
def test_captured_healthy_envelopes_preserve_actual_search_dispatch(
    assistant_message_id: int, calls: int, searches: int, originals: int
) -> None:
    invoked: list[bool] = []

    def search(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        invoked.append(True)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Actual focused search")

    harness, limit = setup("search_corpus", search)
    limit.record_sources(
        EvidenceItem(
            source_id="captured-envelope",
            chunk_id=str(index),
            text=f"Original condition and exception {index}",
        )
        for index in range(originals)
    )
    for index in range(searches):
        limit.record_search(f"captured-search-{index}")
    for _ in range(calls):
        limit.reserve_model("root")
    assert assistant_message_id in {5079, 4914, 4917, 4908, 4896, 4890}
    assert limit.finish_request(harness.view(), harness.context) is None
    receipt = harness._dispatch(
        [CapabilityCall(name="search_corpus", arguments={"query": "unresolved effect"})]
    )[0]
    assert invoked == [True]
    assert receipt.outcome.status == OutcomeStatus.FOUND
    assert "blocked_by_research_limit" not in receipt.outcome.data
    assert limit.progress()["critical_reads_reserved"] == 0


def test_cached_complete_original_reused_without_critical_reservation_or_io() -> None:
    invoked: list[bool] = []
    text = "Whole operative condition, exception and continuation. " * 100

    def read(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        invoked.append(True)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Complete original",
            evidence=[EvidenceItem(source_id="law", chunk_id="chunk", text=text)],
            data={"complete": True},
        )

    harness, limit = setup("read_provision", read)
    first = harness._dispatch(
        [
            CapabilityCall(
                name="read_provision", arguments={"source_id": "law", "article": "3"}
            )
        ]
    )[0]
    force_tail(limit)
    repeated = harness._dispatch(
        [
            CapabilityCall(
                name="read_provision", arguments={"source_id": "law", "article": "3"}
            )
        ]
    )[0]
    assert invoked == [True]
    assert first.evidence_ids == repeated.evidence_ids == [1]
    assert repeated.outcome.data["reused_recorded_read"] is True
    original = harness.evidence.get(1)
    assert original is not None and original.text == text
    result = json.loads(harness._model_tool_result(repeated).content)
    assert result["original_evidence"][0]["text"] == text
    assert repeated.outcome.data["complete"] is True
    assert limit.progress()["critical_reads_reserved"] == 0


def test_fresh_exact_read_reserved_before_handler_then_cache_is_free() -> None:
    observations: list[JsonValue] = []
    saved: list[dict[str, JsonValue]] = []

    def read(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        observations.append(limit.progress()["critical_reads_reserved"])
        assert saved and saved[-1]["state"] == limit.export()["state"]
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual original",
            evidence=[
                EvidenceItem(source_id="law", text="Entire condition and exception")
            ],
        )

    harness, limit = setup("read_provision", read, legacy_closure=True)
    harness.checkpoint = lambda _snapshot: saved.append(limit.export())
    force_tail(limit)
    for _ in range(2):
        receipt = harness._dispatch(
            [
                CapabilityCall(
                    name="read_provision",
                    arguments={"source_id": "law", "article": "11"},
                )
            ]
        )[0]
        assert receipt.outcome.status == OutcomeStatus.FOUND
    assert observations == [1]
    assert limit.progress()["critical_reads_reserved"] == 1


@pytest.mark.parametrize("name", ["search_corpus", "spawn_researcher"])
def test_broad_call_blocked_before_batch_preparation_or_nested_execution(
    name: str,
) -> None:
    executed: list[bool] = []
    prepared: list[bool] = []

    def handler(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        executed.append(True)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Should not execute")

    harness, limit = setup(name, handler)
    harness.context.services["prepare_search_batch"] = lambda _calls, _context: (
        prepared.append(True)
    )
    force_tail(limit)
    call = CapabilityCall(
        name=name,
        arguments={
            "query": "broad",
            "calls": [{"name": "search_corpus", "arguments": {"query": "nested"}}],
        },
    )
    receipt = harness._dispatch([call])[0]
    assert not executed and not prepared
    assert receipt.outcome.status == OutcomeStatus.PARTIAL
    assert receipt.outcome.data["blocked_by_research_limit"] is True
    assert receipt.outcome.data["absence_proven"] is False
    assert receipt.evidence_ids == []
    assert harness.evidence.summaries() == []


@pytest.mark.parametrize("cropped,long_tail", [(False, True), (True, False)])
def test_protected_modes_and_short_cropped_path_keep_existing_dispatch(
    cropped: bool, long_tail: bool
) -> None:
    invoked: list[bool] = []

    def handler(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        invoked.append(True)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Actual search")

    harness, limit = setup("search_corpus", handler, cropped=cropped)
    if long_tail:
        force_tail(limit)
    assert (
        harness._dispatch(
            [
                CapabilityCall(
                    name="search_corpus", arguments={"query": "specific issue"}
                )
            ]
        )[0].outcome.status
        == OutcomeStatus.FOUND
    )
    assert invoked == [True]


def test_foreign_controller_cannot_authorize_this_run() -> None:
    harness, _ = setup(
        "search_corpus",
        lambda _a, _c: ToolOutcome(status=OutcomeStatus.FOUND, summary="Search"),
    )
    harness.context.services["cropped_research_controller"] = CroppedResearchLimit(
        RunContext(run_id="foreign"), "Actual question"
    )
    with pytest.raises(ValueError, match="run, scope, request or policy"):
        harness._dispatch(
            [CapabilityCall(name="search_corpus", arguments={"query": "q"})]
        )


@pytest.mark.parametrize(
    "cropped,name,expected",
    [
        (True, "list_researchers", 1),
        (False, "list_researchers", 0),
        (True, "wait_researcher", 1),
        (False, "wait_researcher", 1),
    ],
)
def test_host_free_wait_before_next_model_decision_and_no_duplicate_wait(
    cropped: bool, name: str, expected: int
) -> None:
    waits: list[str | None] = []
    harness, _ = setup(
        name,
        lambda _a, _c: ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Actual worker status"
        ),
        cropped=cropped,
    )
    harness.context.services["wait_for_task_change"] = lambda task_id: waits.append(
        task_id
    )
    decisions = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            return Decision(
                calls=[CapabilityCall(name=name, arguments={"task_id": "worker"})]
            )
        assert len(waits) == expected
        return Decision(answer="Complete")

    harness.decide = decide
    assert harness.run().answer == "Complete"
    assert waits == ["worker"] * expected


@pytest.mark.parametrize("cropped", [True, False])
def test_cropped_host_collects_real_worker_original_before_paid_decision(
    cropped: bool,
) -> None:
    hosted: list[bool] = []
    text = "Actual worker original with required operative exception."
    harness, _ = setup(
        "wait_researcher",
        lambda _a, _c: ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Actual collected finding",
            evidence=[EvidenceItem(source_id="worker-law", text=text)],
        ),
        cropped=cropped,
    )

    def host(current: Harness) -> Decision | None:
        hosted.append(True)
        return (
            Decision(
                calls=[
                    CapabilityCall(
                        name="wait_researcher", arguments={"task_id": "worker"}
                    )
                ]
            )
            if not current.receipts
            else None
        )

    def decide(view: HarnessView) -> Decision:
        assert bool(view.original_evidence) is cropped
        if cropped:
            assert view.original_evidence[0]["text"] == text
        return Decision(answer="Complete")

    harness.context.services["cropped_host_decision"] = host
    harness.decide = decide
    assert harness.run().answer == "Complete"
    assert harness.context.budget.snapshot()["decisions"] == 1
    assert len(hosted) == (2 if cropped else 0)


def test_invalid_host_decision_is_rejected_before_model_or_tools() -> None:
    harness, _ = setup(
        "wait_researcher",
        lambda _a, _c: ToolOutcome(status=OutcomeStatus.FOUND, summary="Wait"),
    )
    harness.context.services["cropped_host_decision"] = lambda _harness: (
        "unverified answer"
    )
    with pytest.raises(TypeError, match="Decision or None"):
        harness.run()
    assert harness.context.budget.snapshot()["decisions"] == 0


@pytest.mark.parametrize("cropped", [True, False])
def test_rewrite_allowance_preserves_canonical_reads_and_honest_search_gap(
    cropped: bool,
) -> None:
    invoked: list[str] = []
    harness, _ = setup(
        "search_corpus",
        lambda _a, _c: (
            invoked.append("search")
            or ToolOutcome(status=OutcomeStatus.FOUND, summary="Search")
        ),
        cropped=cropped,
    )
    harness.registry.register(
        ToolSpec(
            name="read_provision",
            description="Canonical clause read",
            parameters={"type": "object"},
            handler=lambda _a, _c: (
                invoked.append("read")
                or ToolOutcome(
                    status=OutcomeStatus.FOUND,
                    summary="Whole canonical clause",
                    evidence=[
                        EvidenceItem(
                            source_id="law", text="Operative rule and its exception."
                        )
                    ],
                )
            ),
        )
    )

    def prepare(
        _calls: list[CapabilityCall], _context: RunContext
    ) -> dict[str, object]:
        raise CroppedResearchBudgetReached("Rewrite allowance reached")

    harness.context.services["prepare_search_batch"] = prepare
    calls = [
        CapabilityCall(name="search_corpus", arguments={"query": "q"}),
        CapabilityCall(
            name="read_provision", arguments={"source_id": "law", "article": "3"}
        ),
    ]
    if not cropped:
        with pytest.raises(CroppedResearchBudgetReached):
            harness._dispatch(calls)
        assert not invoked
        return
    receipts = harness._dispatch(calls)
    assert invoked == ["read"]
    assert receipts[0].outcome.status is OutcomeStatus.PARTIAL
    assert receipts[0].outcome.data["not_read"] is True
    assert receipts[0].outcome.data["absence_proven"] is False
    assert receipts[1].evidence_ids
    original = harness.evidence.get(receipts[1].evidence_ids[0])
    assert original is not None
    assert original.text == "Operative rule and its exception."
