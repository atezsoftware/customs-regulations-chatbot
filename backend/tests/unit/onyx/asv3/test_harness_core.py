import contextvars
import json
import threading
import time
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    Artifact,
    CapabilityCall,
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    SharedBudget,
    TaskStatus,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.progress import ProgressReporter
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.workers import WorkerPool
from onyx.llm.models import AssistantMessage, FunctionCall, ToolCall


def test_dispatch_validates_schema_and_corpus_boundary_before_execution() -> None:
    invoked: list[str] = []

    def handler(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        invoked.append("executed")
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="ok")

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="external",
                description="external",
                external=True,
                parameters={
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                    "required": ["count"],
                },
                handler=handler,
            )
        ]
    )
    context = RunContext()
    denied = registry.dispatch(
        CapabilityCall(name="external", arguments={"count": 1}), context
    )
    assert denied.status == OutcomeStatus.DENIED
    assert registry.definitions(context) == []
    context.corpus_only = False
    invalid = registry.dispatch(
        CapabilityCall(name="external", arguments={"count": "1"}), context
    )
    assert invalid.status == OutcomeStatus.INVALID
    assert invoked == []
    assert context.budget.snapshot()["tools"] == 0


def test_vertex_normalization_of_tool_definition_cannot_corrupt_registry() -> None:
    from litellm.llms.vertex_ai.common_utils import add_object_type

    parameters: dict[str, JsonValue] = {
        "type": "object",
        "properties": {"mode": {"enum": ["keyword", "hybrid"]}},
    }
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters=parameters,
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="ok"
                ),
            )
        ]
    )
    definition = registry.definitions(RunContext())[0]
    function = definition["function"]
    assert isinstance(function, dict)
    emitted = function["parameters"]
    assert isinstance(emitted, dict)
    add_object_type(emitted)
    assert emitted != parameters
    result = registry.dispatch(
        CapabilityCall(name="read", arguments={"mode": "keyword"}), RunContext()
    )
    assert result.status == OutcomeStatus.FOUND
    assert cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], parameters["properties"])["mode"],
    ) == {"enum": ["keyword", "hybrid"]}


def test_harness_queues_excess_calls_and_propagates_context_and_citations() -> None:
    inherited = contextvars.ContextVar("test_asv3_scope", default="missing")
    token = inherited.set("tenant-A")
    lock = threading.Lock()
    release = threading.Event()
    active = peak = 0
    invoked: list[int] = []

    def handler(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        nonlocal active, peak
        assert inherited.get() == "tenant-A"
        number = args["number"]
        assert isinstance(number, int)
        with lock:
            active += 1
            peak = max(peak, active)
            if active == 2:
                release.set()
        assert release.wait(2)
        with lock:
            invoked.append(number)
            active -= 1
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="source",
            evidence=[
                EvidenceItem(
                    source_id="law", chunk_id=str(number), text=f"Article {number}"
                )
            ],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters={
                    "type": "object",
                    "properties": {"number": {"type": "integer"}},
                    "required": ["number"],
                },
                handler=handler,
            )
        ]
    )
    step = 0

    def decide(view: HarnessView) -> Decision:
        nonlocal step
        step += 1
        if step == 1:
            return Decision(
                questions=["all articles"],
                calls=[
                    CapabilityCall(name="read", arguments={"number": n})
                    for n in range(5)
                ],
            )
        assert len(view.receipts) == 5
        assert len(view.evidence) == 5
        return Decision(answer="All articles are covered")

    ledger = EvidenceLedger()
    harness = Harness(
        request="read all",
        context=RunContext(),
        registry=registry,
        decide=decide,
        evidence=ledger,
        max_workers=2,
    )
    try:
        result = harness.run()
    finally:
        inherited.reset(token)
    assert result.status == OutcomeStatus.FOUND
    assert sorted(invoked) == list(range(5))
    assert peak == 2
    assert sorted(
        n for receipt in result.receipts for n in receipt.evidence_ids
    ) == list(range(1, 6))
    for receipt in result.receipts:
        assert len(receipt.evidence_ids) == 1
        item = ledger.get(receipt.evidence_ids[0])
        assert item is not None and item.chunk_id == str(
            receipt.call.arguments["number"]
        )


def test_research_deadline_preserves_reserve_and_completed_sibling_evidence() -> None:
    release, late_finished, early_finished = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def handler(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        if args["slow"]:
            assert release.wait(2)
            late_finished.set()
        else:
            early_finished.set()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="source",
            evidence=[
                EvidenceItem(
                    source_id="late" if args["slow"] else "early", text="Original text"
                )
            ],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters={
                    "type": "object",
                    "properties": {"slow": {"type": "boolean"}},
                    "required": ["slow"],
                },
                handler=handler,
            )
        ]
    )
    context = RunContext(timeout_seconds=0.8, research_reserve_seconds=0.6)
    ledger = EvidenceLedger()

    def decide(_view: HarnessView) -> Decision:
        calls = [
            CapabilityCall(name="read", arguments={"slow": True}, call_id="slow"),
            CapabilityCall(name="read", arguments={"slow": False}, call_id="fast"),
        ]
        return Decision(
            calls=calls,
            assistant_message=AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call.call_id,
                        function=FunctionCall(
                            name=call.name, arguments=json.dumps(call.arguments)
                        ),
                    )
                    for call in calls
                ],
            ),
        )

    harness = Harness(
        request="deadline reserve",
        context=context,
        registry=registry,
        decide=decide,
        evidence=ledger,
    )
    try:
        result = harness.run()
        assert early_finished.is_set()
        assert result.status == OutcomeStatus.TRUNCATED
        assert context.deadline - time.monotonic() > 0.45
        receipts = {receipt.call.call_id: receipt for receipt in result.receipts}
        assert receipts["slow"].outcome.status == OutcomeStatus.TRUNCATED
        assert receipts["fast"].outcome.status == OutcomeStatus.FOUND
        assert receipts["fast"].evidence_ids == [1]
        assert cast(EvidenceItem, ledger.get(1)).source_id == "early"
        assert [message.tool_call_id for message in harness.turns[0].results] == [
            "slow",
            "fast",
        ]
    finally:
        release.set()
    assert late_finished.wait(2)
    assert ledger.get(2) is None


def test_execute_revokes_payload_completed_after_research_deadline() -> None:
    context = RunContext(timeout_seconds=1, research_reserve_seconds=0.9)
    invoked: list[bool] = []

    def handler(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        invoked.append(True)
        time.sleep(0.15)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="late",
            evidence=[EvidenceItem(source_id="late", text="Must not be accepted")],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters={"type": "object"},
                handler=handler,
            )
        ]
    )
    harness = Harness(
        request="late result",
        context=context,
        registry=registry,
        decide=lambda _view: Decision(answer="done"),
    )
    receipt = harness._execute(CapabilityCall(name="read"), context)
    assert invoked == [True]
    assert receipt.outcome.status == OutcomeStatus.TRUNCATED
    assert receipt.outcome.evidence == []
    assert receipt.outcome.artifacts == []


def test_cancel_rejects_late_worker_output_and_preserves_sibling() -> None:
    entered, release = threading.Event(), threading.Event()

    def runner(task: str, _context: RunContext, _updates: object) -> ToolOutcome:
        if task == "slow":
            entered.set()
            assert release.wait(2)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary=task,
            evidence=[EvidenceItem(source_id="source", text=task)],
        )

    pool = WorkerPool(RunContext(), runner, max_workers=2)
    try:
        slow = pool.spawn("slow")
        assert entered.wait(2)
        sibling = pool.spawn("fast")
        assert pool.wait(sibling, 2)[0].status == TaskStatus.COMPLETED
        pool.cancel(slow)
        release.set()
        snapshots = {task.task_id: task for task in pool.list()}
        assert snapshots[slow].status == TaskStatus.CANCELLED
        assert snapshots[slow].outcome is None
        assert snapshots[sibling].outcome is not None
    finally:
        release.set()
        pool.close()


def test_evidence_checkpoint_restores_ids_and_rejects_text_tampering() -> None:
    ledger = EvidenceLedger()
    context = RunContext()
    item = EvidenceItem(
        source_id="law", text="original", chunk_id="143", question_ids=["q3"]
    )
    assert ledger.add([item], context) == [1]
    ledger.include([1])
    restored = EvidenceLedger()
    restored.restore(ledger.export(), RunContext())
    assert restored.inspect(1)["included"] is True
    assert restored.add([item], RunContext()) == [1]
    assert restored.add(
        [EvidenceItem(source_id="law", text="other", chunk_id="142")], RunContext()
    ) == [2]
    checkpoint = ledger.export()
    records = checkpoint["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    corrupted = record["item"]
    assert isinstance(corrupted, dict)
    corrupted["text"] = "modified"
    with pytest.raises(ValueError, match="hash"):
        EvidenceLedger().restore(checkpoint, RunContext())


def test_evidence_context_output_is_strictly_bounded_and_original_is_readable() -> None:
    ledger = EvidenceLedger()
    context = RunContext()
    ledger.add(
        [EvidenceItem(source_id=f"source-{n}", text='\\"' * 10000) for n in range(100)],
        context,
    )
    assert len(json.dumps(ledger.summaries(max_chars=500), ensure_ascii=False)) <= 500
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {"run_id": context.run_id}):
        registry.register(spec)
    result = registry.dispatch(
        CapabilityCall(
            name="read_evidence",
            arguments={"citation": 1, "start_char": 100, "num_chars": 20},
        ),
        context,
    )
    assert result.status == OutcomeStatus.TRUNCATED
    assert result.data["text"] == cast(EvidenceItem, ledger.get(1)).text[100:120]


def test_progress_narration_and_resume_keep_language_sequence() -> None:
    reporter = ProgressReporter(
        "run",
        "fr",
        translate=lambda _phase, _language: ("Recherche", "Je consulte les sources."),
    )
    first = reporter.report(
        "started",
        title="Analyse du cas",
        message="Je vérifie le délai applicable à ces marchandises.",
    )
    assert first.language == "fr"
    assert first.message.startswith("Je vérifie")
    resumed = ProgressReporter(
        "run", "fr", translate=lambda _phase, _language: ("Recherche", "Sources")
    )
    resumed.restore(reporter.export())
    assert resumed.report("tools").sequence == 2
    assert resumed.snapshot()[0] == first


def test_worker_checkpoint_marks_unfinished_tasks_interrupted() -> None:
    def runner(task: str, _context: RunContext, _updates: object) -> ToolOutcome:
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=task)

    context = RunContext(run_id="resumed")
    pool = WorkerPool(context, runner)
    try:
        pool.restore(
            {
                "version": 1,
                "run_id": "resumed",
                "tasks": [
                    {"task_id": "old", "task": "check source", "status": "running"}
                ],
            }
        )
        assert pool.list()[0].status == TaskStatus.INTERRUPTED
        new = pool.followup("old", "Continue with a fresh publication snapshot")
        assert new != "old"
        assert pool.wait(new, 2)[0].status == TaskStatus.COMPLETED
    finally:
        pool.close()


def test_shared_budget_is_inherited_by_child_runs() -> None:
    budget = SharedBudget(max_tools=1)
    context = RunContext(budget=budget)
    context.child().budget.consume("tools")
    with pytest.raises(RuntimeError, match="budget"):
        context.budget.consume("tools")


def test_composition_does_not_hold_capacity_needed_by_nested_calls() -> None:
    context = RunContext(budget=SharedBudget(max_inflight_tools=1), timeout_seconds=1)
    registry = CapabilityRegistry()

    def leaf(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="leaf")

    def compose(_args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        return registry.dispatch(CapabilityCall(name="leaf"), child)

    registry.register(
        ToolSpec(
            name="leaf", description="leaf", parameters={"type": "object"}, handler=leaf
        )
    )
    registry.register(
        ToolSpec(
            name="compose",
            description="compose",
            parameters={"type": "object"},
            handler=compose,
            orchestrates=True,
        )
    )
    assert (
        registry.dispatch(CapabilityCall(name="compose"), context).status
        == OutcomeStatus.FOUND
    )


def test_finalization_guard_returns_open_research_to_model() -> None:
    decisions = 0

    def decide(view: HarnessView) -> Decision:
        nonlocal decisions
        decisions += 1
        if decisions == 2:
            assert view.receipts[-1].outcome.status == OutcomeStatus.PARTIAL
        return Decision(answer="draft")

    def guard() -> ToolOutcome | None:
        if decisions == 1:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="A decisive researcher is still running",
            )
        return None

    harness = Harness(
        request="all questions",
        context=RunContext(),
        registry=CapabilityRegistry(),
        decide=decide,
        finalize_guard=guard,
    )
    assert harness.run().answer == "draft"
    assert decisions == 2


def test_evidence_context_preserves_late_question_identity_under_pressure() -> None:
    context = RunContext()
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="first-law",
                text="a" * 10000,
                question_ids=["q1"],
                chunk_id=str(n),
            )
            for n in range(50)
        ],
        context,
    )
    ledger.add(
        [
            EvidenceItem(
                source_id="repair-law",
                text="cost-only-rule",
                question_ids=["q3"],
                chunk_id="143",
            )
        ],
        context,
    )
    summary = ledger.summaries(max_chars=900)
    assert any(item["chunk_id"] == "143" for item in summary)
    assert len(json.dumps(summary, ensure_ascii=False)) <= 900


def test_binary_artifact_retained_once_and_never_in_view_or_checkpoint() -> None:
    from onyx.asv3.artifacts import ArtifactStore
    from onyx.asv3.models import Artifact, ToolReceipt

    context = RunContext()
    harness = Harness(
        request="source",
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(answer="done"),
    )
    payload = "IMAGE_PAYLOAD_MARKER" + "A" * 500000
    image = Artifact(
        artifact_id="page-1",
        name="page",
        media_type="image/png",
        metadata={"base64": payload, "page": 1},
    )
    references = harness.artifacts.add([image], context)
    assert references[0].metadata == {"base64_omitted": True, "page": 1}
    assert context.budget.snapshot()["artifact_bytes"] == len(payload)
    assert harness.artifacts.add([image], context) == references
    assert context.budget.snapshot()["artifact_bytes"] == len(payload)
    assert isinstance(context.services["artifacts"], ArtifactStore)
    stored = harness.artifacts.get("page-1")
    assert stored is not None and stored.metadata["base64"] is payload
    # Even old receipts created outside acceptance are made safe at every boundary.
    harness.receipts.append(
        ToolReceipt(
            call=CapabilityCall(name="page"),
            outcome=ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="page",
                artifacts=[image],
                data={"base64": payload},
            ),
            elapsed_seconds=0,
        )
    )
    assert "IMAGE_PAYLOAD_MARKER" not in harness.view().model_dump_json()
    saved = harness.snapshot()
    assert "IMAGE_PAYLOAD_MARKER" not in json.dumps(saved)
    restored = Harness(
        request="source",
        context=RunContext(run_id=context.run_id),
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(answer="done"),
    )
    restored.restore(saved)
    assert restored.receipts[0].outcome.artifacts[0].artifact_id == "page-1"
    assert restored.context.budget.snapshot()["artifact_bytes"] == 0


def test_researchers_share_atomic_final_decision_reserve() -> None:
    from onyx.asv3.models import RunStopped

    budget = SharedBudget(max_decisions=8, final_decision_reserve=3)
    budget.consume("decisions")  # Language selection.
    contexts = [RunContext(budget=budget), RunContext(budget=budget)]
    admitted: list[bool] = []

    def consume(context: RunContext) -> None:
        for _ in range(5):
            try:
                context.budget.consume_research_decision()
                admitted.append(True)
            except RunStopped:
                break

    threads = [
        threading.Thread(target=consume, args=(context,)) for context in contexts
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(admitted) == 4
    assert budget.snapshot()["decisions"] == 5
    for _ in range(3):
        budget.consume("decisions")
    assert budget.snapshot()["decisions"] == 8
    with pytest.raises(RunStopped):
        budget.consume("decisions")


def test_worker_checkpoints_reference_shared_originals_without_full_ledger() -> None:
    context = RunContext()
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    original = "ORIGINAL_SOURCE_ONLY" + "Text " * 100000

    def run(_task: str, _context: RunContext, _updates: object) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="completed",
            evidence=[EvidenceItem(source_id="law", text=original)],
        )

    pool = WorkerPool(context, run)
    try:
        task = pool.spawn("read law")
        outcome = pool.wait(task, 2)[0].outcome
        assert outcome is not None
        assert outcome.evidence == []
        assert outcome.data["evidence_numbers"] == [1]
        stored = ledger.get(1)
        assert stored is not None and stored.text == original
        assert "ORIGINAL_SOURCE_ONLY" not in json.dumps(pool.export())
        assert len(json.dumps(pool.export())) < 2000
    finally:
        pool.close()


def test_four_independent_researchers_run_concurrently_with_shared_budget() -> None:
    context = RunContext(scope={"document_set": "PC"})
    ready = threading.Barrier(5)
    release = threading.Event()
    contexts: list[RunContext] = []
    lock = threading.Lock()

    def runner(task: str, child: RunContext, _updates: object) -> ToolOutcome:
        with lock:
            contexts.append(child)
        if task != "queued":
            ready.wait(3)
            assert release.wait(3)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=task)

    pool = WorkerPool(context, runner)
    try:
        tasks = [pool.spawn(f"independent {n}") for n in range(4)]
        ready.wait(3)
        queued = pool.spawn("queued")
        snapshots = {task.task_id: task for task in pool.list()}
        assert all(snapshots[task].status == TaskStatus.RUNNING for task in tasks)
        assert snapshots[queued].status == TaskStatus.QUEUED
        assert len(contexts) == 4
        assert len({id(child) for child in contexts}) == 4
        assert all(child.budget is context.budget for child in contexts)
        contexts[0].scope["private"] = "first"
        assert all("private" not in child.scope for child in contexts[1:])
        assert "private" not in context.scope
        release.set()
        assert all(
            pool.wait(task, 3)[0].status == TaskStatus.COMPLETED
            for task in [*tasks, queued]
        )
    finally:
        release.set()
        pool.close()


def test_saturated_parent_workers_can_delegate_and_wait_without_deadlock() -> None:
    context = RunContext()
    parents_ready = threading.Barrier(3)
    pool: WorkerPool
    nested_contexts: list[RunContext] = []

    def runner(task: str, child: RunContext, _updates: object) -> ToolOutcome:
        if child.depth == 1:
            parents_ready.wait(3)
            nested = pool.spawn("nested " + task, request_context=child)
            result = pool.wait(nested, 3)[0]
            assert result.status == TaskStatus.COMPLETED
            assert (
                result.outcome is not None
                and result.outcome.summary == "nested " + task
            )
        else:
            nested_contexts.append(child)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=task)

    pool = WorkerPool(context, runner, max_workers=3, max_nested_workers=2)
    try:
        parents = [pool.spawn(f"parent {n}") for n in range(3)]
        assert all(
            pool.wait(task, 3)[0].status == TaskStatus.COMPLETED for task in parents
        )
        assert len(nested_contexts) == 3
        assert all(
            child.depth == 2 and child.budget is context.budget
            for child in nested_contexts
        )
        assert len(pool.list()) == 6
    finally:
        pool.close()


def test_expired_deadline_does_not_report_a_user_cancellation() -> None:
    context = RunContext(timeout_seconds=-1)
    assert not context.is_cancelled()
    with pytest.raises(RunStopped, match="deadline"):
        context.check_active()
    context.cancel()
    assert context.is_cancelled()


def test_progress_callback_does_not_block_concurrent_checkpoint_export() -> None:
    callback_entered = threading.Event()
    release_callback = threading.Event()
    exported = threading.Event()

    def emit(_event: object) -> None:
        callback_entered.set()
        assert release_callback.wait(2)

    reporter = ProgressReporter("run", "tr", emit)
    reporting = threading.Thread(target=lambda: reporter.report("worker"))
    exporting = threading.Thread(target=lambda: (reporter.export(), exported.set()))
    reporting.start()
    try:
        assert callback_entered.wait(2)
        exporting.start()
        assert exported.wait(1), "Checkpoint export must not wait for its own emitter"
    finally:
        release_callback.set()
        reporting.join(2)
        if exporting.ident is not None:
            exporting.join(2)


def test_tool_budget_exhaustion_is_truncated_not_cancelled() -> None:
    budget = SharedBudget(max_tools=0)
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read",
                description="read",
                parameters={"type": "object"},
                handler=lambda _args, _context: ToolOutcome(
                    status=OutcomeStatus.FOUND, summary="read"
                ),
            )
        ]
    )
    outcome = registry.dispatch(CapabilityCall(name="read"), RunContext(budget=budget))
    assert outcome.status == OutcomeStatus.TRUNCATED


def test_wait_all_ignores_previously_completed_sibling_until_pending_changes() -> None:
    entered, release = threading.Event(), threading.Event()

    def runner(task: str, _child: RunContext, _updates: object) -> ToolOutcome:
        if task == "pending":
            entered.set()
            assert release.wait(3)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary=task)

    pool = WorkerPool(RunContext(), runner, max_workers=2)
    waiter_finished = threading.Event()
    try:
        done = pool.spawn("done")
        assert pool.wait(done, 3)[0].status == TaskStatus.COMPLETED
        pending = pool.spawn("pending")
        assert entered.wait(3)
        waiter = threading.Thread(
            target=lambda: (pool.wait(timeout_seconds=3), waiter_finished.set())
        )
        waiter.start()
        assert not waiter_finished.wait(0.05)
        release.set()
        assert waiter_finished.wait(3)
        waiter.join(3)
        assert pool.wait(pending, 3)[0].status == TaskStatus.COMPLETED
    finally:
        release.set()
        pool.close()


def test_artifact_resident_eviction_preserves_locators_and_can_reopen() -> None:
    from onyx.asv3.artifacts import ArtifactStore
    from onyx.asv3.models import Artifact

    context = RunContext(budget=SharedBudget(max_artifact_bytes=150))
    store = ArtifactStore()
    pages = [
        Artifact(
            artifact_id=str(n),
            name="page",
            media_type="image/png",
            source_ids=["source"],
            metadata={"base64": "A" * 100, "page": n},
        )
        for n in range(3)
    ]
    store.add(pages, context)
    assert context.budget.snapshot()["artifact_bytes"] == 100
    evicted = store.get("0")
    assert evicted is not None and evicted.metadata["base64_omitted"] is True
    assert evicted.metadata["page"] == 0 and evicted.source_ids == ["source"]
    store.add([pages[0]], context)
    reopened = store.get("0")
    assert reopened is not None and reopened.metadata["base64"] == "A" * 100
    assert context.budget.snapshot()["artifact_bytes"] == 100


def test_fast_sibling_is_durable_while_slow_tool_blocks_and_native_pair_order_is_preserved() -> (
    None
):
    slow_entered, release, fast_checkpoint = (threading.Event() for _ in range(3))
    snapshots: list[dict[str, JsonValue]] = []
    callbacks: list[str] = []
    errors: list[BaseException] = []
    calls = [
        CapabilityCall(name="read", arguments={"slow": value}, call_id=name)
        for name, value in [("slow", True), ("fast", False)]
    ]

    def handler(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        if args["slow"]:
            slow_entered.set()
            assert release.wait(5)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="original source",
            evidence=[
                EvidenceItem(
                    source_id="law",
                    chunk_id="critical",
                    text="Original controlling text",
                )
            ],
            artifacts=[
                Artifact(
                    artifact_id="shared-artifact",
                    name="source locator",
                    metadata={"content": "original"},
                )
            ],
        )

    def checkpoint(snapshot: dict[str, JsonValue]) -> None:
        snapshots.append(json.loads(json.dumps(snapshot)))
        receipts = snapshot["receipts"]
        assert isinstance(receipts, list)
        if any(
            isinstance(item, dict)
            and cast(dict[str, JsonValue], item["call"])["call_id"] == "fast"
            for item in receipts
        ):
            fast_checkpoint.set()

    step = 0

    def decide(_view: HarnessView) -> Decision:
        nonlocal step
        step += 1
        if step > 1:
            return Decision(answer="Sources received")
        return Decision(
            calls=calls,
            assistant_message=AssistantMessage(
                content=None,
                tool_calls=[
                    ToolCall(
                        id=call.call_id,
                        function=FunctionCall(
                            name=call.name, arguments=json.dumps(call.arguments)
                        ),
                    )
                    for call in calls
                ],
            ),
        )

    harness = Harness(
        request="critical sibling",
        context=RunContext(timeout_seconds=10),
        registry=CapabilityRegistry(
            [
                ToolSpec(
                    name="read",
                    description="read",
                    parameters={"type": "object"},
                    handler=handler,
                )
            ]
        ),
        decide=decide,
        checkpoint=checkpoint,
        on_receipt=lambda receipt: callbacks.append(receipt.call.call_id),
    )

    def run() -> None:
        try:
            harness.run()
        except BaseException as error:
            errors.append(error)

    runner = threading.Thread(target=run)
    runner.start()
    try:
        assert slow_entered.wait(3)
        assert fast_checkpoint.wait(3)
        assert runner.is_alive()
        current = snapshots[-1]
        assert current["pending_call_count"] == 1
        assert (
            current["turns"] == []
        )  # A partial batch creates no orphan native messages.
        evidence = current["evidence"]
        assert (
            isinstance(evidence, dict)
            and len(cast(list[JsonValue], evidence["records"])) == 1
        )
        assert callbacks == ["fast"]
        assert harness.context.budget.snapshot()["evidence_bytes"] == len(
            "Original controlling text"
        )
        assert harness.context.budget.snapshot()["artifact_bytes"] == len("original")
    finally:
        release.set()
        runner.join(3)
    assert not runner.is_alive() and errors == []
    assert callbacks == ["fast", "slow"]
    assert len(harness.receipts) == 2
    assert [receipt.evidence_ids for receipt in harness.receipts] == [[1], [1]]
    assert harness.context.budget.snapshot()["evidence_bytes"] == len(
        "Original controlling text"
    )
    assert harness.context.budget.snapshot()["artifact_bytes"] == len("original")
    assert [result.tool_call_id for result in harness.turns[0].results] == [
        "slow",
        "fast",
    ]
    assert snapshots[0]["pending_call_count"] == 2
    assert snapshots[-1]["pending_calls"] == []


def test_cancelled_batch_keeps_fast_checkpoint_and_rejects_all_late_writes() -> None:
    release, fast_checkpoint, late_finished = (threading.Event() for _ in range(3))
    snapshots: list[dict[str, JsonValue]] = []
    callbacks: list[str] = []

    def handler(args: dict[str, JsonValue], _child: RunContext) -> ToolOutcome:
        slow = bool(args["slow"])
        if slow:
            assert release.wait(5)
            late_finished.set()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="source",
            evidence=[
                EvidenceItem(source_id="late" if slow else "fast", text="original")
            ],
        )

    def checkpoint(snapshot: dict[str, JsonValue]) -> None:
        snapshots.append(snapshot)
        if snapshot["receipts"]:
            fast_checkpoint.set()

    context = RunContext(timeout_seconds=10)
    harness = Harness(
        request="cancel in batch",
        context=context,
        registry=CapabilityRegistry(
            [
                ToolSpec(
                    name="read",
                    description="read",
                    parameters={"type": "object"},
                    handler=handler,
                )
            ]
        ),
        decide=lambda _view: Decision(
            calls=[
                CapabilityCall(name="read", arguments={"slow": n == 0}, call_id=str(n))
                for n in range(2)
            ]
        ),
        checkpoint=checkpoint,
        on_receipt=lambda receipt: callbacks.append(receipt.call.call_id),
    )
    runner = threading.Thread(target=harness.run)
    runner.start()
    try:
        assert fast_checkpoint.wait(3)
        saved = len(snapshots)
        context.cancel()
        runner.join(3)
        assert not runner.is_alive()
        release.set()
        assert late_finished.wait(3)
        assert len(snapshots) == saved
        assert callbacks == ["1"]
        assert harness.evidence.get(1) is not None and harness.evidence.get(2) is None
        assert len(harness.receipts) == 1 and harness.turns == []
    finally:
        release.set()
        runner.join(3)


def test_restore_pending_metadata_marks_interrupted_without_reexecution_or_budget_charge() -> (
    None
):
    context = RunContext(run_id="interrupted")
    source = Harness(
        request="restore",
        context=context,
        registry=CapabilityRegistry(),
        decide=lambda _view: Decision(answer="done"),
    )
    snapshot = source.snapshot()
    snapshot["pending_calls"] = [
        {"call_id": "pending", "name": "read", "arguments": {"source_id": "law"}}
    ]
    snapshot["pending_call_count"] = 1
    invoked: list[bool] = []

    def handler(_args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        invoked.append(True)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="must be explicit")

    restored = Harness(
        request="restore",
        context=RunContext(run_id="interrupted"),
        registry=CapabilityRegistry(
            [
                ToolSpec(
                    name="read",
                    description="read",
                    parameters={"type": "object"},
                    handler=handler,
                )
            ]
        ),
        decide=lambda _view: Decision(answer="done"),
    )
    restored.restore(snapshot)
    receipt = restored.receipts[0]
    assert receipt.call.call_id == "pending"
    assert receipt.outcome.status == OutcomeStatus.CANCELLED
    assert receipt.outcome.data["interrupted"] is True
    assert restored.context.budget.snapshot()["tools"] == 0
    assert restored.snapshot()["pending_calls"] == []
    assert restored.turns == []
    restored.run()
    assert invoked == []
