from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, cast

from pydantic import JsonValue

from onyx.asv3.artifacts import ArtifactStore, artifact_reference, compact_json
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    HarnessResult,
    HarnessView,
    OutcomeStatus,
    ResearchTurn,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.progress import ProgressReporter
from onyx.asv3.registry import CapabilityRegistry
from onyx.llm.models import ToolMessage
from onyx.tracing.answer_graph import graph_step

DecisionMaker = Callable[[HarnessView], Decision]
CheckpointWriter = Callable[[dict[str, JsonValue]], None]


class Harness:
    def __init__(
        self,
        *,
        request: str,
        context: RunContext,
        registry: CapabilityRegistry,
        decide: DecisionMaker,
        evidence: EvidenceLedger | None = None,
        progress: ProgressReporter | None = None,
        on_receipt: Callable[[ToolReceipt], None] | None = None,
        checkpoint: CheckpointWriter | None = None,
        max_workers: int = 4,
        max_context_chars: int = 60000,
        finalize_guard: Callable[[], ToolOutcome | None] | None = None,
        draft_guard: Callable[[str], ToolOutcome | None] | None = None,
        report_terminal: bool = True,
    ) -> None:
        self.request = request
        self.context = context
        self.registry = registry
        self.decide = decide
        self.evidence = evidence or EvidenceLedger()
        self.progress = progress
        self.on_receipt = on_receipt
        self.checkpoint = checkpoint
        self.max_workers = max_workers
        self.max_context_chars = max_context_chars
        self.finalize_guard = finalize_guard
        self.draft_guard = draft_guard
        self.report_terminal = report_terminal
        artifacts = self.context.services.get("artifacts")
        self.artifacts = (
            artifacts if isinstance(artifacts, ArtifactStore) else ArtifactStore()
        )
        self.context.services["artifacts"] = self.artifacts
        self.questions: list[str] = []
        self.facts: list[str] = []
        self.receipts: list[ToolReceipt] = []
        self.turns: list[ResearchTurn] = []
        self.last_draft: str | None = None
        self._blocked_attempts: dict[str, int] = {}
        self._seen_calls: set[str] = set()
        self._seen_failures: set[str] = set()
        self._committed_calls: set[str] = set()
        self._pending_calls: dict[str, dict[str, JsonValue]] = {}
        self.context.services["registry"] = registry
        self.context.services["evidence"] = self.evidence

    @staticmethod
    def _receipt_reference(receipt: ToolReceipt) -> ToolReceipt:
        data = compact_json(receipt.outcome.data)
        assert isinstance(data, dict)
        return receipt.model_copy(
            update={
                "outcome": receipt.outcome.model_copy(
                    update={
                        "evidence": [],
                        "data": data,
                        "artifacts": [
                            artifact_reference(item)
                            for item in receipt.outcome.artifacts
                        ],
                    }
                )
            }
        )

    def snapshot(self) -> dict[str, JsonValue]:
        return {
            "version": 1,
            "run_id": self.context.run_id,
            "request": self.request,
            "language": self.context.language,
            "questions": list(self.questions),
            "facts": list(self.facts),
            "budget": self.context.budget.snapshot(),
            "receipts": [
                self._receipt_reference(item).model_dump(mode="json")
                for item in self.receipts
            ],
            "evidence": self.evidence.export(),
            "seen_calls": sorted(self._seen_calls),
            "seen_failures": sorted(self._seen_failures),
            "pending_calls": list(self._pending_calls.values())[:128],
            "pending_call_count": len(self._pending_calls),
            "pending_calls_omitted": max(0, len(self._pending_calls) - 128),
            "turns": [turn.model_dump(mode="json") for turn in self.turns],
            "last_draft": self.last_draft,
        }

    def restore(self, snapshot: dict[str, JsonValue]) -> None:
        if (
            snapshot.get("version") != 1
            or snapshot.get("run_id") != self.context.run_id
        ):
            raise ValueError("Checkpoint version or run identity mismatch")
        if (
            snapshot.get("request") != self.request
            or snapshot.get("language") != self.context.language
        ):
            raise ValueError("Checkpoint request or language mismatch")

        def array(name: str) -> list[JsonValue]:
            value = snapshot.get(name, [])
            if not isinstance(value, list):
                raise ValueError(f"Invalid checkpoint {name}")
            return value

        self.questions = [str(item) for item in array("questions")]
        self.facts = [str(item) for item in array("facts")]
        self.receipts = [
            self._receipt_reference(ToolReceipt.model_validate(item))
            for item in array("receipts")
        ]
        self._seen_calls = {str(item) for item in array("seen_calls")}
        self._seen_failures = {str(item) for item in array("seen_failures")}
        self.turns = [ResearchTurn.model_validate(item) for item in array("turns")]
        self._trim_turns()
        draft = snapshot.get("last_draft")
        self.last_draft = draft if isinstance(draft, str) else None
        evidence = snapshot.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Checkpoint evidence is missing")
        self.evidence.restore(evidence, self.context)
        budget = snapshot.get("budget")
        if isinstance(budget, dict):
            self.context.budget.restore({**budget, "artifact_bytes": 0})
        self._committed_calls = {receipt.call.call_id for receipt in self.receipts}
        self._seen_calls.update(self._committed_calls)
        self._pending_calls.clear()
        pending = array("pending_calls")
        if len(pending) > 128:
            raise ValueError("Pending tool metadata exceeds its checkpoint bound")
        for item in pending:
            if not isinstance(item, dict):
                raise ValueError("Invalid pending tool metadata")
            call = CapabilityCall.model_validate(item)
            if call.call_id in self._committed_calls:
                continue
            self.receipts.append(
                ToolReceipt(
                    call=call,
                    elapsed_seconds=0,
                    outcome=ToolOutcome(
                        status=OutcomeStatus.CANCELLED,
                        summary="Interrupted tool call was not automatically resumed; choose the next research action explicitly",
                        data={"interrupted": True, "pending_metadata_only": True},
                    ),
                )
            )
            self._seen_calls.add(call.call_id)
            self._committed_calls.add(call.call_id)
        omitted = snapshot.get("pending_calls_omitted", 0)
        if isinstance(omitted, bool) or not isinstance(omitted, int) or omitted < 0:
            raise ValueError("Invalid omitted pending tool count")
        if omitted:
            self.receipts.append(
                ToolReceipt(
                    call=CapabilityCall(name="interrupted_batch"),
                    elapsed_seconds=0,
                    outcome=ToolOutcome(
                        status=OutcomeStatus.CANCELLED,
                        summary="Additional unfinished tool metadata was bounded; no tool was automatically resumed",
                        data={"interrupted": True, "omitted_pending_calls": omitted},
                    ),
                )
            )

    def _save(self) -> None:
        if (
            self.checkpoint
            and not self.context.is_cancelled()
            and time.monotonic() < self.context.deadline
        ):
            self.checkpoint(self.snapshot())

    def view(self) -> HarnessView:
        from onyx.asv3.supplemental_tools import ScenarioState

        scenario = self.context.services.get("scenario_state")
        if isinstance(scenario, ScenarioState):
            shared = scenario.snapshot()
            questions, facts = shared.get("questions"), shared.get("facts")
            if isinstance(questions, list):
                self.questions = list(
                    dict.fromkeys(self.questions + [str(item) for item in questions])
                )
            if isinstance(facts, list):
                self.facts = list(
                    dict.fromkeys(self.facts + [str(item) for item in facts])
                )
        # Receipts retain structured IDs/status while evidence is separately addressable.
        receipts: list[ToolReceipt] = []
        remaining = self.max_context_chars // 3
        for receipt in reversed(self.receipts):
            compacted = compact_json(receipt.outcome.data, max_chars=3000)
            assert isinstance(compacted, dict)
            bounded = receipt.model_copy(
                deep=False,
                update={
                    "outcome": receipt.outcome.model_copy(
                        deep=False,
                        update={
                            "evidence": [],
                            "summary": receipt.outcome.summary[:1000],
                            "artifacts": [
                                artifact_reference(item)
                                for item in receipt.outcome.artifacts
                            ],
                            "data": compacted,
                        },
                    )
                },
            )
            size = len(bounded.model_dump_json())
            if size > remaining:
                break
            remaining -= size
            receipts.append(bounded)
        return HarnessView(
            request=self.request,
            questions=list(self.questions),
            facts=list(self.facts),
            receipts=list(reversed(receipts)),
            evidence=self.evidence.summaries(max_chars=self.max_context_chars // 2),
            tools=self.registry.definitions(self.context),
            turns=list(self.turns),
        )

    def _trim_turns(self) -> None:
        while self.turns and (
            len(self.turns) > 6
            or sum(len(turn.model_dump_json()) for turn in self.turns) > 120000
        ):
            self.turns.pop(0)

    def _execute(self, call: CapabilityCall, context: RunContext) -> ToolReceipt:
        start = time.monotonic()
        try:
            context.check_research_active()
            with graph_step(
                "asv3.tool", {"name": call.name, "arguments": call.arguments}
            ) as step:
                outcome = self.registry.dispatch(call, context)
                step.output_value = {
                    "status": outcome.status.value,
                    "summary": outcome.summary,
                    "data": compact_json(outcome.data),
                    "sources": [
                        {
                            "source_id": item.source_id,
                            "chunk_id": item.chunk_id,
                            "text_hash": item.text_hash,
                            "chars": len(item.text),
                        }
                        for item in outcome.evidence
                    ],
                }
            # The handler may not support cancellation; its late payload is still revoked.
            context.check_research_active()
        except RunStopped as error:
            outcome = ToolOutcome(
                status=OutcomeStatus.CANCELLED
                if context.is_cancelled()
                else OutcomeStatus.TRUNCATED,
                summary=str(error),
            )
        return ToolReceipt(
            call=call, outcome=outcome, elapsed_seconds=time.monotonic() - start
        )

    def _commit_receipt(self, receipt: ToolReceipt) -> None:
        if receipt.call.call_id in self._committed_calls:
            return
        self.context.check_active()
        receipt.evidence_ids = self.evidence.add(receipt.outcome.evidence, self.context)
        if receipt.outcome.status in (
            OutcomeStatus.UNAVAILABLE,
            OutcomeStatus.INVALID,
            OutcomeStatus.ERROR,
            OutcomeStatus.DENIED,
        ):
            self._seen_failures.add(
                json.dumps(
                    {"name": receipt.call.name, "arguments": receipt.call.arguments},
                    sort_keys=True,
                )
            )
        receipt.outcome.artifacts = self.artifacts.add(
            receipt.outcome.artifacts, self.context
        )
        compacted = compact_json(receipt.outcome.data)
        assert isinstance(compacted, dict)
        receipt.outcome.data = compacted
        self.context.check_active()
        if self.on_receipt:
            self.on_receipt(receipt)
        receipt.outcome = receipt.outcome.model_copy(update={"evidence": []})
        self.receipts.append(receipt)
        self._seen_calls.add(receipt.call.call_id)
        self._committed_calls.add(receipt.call.call_id)
        self._pending_calls.pop(receipt.call.call_id, None)
        self._save()

    def _dispatch(self, calls: list[CapabilityCall]) -> list[ToolReceipt]:
        executor = ThreadPoolExecutor(
            max_workers=self.max_workers, thread_name_prefix="asv3-tool"
        )
        futures: list[tuple[CapabilityCall, RunContext, Future[ToolReceipt]]] = []
        ready: dict[str, ToolReceipt] = {}
        try:
            for call in calls:
                signature = json.dumps(
                    {"name": call.name, "arguments": call.arguments}, sort_keys=True
                )
                if call.call_id in self._seen_calls or signature in self._seen_failures:
                    ready[call.call_id] = ToolReceipt(
                        call=call,
                        elapsed_seconds=0,
                        outcome=ToolOutcome(
                            status=OutcomeStatus.INVALID,
                            summary="Repeated failed call: change the arguments or method",
                        ),
                    )
                    continue
                self._seen_calls.add(call.call_id)
                arguments = compact_json(call.arguments, max_chars=1500)
                assert isinstance(arguments, dict)
                self._pending_calls[call.call_id] = {
                    "call_id": call.call_id,
                    "name": call.name,
                    "arguments": arguments,
                }
            # Persist locators before execution; an interrupted batch is never replayed.
            self._save()
            for call in calls:
                if call.call_id in ready:
                    self._commit_receipt(ready[call.call_id])
                    continue
                child = self.context.child()
                # Tools are not delegated researchers; keep their delegation depth unchanged.
                child.depth = self.context.depth
                captured = contextvars.copy_context()
                future = cast(
                    Future[ToolReceipt],
                    executor.submit(captured.run, self._execute, call, child),
                )
                futures.append((call, child, future))
            pending = {
                call.call_id: (call, child, future) for call, child, future in futures
            }
            while pending:
                self.context.check_active()
                # A slow first call cannot hide an already completed sibling's evidence.
                for call_id, (_, _, future) in list(pending.items()):
                    if future.done():
                        ready[call_id] = future.result()
                        self._commit_receipt(ready[call_id])
                        del pending[call_id]
                if not pending:
                    break
                try:
                    self.context.check_research_active()
                except RunStopped as error:
                    for call_id, (call, child, future) in pending.items():
                        child.cancel()
                        future.cancel()
                        ready[call_id] = ToolReceipt(
                            call=call,
                            elapsed_seconds=0,
                            outcome=ToolOutcome(
                                status=OutcomeStatus.TRUNCATED,
                                summary=str(error),
                            ),
                        )
                        self._commit_receipt(ready[call_id])
                    break
                time.sleep(
                    min(0.02, max(0, self.context.research_deadline - time.monotonic()))
                )
            return [ready[call.call_id] for call in calls]
        finally:
            for _, child, future in futures:
                if not future.done():
                    child.cancel()
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    def run(self) -> HarnessResult:
        if self.progress:
            self.progress.report("started")
        status = OutcomeStatus.PARTIAL
        answer: str | None = None
        try:
            while True:
                self.context.check_research_active()
                self.context.budget.consume_research_decision()
                decision = self.decide(self.view())
                self.context.check_active()
                self.questions = list(
                    dict.fromkeys(self.questions + decision.questions)
                )
                self.facts = list(dict.fromkeys(self.facts + decision.facts))
                if decision.calls:
                    if len({call.call_id for call in decision.calls}) != len(
                        decision.calls
                    ):
                        raise ValueError("Decision contains duplicate call IDs")
                    if self.progress:
                        self.progress.report("tools")
                    results: list[ToolMessage] = []
                    for receipt in self._dispatch(decision.calls):
                        self.context.check_active()
                        results.append(
                            ToolMessage(
                                tool_call_id=receipt.call.call_id,
                                content=receipt.model_dump_json(),
                            )
                        )
                    if decision.assistant_message is not None:
                        self.turns.append(
                            ResearchTurn(
                                assistant=decision.assistant_message, results=results
                            )
                        )
                        self._trim_turns()
                    self._save()
                    continue
                if decision.answer is not None and decision.answer.strip():
                    self.last_draft = decision.answer
                    blocked = self.finalize_guard() if self.finalize_guard else None
                    if blocked is None and self.draft_guard:
                        blocked = self.draft_guard(decision.answer)
                    if blocked is not None:
                        receipt = ToolReceipt(
                            call=CapabilityCall(name="finalization_status"),
                            outcome=blocked,
                            elapsed_seconds=0,
                        )
                        self.receipts.append(receipt)
                        if self.on_receipt:
                            self.on_receipt(receipt)
                        self._save()
                        fingerprint = json.dumps(
                            {
                                "evidence": [
                                    (item["citation"], item["text_hash"])
                                    for item in self.evidence.summaries()
                                ],
                                "gap": blocked.model_dump(mode="json"),
                            },
                            sort_keys=True,
                        )
                        self._blocked_attempts[fingerprint] = (
                            self._blocked_attempts.get(fingerprint, 0) + 1
                        )
                        if self._blocked_attempts[fingerprint] >= 3:
                            status = OutcomeStatus.PARTIAL
                            break
                        continue
                    answer = decision.answer
                    status = OutcomeStatus.FOUND
                    break
                self._save()
        except RunStopped as error:
            status = (
                OutcomeStatus.CANCELLED
                if "cancel" in str(error).lower()
                else OutcomeStatus.TRUNCATED
            )
        finally:
            self._save()
            if self.progress and self.report_terminal:
                phase = (
                    "completed"
                    if status == OutcomeStatus.FOUND
                    else "cancelled"
                    if status == OutcomeStatus.CANCELLED
                    else "failed"
                )
                self.progress.report(
                    phase,
                    status="completed"
                    if status == OutcomeStatus.FOUND
                    else "cancelled"
                    if status == OutcomeStatus.CANCELLED
                    else "failed",
                )
        return HarnessResult(
            answer=answer,
            status=status,
            receipts=self.receipts,
            questions=self.questions,
            facts=self.facts,
        )
