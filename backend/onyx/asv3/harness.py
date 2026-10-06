from __future__ import annotations

import contextvars
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, cast

from pydantic import JsonValue

from onyx.asv3.artifacts import ArtifactStore, artifact_reference, compact_json
from onyx.asv3.citation_numbers import extract_citation_numbers
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
from onyx.asv3.progress import ProgressReporter, action_narration, public_action_id
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import EvidenceWorkingSet, ResearchState
from onyx.asv3.working_memory import WorkingMemory
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
        adaptive_tool_parallelism: bool = False,
        max_context_chars: int = 60000,
        finalize_guard: Callable[[], ToolOutcome | None] | None = None,
        draft_guard: Callable[[str], ToolOutcome | None] | None = None,
        partial_submission: Callable[[], str | None] | None = None,
        report_terminal: bool = True,
        on_decision: Callable[[Decision], None] | None = None,
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
        self.adaptive_tool_parallelism = adaptive_tool_parallelism
        self.max_context_chars = max_context_chars
        self.finalize_guard = finalize_guard
        self.draft_guard = draft_guard
        self.partial_submission = partial_submission
        self.report_terminal = report_terminal
        self.on_decision = on_decision
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
        self.stop_reason: str | None = None
        self.publication_gap: ToolOutcome | None = None
        self._blocked_attempts: dict[str, int] = {}
        self._seen_calls: set[str] = set()
        self._seen_failures: set[str] = set()
        self._committed_calls: set[str] = set()
        self._pending_calls: dict[str, dict[str, JsonValue]] = {}
        self._progress_calls: set[str] = set()
        self._completed_reads: dict[str, ToolReceipt] = {}
        self.evidence_working_set = EvidenceWorkingSet(
            max_ranges=None
            if self.context.services.get("lean_native_mode") is True
            else 128
        )
        self.working_memory = WorkingMemory(self.context.scope)
        self.context.services["working_memory"] = self.working_memory
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
            "stop_reason": self.stop_reason,
            "publication_gap": self.publication_gap.model_dump(mode="json")
            if self.publication_gap
            else None,
            "working_memory": self.working_memory.export(),
            "evidence_working_set": self.evidence_working_set.export(),
            "research_state": state.export()
            if isinstance(
                state := self.context.services.get("research_state"), ResearchState
            )
            else {},
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
        reason = snapshot.get("stop_reason")
        self.stop_reason = reason if isinstance(reason, str) else None
        gap = snapshot.get("publication_gap")
        self.publication_gap = ToolOutcome.model_validate(gap) if gap else None
        evidence = snapshot.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Checkpoint evidence is missing")
        self.evidence.restore(evidence, self.context)
        research = self.context.services.get("research_state")
        saved_research = snapshot.get("research_state")
        if (
            isinstance(research, ResearchState)
            and isinstance(saved_research, dict)
            and saved_research
        ):
            research.restore(saved_research, self.evidence)
        ranges = snapshot.get("evidence_working_set")
        if isinstance(ranges, list):
            self.evidence_working_set.restore(ranges, self.evidence)
        budget = snapshot.get("budget")
        if isinstance(budget, dict):
            self.context.budget.restore({**budget, "artifact_bytes": 0})
        self._committed_calls = {receipt.call.call_id for receipt in self.receipts}
        working = snapshot.get("working_memory")
        if isinstance(working, dict):
            self.working_memory.restore(working)
        else:
            for receipt in self.receipts:
                self.working_memory.observe(receipt)
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
        state = self.context.services.get("research_state")
        if isinstance(state, ResearchState):
            self.questions = list(state.questions)
        focus = self.context.services.get("task_need_ids")
        preferred = (
            state.preferred_citations(
                [str(n) for n in focus] if isinstance(focus, list) else None
            )
            if isinstance(state, ResearchState)
            else []
        )
        required = list(
            dict.fromkeys(
                [
                    *(
                        n
                        for n in self.context.services.get(
                            "independent_evidence_numbers", []
                        )
                        if type(n) is int
                    ),
                    *(
                        state.source_conditions.citations()
                        if isinstance(state, ResearchState)
                        else []
                    ),
                    *(
                        extract_citation_numbers(self.last_draft or "")
                        if self.publication_gap
                        else []
                    ),
                ]
            )
        )
        originals = self.evidence_working_set.view(
            self.evidence,
            preferred=preferred,
            required=required,
            max_chars=None
            if self.context.services.get("lean_native_mode") is True
            else 32000,
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
            evidence=self.evidence.summaries(
                max_chars=min(6000, self.max_context_chars // 2)
            ),
            tools=self.registry.definitions(self.context),
            research_state=state.view() if isinstance(state, ResearchState) else {},
            original_evidence=cast(list[dict[str, JsonValue]], originals["records"]),
            original_evidence_omitted=cast(list[JsonValue], originals["omitted"]),
            required_evidence_numbers=required,
            turns=list(self.turns),
            draft_to_repair=self.last_draft if self.publication_gap else None,
            publication_gap=(
                {"summary": self.publication_gap.summary, **self.publication_gap.data}
                if self.publication_gap
                else None
            ),
        )

    def _trim_turns(self) -> None:
        if self.context.services.get("lean_native_mode"):
            return
        while self.turns and (
            len(self.turns) > 6
            or sum(len(turn.model_dump_json()) for turn in self.turns) > 120000
        ):
            self.turns.pop(0)

    def _tool_progress_phase(self, call: CapabilityCall) -> str | None:
        if call.argument_error is not None:
            return None
        if (
            call.name == "research_questions"
            and self.context.services.get("experimental_parallel") is True
        ):
            return None
        if call.name in {
            "record_scenario",
            "update_research",
            "inspect_research",
            "report_progress",
            "discover_tools",
            "read_research_state",
            "spawn_researcher",
            "followup_researcher",
            "send_update",
            "list_researchers",
            "wait_researcher",
            "cancel_researcher",
        }:
            return None
        return "final" if call.name == "verify_claim" else "tools"

    def _execute(self, call: CapabilityCall, context: RunContext) -> ToolReceipt:
        start = time.monotonic()
        try:
            if call.name == "assemble_answers" and context.depth == 0:
                context.check_active()
            else:
                context.check_research_active()
            phase = self._tool_progress_phase(call)
            if self.progress and phase:
                self._progress_calls.add(call.call_id)
                narration = action_narration(call.arguments, context)
                title, message = narration if narration else (None, None)
                if self.context.depth == 0:
                    self.progress.report(phase, title=title, message=message)
                self.progress.report(
                    phase,
                    task_id=public_action_id(call.call_id),
                    parent_task_id=context.services.get("task_id")
                    if isinstance(context.services.get("task_id"), str)
                    else None,
                    title=title,
                    message=message,
                )
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
            if (
                call.name in {"research_questions", "assemble_answers"}
                and context.depth == 0
            ):
                context.check_active()
            else:
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
        state = self.context.services.get("research_state")
        need = receipt.call.arguments.get("_need_id")
        focus = self.context.services.get("task_need_ids")
        need_ids = (
            [need]
            if isinstance(need, str)
            else [str(n) for n in focus]
            if isinstance(focus, list)
            else []
        )
        if isinstance(state, ResearchState):
            for item in receipt.outcome.evidence:
                item.question_ids = list(
                    dict.fromkeys(
                        [
                            *item.question_ids,
                            *(q for n in need_ids for q in state.question_ids(n)),
                        ]
                    )
                )
        receipt.evidence_ids = self.evidence.add(receipt.outcome.evidence, self.context)
        reopened_ids: list[int] = []
        for read in receipt.outcome.original_reads:
            item = self.evidence.get(read.citation)
            if (
                item is None
                or item.text_hash != read.text_hash
                or read.end_char > len(item.text)
            ):
                raise ValueError("Reopened range differs from this run's original")
            reopened_ids.append(read.citation)
        for read in receipt.outcome.original_reads:
            self.evidence_working_set.remember(
                read.citation, read.start_char, read.end_char
            )
        receipt.evidence_ids = list(
            dict.fromkeys([*receipt.evidence_ids, *reopened_ids])
        )
        if receipt.call.name == "read_evidence" and not receipt.outcome.original_reads:
            number = receipt.call.arguments.get("citation")
            item = self.evidence.get(number) if type(number) is int else None
            if (
                item is not None
                and receipt.outcome.data.get("text_hash") == item.text_hash
            ):
                receipt.evidence_ids = [cast(int, number)]
                start = int(str(receipt.call.arguments.get("start_char", 0)))
                end = min(
                    len(item.text),
                    start + int(str(receipt.call.arguments.get("num_chars", 16000))),
                )
                self.evidence_working_set.remember(cast(int, number), start, end)
        else:
            for number in receipt.evidence_ids:
                if number in reopened_ids:
                    continue
                item = self.evidence.get(number)
                if item is not None:
                    self.evidence_working_set.remember(number, 0, len(item.text))
        if isinstance(state, ResearchState):
            for need_id in need_ids:
                state.attach(need_id, receipt.evidence_ids)
        if (
            receipt.outcome.data.get("invalid_outcome_metadata") is not True
            and receipt.outcome.data.get("research_binding_error") is not True
            and receipt.outcome.status
            in (
                OutcomeStatus.UNAVAILABLE,
                OutcomeStatus.INVALID,
                OutcomeStatus.ERROR,
                OutcomeStatus.DENIED,
            )
        ):
            self._seen_failures.add(self._call_signature(receipt.call))
        receipt.outcome.artifacts = self.artifacts.add(
            receipt.outcome.artifacts, self.context
        )
        compacted = compact_json(receipt.outcome.data)
        assert isinstance(compacted, dict)
        receipt.outcome.data = compacted
        self.context.check_active()
        if self.on_receipt:
            self.on_receipt(receipt)
        self.working_memory.observe(receipt)
        receipt.outcome = receipt.outcome.model_copy(update={"evidence": []})
        self.receipts.append(receipt)
        self._seen_calls.add(receipt.call.call_id)
        self._committed_calls.add(receipt.call.call_id)
        self._pending_calls.pop(receipt.call.call_id, None)
        if (
            receipt.evidence_ids
            and receipt.outcome.status in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}
            and receipt.call.name
            in {
                "read_evidence",
                "read_chunk",
                "read_chunk_context",
                "read_provision",
                "read_source_range",
            }
        ):
            self._completed_reads[self._read_signature(receipt.call)] = receipt
        if self.progress and receipt.call.call_id in self._progress_calls:
            phase = self._tool_progress_phase(receipt.call)
            if phase:
                status = (
                    "cancelled"
                    if receipt.outcome.status == OutcomeStatus.CANCELLED
                    else "failed"
                    if receipt.outcome.status
                    in {
                        OutcomeStatus.TRUNCATED,
                        OutcomeStatus.INVALID,
                        OutcomeStatus.ERROR,
                        OutcomeStatus.DENIED,
                        OutcomeStatus.UNAVAILABLE,
                    }
                    else "completed"
                )
                self.progress.report(
                    phase,
                    status=status,
                    task_id=public_action_id(receipt.call.call_id),
                    parent_task_id=self.context.services.get("task_id")
                    if isinstance(self.context.services.get("task_id"), str)
                    else None,
                    title=(
                        narration[0]
                        if (
                            narration := action_narration(
                                receipt.call.arguments, self.context
                            )
                        )
                        else None
                    ),
                    message=narration[1] if narration else None,
                )
            self._progress_calls.discard(receipt.call.call_id)
        self._save()

    def _read_signature(self, call: CapabilityCall) -> str:
        arguments = {
            key: value
            for key, value in call.arguments.items()
            if key
            not in {
                "_public_update",
                "_need_id",
                "_language",
                "_notifications",
                "_external_requested",
                "_outcomes",
                "_coverage",
            }
        }
        number = arguments.get("citation")
        if (
            call.name == "read_evidence"
            and isinstance(number, int)
            and not isinstance(number, bool)
        ):
            item = self.evidence.get(number)
            if item is not None:
                start = int(str(arguments.get("start_char", 0)))
                arguments = {
                    "citation": number,
                    "text_hash": item.text_hash,
                    "start_char": start,
                    "end_char": min(
                        len(item.text),
                        start + int(str(arguments.get("num_chars", 16000))),
                    ),
                }
        return json.dumps(
            {
                "name": call.name,
                "arguments": arguments,
            },
            sort_keys=True,
        )

    @staticmethod
    def _call_signature(call: CapabilityCall) -> str:
        return json.dumps(
            {
                "name": call.name,
                "arguments": {
                    key: value
                    for key, value in call.arguments.items()
                    if key
                    not in {
                        "_public_update",
                        "_need_id",
                        "_language",
                        "_notifications",
                        "_external_requested",
                        "_outcomes",
                        "_coverage",
                    }
                },
                **(
                    {
                        "invalid_arguments_hash": call.invalid_arguments_hash,
                        "argument_error": call.argument_error,
                    }
                    if call.argument_error is not None
                    else {}
                ),
            },
            sort_keys=True,
        )

    def _dispatch(self, calls: list[CapabilityCall]) -> list[ToolReceipt]:
        # Local mutations establish bindings before dependent I/O in the same decision.
        mutation_names = {"update_research", "repair_question_answer"}
        mutations = [call for call in calls if call.name in mutation_names]
        if mutations and len(mutations) != len(calls):
            receipts = self._dispatch(mutations)
            failed_repairs = [
                receipt.call.call_id
                for receipt in receipts
                if receipt.call.name == "repair_question_answer"
                and receipt.outcome.status != OutcomeStatus.FOUND
            ]
            remaining = []
            for call in calls:
                if call.name in mutation_names:
                    continue
                if call.name == "assemble_answers" and failed_repairs:
                    receipt = ToolReceipt(
                        call=call,
                        elapsed_seconds=0,
                        outcome=ToolOutcome(
                            status=OutcomeStatus.PARTIAL,
                            summary="Resolve the rejected targeted repair before assembling the answer.",
                            data={"rejected_repair_call_ids": failed_repairs},
                        ),
                    )
                    self._commit_receipt(receipt)
                    receipts.append(receipt)
                else:
                    remaining.append(call)
            if remaining:
                receipts.extend(self._dispatch(remaining))
            by_id = {receipt.call.call_id: receipt for receipt in receipts}
            return [by_id[call.call_id] for call in calls]
        executor = ThreadPoolExecutor(
            max_workers=max(1, len(calls))
            if self.adaptive_tool_parallelism
            else self.max_workers,
            thread_name_prefix="asv3-tool",
        )
        futures: list[tuple[CapabilityCall, RunContext, Future[ToolReceipt]]] = []
        ready: dict[str, ToolReceipt] = {}
        try:
            for call in calls:
                binding_gap = self.registry.research_binding_gap(call, self.context)
                if binding_gap is not None:
                    ready[call.call_id] = ToolReceipt(
                        call=call, outcome=binding_gap, elapsed_seconds=0
                    )
                    continue
                metadata_gap = self.registry.outcome_metadata_gap(call, self.context)
                if metadata_gap is not None:
                    ready[call.call_id] = ToolReceipt(
                        call=call, outcome=metadata_gap, elapsed_seconds=0
                    )
                    continue
                signature = self._call_signature(call)
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
                cached = (
                    self._completed_reads.get(self._read_signature(call))
                    if call.argument_error is None
                    else None
                )
                if cached is not None:
                    ready[call.call_id] = ToolReceipt(
                        call=call,
                        elapsed_seconds=0,
                        outcome=cached.outcome.model_copy(
                            update={
                                "evidence": [
                                    item
                                    for number in cached.evidence_ids
                                    if (item := self.evidence.get(number)) is not None
                                ],
                                "data": {
                                    **cached.outcome.data,
                                    "reused_recorded_read": True,
                                },
                            }
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
            prepared_searches: dict[str, object] = {}
            prepare_search_batch = self.context.services.get("prepare_search_batch")
            if callable(prepare_search_batch):
                batch = prepare_search_batch(
                    [
                        call
                        for call in calls
                        if call.call_id not in ready
                        and call.name == "search_corpus"
                        and call.argument_error is None
                    ],
                    self.context,
                )
                if isinstance(batch, dict):
                    prepared_searches = batch
            for call in calls:
                if call.call_id in ready:
                    self._commit_receipt(ready[call.call_id])
                    continue
                child = self.context.child()
                child.services["applied_outcome_metadata_call"] = call
                # Tools are not delegated researchers; keep their delegation depth unchanged.
                child.depth = self.context.depth
                if call.call_id in prepared_searches:
                    child.services["search_batch_tool"] = prepared_searches[
                        call.call_id
                    ]
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
                    if self.context.depth == 0 and all(
                        call.name in {"research_questions", "assemble_answers"}
                        for call, _, _ in pending.values()
                    ):
                        self.context.check_active()
                    else:
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
                time.sleep(0.02)
            return [ready[call.call_id] for call in calls]
        finally:
            for _, child, future in futures:
                if not future.done():
                    child.cancel()
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)

    def _model_tool_result(self, receipt: ToolReceipt) -> ToolMessage:
        outcome = receipt.outcome.model_dump(mode="json")
        # A compact audit receipt is not the original passage requested by the model.
        if receipt.call.name == "read_evidence" and not receipt.outcome.original_reads:
            data = outcome.get("data")
            number = receipt.call.arguments.get("citation")
            if (
                isinstance(data, dict)
                and isinstance(number, int)
                and data.get("citation") == number
            ):
                item = self.evidence.get(number)
                if item is not None and data.get("text_hash") == item.text_hash:
                    start = int(cast(int, receipt.call.arguments.get("start_char", 0)))
                    count = int(
                        cast(int, receipt.call.arguments.get("num_chars", 16000))
                    )
                    data["text"] = item.text[start : start + count]
        reopened = {read.citation for read in receipt.outcome.original_reads}
        originals = json.loads(
            self.evidence.serialize_records(
                [number for number in receipt.evidence_ids if number not in reopened],
                max_chars=20000,
            )
        )
        data = outcome.get("data")
        if receipt.outcome.original_reads:
            for read in receipt.outcome.original_reads:
                item = self.evidence.get(read.citation)
                if item is None or item.text_hash != read.text_hash:
                    raise ValueError("Reopened original changed before model delivery")
                originals.append(
                    {
                        "citation": read.citation,
                        "source_id": item.source_id,
                        "chunk_id": item.chunk_id,
                        "text_hash": item.text_hash,
                        "text": item.text[read.start_char : read.end_char],
                        "start_char": read.start_char,
                        "end_char": read.end_char,
                        "total_chars": len(item.text),
                        "truncated": read.start_char != 0
                        or read.end_char != len(item.text),
                        "question_ids": item.question_ids,
                    }
                )
            identities = {
                (read.citation, read.text_hash)
                for read in receipt.outcome.original_reads
            }

            def without_duplicate_text(value: JsonValue) -> JsonValue:
                if isinstance(value, dict):
                    own_original = (
                        (value.get("citation"), value.get("text_hash")) in identities
                        if isinstance(value.get("citation"), int)
                        and isinstance(value.get("text_hash"), str)
                        else False
                    )
                    return {
                        key: without_duplicate_text(child)
                        for key, child in value.items()
                        if key != "text" or not own_original
                    }
                if isinstance(value, list):
                    return [without_duplicate_text(child) for child in value]
                return value

            outcome["data"] = without_duplicate_text(outcome.get("data", {}))
        elif receipt.call.name == "read_evidence" and isinstance(data, dict):
            # Respect the requested range; a short peek is not delivery of the full block.
            originals = [
                {
                    **data,
                    "end_char": int(str(data.get("start_char", 0)))
                    + len(str(data.get("text", ""))),
                    "truncated": receipt.outcome.status != OutcomeStatus.FOUND,
                }
            ]
            outcome["data"] = {
                key: value for key, value in data.items() if key != "text"
            }
        return ToolMessage(
            tool_call_id=receipt.call.call_id,
            content=json.dumps(
                {
                    "outcome": outcome,
                    "evidence_ids": receipt.evidence_ids,
                    "original_evidence": originals,
                },
                ensure_ascii=False,
            ),
        )

    def run(self) -> HarnessResult:
        self.stop_reason = None
        if (
            self.progress
            and self.context.depth == 0
            and self.context.services.get("final_repair") is not True
        ):
            self.progress.report("started")
        status = OutcomeStatus.PARTIAL
        answer: str | None = None
        try:
            while True:
                assembled = self.context.services.get("assembled_answer")
                if (
                    not self.context.depth
                    and self.context.services.get("experimental_parallel") is True
                    and isinstance(assembled, str)
                    and assembled.strip()
                ):
                    self.context.check_active()
                    blocked = self.draft_guard(assembled) if self.draft_guard else None
                    if blocked is None:
                        self.last_draft = answer = assembled
                        status = (
                            OutcomeStatus.PARTIAL
                            if self.context.services.get("independent_partial") is True
                            else OutcomeStatus.FOUND
                        )
                        self.stop_reason = "independent_answers_assembled"
                        break
                    self.publication_gap = blocked
                    self.context.services.pop("assembled_answer", None)
                if self.context.services.get("question_research_started") is True:
                    self.context.check_active()
                else:
                    self.context.check_research_active()
                budget = self.context.budget
                limit = budget.limits["tools"]
                if (
                    self.context.services.get("research_state")
                    and self.context.services.get("final_repair") is not True
                    and self.context.services.get("independent_question") is not True
                    and limit >= 16
                ):
                    limit -= 8
                if (
                    not budget.unlimited_execution
                    and budget.snapshot()["tools"] >= limit
                ):
                    raise RunStopped(
                        "Shared tools budget exhausted; recorded originals retained for finalization"
                    )
                self.context.consume_research_decision()
                decision = self.decide(self.view())
                self.context.check_active()
                if self.on_decision:
                    self.on_decision(decision)
                research_state = self.context.services.get("research_state")
                self.questions = (
                    list(research_state.questions)
                    if isinstance(research_state, ResearchState)
                    else list(dict.fromkeys(self.questions + decision.questions))
                )
                self.facts = list(dict.fromkeys(self.facts + decision.facts))
                if decision.calls:
                    if len({call.call_id for call in decision.calls}) != len(
                        decision.calls
                    ):
                        raise ValueError("Decision contains duplicate call IDs")
                    results: list[ToolMessage] = []
                    for receipt in self._dispatch(decision.calls):
                        self.context.check_active()
                        results.append(self._model_tool_result(receipt))
                    if decision.assistant_message is not None:
                        self.turns.append(
                            ResearchTurn(
                                assistant=decision.assistant_message, results=results
                            )
                        )
                        self._trim_turns()
                    self._save()
                    submitted = self.context.services.get("submitted_answer")
                    if (
                        (
                            not self.context.depth
                            or self.context.services.get("independent_question") is True
                        )
                        and isinstance(submitted, str)
                        and submitted.strip()
                    ):
                        self.last_draft = answer = submitted
                        status = OutcomeStatus.FOUND
                        self.stop_reason = "model_submitted_answer"
                        break
                    assembled = self.context.services.get("assembled_answer")
                    if (
                        not self.context.depth
                        and isinstance(assembled, str)
                        and assembled.strip()
                    ):
                        self.last_draft = answer = assembled
                        status = (
                            OutcomeStatus.PARTIAL
                            if self.context.services.get("independent_partial") is True
                            else OutcomeStatus.FOUND
                        )
                        self.stop_reason = "independent_answers_assembled"
                        break
                    partial_answer = (
                        self.partial_submission() if self.partial_submission else None
                    )
                    if isinstance(partial_answer, str) and partial_answer.strip():
                        # Submission ends research, not source-backed publication checks.
                        self.last_draft = answer = partial_answer
                        status = OutcomeStatus.PARTIAL
                        self.stop_reason = "model_requested_partial_publication"
                        break
                    if all(
                        call.name in {"wait_researcher", "list_researchers"}
                        for call in decision.calls
                    ) and any(
                        call.name == "wait_researcher" for call in decision.calls
                    ):
                        wait = self.context.services.get("wait_for_task_change")
                        if callable(wait):
                            selected = decision.calls[-1].arguments.get("task_id")
                            wait(selected if isinstance(selected, str) else None)
                    continue
                if decision.answer is not None and decision.answer.strip():
                    self.last_draft = decision.answer
                    blocked = self.finalize_guard() if self.finalize_guard else None
                    if blocked is None and self.draft_guard:
                        blocked = self.draft_guard(decision.answer)
                    if blocked is not None:
                        self.publication_gap = blocked
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
                        if (
                            self._blocked_attempts[fingerprint] >= 3
                            and self.context.services.get("independent_question")
                            is not True
                            and not (
                                self.context.services.get("research_profile")
                                == "experimental"
                                and (
                                    blocked.data.get("pending_related_source_review")
                                    is True
                                    or (
                                        isinstance(
                                            blocked.data.get(
                                                "retained_authority_requirements"
                                            ),
                                            list,
                                        )
                                        and bool(
                                            blocked.data[
                                                "retained_authority_requirements"
                                            ]
                                        )
                                    )
                                )
                            )
                        ):
                            status = OutcomeStatus.PARTIAL
                            self.stop_reason = "repeated_publication_gap"
                            break
                        continue
                    answer = decision.answer
                    status = OutcomeStatus.FOUND
                    self.publication_gap = None
                    self.stop_reason = (
                        "verified_draft" if self.draft_guard else "answer_ready"
                    )
                    break
                self._save()
        except RunStopped as error:
            self.stop_reason = str(error)
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
            stop_reason=self.stop_reason,
            publication_gap=self.publication_gap,
        )
