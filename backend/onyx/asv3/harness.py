from __future__ import annotations

import contextvars
import copy
import hashlib
import json
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from typing import Callable, cast

from pydantic import JsonValue

from onyx.asv3.artifacts import ArtifactStore, artifact_reference, compact_json
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.cropped_article_bindings import (
    CroppedArticleBindings,
    without_ordered_source_read_data,
)
from onyx.asv3.cropped_cost_budget import CroppedResearchBudgetReached
from onyx.asv3.cropped_crossrefs import CroppedCrossReferencePolicy
from onyx.asv3.cropped_research_limit import CroppedResearchLimit
from onyx.asv3.cropped_worker_handoff import (
    WORKER_HANDOFF_REASON,
    CroppedWorkerHandoff,
    contains_worker_handoff,
    should_handoff_worker,
)
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
from onyx.asv3.native_cache_projection import lossless_original_transport_enabled
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.parallel_execution import parallel_execution_enabled
from onyx.asv3.progress import ProgressReporter, action_narration, public_action_id
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.research_state import EvidenceWorkingSet, ResearchState
from onyx.asv3.retained_answer import retained_answer_enabled
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.asv3.working_memory import WorkingMemory
from onyx.llm.models import ToolMessage
from onyx.tracing.answer_graph import graph_step

DecisionMaker = Callable[[HarnessView], Decision]
CheckpointWriter = Callable[[dict[str, JsonValue]], None]
_REPEATED_FAILED_CALL = "Repeated failed call: change the arguments or method"


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
        complete_submission: Callable[[], str | None] | None = None,
        partial_submission: Callable[[], str | None] | None = None,
        report_terminal: bool = True,
        report_started: bool = True,
        on_decision: Callable[[Decision], None] | None = None,
        checkpoint_snapshot: Callable[[], dict[str, JsonValue]] | None = None,
        repair_decision: Callable[[Harness], Decision] | None = None,
    ) -> None:
        self.request = request
        self.context = context
        self.registry = registry
        self.decide = decide
        self.evidence = evidence or EvidenceLedger()
        self.progress = progress
        self.on_receipt = on_receipt
        self.checkpoint = checkpoint
        self.checkpoint_snapshot = checkpoint_snapshot
        self.max_workers = max_workers
        self.adaptive_tool_parallelism = adaptive_tool_parallelism
        self.max_context_chars = max_context_chars
        self.finalize_guard = finalize_guard
        self.draft_guard = draft_guard
        self.complete_submission = complete_submission
        self.partial_submission = partial_submission
        self.report_terminal = report_terminal
        self.report_started = report_started
        self.on_decision = on_decision
        self.repair_decision = repair_decision
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
        self._unavailable_governing_attempts: dict[str, int] = {}
        self._cheap_rejection_fingerprint: str | None = None
        self._cheap_rejection_attempts = 0
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

    def _receipt_data(
        self, receipt: ToolReceipt, *, max_chars: int = 12000
    ) -> dict[str, JsonValue]:
        data = (
            copy.deepcopy(receipt.outcome.data)
            if self.context.services.get("asv3_exp_cropped") is True
            and contains_worker_handoff(
                receipt.outcome.data, ledger=self.evidence, context=self.context
            )
            else compact_json(receipt.outcome.data, max_chars=max_chars)
        )
        assert isinstance(data, dict)
        return data

    def _receipt_reference(self, receipt: ToolReceipt) -> ToolReceipt:
        data = self._receipt_data(receipt)
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
            **self._tuned_submission_snapshot(),
            **self._cropped_draft_delivery_snapshot(),
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
            **self._cheap_retry_checkpoint(),
        }

    def snapshot_control(self) -> dict[str, JsonValue]:
        """Detach owner-thread control; shared research is captured separately."""
        return copy.deepcopy(
            {
                "version": 1,
                **self._tuned_submission_snapshot(),
                **self._cropped_draft_delivery_snapshot(),
                "run_id": self.context.run_id,
                "request": self.request,
                "language": self.context.language,
                "questions": list(self.questions),
                "facts": list(self.facts),
                "receipts": [
                    self._receipt_reference(item).model_dump(mode="json")
                    for item in self.receipts
                ],
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
                **self._cheap_retry_checkpoint(),
            }
        )

    def _cheap_retry_checkpoint(self) -> dict[str, JsonValue]:
        if self.context.services.get("asv3_exp_cheap") is not True:
            return {}
        return {
            "cheap_publication_retry": {
                "version": 1,
                "fingerprint": self._cheap_rejection_fingerprint,
                "attempts": self._cheap_rejection_attempts,
            }
        }

    def _cropped_draft_delivery_snapshot(self) -> dict[str, JsonValue]:
        call_id = self.context.services.get("last_model_call_id")
        if (
            self.context.services.get("asv3_exp_cropped") is not True
            or not self.last_draft
            or not isinstance(call_id, str)
            or self.evidence.delivery_flow(call_id) is None
        ):
            return {}
        return {
            "cropped_draft_delivery": {
                "draft_sha256": hashlib.sha256(self.last_draft.encode()).hexdigest(),
                "model_call_id": call_id,
            }
        }

    def _cheap_publication_fingerprint(self, candidate: str, gap: ToolOutcome) -> str:
        originals: list[JsonValue] = []
        for number in self.evidence.citation_numbers():
            item = self.evidence.get(number)
            assert item is not None
            originals.append(
                {
                    "citation": number,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                    "metadata": item.metadata,
                    "search_doc": {
                        "document_id": item.search_doc.document_id,
                        "source_type": item.search_doc.source_type.value,
                        "metadata": item.search_doc.metadata,
                    }
                    if item.search_doc is not None
                    else None,
                }
            )
        outcomes = self.context.services.get("outcome_map")
        semantic_outcomes = (
            outcomes.export() if isinstance(outcomes, OutcomeMap) else {}
        )
        semantic_outcomes.pop("revision", None)
        research = self.context.services.get("research_state")
        semantic_research = (
            research.export() if isinstance(research, ResearchState) else {}
        )
        semantic_research.pop("revision", None)
        observations = {
            json.dumps(
                {
                    "name": receipt.call.name,
                    "arguments": {
                        key: value
                        for key, value in receipt.call.arguments.items()
                        if key not in {"_public_update", "_language", "_notifications"}
                    },
                    "outcome": receipt.outcome.model_dump(mode="json"),
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            for receipt in self.receipts
            if receipt.call.name
            not in {
                "finalization_status",
                "submit_answer",
                "submit_partial_answer",
                "submit_retained_answer",
                "submit_retained_partial_answer",
                "assemble_answers",
            }
        }
        return hashlib.sha256(
            json.dumps(
                {
                    "candidate": candidate,
                    "gap": gap.model_dump(mode="json"),
                    "originals": originals,
                    "facts": self.facts,
                    "questions": self.questions,
                    "outcomes": semantic_outcomes,
                    "research": semantic_research,
                    "observations": sorted(observations),
                },
                sort_keys=True,
                ensure_ascii=False,
            ).encode()
        ).hexdigest()

    def _remember_cheap_rejection(self, candidate: str, gap: ToolOutcome) -> None:
        if self.context.services.get("asv3_exp_cheap") is not True:
            return
        fingerprint = self._cheap_publication_fingerprint(candidate, gap)
        if fingerprint != self._cheap_rejection_fingerprint:
            self._cheap_rejection_fingerprint = fingerprint
            self._cheap_rejection_attempts = 0
        # Initial rejection, two fresh repairs and one ordinary-context fallback.
        self._cheap_rejection_attempts = min(4, self._cheap_rejection_attempts + 1)

    def _cheap_rejection_exhausted(self) -> bool:
        return (
            self.context.services.get("asv3_exp_cheap") is True
            and self._cheap_rejection_attempts >= 4
            and self.last_draft is not None
            and self.publication_gap is not None
            and self._cheap_rejection_fingerprint
            == self._cheap_publication_fingerprint(
                self.last_draft, self.publication_gap
            )
        )

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
        if self.context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT:
            attempts = snapshot.get("tuned_unavailable_governing_attempts", {})
            if not isinstance(attempts, dict) or any(
                not key or type(count) is not int or count < 0
                for key, count in attempts.items()
            ):
                raise ValueError("Invalid unavailable governing submission checkpoint")
            self._unavailable_governing_attempts = cast(dict[str, int], attempts)
        if self.context.services.get("asv3_exp_cheap") is True:
            retries = snapshot.get("cheap_publication_retry")
            if not isinstance(retries, dict):
                raise ValueError("Cheap publication retry checkpoint is missing")
            fingerprint, attempts = retries.get("fingerprint"), retries.get("attempts")
            if (
                set(retries) != {"version", "fingerprint", "attempts"}
                or type(retries.get("version")) is not int
                or retries.get("version") != 1
                or type(attempts) is not int
                or not 0 <= attempts <= 4
                or (
                    fingerprint is not None
                    and (
                        not isinstance(fingerprint, str)
                        or re.fullmatch(r"[a-f0-9]{64}", fingerprint) is None
                    )
                )
                or (fingerprint is None and attempts != 0)
            ):
                raise ValueError("Invalid Cheap publication retry checkpoint")
            self._cheap_rejection_fingerprint = cast(str | None, fingerprint)
            self._cheap_rejection_attempts = attempts
        evidence = snapshot.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Checkpoint evidence is missing")
        self.evidence.restore(evidence, self.context)
        if self.context.services.get("asv3_exp_cropped") is True:
            self.context.services.pop("last_model_call_id", None)
            delivery = snapshot.get("cropped_draft_delivery")
            if delivery is not None:
                call_id = (
                    delivery.get("model_call_id")
                    if isinstance(delivery, dict)
                    else None
                )
                if (
                    not isinstance(delivery, dict)
                    or set(delivery) != {"draft_sha256", "model_call_id"}
                    or not self.last_draft
                    or delivery.get("draft_sha256")
                    != hashlib.sha256(self.last_draft.encode()).hexdigest()
                    or not isinstance(call_id, str)
                    or self.evidence.delivery_flow(call_id) is None
                ):
                    raise ValueError("Cropped draft delivery checkpoint changed")
                self.context.services["last_model_call_id"] = call_id
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
                self.working_memory.observe(receipt, context=self.context)
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
            preparation = (
                graph_step("asv3.checkpoint_preparation", {})
                if parallel_execution_enabled(self.context)
                else nullcontext()
            )
            with preparation as checkpoint_step:
                snapshot_started = time.monotonic()
                snapshot = (
                    self.checkpoint_snapshot()
                    if self.checkpoint_snapshot is not None
                    else self.snapshot()
                )
                snapshot_finished = time.monotonic()
                self.checkpoint(snapshot)
                if checkpoint_step is not None:
                    checkpoint_step.output_value = {
                        "snapshot_seconds": snapshot_finished - snapshot_started,
                        "write_seconds": time.monotonic() - snapshot_finished,
                    }

    def save_repair_checkpoint(self) -> None:
        """Persist a reserved focused attempt before any research or writer call."""
        if self.checkpoint is not None:
            self._save()
        else:
            callback = self.context.services.get("cropped_repair_checkpoint")
            if callable(callback):
                callback()

    def dispatch_repair_reads(self, calls: list[CapabilityCall]) -> list[ToolReceipt]:
        """Keep targeted research on the ordinary ACL, receipt and evidence path."""
        return self._dispatch(calls)

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
            compacted = self._receipt_data(receipt, max_chars=3000)
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
        view = HarnessView(
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

        if self.context.services.get("asv3_exp_cropped") is True and isinstance(
            state, ResearchState
        ):
            from onyx.asv3.cropped_task_packets import packet_for_view

            projection = packet_for_view(
                view, self.receipts, state, self.context, self.evidence
            )
            if projection is not None:
                view, _manifest = projection
        return view

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
        return (
            "final"
            if call.name == "verify_claim"
            and not parallel_execution_enabled(self.context)
            else "tools"
        )

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
        if self.context.services.get("asv3_workflow_variant") == "asv3_tuned":
            from onyx.asv3.article_bindings import ArticleBindings

            bindings = self.context.services.get("tuned_article_bindings")
            if isinstance(bindings, ArticleBindings):
                bindings.consume_receipt(receipt, self.context, self.evidence)
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
        bindings = self.context.services.get("cropped_article_bindings")
        if self.context.services.get("asv3_exp_cropped") is True and isinstance(
            bindings, CroppedArticleBindings
        ):
            bindings.record_receipt(receipt, self.context, self.evidence)
            receipt.outcome.data = without_ordered_source_read_data(
                receipt.outcome.data
            )
        crossrefs = self.context.services.get("cropped_cross_references")
        if self.context.services.get("asv3_exp_cropped") is True and isinstance(
            crossrefs, CroppedCrossReferencePolicy
        ):
            crossrefs.observe(receipt, self.context, self.evidence)
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
        compacted = self._receipt_data(receipt)
        assert isinstance(compacted, dict)
        receipt.outcome.data = compacted
        candidate = receipt.call.arguments.get("answer")
        if (
            retained_answer_enabled(self.context)
            and receipt.call.name in {"submit_answer", "submit_partial_answer"}
            and receipt.outcome.status
            in {OutcomeStatus.INVALID, OutcomeStatus.PARTIAL, OutcomeStatus.DENIED}
            and isinstance(candidate, str)
            and candidate.strip()
            and not (
                self.context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
                and receipt.call.name == "submit_partial_answer"
                and receipt.outcome.status == OutcomeStatus.PARTIAL
                and self.partial_submission is not None
                and self.partial_submission() == candidate
            )
        ):
            self.last_draft = candidate
            self.publication_gap = receipt.outcome
            self._remember_cheap_rejection(candidate, receipt.outcome)
        self.context.check_active()
        if self.on_receipt:
            self.on_receipt(receipt)
        self.working_memory.observe(receipt, context=self.context)
        receipt.outcome = receipt.outcome.model_copy(update={"evidence": []})
        self.receipts.append(receipt)
        self._seen_calls.add(receipt.call.call_id)
        self._committed_calls.add(receipt.call.call_id)
        self._pending_calls.pop(receipt.call.call_id, None)
        if self.context.services.get("asv3_exp_cropped") is True:
            from onyx.asv3.cropped_composed_topics import commit_cropped_composed_topics

            commit_cropped_composed_topics(receipt, self.context, self.evidence)
        if (
            receipt.evidence_ids
            and receipt.outcome.status in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}
            and receipt.call.name
            in {
                "read_evidence",
                "read_chunk",
                "read_chunk_context",
                "read_provision",
                "read_named_provision",
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

    def _can_revalidate_delivered_review(
        self, call: CapabilityCall, signature: str
    ) -> bool:
        """A newly delivered original can invalidate a cached delivery rejection."""
        if (
            not parallel_execution_enabled(self.context)
            and self.context.services.get("asv3_exp_cheap") is not True
        ) or call.name not in {
            "submit_answer",
            "submit_partial_answer",
            "assemble_answers",
        }:
            return False
        model_call = self.context.services.get("last_model_call_id")
        raw_reviews = call.arguments.get("_related_source_reviews")
        if not isinstance(model_call, str) or not isinstance(raw_reviews, list):
            return False
        witnesses: set[int] = set()
        for review in raw_reviews:
            if not isinstance(review, dict):
                return False
            ranges = review.get("witnesses", [])
            if not isinstance(ranges, list):
                return False
            for witness in ranges:
                if not isinstance(witness, dict):
                    return False
                citation = witness.get("citation")
                if type(citation) is not int:
                    return False
                witnesses.add(cast(int, citation))
        if not witnesses or not witnesses <= self.evidence.completely_delivered(
            model_call
        ):
            return False
        for previous in reversed(self.receipts):
            if (
                previous.call.name != call.name
                or self._call_signature(previous.call) != signature
            ):
                continue
            outcome = previous.outcome
            # A cached refusal does not replace the underlying validation receipt.
            if (
                outcome.status == OutcomeStatus.INVALID
                and outcome.summary == _REPEATED_FAILED_CALL
                and not outcome.data
                and previous.elapsed_seconds == 0
            ):
                continue
            diagnostic = outcome.data.get("related_source_review_error")
            if (
                outcome.status != OutcomeStatus.INVALID
                or outcome.data.get("invalid_related_source_review") is not True
                or not isinstance(diagnostic, dict)
            ):
                return False
            errors = diagnostic.get("validation_errors")
            if (
                diagnostic.get("code") != "not_fully_delivered"
                or not isinstance(errors, list)
                or not errors
            ):
                return False
            return all(
                isinstance(error, dict) and error.get("code") == "not_fully_delivered"
                for error in errors
            )
        return False

    def _dispatch(self, calls: list[CapabilityCall]) -> list[ToolReceipt]:
        terminal_batch = self.context.services.get("hosted_terminal_batch")
        if (
            parallel_execution_enabled(self.context)
            and isinstance(terminal_batch, dict)
            and len(calls) > 1
        ):
            terminal_id = terminal_batch.get("terminal_call_id")
            predecessors = [call for call in calls if call.call_id != terminal_id]
            terminal = [call for call in calls if call.call_id == terminal_id]
            if (
                len(terminal) == 1
                and terminal[0].name == "submit_answer"
                and all(
                    call.name in {"update_research", "verify_claim"}
                    for call in predecessors
                )
                and [call.call_id for call in predecessors]
                == terminal_batch.get("predecessor_call_ids")
                and terminal_batch.get("model_call_id")
                == self.context.services.get("last_model_call_id")
            ):
                receipts = self._dispatch(predecessors)
                if all(
                    receipt.outcome.status == OutcomeStatus.FOUND
                    and receipt.call.call_id in self._committed_calls
                    for receipt in receipts
                ):
                    self.context.services["hosted_terminal_completed"] = dict(
                        terminal_batch
                    )
                    receipts.extend(self._dispatch(terminal))
                else:
                    blocked = ToolReceipt(
                        call=terminal[0],
                        elapsed_seconds=0,
                        outcome=ToolOutcome(
                            status=OutcomeStatus.PARTIAL,
                            summary="Correct the rejected same-decision update before submitting the retained answer.",
                            data={
                                "rejected_predecessor_call_ids": [
                                    receipt.call.call_id
                                    for receipt in receipts
                                    if receipt.outcome.status != OutcomeStatus.FOUND
                                ]
                            },
                        ),
                    )
                    self._commit_receipt(blocked)
                    receipts.append(blocked)
                by_id = {receipt.call.call_id: receipt for receipt in receipts}
                return [by_id[call.call_id] for call in calls]
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
                if call.call_id in self._seen_calls or (
                    signature in self._seen_failures
                    and not self._can_revalidate_delivered_review(call, signature)
                ):
                    ready[call.call_id] = ToolReceipt(
                        call=call,
                        elapsed_seconds=0,
                        outcome=ToolOutcome(
                            status=OutcomeStatus.INVALID,
                            summary=_REPEATED_FAILED_CALL,
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
                registry_cached = self.registry.cached_cropped_read(call, self.context)
                if registry_cached is not None:
                    ready[call.call_id] = ToolReceipt(
                        call=call, elapsed_seconds=0, outcome=registry_cached
                    )
                    continue
                cropped_limit = self.context.services.get("cropped_research_controller")
                if (
                    self.context.services.get("asv3_exp_cropped") is True
                    and isinstance(cropped_limit, CroppedResearchLimit)
                    and call.argument_error is None
                ):
                    cropped_limit.check_context(self.context)
                    admission = cropped_limit.admit_tool(
                        call.name,
                        call.arguments,
                        critical_target_key=hashlib.sha256(
                            self._read_signature(call).encode("utf-8")
                        ).hexdigest(),
                        reserve=False,
                    )
                    if not admission.allowed and admission.reason != (
                        "critical_read_already_reserved_reuse_recorded_evidence"
                    ):
                        ready[call.call_id] = ToolReceipt(
                            call=call,
                            elapsed_seconds=0,
                            outcome=cropped_limit.blocked_tool(admission),
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
            # Checkpoint locators before execution; recorded interrupted batches are never replayed.
            self._save()
            prepared_searches: dict[str, object] = {}
            prepare_search_batch = self.context.services.get("prepare_search_batch")
            search_calls = [
                call
                for call in calls
                if call.call_id not in ready
                and call.name == "search_corpus"
                and call.argument_error is None
            ]
            if callable(prepare_search_batch) and (
                search_calls
                or self.context.services.get("asv3_exp_cropped") is not True
            ):
                try:
                    batch = prepare_search_batch(
                        search_calls,
                        self.context,
                    )
                except CroppedResearchBudgetReached as error:
                    if self.context.services.get("asv3_exp_cropped") is not True:
                        raise
                    batch = {}
                    for call in search_calls:
                        ready[call.call_id] = ToolReceipt(
                            call=call,
                            elapsed_seconds=0,
                            outcome=ToolOutcome(
                                status=OutcomeStatus.PARTIAL,
                                summary=str(error),
                                data={
                                    "blocked_by_research_limit": True,
                                    "not_read": True,
                                    "absence_proven": False,
                                },
                            ),
                        )
                if isinstance(batch, dict):
                    prepared_searches = batch
            for call in calls:
                if call.call_id in ready:
                    self._commit_receipt(ready[call.call_id])
                    continue
                child = self.context.child()
                child.services["applied_outcome_metadata_call"] = call
                if self.context.services.get("asv3_exp_cropped") is True:
                    child.services["cropped_tool_checkpoint"] = (
                        self.save_repair_checkpoint
                    )
                completed = self.context.services.get("hosted_terminal_completed")
                if (
                    parallel_execution_enabled(self.context)
                    and isinstance(completed, dict)
                    and completed.get("terminal_call_id") == call.call_id
                ):
                    child.services["hosted_terminal_execution_id"] = call.call_id
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
                max_chars=None
                if lossless_original_transport_enabled(self.context)
                else 20000,
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

    def _tuned_submission_snapshot(self) -> dict[str, JsonValue]:
        if self.context.services.get("asv3_workflow_variant") != ASV3_TUNED_VARIANT:
            return {}
        return {
            "tuned_unavailable_governing_attempts": dict(
                self._unavailable_governing_attempts
            )
        }

    def _repeated_unavailable_governing_gap(self) -> bool:
        if (
            self.context.services.get("asv3_workflow_variant") != ASV3_TUNED_VARIANT
            or self.publication_gap is None
        ):
            return False
        unavailable = self.context.services.get("tuned_unavailable_governing_keys")
        if not callable(unavailable):
            return False
        keys = unavailable(self.publication_gap.data)
        return self._record_unavailable_governing_attempts(keys)

    def _record_unavailable_governing_attempts(self, keys: set[str]) -> bool:
        for key in keys:
            self._unavailable_governing_attempts[key] = (
                self._unavailable_governing_attempts.get(key, 0) + 1
            )
        return any(self._unavailable_governing_attempts[key] >= 3 for key in keys)

    def _repeated_blocked_governing_research(self, receipts: list[ToolReceipt]) -> bool:
        if (
            self.context.services.get("asv3_workflow_variant") != ASV3_TUNED_VARIANT
            or not receipts
        ):
            return False

        def blocked_keys(outcome: dict[str, JsonValue]) -> set[str]:
            data = outcome.get("data")
            if not isinstance(data, dict):
                return set()
            key = data.get("governing_original_target_key")
            if (
                outcome.get("status") == OutcomeStatus.DENIED.value
                and data.get("governing_original_retry_blocked") is True
                and data.get("acquisition_status") in ("unavailable", "failed")
                and isinstance(key, str)
                and key
            ):
                return {key}
            return set()

        keys: set[str] = set()
        for receipt in receipts:
            outcome = receipt.outcome
            if outcome.evidence or outcome.original_reads or outcome.artifacts:
                return False
            if receipt.call.name == "compose_tool_calls":
                steps = outcome.data.get("steps")
                if not isinstance(steps, dict) or not steps:
                    return False
                declared = receipt.call.arguments.get("steps")
                if not isinstance(declared, list):
                    return False
                dependencies: dict[str, set[str]] = {}
                for item in declared:
                    if not isinstance(item, dict) or not isinstance(
                        item.get("id"), str
                    ):
                        return False
                    needs = item.get("depends_on", [])
                    if not isinstance(needs, list) or any(
                        not isinstance(need, str) for need in needs
                    ):
                        return False
                    dependencies[item["id"]] = set(cast(list[str], needs))
                if dependencies.keys() != steps.keys():
                    return False
                blocked: set[str] = set()
                cancelled: set[str] = set()
                for step_id, step in steps.items():
                    step_keys = blocked_keys(step) if isinstance(step, dict) else set()
                    if step_keys:
                        blocked.add(step_id)
                        keys.update(step_keys)
                    elif (
                        isinstance(step, dict)
                        and step.get("status") == OutcomeStatus.CANCELLED.value
                        and step.get("summary") == "A required dependency failed."
                        and dependencies[step_id]
                    ):
                        cancelled.add(step_id)
                    else:
                        return False
                while cancelled:
                    settled = {
                        step_id
                        for step_id in cancelled
                        if dependencies[step_id] <= blocked
                    }
                    if not settled:
                        return False
                    blocked.update(settled)
                    cancelled.difference_update(settled)
            else:
                receipt_keys = blocked_keys(outcome.model_dump(mode="json"))
                if not receipt_keys:
                    return False
                keys.update(receipt_keys)
        return self._record_unavailable_governing_attempts(keys)

    def run(self) -> HarnessResult:
        self.stop_reason = None
        if (
            self.progress
            and self.report_started
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
                if should_handoff_worker(self.context):
                    self.context.check_active()
                    status = OutcomeStatus.PARTIAL
                    self.stop_reason = WORKER_HANDOFF_REASON
                    break
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
                decision: Decision | None = None
                host_decision = self.context.services.get("cropped_host_decision")
                if self.context.services.get("asv3_exp_cropped") is True and callable(
                    host_decision
                ):
                    decision = host_decision(self)
                    if decision is not None and not isinstance(decision, Decision):
                        raise TypeError(
                            "Cropped host decision must return Decision or None"
                        )
                if decision is None:
                    self.context.consume_research_decision()
                    decision = (
                        self.repair_decision(self)
                        if self.repair_decision is not None
                        and self.publication_gap is not None
                        else self.decide(self.view())
                    )
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
                    decision_receipts: list[ToolReceipt] = []
                    rejected_submission = False
                    for receipt in self._dispatch(decision.calls):
                        self.context.check_active()
                        decision_receipts.append(receipt)
                        results.append(self._model_tool_result(receipt))
                        rejected_submission = rejected_submission or (
                            receipt.call.name
                            in {"submit_answer", "submit_partial_answer"}
                            and receipt.outcome.status
                            in {
                                OutcomeStatus.INVALID,
                                OutcomeStatus.PARTIAL,
                                OutcomeStatus.DENIED,
                            }
                        )
                    repeated_research = self._repeated_blocked_governing_research(
                        decision_receipts
                    )
                    if decision.assistant_message is not None:
                        self.turns.append(
                            ResearchTurn(
                                assistant=decision.assistant_message, results=results
                            )
                        )
                        self._trim_turns()
                    self._save()
                    submitted = (
                        self.complete_submission()
                        if self.complete_submission is not None
                        else self.context.services.get("submitted_answer")
                    )
                    if (
                        (
                            self.complete_submission is not None
                            or not self.context.depth
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
                    if (
                        rejected_submission
                        and self.context.services.get("asv3_workflow_variant")
                        == ASV3_TUNED_VARIANT
                    ):
                        repeated = self._repeated_unavailable_governing_gap()
                        self._save()
                        if repeated:
                            status = OutcomeStatus.PARTIAL
                            self.stop_reason = "repeated_unavailable_governing_original"
                            break
                    if repeated_research:
                        status = OutcomeStatus.PARTIAL
                        self.stop_reason = "repeated_unavailable_governing_original"
                        break
                    if self._cheap_rejection_exhausted():
                        status = OutcomeStatus.PARTIAL
                        self.stop_reason = "repeated_publication_gap"
                        break
                    wait_only = all(
                        call.name in {"wait_researcher", "list_researchers"}
                        for call in decision.calls
                    )
                    has_wait = any(
                        call.name == "wait_researcher" for call in decision.calls
                    )
                    if wait_only and (
                        has_wait
                        or self.context.services.get("asv3_exp_cropped") is True
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
                        repeated = self._repeated_unavailable_governing_gap()
                        self._remember_cheap_rejection(decision.answer, blocked)
                        self._save()
                        if repeated:
                            status = OutcomeStatus.PARTIAL
                            self.stop_reason = "repeated_unavailable_governing_original"
                            break
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
                        if self._cheap_rejection_exhausted() or (
                            self.context.services.get("asv3_exp_cheap") is not True
                            and self._blocked_attempts[fingerprint] >= 3
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
        except CroppedWorkerHandoff:
            self.stop_reason = WORKER_HANDOFF_REASON
            status = OutcomeStatus.PARTIAL
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
