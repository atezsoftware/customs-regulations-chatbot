from __future__ import annotations

import contextvars
import json
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import cast

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, RunContext, RunStopped, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.legal_composite.models import ResearchPlan, SourceAction, WorkflowPolicy
from onyx.tracing.answer_graph import graph_step


class InvalidSourceAction(RunStopped):
    pass


class CanonicalAcquirer:
    def __init__(
        self,
        registry: CapabilityRegistry,
        context: RunContext,
        ledger: EvidenceLedger,
        policy: WorkflowPolicy,
        *,
        registry_for_action: Callable[[SourceAction], CapabilityRegistry] | None = None,
        expand_actions: Callable[[list[SourceAction], ResearchPlan], list[SourceAction]]
        | None = None,
        lane_inventory: dict[str, JsonValue] | None = None,
        on_batch_progress: Callable[[list[SourceAction], int, int], None] | None = None,
        host_registry: CapabilityRegistry | None = None,
    ) -> None:
        self.registry = registry
        self.host_registry = host_registry or registry
        self.context = context
        self.ledger = ledger
        self.policy = policy
        self.registry_for_action = registry_for_action or (lambda _action: registry)
        self.expand_actions = expand_actions or (lambda actions, _plan: actions)
        self.lane_inventory = lane_inventory or {}
        self.on_batch_progress = on_batch_progress or (
            lambda _actions, _pending, _completed: None
        )
        self.search_calls = 0
        self.last_receipts: list[dict[str, JsonValue]] = []
        self._completed: dict[str, dict[str, JsonValue]] = {}

    def definitions(self) -> list[dict[str, JsonValue]]:
        return self.registry.definitions(self.context)

    def _bind_originals(self, numbers: list[int], need_ids: list[str]) -> None:
        retained = []
        for number in dict.fromkeys(numbers):
            item = self.ledger.get(number)
            if item is not None:
                item.question_ids = list(dict.fromkeys(item.question_ids + need_ids))
                retained.append(item)
        self.ledger.add(retained, self.context)

    def _dispatch_action(
        self,
        action: SourceAction,
        call: CapabilityCall,
        child: RunContext,
        host_actions: bool = False,
    ) -> ToolOutcome:
        with graph_step(
            "legal_composite.source_task",
            {
                "tool": call.name,
                "source_kind": action.source_kind,
                "need_ids": action.need_ids,
                "query": call.arguments.get("query"),
            },
        ) as step:
            registry = (
                self.host_registry
                if host_actions and action.source_kind is None
                else self.registry_for_action(action)
            )
            outcome = registry.dispatch(call, child)
            step.output_value = {
                "status": outcome.status.value,
                "original_count": len(outcome.evidence),
            }
            return outcome

    def acquire(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        return self._acquire(self.expand_actions(actions, plan), plan)

    def acquire_host_actions(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        """Run observed dependency targets without expanding each query to every lane."""
        return self._acquire(actions, plan, host_actions=True)

    def _acquire(
        self,
        actions: list[SourceAction],
        plan: ResearchPlan,
        *,
        host_actions: bool = False,
    ) -> list[dict[str, JsonValue]]:
        result: list[dict[str, JsonValue]] = []
        self.last_receipts = result
        self.context.check_research_active()
        need_ids = {need.need_id for need in plan.needs}
        calls: list[tuple[SourceAction, CapabilityCall, str]] = []
        for action in actions:
            if set(action.need_ids) - need_ids:
                raise InvalidSourceAction(
                    "Source action refers to an unknown frozen need"
                )
            selected_registry = (
                self.host_registry
                if host_actions and action.source_kind is None
                else self.registry_for_action(action)
            )
            spec = selected_registry.get(action.tool)
            if spec is None or spec.external or spec.orchestrates:
                raise InvalidSourceAction(
                    "Source action is outside the canonical capability allowlist"
                )
            arguments = dict(action.arguments)
            if action.tool == "search_corpus":
                # The planner already supplies focused queries without query expansion.
                arguments["expand_query"] = False
            signature = json.dumps(
                {
                    "tool": action.tool,
                    "source_kind": action.source_kind,
                    "arguments": {
                        key: value
                        for key, value in arguments.items()
                        if key != "_public_update"
                    },
                },
                sort_keys=True,
            )
            if signature in self._completed:
                recorded = self._completed[signature].get("citations", [])
                assert isinstance(recorded, list)
                self._bind_originals(
                    [number for number in recorded if isinstance(number, int)],
                    action.need_ids,
                )
                result.append(
                    {
                        **self._completed[signature],
                        "need_ids": list(action.need_ids),
                        "reused": True,
                        **({"host_arguments": arguments} if host_actions else {}),
                    }
                )
                continue
            duplicate = next(
                (
                    index
                    for index, (_, _, existing_signature) in enumerate(calls)
                    if existing_signature == signature
                ),
                None,
            )
            if duplicate is not None:
                existing_action, existing_call, _ = calls[duplicate]
                calls[duplicate] = (
                    existing_action.model_copy(
                        update={
                            "need_ids": list(
                                dict.fromkeys(
                                    existing_action.need_ids + action.need_ids
                                )
                            )
                        }
                    ),
                    existing_call,
                    signature,
                )
                continue
            calls.append(
                (
                    action,
                    CapabilityCall(name=action.tool, arguments=arguments),
                    signature,
                )
            )
        requested_searches = sum(call.name == "search_corpus" for _, call, _ in calls)
        if self.search_calls + requested_searches > self.policy.max_search_calls:
            raise RunStopped(
                "Focused search budget exhausted; unresolved needs remain open"
            )
        if self.context.budget.snapshot()["tools"] + len(calls) > self.policy.max_tools:
            raise RunStopped(
                "Canonical acquisition budget exhausted; unresolved needs remain open"
            )
        self.search_calls += requested_searches
        executor = ThreadPoolExecutor(
            max_workers=self.policy.max_parallel_tools,
            thread_name_prefix="legal-composite-source",
        )
        pending: dict[
            Future[ToolOutcome], tuple[SourceAction, CapabilityCall, str, RunContext]
        ] = {}
        batch_actions = [action for action, _call, _signature in calls]
        completed_count = 0
        try:
            for action, call, signature in calls:
                child = self.context.child()
                child.depth = self.context.depth
                captured = contextvars.copy_context()
                future = cast(
                    Future[ToolOutcome],
                    executor.submit(
                        captured.run,
                        self._dispatch_action,
                        action,
                        call,
                        child,
                        host_actions,
                    ),
                )
                pending[future] = action, call, signature, child
            if pending:
                self.on_batch_progress(batch_actions, len(pending), completed_count)
            while pending:
                self.context.check_research_active()
                completed, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in completed:
                    action, call, signature, _child = pending.pop(future)
                    outcome = future.result()
                    for item in outcome.evidence:
                        item.question_ids = list(action.need_ids)
                        if action.source_kind is not None:
                            item.metadata["legal_composite_source_kind"] = (
                                action.source_kind.value
                            )
                    numbers = self.ledger.add(outcome.evidence, self.context)
                    numbers.extend(
                        read.citation
                        for read in outcome.original_reads
                        if read.citation not in numbers
                    )
                    self._bind_originals(numbers, action.need_ids)
                    row: dict[str, JsonValue] = {
                        "tool": call.name,
                        "source_kind": action.source_kind,
                        "need_ids": list(action.need_ids),
                        "status": outcome.status.value,
                        "summary": outcome.summary,
                        "data": outcome.data,
                        "citations": numbers,
                        **({"host_arguments": call.arguments} if host_actions else {}),
                    }
                    # A bounded receipt is navigation only; full originals stay in the ledger.
                    if len(json.dumps(row, ensure_ascii=False)) > 10_000:
                        row["data"] = {
                            "receipt_omitted": True,
                            "instruction": "Use recorded originals or a focused source read.",
                            **(
                                {
                                    key: value
                                    for key, value in outcome.data.items()
                                    if key
                                    in {
                                        "has_more",
                                        "next_position",
                                        "evidence_next_position",
                                        "scan_truncated",
                                        "evidence_truncated",
                                        "next_offset",
                                        "sources",
                                        "candidates",
                                        "edge_id",
                                        "navigation_only",
                                        "absence_proven",
                                    }
                                }
                                if host_actions
                                else {}
                            ),
                        }
                    self._completed[signature] = row
                    result.append(row)
                    completed_count += 1
                    self.on_batch_progress(batch_actions, len(pending), completed_count)
                    with graph_step(
                        "legal_composite.acquisition",
                        {
                            "tool": call.name,
                            "need_ids": action.need_ids,
                            "source_kind": action.source_kind,
                        },
                    ) as step:
                        step.output_value = {
                            "status": outcome.status.value,
                            "citations": numbers,
                        }
            return result
        finally:
            # SDK/DB operations may finish later; they cannot commit late evidence to the ledger.
            for future, (action, call, _, child) in pending.items():
                child.cancel()
                future.cancel()
                result.append(
                    {
                        "tool": call.name,
                        "need_ids": list(action.need_ids),
                        "status": "truncated",
                        "summary": "Acquisition stopped before a complete result; this is not evidence of corpus absence.",
                        "citations": [],
                        **({"host_arguments": call.arguments} if host_actions else {}),
                    }
                )
            executor.shutdown(wait=False, cancel_futures=True)
