"""Bounded source operations; discovery and canonical reading remain explicit."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from uuid import uuid4

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, RunStopped
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.search_adapter import ScopedSearchAdapter
from onyx.legal_review.models import IssuePlan, SourceAction, WorkflowPolicy
from onyx.tracing.answer_graph import graph_step


class SourceAcquirer:
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        context: RunContext,
        ledger: EvidenceLedger,
        policy: WorkflowPolicy,
        search_adapter: ScopedSearchAdapter,
    ) -> None:
        self.registry = registry
        self.context = context
        self.ledger = ledger
        self.policy = policy
        self.search_adapter = search_adapter
        self.searches = 0
        self.receipts: list[dict[str, JsonValue]] = []

    def definitions(self) -> list[dict[str, JsonValue]]:
        return self.registry.definitions(self.context)

    def acquire(
        self, actions: list[SourceAction], plan: IssuePlan, *, finalizing: bool = False
    ) -> None:
        known = {issue.issue_id for issue in plan.issues}
        calls: list[tuple[SourceAction, CapabilityCall]] = []
        for action in actions:
            if set(action.issue_ids) - known:
                raise ValueError("Source action refers to an unknown issue")
            arguments = dict(action.arguments)
            if action.tool == "search_corpus":
                if self.searches >= self.policy.max_searches:
                    raise RunStopped("Legal review search budget exhausted")
                self.searches += 1
                arguments.update(
                    {
                        "coverage_item": ", ".join(action.issue_ids),
                        "evidence_target": "Complete operative originals and material limiting or contrary authority for the requested outcomes",
                        "expand_query": False,
                    }
                )
            calls.append(
                (
                    action,
                    CapabilityCall(
                        call_id=str(uuid4()), name=action.tool, arguments=arguments
                    ),
                )
            )
        prepared = self.search_adapter.prepare_batch(
            [call for _, call in calls], self.context
        )
        # Repair may use evidence operations from the retained finalization window.
        phase_context = (
            RunContext(
                run_id=self.context.run_id,
                scope=self.context.scope,
                services=self.context.services,
                budget=self.context.budget,
                deadline=self.context.deadline,
                research_deadline=self.context.deadline,
                cancelled=self.context.is_cancelled,
            )
            if finalizing
            else self.context
        )
        pool = ThreadPoolExecutor(max_workers=self.policy.max_parallel_tools)
        pending: dict[Future, tuple[SourceAction, CapabilityCall, float]] = {}
        try:
            for action, call in calls:
                child = phase_context.child()
                if call.call_id in prepared:
                    child.services["search_batch_tool"] = prepared[call.call_id]
                future = pool.submit(
                    copy_context().run, self.registry.dispatch, call, child
                )
                pending[future] = (action, call, time.monotonic())
            while pending:
                phase_context.check_research_active()
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    action, call, started = pending.pop(future)
                    outcome = future.result()
                    if outcome.status is OutcomeStatus.CANCELLED:
                        raise RunStopped("Research cancelled")
                    for item in outcome.evidence:
                        item.question_ids = list(action.issue_ids)
                    numbers = self.ledger.add(outcome.evidence, self.context)
                    receipt: dict[str, JsonValue] = {
                        "call_id": call.call_id,
                        "tool": call.name,
                        "arguments": call.arguments,
                        "issue_ids": list(action.issue_ids),
                        "status": outcome.status.value,
                        "summary": outcome.summary,
                        "data": outcome.data,
                        "evidence_ids": numbers,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    self.receipts.append(receipt)
                    with graph_step("legal_review.source_operation", receipt) as step:
                        step.output_value = receipt
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
