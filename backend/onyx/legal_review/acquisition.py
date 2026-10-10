"""Bounded source operations; discovery and canonical reading remain explicit."""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextvars import copy_context
from typing import cast
from uuid import uuid4

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.search_adapter import ScopedSearchAdapter
from onyx.db.asv3_candidate_inventory import asv3_source_inventory_scope
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
        broker: CorpusBroker | None = None,
    ) -> None:
        self.registry = registry
        self.context = context
        self.ledger = ledger
        self.policy = policy
        self.search_adapter = search_adapter
        self.broker = broker
        self._expanded: set[int] = set()
        self.searches = 0
        self.receipts: list[dict[str, JsonValue]] = []

    def definitions(self) -> list[dict[str, JsonValue]]:
        return self.registry.definitions(self.context)

    def complete_articles(self, citations: set[int], finalizing: bool) -> bool:
        """Read structural article siblings for used search hits without another model call."""
        if self.broker is None:
            return False
        items = [
            item
            for citation in sorted(citations - self._expanded)
            if (item := self.ledger.get(citation)) is not None
            and item.search_doc is not None
            and item.metadata.get("article_closure_complete") is False
        ]
        if not items:
            return False
        end = (
            self.context.deadline - self.policy.publication_reserve_seconds - 60
            if finalizing
            else self.context.research_deadline
        )
        phase = self.context.child()
        phase.deadline = min(phase.deadline, end)
        phase.research_deadline = phase.deadline
        phase.check_active()
        before = set(self.ledger.citation_numbers())
        groups: dict[str, list[EvidenceItem]] = {}
        for item in items:
            groups.setdefault(item.source_id, []).append(item)

        def read_group(members: list[EvidenceItem]) -> list[EvidenceItem]:
            assert self.broker is not None
            with asv3_source_inventory_scope(
                scope_key=f"{self.context.run_id}:legal_review_article_context:{members[0].source_id}",
                check_active=phase.check_active,
            ):
                hydrated = self.broker.hydrate_search_results(
                    [
                        item.search_doc
                        for item in members
                        if item.search_doc is not None
                    ],
                    phase,
                )
            return [item for values in hydrated.values() for item in values]

        pool = ThreadPoolExecutor(max_workers=self.policy.max_parallel_tools)
        pending: dict[Future[list[EvidenceItem]], tuple[list[EvidenceItem], float]] = {}
        try:
            for members in groups.values():
                future = cast(
                    Future[list[EvidenceItem]],
                    pool.submit(copy_context().run, read_group, members),
                )
                pending[future] = (
                    members,
                    time.monotonic(),
                )
            while pending:
                phase.check_active()
                done, _ = wait(pending, timeout=0.05, return_when=FIRST_COMPLETED)
                for future in done:
                    members, started = pending.pop(future)
                    originals = future.result()
                    source_citations = {
                        number
                        for number in citations
                        if (item := self.ledger.get(number)) is not None
                        and item.source_id == members[0].source_id
                    }
                    issue_ids = sorted(
                        {identity for item in members for identity in item.question_ids}
                    )
                    for item in originals:
                        item.question_ids = issue_ids
                    numbers = self.ledger.add(originals, phase)
                    self._expanded.update(source_citations)
                    receipt: dict[str, JsonValue] = {
                        "call_id": str(uuid4()),
                        "tool": "read_article_context",
                        "arguments": {
                            "source_id": members[0].source_id,
                            "citations": sorted(source_citations),
                        },
                        "issue_ids": issue_ids,
                        "evidence_ids": numbers,
                        "status": "found" if originals else "not_found",
                        "data": {
                            "closures": [
                                {
                                    "source_id": item.source_id,
                                    "chunk_id": item.chunk_id,
                                    "complete": item.metadata.get(
                                        "article_closure_complete"
                                    ),
                                    "continuation": item.metadata.get(
                                        "article_closure_continuation"
                                    ),
                                }
                                for item in originals
                            ]
                        },
                        "elapsed_seconds": time.monotonic() - started,
                    }
                    self.receipts.append(receipt)
                    with graph_step("legal_review.article_context", receipt) as step:
                        step.output_value = receipt
        finally:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
        return bool(set(self.ledger.citation_numbers()) - before)

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
                self.searches += 1
                arguments.setdefault("coverage_item", ", ".join(action.issue_ids))
                arguments.setdefault("evidence_target", arguments["query"])
                arguments["expand_query"] = False
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
                research_deadline=self.context.deadline
                - self.policy.publication_reserve_seconds
                - 60,
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
                child.deadline = min(child.deadline, child.research_deadline)
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
                        "research_need_ids": list(action.research_need_ids),
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
