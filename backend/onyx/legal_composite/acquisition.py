from __future__ import annotations

import contextvars
import json
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from hashlib import sha256
from threading import Lock
from typing import cast

import jsonschema
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import SourceKind
from onyx.legal_composite.models import ResearchPlan, SourceAction, WorkflowPolicy
from onyx.tracing.answer_graph import graph_step

AcquisitionCall = tuple[SourceAction, CapabilityCall, str]


class TaskDurationStats(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int = Field(strict=True, ge=1)
    total_seconds: float = Field(ge=0, allow_inf_nan=False)
    max_seconds: float = Field(ge=0, allow_inf_nan=False)


class AcquisitionTimingSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    completed: dict[str, TaskDurationStats]
    diagnostics: dict[str, dict[str, TaskDurationStats]]


_TIMED_CANONICAL_TOOLS = frozenset(
    {
        "search_corpus",
        "resolve_source",
        "read_source_range",
        "read_chunk",
        "read_chunk_context",
        "read_provision",
        "search_source_text",
        "query_corpus",
        "follow_reference",
        "diagnose_source",
        "compare_versions",
        "read_named_provision",
        "dependency_related_sources",
        "read_evidence",
    }
)

_SINGLE_PROVISION_SELECTOR = re.compile(
    r"(?:(?:ek|geçici|gecici|mükerrer|mukerrer)\s+)?"
    r"(?:(?:madde|md\.?|m\.|article|art\.?)\s*:?\s*)?"
    r"[0-9]+[a-zçğıöşü]?"
    r"(?:\s*/\s*(?:[0-9]+(?:\s*/\s*[a-zçğıöşü])?|[a-zçğıöşü]))?",
    re.IGNORECASE,
)


class InvalidSourceAction(RunStopped):
    pass


class CanonicalEvidenceStage:
    """Retain verified originals while the owning source action is still active."""

    def __init__(
        self, ledger: EvidenceLedger, context: RunContext, need_ids: list[str]
    ) -> None:
        self.ledger = ledger
        self.context = context
        self.need_ids = tuple(need_ids)
        self._lock = Lock()
        self._active = True
        self._citations: list[int] = []

    def retain(
        self, items: list[EvidenceItem], existing_citations: list[int] | None = None
    ) -> list[int]:
        with self._lock:
            if not self._active:
                raise RunStopped("Canonical acquisition is closed")
            self.context.check_research_active()
            originals = list(items)
            for number in existing_citations or []:
                item = self.ledger.get(number)
                if item is not None:
                    originals.append(item)
            bound = [
                item.model_copy(
                    deep=True,
                    update={
                        "question_ids": list(
                            dict.fromkeys([*item.question_ids, *self.need_ids])
                        )
                    },
                )
                for item in originals
            ]
            numbers: list[int] = []
            for item in bound:
                self.context.check_research_active()
                recorded = self.ledger.add([item], self.context)
                numbers.extend(recorded)
                self._citations = list(dict.fromkeys([*self._citations, *recorded]))
            return numbers

    def citations(self) -> list[int]:
        with self._lock:
            return list(self._citations)

    def close(self) -> None:
        with self._lock:
            self._active = False
            self.context.cancel()


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
        coalesce_progress: bool = False,
        host_registry: CapabilityRegistry | None = None,
        source_kinds: dict[str, SourceKind] | None = None,
        prioritize_observed_reads: bool = False,
        capture_task_timings: bool = False,
    ) -> None:
        if type(capture_task_timings) is not bool:
            raise ValueError("capture_task_timings must be a boolean")
        self.registry = registry
        self.host_registry = host_registry or registry
        self.source_kinds = dict(source_kinds or {})
        self.context = context
        self.ledger = ledger
        self.policy = policy
        self.registry_for_action = registry_for_action or (lambda _action: registry)
        self.expand_actions = expand_actions or (lambda actions, _plan: actions)
        self.lane_inventory = lane_inventory or {}
        self.on_batch_progress = on_batch_progress or (
            lambda _actions, _pending, _completed: None
        )
        self.coalesce_progress = coalesce_progress
        self.prioritize_observed_reads = prioritize_observed_reads
        self.capture_task_timings = capture_task_timings
        self._timing_lock = Lock()
        self._completed_timings: dict[str, TaskDurationStats] = {}
        self._diagnostic_timings: dict[str, dict[str, TaskDurationStats]] = {}
        self.search_calls = 0
        self.last_receipts: list[dict[str, JsonValue]] = []
        self._completed: dict[str, dict[str, JsonValue]] = {}

    def definitions(self) -> list[dict[str, JsonValue]]:
        return self.registry.definitions(self.context)

    def task_timing_snapshot(self) -> AcquisitionTimingSnapshot:
        """Snapshot settled worker durations; executor queue time is excluded."""
        with self._timing_lock:
            return AcquisitionTimingSnapshot(
                completed=dict(self._completed_timings),
                diagnostics={
                    tool: dict(statuses)
                    for tool, statuses in self._diagnostic_timings.items()
                },
            )

    def _record_task_timing(
        self,
        tool: str,
        started: float,
        outcome: ToolOutcome | None,
        stage: CanonicalEvidenceStage,
        failure_status: str,
    ) -> str | None:
        # Observation must not replace the dispatch or canonical-stage exception.
        try:
            elapsed = time.monotonic() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                return None
            safe_tool = tool if tool in _TIMED_CANONICAL_TOOLS else "other"
            status = (
                ("cancelled" if failure_status == "cancelled" else "error")
                if outcome is None
                else outcome.status.value
            )
            original_count = 0
            completed = False
            if outcome is not None and outcome.status == OutcomeStatus.FOUND:
                numbers = stage.citations()
                original_count = len(numbers)
                completed = bool(numbers) and safe_tool != "other"
                for number in numbers:
                    item = self.ledger.get(number)
                    if item is None or item.search_doc is None or not item.chunk_id:
                        completed = False
                        break
                    canonical = item.metadata.get("canonical_metadata")
                    layers = (
                        item.metadata,
                        item.search_doc.metadata,
                        canonical if isinstance(canonical, dict) else {},
                    )
                    if (
                        not item.text.strip()
                        or item.search_doc.document_id != item.source_id
                        or item.search_doc.metadata.get("regulatory_chunk_id")
                        != item.chunk_id
                        or sha256(item.text.encode()).hexdigest() != item.text_hash
                        or any(
                            layer.get(key)
                            for layer in layers
                            for key in ("external", "derived", "untrusted", "truncated")
                        )
                    ):
                        completed = False
                        break
                if not completed:
                    status = "empty" if not numbers else "noncanonical"
            if safe_tool == "other":
                status = "unsupported_tool"
            with self._timing_lock:
                target = (
                    self._completed_timings
                    if completed
                    else self._diagnostic_timings.setdefault(safe_tool, {})
                )
                key = safe_tool if completed else status
                previous = target.get(key)
                target[key] = TaskDurationStats(
                    count=(previous.count if previous else 0) + 1,
                    total_seconds=(previous.total_seconds if previous else 0) + elapsed,
                    max_seconds=max(previous.max_seconds if previous else 0, elapsed),
                )
            return (
                f"tool={safe_tool} status={status} elapsed_seconds={elapsed:.6f} "
                f"original_count={original_count} canonical_complete={int(completed)}"
            )
        except Exception:
            return None

    def pending_call_counts(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> dict[str, int]:
        """Count expanded pending calls without executing or rebinding originals."""
        from onyx.asv3.legal_source_reviews import related_source_reviews_enabled

        known_needs = {need.need_id for need in plan.needs}
        pending: dict[str, SourceAction] = {}
        metadata_keys = {
            "_public_update",
            "_need_id",
            "_language",
            "_notifications",
            "_external_requested",
            "_outcomes",
            "_coverage",
            *(
                {"_related_source_reviews"}
                if related_source_reviews_enabled(self.context)
                else set()
            ),
        }
        for action in self.expand_actions(actions, plan):
            if set(action.need_ids) - known_needs:
                raise InvalidSourceAction(
                    "Source action refers to an unknown frozen need"
                )
            spec = self.registry_for_action(action).get(action.tool)
            if spec is None or spec.external or spec.orchestrates:
                raise InvalidSourceAction(
                    "Source action is outside the canonical capability allowlist"
                )
            arguments = dict(action.arguments)
            if action.tool == "search_corpus":
                arguments["expand_query"] = False
            if not jsonschema.Draft202012Validator(spec.parameters).is_valid(
                {
                    key: value
                    for key, value in arguments.items()
                    if key not in metadata_keys
                }
            ):
                raise InvalidSourceAction(
                    "Source action arguments do not match the canonical capability schema"
                )
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
                continue
            previous = pending.get(signature)
            pending[signature] = (
                previous.model_copy(
                    update={
                        "need_ids": list(
                            dict.fromkeys(previous.need_ids + action.need_ids)
                        )
                    }
                )
                if previous is not None
                else action
            )
        counts: dict[str, int] = {}
        for action in pending.values():
            counts[action.tool] = counts.get(action.tool, 0) + 1
        return counts

    def safe_observed_read_actions(
        self,
        actions: list[SourceAction],
        plan: ResearchPlan,
        delivered: set[int],
    ) -> tuple[list[SourceAction], list[SourceAction]]:
        """Partition exact read navigation from delivered originals without acquiring."""
        self.pending_call_counts(actions, plan)

        targets: set[str] = set()
        for action in actions:
            source_id = action.arguments.get("source_id")
            if action.tool in {
                "read_provision",
                "read_chunk",
                "read_chunk_context",
            } and isinstance(source_id, str):
                targets.add(source_id)
        observed: dict[str, set[str]] = {}
        for number in delivered:
            if type(number) is not int:
                continue
            item = self.ledger.get(number)
            if (
                item is None
                or item.source_id not in targets
                or item.search_doc is None
                or not item.chunk_id
            ):
                continue
            canonical = item.metadata.get("canonical_metadata")
            layers = (
                item.metadata,
                item.search_doc.metadata,
                canonical if isinstance(canonical, dict) else {},
            )
            if (
                item.search_doc.document_id != item.source_id
                or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
                or sha256(item.text.encode()).hexdigest() != item.text_hash
                or any(
                    layer.get(flag)
                    for layer in layers
                    for flag in ("external", "derived", "untrusted", "truncated")
                )
            ):
                continue
            observed.setdefault(item.source_id, set()).add(item.chunk_id)
        selected: list[SourceAction] = []
        remaining: list[SourceAction] = []
        for action in actions:
            source_id = action.arguments.get("source_id")
            safe = False
            if isinstance(source_id, str) and source_id in observed:
                if action.tool == "read_provision":
                    article = action.arguments.get("article")
                    # The proposed provision is navigation; its returned body still needs proof.
                    safe = (
                        isinstance(article, str)
                        and _SINGLE_PROVISION_SELECTOR.fullmatch(article.strip())
                        is not None
                    )
                elif action.tool in {"read_chunk", "read_chunk_context"}:
                    chunk_id = action.arguments.get("chunk_id")
                    safe = isinstance(chunk_id, str) and chunk_id in observed[source_id]
            (selected if safe else remaining).append(action)
        return selected, remaining

    def _scheduled_calls(
        self, calls: list[AcquisitionCall], *, host_actions: bool = False
    ) -> list[AcquisitionCall]:
        if not self.prioritize_observed_reads:
            return calls
        targets: set[str] = set()
        for action, _call, _signature in calls:
            source_id = action.arguments.get("source_id")
            if action.tool in {
                "read_provision",
                "read_chunk",
                "read_chunk_context",
            } and isinstance(source_id, str):
                targets.add(source_id)
        if not targets:
            return calls
        from onyx.asv3.corpus_tools import article_references
        from onyx.asv3.legal_source_reviews import related_source_reviews_enabled

        metadata_keys = {
            "_public_update",
            "_need_id",
            "_language",
            "_notifications",
            "_external_requested",
            "_outcomes",
            "_coverage",
            *(
                {"_related_source_reviews"}
                if related_source_reviews_enabled(self.context)
                else set()
            ),
        }

        observed: dict[str, set[str]] = {}
        for row in self.ledger.provision_metadata():
            row_source = row["source_id"]
            if (
                not isinstance(row_source, str)
                or row_source not in targets
                or row["citable"] is not True
            ):
                continue
            number = row["citation"]
            assert isinstance(number, int)
            item = self.ledger.get(number)
            if item is None or item.search_doc is None or not item.chunk_id:
                continue
            canonical = item.metadata.get("canonical_metadata")
            layers = (
                item.metadata,
                item.search_doc.metadata,
                canonical if isinstance(canonical, dict) else {},
            )
            if (
                item.search_doc.document_id != item.source_id
                or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
                or any(
                    layer.get(key) is True
                    for layer in layers
                    for key in ("external", "derived", "untrusted", "truncated")
                )
            ):
                continue
            observed.setdefault(item.source_id, set()).add(item.chunk_id)

        def priority(entry: AcquisitionCall) -> int:
            action, call, _signature = entry
            source_id = action.arguments.get("source_id")
            if not isinstance(source_id, str) or source_id not in observed:
                return 1
            registry = (
                self.host_registry
                if host_actions and action.source_kind is None
                else self.registry_for_action(action)
            )
            spec = registry.get(action.tool)
            if spec is None or not jsonschema.Draft202012Validator(
                spec.parameters
            ).is_valid(
                {
                    key: value
                    for key, value in call.arguments.items()
                    if key not in metadata_keys
                }
            ):
                return 1
            if action.tool == "read_provision":
                article = action.arguments.get("article")
                if isinstance(article, str) and len(article_references(article)) == 1:
                    return 0
            if action.tool in {"read_chunk", "read_chunk_context"}:
                chunk_id = action.arguments.get("chunk_id")
                if isinstance(chunk_id, str) and chunk_id in observed[source_id]:
                    return 0
            return 1

        # Queue priority does not authorize a read or change any search's identity.
        return sorted(calls, key=priority)

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
        started: float | None = None
        if self.capture_task_timings:
            try:
                started = time.monotonic()
            except Exception:
                pass
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
            stage = child.services.get("legal_composite_original_stage")
            assert isinstance(stage, CanonicalEvidenceStage)
            outcome: ToolOutcome | None = None
            completed = False
            failure_status = "error"
            try:
                try:
                    outcome = registry.dispatch(call, child)
                    stage.retain(
                        outcome.evidence,
                        [read.citation for read in outcome.original_reads],
                    )
                finally:
                    stage.close()
                completed = True
            except RunStopped:
                failure_status = "cancelled"
                raise
            finally:
                if started is not None:
                    step.summary = self._record_task_timing(
                        call.name,
                        started,
                        outcome if completed else None,
                        stage,
                        failure_status,
                    )
            assert outcome is not None
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
        calls: list[AcquisitionCall] = []
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
        scheduled_calls = self._scheduled_calls(calls, host_actions=host_actions)
        executor = ThreadPoolExecutor(
            max_workers=self.policy.max_parallel_tools,
            thread_name_prefix="legal-composite-source",
        )
        pending: dict[
            Future[ToolOutcome], tuple[SourceAction, CapabilityCall, str, RunContext]
        ] = {}
        batch_actions = [action for action, _call, _signature in calls]
        completed_count = 0
        last_progress = time.monotonic()
        stages: list[CanonicalEvidenceStage] = []
        try:
            for action, call, signature in scheduled_calls:
                child = self.context.child()
                child.depth = self.context.depth
                stage = CanonicalEvidenceStage(self.ledger, child, action.need_ids)
                child.services["legal_composite_original_stage"] = stage
                stages.append(stage)
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
                completed = {future for future in pending if future.done()}
                if not completed:
                    self.context.check_research_active()
                    completed, _ = wait(
                        pending, timeout=0.05, return_when=FIRST_COMPLETED
                    )
                if not completed and time.monotonic() - last_progress >= 5:
                    self.on_batch_progress(batch_actions, len(pending), completed_count)
                    last_progress = time.monotonic()
                stopped: RunStopped | None = None
                for future in completed:
                    action, call, signature, child = pending.pop(future)
                    stage = child.services.get("legal_composite_original_stage")
                    assert isinstance(stage, CanonicalEvidenceStage)
                    try:
                        outcome = future.result()
                    except RunStopped as error:
                        result.append(
                            {
                                "tool": call.name,
                                "source_kind": action.source_kind,
                                "need_ids": list(action.need_ids),
                                "status": "truncated",
                                "summary": "Acquisition stopped before a complete result; retained originals are partial evidence, not proof of corpus absence.",
                                "citations": stage.citations(),
                                **(
                                    {"host_arguments": call.arguments}
                                    if host_actions
                                    else {}
                                ),
                            }
                        )
                        stopped = error
                        continue
                    numbers = stage.citations()
                    row: dict[str, JsonValue] = {
                        "tool": call.name,
                        "source_kind": action.source_kind,
                        "original_source_kinds": {
                            item.source_id: self.source_kinds.get(
                                item.source_id, SourceKind.UNKNOWN
                            ).value
                            for number in numbers
                            if (item := self.ledger.get(number)) is not None
                        },
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
                    if not self.coalesce_progress:
                        self.on_batch_progress(
                            batch_actions, len(pending), completed_count
                        )
                        last_progress = time.monotonic()
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
                if completed and self.coalesce_progress:
                    self.on_batch_progress(batch_actions, len(pending), completed_count)
                    last_progress = time.monotonic()
                if stopped is not None:
                    raise stopped
            return result
        finally:
            # SDK/DB operations may finish later; they cannot commit late evidence to the ledger.
            for stage in stages:
                stage.close()
            for future, (action, call, _, child) in pending.items():
                child.cancel()
                future.cancel()
                stage = child.services.get("legal_composite_original_stage")
                assert isinstance(stage, CanonicalEvidenceStage)
                result.append(
                    {
                        "tool": call.name,
                        "source_kind": action.source_kind,
                        "need_ids": list(action.need_ids),
                        "status": "truncated",
                        "summary": "Acquisition stopped before a complete result; retained originals are partial evidence, not proof of corpus absence.",
                        "citations": stage.citations(),
                        **({"host_arguments": call.arguments} if host_actions else {}),
                    }
                )
            executor.shutdown(wait=False, cancel_futures=True)
