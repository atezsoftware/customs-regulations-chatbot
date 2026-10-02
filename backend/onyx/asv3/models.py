from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.context.search.models import SearchDoc


class OutcomeStatus(StrEnum):
    FOUND = "found"
    PARTIAL = "partial"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    DENIED = "denied"
    TRUNCATED = "truncated"
    VERSION_UNKNOWN = "version_unknown"
    CANCELLED = "cancelled"
    INVALID = "invalid"
    ERROR = "error"


class Artifact(BaseModel):
    artifact_id: str
    name: str
    media_type: str = "application/json"
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    source_id: str
    text: str
    chunk_id: str | None = None
    search_doc: SearchDoc | None = None
    question_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    text_hash: str = ""

    @model_validator(mode="after")
    def verify_hash(self) -> EvidenceItem:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_hash and self.text_hash != digest:
            raise ValueError("Evidence text does not match its hash")
        self.text_hash = digest
        return self

    @property
    def identity(self) -> tuple[str, str | None, str]:
        return self.source_id, self.chunk_id, self.text_hash


class ToolOutcome(BaseModel):
    status: OutcomeStatus
    summary: str
    data: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class CapabilityCall(BaseModel):
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: str(uuid4()))


class Decision(BaseModel):
    calls: list[CapabilityCall] = Field(default_factory=list)
    answer: str | None = None
    questions: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    description: str
    parameters: dict[str, JsonValue]
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome]
    parallel_safe: bool = True
    external: bool = False
    orchestrates: bool = False

    def definition(self) -> dict[str, JsonValue]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolReceipt(BaseModel):
    call: CapabilityCall
    outcome: ToolOutcome
    elapsed_seconds: float
    evidence_ids: list[int] = Field(default_factory=list)


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class TaskSnapshot(BaseModel):
    task_id: str
    task: str
    status: TaskStatus
    parent_task_id: str | None = None
    updates: list[str] = Field(default_factory=list)
    outcome: ToolOutcome | None = None


class HarnessResult(BaseModel):
    answer: str | None
    status: OutcomeStatus
    receipts: list[ToolReceipt]
    questions: list[str]
    facts: list[str]


class HarnessView(BaseModel):
    request: str
    questions: list[str]
    facts: list[str]
    receipts: list[ToolReceipt]
    evidence: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]


class RunStopped(RuntimeError):
    pass


class SharedBudget:
    def __init__(
        self,
        max_tools: int = 64,
        max_decisions: int = 32,
        max_evidence_bytes: int = 2_000_000,
        max_inflight_tools: int = 4,
        max_inflight_models: int = 4,
        max_artifact_bytes: int = 4_000_000,
        final_decision_reserve: int = 3,
    ) -> None:
        import threading

        self._lock = threading.Lock()
        self.tool_slots = threading.BoundedSemaphore(max_inflight_tools)
        self.model_slots = threading.BoundedSemaphore(max_inflight_models)
        self.final_decision_reserve = min(final_decision_reserve, max_decisions)
        self.limits = {
            "tools": max_tools,
            "decisions": max_decisions,
            "evidence_bytes": max_evidence_bytes,
            "artifact_bytes": max_artifact_bytes,
        }
        self.used: dict[str, int] = dict.fromkeys(self.limits, 0)

    def consume(self, kind: str, amount: int = 1) -> None:
        with self._lock:
            if amount < 0 or self.used[kind] + amount > self.limits[kind]:
                raise RunStopped(f"Shared {kind} budget exhausted")
            self.used[kind] += amount

    def release(self, kind: str, amount: int) -> None:
        with self._lock:
            if amount < 0 or amount > self.used[kind]:
                raise ValueError("Invalid resource budget release")
            self.used[kind] -= amount

    def consume_research_decision(self) -> None:
        with self._lock:
            if (
                self.used["decisions"]
                >= self.limits["decisions"] - self.final_decision_reserve
            ):
                raise RunStopped(
                    "Research decision budget exhausted; finalization reserve retained"
                )
            self.used["decisions"] += 1

    def restore(self, values: dict[str, JsonValue]) -> None:
        with self._lock:
            for kind, value in values.items():
                if (
                    kind not in self.limits
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or not 0 <= value <= self.limits[kind]
                ):
                    raise ValueError("Invalid checkpoint budget")
            self.used.update(
                {
                    kind: int(value)
                    for kind, value in values.items()
                    if isinstance(value, int)
                }
            )

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.used)


class RunContext:
    def __init__(
        self,
        *,
        run_id: str | None = None,
        language: str = "tr",
        scope: dict[str, JsonValue] | None = None,
        services: dict[str, object] | None = None,
        budget: SharedBudget | None = None,
        timeout_seconds: float = 180,
        cancelled: Callable[[], bool] | None = None,
        depth: int = 0,
        max_depth: int = 2,
        corpus_only: bool = True,
        deadline: float | None = None,
        research_deadline: float | None = None,
        research_reserve_seconds: float = 0,
    ) -> None:
        import copy
        import threading
        import time

        self.run_id = run_id or str(uuid4())
        self.language = language
        self.scope = copy.deepcopy(scope or {})
        self.services = dict(services or {})
        self.budget = budget or SharedBudget()
        self.deadline = (
            deadline if deadline is not None else time.monotonic() + timeout_seconds
        )
        self.research_deadline = (
            research_deadline
            if research_deadline is not None
            else self.deadline - research_reserve_seconds
        )
        self._cancelled = cancelled or (lambda: False)
        self._stop = threading.Event()
        self.depth = depth
        self.max_depth = max_depth
        self.corpus_only = corpus_only

    def check_active(self) -> None:
        import time

        if self._stop.is_set() or self._cancelled():
            raise RunStopped("Research cancelled")
        if time.monotonic() >= self.deadline:
            raise RunStopped("Research deadline exceeded")

    def check_research_active(self) -> None:
        import time

        self.check_active()
        if time.monotonic() >= self.research_deadline:
            raise RunStopped("Research deadline exceeded; finalization time retained")

    def cancel(self) -> None:
        self._stop.set()

    def child(self) -> RunContext:
        return RunContext(
            run_id=self.run_id,
            language=self.language,
            scope=self.scope,
            services=self.services,
            budget=self.budget,
            deadline=self.deadline,
            research_deadline=self.research_deadline,
            cancelled=self.is_cancelled,
            depth=self.depth + 1,
            max_depth=self.max_depth,
            corpus_only=self.corpus_only,
        )

    def is_cancelled(self) -> bool:
        return self._stop.is_set() or self._cancelled()


ToolSpec.model_rebuild()
