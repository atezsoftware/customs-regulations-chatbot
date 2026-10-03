from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import StrEnum
from typing import Callable, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from onyx.context.search.models import SearchDoc
from onyx.llm.models import AssistantMessage, ToolMessage


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


_EVIDENCE_FIELDS = frozenset(
    "source_id file_id document_id regulatory_chunk_id canonical_chunk_id chunk_id "
    "canonical_identity original_document_id source_sha256 text_hash payload_sha256 "
    "publication_id publication_revision publication_revision_id revision revision_id "
    "publication_version index_name index_uuid query_index_name query_index_uuid "
    "version version_unknown status validity_start validity_end effective_start "
    "effective_end read_as_of_date as_of_date regulatory_validity_start_date "
    "regulatory_validity_end_date position semantic_position projection_ordinal "
    "heading_path regulatory_heading_path article_no article_title paragraph_no "
    "clause_label chunk_type derived_role canonical_role extraction extraction_method "
    "derived external untrusted legal_authority external_tool_name external_tool_id "
    "truncated retrieval_method section_context additional_context "
    "article_closure_complete article_closure_continuation article_closure_remaining_count "
    "source_outline_truncated follow_context_tool retrieved_projection_ordinal "
    "retrieved_center asv3_native_locator asv3_citation_preview_url semantic_key "
    "source_regulatory_chunk_ids replaced_regulatory_chunk_ids regulation_id "
    "regulation_name regulation_number regulation_type title name source_name "
    "target_id target_article_no target_paragraph_no target_clause_label "
    "regulatory_document_id source_canonical_document_id canonical_document_id "
    "canonical_source_id supersedes_id superseded_by_id representation_id "
    "representation_sha256 binding_id tenant_id document_set_id document_sets "
    "document_type document_date legal_dates chunk_variant semantic_identifier "
    "source_document_id source_publication_id publication_payload_sha256 "
    "projection_payload_sha256 binding_sha256 effective_date supersession_id "
    "provision_identifiers decision_numbers validity_start_date validity_end_date "
    "context_projection_id chunk_index source_type search_settings_id committed_epoch "
    "original_heading_present original_heading_path ordinal asv3_context_kind "
    "asv3_selected_sibling_count asv3_rerank_excerpt_truncated "
    "asv3_operative_unit_complete asv3_navigation_position_authority".split()
)
_LOCATOR_FIELDS = frozenset(
    "page pages sheet cell row column path original_box normalized_box original_width "
    "original_height coordinate_system start_char end_char start_offset end_offset "
    "start_page end_page line start_line end_line table table_index row_start row_end "
    "column_start column_end start_position end_position".split()
)


def compact_evidence_metadata(metadata: Mapping[str, object]) -> dict[str, JsonValue]:
    """Retain citation/provenance locators; index vectors and raw payloads stay outside research."""

    def selected(value: object) -> JsonValue:
        if isinstance(value, dict):
            return compact_evidence_metadata(
                {name: part for name, part in value.items() if isinstance(name, str)}
            )
        if isinstance(value, (list, tuple)):
            return [
                selected(item)
                for item in value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return None

    result: dict[str, JsonValue] = {}
    for key, value in metadata.items():
        if key in _EVIDENCE_FIELDS:
            result[key] = selected(value)
        elif key == "canonical_metadata" and isinstance(value, dict):
            result[key] = selected(value)
        elif key in {"index", "publication", "provenance", "target"} and isinstance(
            value, dict
        ):
            result[key] = selected(value)
        elif key == "locator" and isinstance(value, dict):
            result[key] = {
                name: selected(part)
                for name, part in value.items()
                if isinstance(name, str) and name in _LOCATOR_FIELDS
            }
        elif key == "source_links" and isinstance(value, dict):
            result[key] = {
                name: link
                for name, link in value.items()
                if isinstance(name, str) and name == "0" and isinstance(link, str)
            }
        elif key == "source_links" and isinstance(value, str):
            result[key] = value
    return result


def model_evidence_metadata(metadata: Mapping[str, object]) -> dict[str, JsonValue]:
    """Model locators are a projection; complete compact provenance stays in the ledger."""
    compact = compact_evidence_metadata(metadata)
    canonical = compact.get("canonical_metadata")
    merged = {**(canonical if isinstance(canonical, dict) else {}), **compact}
    for key in ("publication", "index", "provenance"):
        nested = merged.get(key)
        if isinstance(nested, dict):
            merged = {**nested, **merged}
    keys = frozenset(
        "document_type title article_no paragraph_no clause_label read_as_of_date version_unknown derived "
        "external untrusted legal_authority source_sha256 locator extraction_method "
        "extraction publication_revision publication_revision_id revision_id index_uuid "
        "query_index_uuid index_name query_index_name version revision document_date "
        "legal_dates decision_numbers provision_identifiers target_article_no "
        "target_paragraph_no target_clause_label publication_payload_sha256 payload_sha256 "
        "article_closure_complete article_closure_remaining_count asv3_context_kind "
        "asv3_selected_sibling_count asv3_rerank_excerpt_truncated "
        "asv3_operative_unit_complete".split()
    )
    result = {key: value for key, value in merged.items() if key in keys}
    for canonical_key, aliases in {
        "heading_path": (
            "heading_path",
            "regulatory_heading_path",
            "original_heading_path",
        ),
        "validity_start": (
            "validity_start",
            "regulatory_validity_start_date",
            "validity_start_date",
            "effective_start",
        ),
        "validity_end": (
            "validity_end",
            "regulatory_validity_end_date",
            "validity_end_date",
            "effective_end",
        ),
    }.items():
        for alias in aliases:
            if alias in merged and merged[alias] is not None:
                result[canonical_key] = merged[alias]
                break
    return result


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
        self.metadata = compact_evidence_metadata(self.metadata)
        if self.search_doc is not None:
            citation_metadata = compact_evidence_metadata(
                dict(self.search_doc.metadata)
            )
            self.search_doc = self.search_doc.model_copy(
                update={"metadata": citation_metadata}
            )
        return self

    @property
    def identity(self) -> tuple[str, str | None, str]:
        return self.source_id, self.chunk_id, self.text_hash


class OriginalEvidenceRead(BaseModel):
    """An exact reopened range of an original already in this run's ledger."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    citation: int = Field(strict=True, ge=1)
    text_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_char: int = Field(strict=True, ge=0)
    end_char: int = Field(strict=True, ge=1)

    @model_validator(mode="after")
    def ordered_range(self) -> OriginalEvidenceRead:
        if self.start_char >= self.end_char:
            raise ValueError("Original evidence range must be nonempty")
        return self


class ToolOutcome(BaseModel):
    status: OutcomeStatus
    summary: str
    data: dict[str, JsonValue] = Field(default_factory=dict)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    original_reads: list[OriginalEvidenceRead] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)


class CapabilityCall(BaseModel):
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    call_id: str = Field(default_factory=lambda: str(uuid4()))
    argument_error: str | None = Field(default=None, max_length=350)
    invalid_arguments_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class Decision(BaseModel):
    calls: list[CapabilityCall] = Field(default_factory=list)
    answer: str | None = None
    questions: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)
    assistant_message: AssistantMessage | None = None


class ResearchTurn(BaseModel):
    assistant: AssistantMessage
    results: list[ToolMessage]

    @model_validator(mode="after")
    def match_results(self) -> ResearchTurn:
        calls = self.assistant.tool_calls or []
        if [call.id for call in calls] != [item.tool_call_id for item in self.results]:
            raise ValueError("Research turn tool results do not match assistant calls")
        return self


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    name: str
    description: str
    parameters: dict[str, JsonValue]
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome]
    parallel_safe: bool = True
    consumes_tool_budget: bool = True
    external: bool = False
    orchestrates: bool = False
    requires_research_need: bool = False
    research_need_argument: Literal["_need_id", "need_ids"] = "_need_id"

    def definition(self) -> dict[str, JsonValue]:
        import copy

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": copy.deepcopy(self.parameters),
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
    need_ids: list[str] = Field(default_factory=list)
    public_title: str | None = None
    public_message: str | None = None
    updates: list[str] = Field(default_factory=list)
    outcome: ToolOutcome | None = None


class HarnessResult(BaseModel):
    answer: str | None
    status: OutcomeStatus
    receipts: list[ToolReceipt]
    questions: list[str]
    facts: list[str]
    stop_reason: str | None = None
    publication_gap: ToolOutcome | None = None


class HarnessView(BaseModel):
    request: str
    questions: list[str]
    facts: list[str]
    receipts: list[ToolReceipt]
    evidence: list[dict[str, JsonValue]]
    tools: list[dict[str, JsonValue]]
    turns: list[ResearchTurn] = Field(default_factory=list)
    draft_to_repair: str | None = None
    publication_gap: dict[str, JsonValue] | None = None
    research_state: dict[str, JsonValue] = Field(default_factory=dict)
    original_evidence: list[dict[str, JsonValue]] = Field(default_factory=list)
    original_evidence_omitted: list[JsonValue] = Field(default_factory=list)
    required_evidence_numbers: list[int] = Field(default_factory=list)


class RunStopped(RuntimeError):
    pass


class SharedBudget:
    def __init__(
        self,
        max_tools: int = 64,
        max_decisions: int = 41,
        max_evidence_bytes: int = 2_000_000,
        max_inflight_tools: int = 4,
        max_inflight_models: int = 4,
        max_inflight_sources: int = 2,
        max_artifact_bytes: int = 4_000_000,
        final_decision_reserve: int = 12,
        coordinator_decision_reserve: int = 6,
    ) -> None:
        import threading

        self._lock = threading.Lock()
        self.tool_slots = threading.BoundedSemaphore(max_inflight_tools)
        self.model_slots = threading.BoundedSemaphore(max_inflight_models)
        self.source_slots = threading.BoundedSemaphore(max_inflight_sources)
        self.final_decision_reserve = min(final_decision_reserve, max_decisions)
        # Writer plus assertion, source inventory and condition checks, with format repair.
        self.publication_decision_reserve = min(8, self.final_decision_reserve)
        self.coordinator_decision_reserve = min(
            coordinator_decision_reserve,
            max(0, max_decisions - self.final_decision_reserve - 1),
        )
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

    def consume_research_decision(self, *, researcher: bool = False) -> None:
        with self._lock:
            research_limit = self.limits["decisions"] - self.final_decision_reserve
            if researcher:
                research_limit -= self.coordinator_decision_reserve
            if self.used["decisions"] >= research_limit:
                raise RunStopped(
                    "Research decision budget exhausted; finalization reserve retained"
                )
            self.used["decisions"] += 1

    def consume_repair_decision(self) -> None:
        """Targeted recovery can use the reserve while keeping publication capacity."""
        with self._lock:
            if self.used["decisions"] >= (
                self.limits["decisions"] - self.publication_decision_reserve
            ):
                raise RunStopped(
                    "Targeted repair capacity used; publication reserve retained"
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

    def consume_research_decision(self) -> None:
        if self.services.get("final_repair") is True and not self.depth:
            self.budget.consume_repair_decision()
        else:
            self.budget.consume_research_decision(researcher=self.depth > 0)

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
