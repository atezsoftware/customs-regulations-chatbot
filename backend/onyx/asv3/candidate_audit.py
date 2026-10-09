"""Bounded provenance for ASv3 Guardrails v3 retrieval candidates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.models import RunContext

CandidateAuditStatus = Literal[
    "selected",
    "excluded",
    "unmapped",
    "hydration_failed",
    "delivered",
]


class CandidateAuditRecord(BaseModel):
    """Metadata-only lifecycle record; passage text remains in the evidence ledger."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    search_run_id: str = Field(min_length=1, max_length=128)
    candidate_id: str = Field(min_length=1, max_length=512)
    source_id: str = Field(min_length=1, max_length=512)
    chunk_id: str = Field(min_length=1, max_length=512)
    lane: str = Field(min_length=1, max_length=64)
    mode: str = Field(min_length=1, max_length=64)
    status: CandidateAuditStatus
    reason: str = Field(min_length=1, max_length=128)
    raw_score: float | None = None
    normalized_score: float | None = None
    rerank_position: int | None = Field(default=None, ge=0)
    outcome_ids: list[str] = Field(default_factory=list, max_length=32)
    scope_version: str | None = Field(default=None, max_length=256)

    @property
    def identity(self) -> tuple[str, str]:
        return self.search_run_id, self.candidate_id


class _CandidateAuditCheckpoint(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    version: Literal[1] = 1
    run_id: str
    scope_hash: str
    request_hash: str
    records: list[CandidateAuditRecord]


class CandidateAudit:
    """Keep a bounded auditable record without becoming a second evidence store."""

    def __init__(
        self,
        context: RunContext,
        *,
        request: str,
        max_records: int = 256,
        max_serialized_chars: int = 96_000,
    ) -> None:
        if max_records < 1 or max_serialized_chars < 1:
            raise ValueError("Candidate audit requires positive capacity limits")
        self._context = context
        self.run_id = context.run_id
        self.scope_hash = _hash(context.scope)
        self.request_hash = hashlib.sha256(request.encode("utf-8")).hexdigest()
        self.max_records = max_records
        self.max_serialized_chars = max_serialized_chars
        self._records: dict[tuple[str, str], CandidateAuditRecord] = {}

    def record(self, record: CandidateAuditRecord) -> None:
        """Replace one candidate lifecycle record without ever recording passage text."""
        self._context.check_active()
        candidate = CandidateAuditRecord.model_validate(record)
        key = candidate.identity
        if key not in self._records and len(self._records) >= self.max_records:
            raise ValueError("Candidate audit record capacity exceeded")
        replacement = {**self._records, key: candidate}
        if self._serialized_chars(replacement.values()) > self.max_serialized_chars:
            raise ValueError("Candidate audit serialized capacity exceeded")
        self._records = replacement

    def record_many(self, records: Iterable[CandidateAuditRecord]) -> None:
        for record in records:
            self.record(record)

    def records(self) -> tuple[CandidateAuditRecord, ...]:
        return tuple(self._records.values())

    def export(self) -> dict[str, JsonValue]:
        records = list(self.records())
        return _CandidateAuditCheckpoint(
            run_id=self.run_id,
            scope_hash=self.scope_hash,
            request_hash=self.request_hash,
            records=records,
        ).model_dump(mode="json")

    def restore(self, data: dict[str, JsonValue]) -> None:
        saved = _CandidateAuditCheckpoint.model_validate(data)
        if saved.run_id != self.run_id:
            raise ValueError("Candidate audit run mismatch")
        if saved.scope_hash != self.scope_hash:
            raise ValueError("Candidate audit scope mismatch")
        if saved.request_hash != self.request_hash:
            raise ValueError("Candidate audit request mismatch")
        if len(saved.records) > self.max_records:
            raise ValueError("Candidate audit record capacity exceeded")
        restored = {record.identity: record for record in saved.records}
        if len(restored) != len(saved.records):
            raise ValueError("Candidate audit has duplicate candidate records")
        if self._serialized_chars(restored.values()) > self.max_serialized_chars:
            raise ValueError("Candidate audit serialized capacity exceeded")
        self._records = restored

    @staticmethod
    def _serialized_chars(records: Iterable[CandidateAuditRecord]) -> int:
        return len(
            json.dumps(
                [record.model_dump(mode="json") for record in records],
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
