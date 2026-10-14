"""One request-bound acquisition attempt for each stable Cropped source defect."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    computed_field,
    model_validator,
)

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import folded
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext

_MAX_RECORDS = 512
RepairStatus = Literal["found", "not_found", "incomplete", "failed"]
_RecordStatus = Literal["in_progress", "found", "not_found", "incomplete", "failed"]
_EvidenceNumber = Annotated[int, Field(strict=True, ge=1)]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _string(value: JsonValue) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _article(value: JsonValue) -> str | None:
    text = _string(value)
    if text is None:
        return None
    text = re.sub(r"\s+", "", text).upper()
    return re.sub(r"^0+(?=\d)", "", text)


class CroppedRepairTarget(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    check_kind: str = Field(min_length=1, max_length=256)
    instrument_number: str | None = None
    formal_name: str | None = None
    article: str | None = None
    qualifier: str | None = None
    source_id: str | None = None
    anchor_source_id: str | None = None
    chunk_id: str | None = None
    text_hash: str | None = None
    requirement_id: str | None = None
    source_name: str | None = None
    unit_ids: list[str] = Field(default_factory=list)
    evidence_numbers: list[_EvidenceNumber] = Field(default_factory=list)
    reference_text: str = ""
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @computed_field
    @property
    def key(self) -> str:
        # Citation numbers, delivery hashes and answer wording are observations,
        # not new permission to retry the same request-bound source acquisition.
        return "cropped_repair_" + _digest(
            {
                "check_kind": self.check_kind,
                "instrument_number": self.instrument_number,
                "formal_name": (folded(self.formal_name) if self.formal_name else None)
                if self.instrument_number is None
                else None,
                "article": _article(self.article),
                "qualifier": folded(self.qualifier) if self.qualifier else None,
                "source_id": (
                    None
                    if self.check_kind == "governing_original"
                    and (self.instrument_number or self.formal_name)
                    else self.source_id
                ),
                "anchor_source_id": None
                if self.check_kind == "governing_original"
                else self.anchor_source_id,
                "chunk_id": self.chunk_id,
                "requirement_id": self.requirement_id
                if self.check_kind.startswith("generic:")
                or not (self.instrument_number or self.formal_name or self.source_id)
                else None,
            }
        )

    def checkpoint_value(self) -> dict[str, JsonValue]:
        return self.model_dump(mode="json", exclude={"key"})


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: CroppedRepairTarget
    status: _RecordStatus
    evidence_numbers: list[_EvidenceNumber] = Field(default_factory=list)
    detail: str = Field(default="", max_length=64000)

    @model_validator(mode="after")
    def consistent_result(self) -> _Record:
        if self.status == "in_progress" and (self.evidence_numbers or self.detail):
            raise ValueError("An in-progress repair cannot contain a completed result")
        if self.status == "found" and not self.evidence_numbers:
            raise ValueError("A found repair needs recorded original evidence")
        if (
            self.status in {"not_found", "incomplete", "failed"}
            and not self.detail.strip()
        ):
            raise ValueError(
                "An unavailable repair needs its precise attempt limitation"
            )
        if len(self.evidence_numbers) != len(set(self.evidence_numbers)):
            raise ValueError("Repair evidence numbers must be unique")
        return self

    def checkpoint_value(self) -> dict[str, JsonValue]:
        return {
            "target": self.target.checkpoint_value(),
            "status": self.status,
            "evidence_numbers": list(self.evidence_numbers),
            "detail": self.detail,
        }


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    run_id: str
    scope_hash: str
    request_hash: str
    records: list[_Record] = Field(max_length=_MAX_RECORDS)
    record_integrity: dict[str, str]
    integrations: dict[str, list[str]] = Field(default_factory=dict)
    integrations_integrity: str = ""


def _unit_ids(row: dict[str, JsonValue], units: dict[str, str]) -> list[str]:
    values: list[JsonValue] = [row.get("unit_id")]
    for field in ("answer_unit_ids", "unit_ids"):
        value = row.get(field)
        if isinstance(value, list):
            values.extend(value)
    return list(
        dict.fromkeys(
            value for value in values if isinstance(value, str) and value in units
        )
    )


def _target(
    kind: str,
    row: dict[str, JsonValue],
    field: str,
    units: dict[str, str],
    ledger: EvidenceLedger,
) -> CroppedRepairTarget:
    number = _string(row.get("instrument_number"))
    formal = _string(row.get("formal_name"))
    source_name = (
        f"{number} sayılı Kanun"
        if number
        else formal or _string(row.get("source_name")) or _string(row.get("name"))
    )
    ids = _unit_ids(row, units)
    source_id = _string(row.get("source_id"))
    candidates = row.get(
        "matching_original_evidence", row.get("available_original_citations", [])
    )
    numbers = []
    for citation in candidates if isinstance(candidates, list) else []:
        if type(citation) is not int or citation < 1:
            continue
        item = ledger.get(citation)
        if item is not None and (
            kind != "related_source" or source_id is None or item.source_id == source_id
        ):
            numbers.append(citation)
    return CroppedRepairTarget(
        check_kind=kind,
        instrument_number=number,
        formal_name=formal,
        article=_article(row.get("article", row.get("article_no"))),
        qualifier=_string(row.get("qualifier", row.get("article_qualifier"))),
        source_id=source_id,
        anchor_source_id=_string(row.get("anchor_source_id")),
        chunk_id=_string(row.get("chunk_id")),
        requirement_id=_string(
            row.get("requirement_id", row.get("condition_id", row.get("lead_id")))
        ),
        source_name=source_name,
        unit_ids=ids,
        evidence_numbers=list(dict.fromkeys(numbers)),
        reference_text=_string(row.get("reference_text")) or "",
        details={
            "gap_entries": [{"field": field, "entry": copy.deepcopy(row)}],
            "unit_texts": {key: units[key] for key in ids},
        },
    )


def extract_repair_targets(
    gap: dict[str, JsonValue], draft: str, ledger: EvidenceLedger
) -> list[CroppedRepairTarget]:
    """Extract acquisition identities; this neither verifies nor approves a claim."""
    inventory = assertion_inventory(draft)
    units = {unit["unit_id"]: unit["text"] for unit in inventory}
    targets: list[CroppedRepairTarget] = []
    for field, kind in (
        ("named_authority_gaps", "governing_original"),
        ("unread_cited_statute_references", "governing_original"),
        ("unread_related_sources", "related_source"),
    ):
        rows = gap.get(field, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Malformed structured source repair gap")
        for row in rows:
            assert isinstance(row, dict)
            targets.append(_target(kind, row, field, units, ledger))
    numbers = gap.get("undelivered_citations", [])
    if not isinstance(numbers, list) or any(
        type(n) is not int or n < 1 for n in numbers
    ):
        raise ValueError("Malformed undelivered original citations")
    for number in numbers:
        assert type(number) is int
        item = ledger.get(number)
        ids = [
            unit["unit_id"] for unit in inventory if number in unit["evidence_numbers"]
        ]
        targets.append(
            CroppedRepairTarget(
                check_kind="original_delivery",
                source_id=item.source_id if item is not None else None,
                chunk_id=item.chunk_id if item is not None else None,
                text_hash=hashlib.sha256(item.text.encode("utf-8")).hexdigest()
                if item is not None
                else None,
                requirement_id="unknown_original_identity" if item is None else None,
                source_name=(
                    item.search_doc.semantic_identifier
                    if item is not None and item.search_doc is not None
                    else _string(item.metadata.get("filename"))
                    if item is not None
                    else None
                ),
                unit_ids=ids,
                evidence_numbers=[number],
                details={
                    "gap_entries": [
                        {"field": "undelivered_citations", "entry": number}
                    ],
                    "unit_texts": {key: units[key] for key in ids},
                },
            )
        )
    if not targets and gap:
        fields = sorted(
            key
            for key in gap
            if key
            not in {
                "summary",
                "instruction",
                "publication_check_instructions",
                "citation_format",
                "unit_id",
                "unit_ids",
                "answer_unit_ids",
                "requirement_id",
                "condition_id",
                "lead_id",
                "source_id",
                "anchor_source_id",
                "instrument_number",
                "formal_name",
                "article",
                "qualifier",
                "draft",
                "answer",
                "text",
                "reference_text",
                "citations",
                "evidence_numbers",
                "matching_original_evidence",
                "inline_evidence",
                "origin_citations",
            }
        )
        identified_rows: list[tuple[str, dict[str, JsonValue]]] = []
        for field in fields:
            rows = gap[field]
            for row in rows if isinstance(rows, list) else []:
                if isinstance(row, dict) and any(
                    _string(row.get(key))
                    for key in ("requirement_id", "condition_id", "lead_id")
                ):
                    identified_rows.append((field, row))
        for field, row in identified_rows or [("generic", gap)]:
            targets.append(
                _target("generic:" + _digest(fields), row, field, units, ledger)
            )
    merged: dict[str, CroppedRepairTarget] = {}
    for target in targets:
        previous = merged.get(target.key)
        if previous is None:
            merged[target.key] = target
            continue
        previous_entries = previous.details.get("gap_entries", [])
        target_entries = target.details.get("gap_entries", [])
        assert isinstance(previous_entries, list) and isinstance(target_entries, list)
        texts = {**units}
        ids = list(dict.fromkeys([*previous.unit_ids, *target.unit_ids]))
        merged[target.key] = previous.model_copy(
            update={
                "unit_ids": ids,
                "evidence_numbers": list(
                    dict.fromkeys(
                        [*previous.evidence_numbers, *target.evidence_numbers]
                    )
                ),
                "details": {
                    "gap_entries": [*previous_entries, *target_entries],
                    "unit_texts": {key: texts[key] for key in ids},
                },
            },
            deep=True,
        )
    return list(merged.values())


class CroppedRepairState:
    def __init__(self, context: RunContext, request: str) -> None:
        self._context = context
        self.request = request
        self.run_id = context.run_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self._records: dict[str, _Record] = {}
        self._integrations: dict[str, set[str]] = {}
        self._lock = threading.RLock()

    def _fence(self) -> None:
        if (
            self._context.run_id,
            _digest(self._context.scope),
            _digest(self.request),
        ) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
        ):
            raise ValueError("Cropped repair run or authorized scope changed")

    def reserve(self, target: CroppedRepairTarget) -> bool:
        self._fence()
        with self._lock:
            if target.key in self._records:
                return False
            if len(self._records) >= _MAX_RECORDS:
                raise ValueError("Cropped source repair target capacity exceeded")
            self._records[target.key] = _Record(
                target=target.model_copy(deep=True), status="in_progress"
            )
            return True

    def finish(
        self,
        key: str,
        status: RepairStatus,
        evidence_numbers: list[int],
        detail: str,
    ) -> None:
        self._fence()
        if status not in {"found", "not_found", "incomplete", "failed"}:
            raise ValueError("Invalid completed Cropped repair status")
        with self._lock:
            record = self._records.get(key)
            if record is None or record.status != "in_progress":
                raise ValueError("Cropped repair must be reserved and finish only once")
            self._records[key] = _Record(
                target=record.target,
                status=status,
                evidence_numbers=evidence_numbers,
                detail=detail,
            )

    def close_interrupted_attempts(self) -> int:
        """Close restored reservations before any worker resumes; never during execution."""
        self._fence()
        with self._lock:
            closed = 0
            for key, record in list(self._records.items()):
                if record.status != "in_progress":
                    continue
                self._records[key] = _Record(
                    target=record.target,
                    status="failed",
                    detail="Interrupted focused attempt; no verified finding; absence not proven.",
                )
                closed += 1
            return closed

    def records_for(
        self, targets: list[CroppedRepairTarget]
    ) -> list[dict[str, JsonValue]]:
        self._fence()
        with self._lock:
            rows: list[dict[str, JsonValue]] = []
            seen: set[str] = set()
            for target in targets:
                record = self._records.get(target.key)
                if record is None or target.key in seen:
                    continue
                seen.add(target.key)
                rows.append(
                    {
                        "key": target.key,
                        "target": target.checkpoint_value(),
                        "status": record.status,
                        "evidence_numbers": list(record.evidence_numbers),
                        "detail": record.detail,
                        "absence_proven": False,
                    }
                )
            return copy.deepcopy(rows)

    def reserve_integration(
        self, targets: list[CroppedRepairTarget], owner: str
    ) -> bool:
        """Reserve one owner-local integration, separately from global acquisition."""
        self._fence()
        if not owner.strip() or len(owner) > 256:
            raise ValueError("Cropped integration owner must be a bounded identity")
        keys = set(target.key for target in targets)
        if not keys:
            return False
        with self._lock:
            if keys - self._records.keys():
                raise ValueError("Cropped integration requires reserved source targets")
            if any(self._records[key].status == "in_progress" for key in keys):
                raise ValueError(
                    "Cropped integration requires completed source attempts"
                )
            previous = self._integrations.get(owner, set())
            new_keys = keys - previous
            if not new_keys:
                return False
            if (
                sum(len(value) for value in self._integrations.values()) + len(new_keys)
                > 4096
            ):
                raise ValueError("Cropped integration capacity exceeded")
            self._integrations[owner] = previous | keys
            return True

    def export(self) -> dict[str, JsonValue]:
        self._fence()
        with self._lock:
            integrations: dict[str, JsonValue] = {
                owner: cast(JsonValue, sorted(keys))
                for owner, keys in self._integrations.items()
            }
            return {
                "version": 1,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "records": [
                    record.checkpoint_value() for record in self._records.values()
                ],
                "record_integrity": {
                    key: _digest(record.checkpoint_value())
                    for key, record in self._records.items()
                },
                "integrations": integrations,
                "integrations_integrity": _digest(integrations),
            }

    def restore(self, snapshot: dict[str, JsonValue]) -> None:
        self._fence()
        saved = _Checkpoint.model_validate(snapshot)
        if (saved.run_id, saved.scope_hash, saved.request_hash) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
        ):
            raise ValueError("Cropped repair checkpoint identity or request mismatch")
        records: dict[str, _Record] = {}
        for record in saved.records:
            key = record.target.key
            if key in records or saved.record_integrity.get(key) != _digest(
                record.checkpoint_value()
            ):
                raise ValueError("Cropped repair checkpoint integrity changed")
            records[key] = record.model_copy(deep=True)
        if set(saved.record_integrity) != set(records):
            raise ValueError(
                "Cropped repair checkpoint contains unrelated integrity keys"
            )
        if saved.integrations_integrity != _digest(saved.integrations) and (
            saved.integrations or saved.integrations_integrity
        ):
            raise ValueError("Cropped integration checkpoint integrity changed")
        integrations: dict[str, set[str]] = {}
        for owner, keys in saved.integrations.items():
            if (
                not owner.strip()
                or len(owner) > 256
                or len(set(keys)) != len(keys)
                or set(keys) - records.keys()
            ):
                raise ValueError(
                    "Cropped integration checkpoint references unknown targets"
                )
            integrations[owner] = set(keys)
        if sum(len(keys) for keys in integrations.values()) > 4096:
            raise ValueError("Cropped integration checkpoint capacity exceeded")
        with self._lock:
            if self._records and (
                self._records != records or self._integrations != integrations
            ):
                raise ValueError(
                    "Cropped repair checkpoint cannot replace live attempts"
                )
            self._records = records
            self._integrations = integrations
