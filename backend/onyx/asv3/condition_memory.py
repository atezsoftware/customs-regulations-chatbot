"""Retain source-bound obligations across assessments without copying source text."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_witness_valid
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    MaterialSourceOmission,
    SourceConditionAuditResult,
    SourceConditionCheck,
)
from onyx.asv3.quotations import normalized
from onyx.asv3.scenario import question_determinations


class RetainedSourceCondition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    condition_id: str
    text_hash: str
    requirement: MaterialSourceOmission


class SourceConditionCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: int = 1
    run_id: str
    scope_hash: str
    questions: list[str]
    conditions: list[RetainedSourceCondition] = Field(max_length=256)


class SourceConditionMemory:
    def __init__(self, run_id: str, scope_hash: str, questions: list[str]) -> None:
        self.run_id, self.scope_hash = run_id, scope_hash
        self.questions = tuple(questions)
        self._records: dict[str, RetainedSourceCondition] = {}
        self._inventory_receipts: dict[str, str] = {}
        self._lock = threading.RLock()

    def inventory_receipt(self, identity: str) -> str | None:
        with self._lock:
            return self._inventory_receipts.get(identity)

    def retain_inventory_receipt(self, identity: str, call_id: str) -> None:
        with self._lock:
            self._inventory_receipts[identity] = call_id

    def _record(
        self, item: MaterialSourceOmission, ledger: EvidenceLedger
    ) -> RetainedSourceCondition:
        requirement = MaterialSourceOmission.model_validate(
            item.model_dump(include=set(MaterialSourceOmission.model_fields))
        )
        original = ledger.get(requirement.witness.citation)
        allowed = {
            row["determination_id"]
            for row in question_determinations(list(self.questions))
        }
        if (
            original is None
            or not assertion_witness_valid(
                requirement.witness, {requirement.witness.citation: original.text}
            )
            or set(requirement.determination_ids) - allowed
        ):
            raise ValueError(
                "Retained condition must bind to an original and requested determination"
            )
        identity = json.dumps(
            [
                self.run_id,
                self.scope_hash,
                original.text_hash,
                requirement.witness.model_dump(),
                sorted(requirement.determination_ids),
                normalized(requirement.detail),
            ],
            ensure_ascii=False,
            sort_keys=True,
        )
        return RetainedSourceCondition(
            condition_id="sc-" + hashlib.sha256(identity.encode()).hexdigest()[:24],
            text_hash=original.text_hash,
            requirement=requirement,
        )

    def remember(
        self, items: Sequence[MaterialSourceOmission], ledger: EvidenceLedger
    ) -> None:
        records = [self._record(item, ledger) for item in items]
        with self._lock:
            additions = {
                item.condition_id: item
                for item in records
                if item.condition_id not in self._records
            }
            if len(self._records) + len(additions) > 256:
                raise ValueError(
                    "Retained condition capacity exceeded; material obligations were not evicted"
                )
            self._records.update(additions)

    def citations(self) -> list[int]:
        with self._lock:
            return list(
                dict.fromkeys(
                    row.requirement.witness.citation for row in self._records.values()
                )
            )

    def required_conditions(self) -> list[dict[str, JsonValue]]:
        with self._lock:
            return [row.model_dump(mode="json") for row in self._records.values()]

    def materialize(
        self, audit: SourceConditionAuditResult, ledger: EvidenceLedger
    ) -> tuple[SourceConditionAuditResult, list[str]]:
        """Apply explicit resolutions to immutable requirements, never to renamed rules."""
        with self._lock:
            records = dict(self._records)
        supplied = {row.condition_id: row for row in audit.resolutions}
        defects: list[str] = []
        if len(supplied) != len(audit.resolutions) or supplied.keys() - records.keys():
            defects.append(
                "Condition resolutions contain duplicate or unknown retained IDs."
            )
        conditions = list(audit.conditions)
        repeated: set[str] = set()
        for item in conditions:
            try:
                record = self._record(item, ledger)
            except ValueError as error:
                defects.append(str(error))
                continue
            if record.condition_id in records:
                repeated.add(record.condition_id)
        if repeated & supplied.keys():
            defects.append(
                "Assess a retained condition once, in resolutions or as the same unchanged requirement."
            )
        for identity, record in records.items():
            if identity in repeated:
                continue
            resolution = supplied.get(identity)
            if resolution is None:
                defects.append(
                    f"Retained condition {identity} was not reassessed; absence cannot close it."
                )
                continue
            requirement = record.requirement.model_dump()
            if resolution.witness is not None:
                requirement["witness"] = resolution.witness.model_dump()
            conditions.append(
                SourceConditionCheck(
                    **requirement,
                    **resolution.model_dump(exclude={"condition_id", "witness"}),
                )
            )
        # Publication uses the same complete requirement set, with exact original witnesses.
        return audit.model_copy(
            update={"conditions": conditions, "resolutions": []}
        ), defects

    def export(self) -> dict[str, JsonValue]:
        return SourceConditionCheckpoint(
            run_id=self.run_id,
            scope_hash=self.scope_hash,
            questions=list(self.questions),
            conditions=[
                RetainedSourceCondition.model_validate(row)
                for row in self.required_conditions()
            ],
        ).model_dump(mode="json")

    def restore(self, data: dict[str, JsonValue], ledger: EvidenceLedger) -> None:
        saved = SourceConditionCheckpoint.model_validate(data)
        if (saved.version, saved.run_id, saved.scope_hash, saved.questions) != (
            1,
            self.run_id,
            self.scope_hash,
            list(self.questions),
        ):
            raise ValueError("Retained condition request or scope mismatch")
        records = [self._record(row.requirement, ledger) for row in saved.conditions]
        if any(
            row != restored for row, restored in zip(saved.conditions, records)
        ) or len({row.condition_id for row in records}) != len(records):
            raise ValueError("Retained condition original identity changed")
        with self._lock:
            self._records = {row.condition_id: row for row in records}
