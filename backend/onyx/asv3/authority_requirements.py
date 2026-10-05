"""Retain actual rejected governing-source dependencies across answer revisions."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import (
    _formal_law_name,
    _named_native_references,
    _precise_original_gap,
    _reference_name,
    folded,
    native_named_authority_gap,
    statute_references,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.regulatory.heading_path import (
    extract_regulatory_provision_reference_occurrences,
)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class _Requirement(BaseModel):
    model_config = ConfigDict(extra="forbid")
    requirement_id: str = Field(pattern=r"^authority_[a-f0-9]{64}$")
    owner: str = Field(min_length=1)
    instrument_number: str | None
    formal_name: str | None
    article: str | None
    qualifier: str | None
    reference_text: str = Field(min_length=1)
    origin_unit_id: str = Field(min_length=1)
    outcome_ids: list[str] = Field(default_factory=list)


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    run_id: str
    scope_hash: str
    request_hash: str
    records: list[_Requirement]
    record_integrity: dict[str, str]


class AuthorityRequirements:
    def __init__(self, context: RunContext, request: str) -> None:
        self.run_id = context.run_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self._records: dict[str, _Requirement] = {}
        self._lock = threading.RLock()

    def _fence(self, context: RunContext) -> None:
        if (context.run_id, _digest(context.scope)) != (
            self.run_id,
            self.scope_hash,
        ):
            raise ValueError("Authority requirement run or scope changed")

    @staticmethod
    def _owner(context: RunContext) -> str:
        task = context.services.get("task_id")
        return task if isinstance(task, str) and task else "coordinator"

    @staticmethod
    def _identity(record: _Requirement) -> list[str | None]:
        return [
            record.owner,
            record.instrument_number,
            record.formal_name if record.instrument_number is None else None,
            record.article,
            record.qualifier,
        ]

    def _remember(
        self,
        answer: str,
        gap: dict[str, JsonValue] | None,
        context: RunContext,
    ) -> None:
        if gap is None:
            return
        missing = gap.get("named_authority_gaps")
        if not isinstance(missing, list):
            return
        units = {unit["unit_id"] for unit in assertion_inventory(answer)}
        owner = self._owner(context)
        task_outcomes = context.services.get("task_outcome_ids")
        outcome_ids = (
            [item for item in task_outcomes if isinstance(item, str)]
            if isinstance(task_outcomes, list)
            else []
        )
        for entry in missing:
            if not isinstance(entry, dict):
                raise ValueError("Invalid named authority requirement")
            unit, reference = entry.get("unit_id"), entry.get("reference_text")
            if (
                not isinstance(unit, str)
                or unit not in units
                or not isinstance(reference, str)
                or not reference.strip()
            ):
                raise ValueError(
                    "An authority requirement needs its rejected answer unit"
                )
            number, article = entry.get("instrument_number"), entry.get("article")
            if (number is not None and not isinstance(number, str)) or (
                article is not None and not isinstance(article, str)
            ):
                raise ValueError("Invalid governing instrument identity")
            parsed = next(
                (
                    item
                    for item in statute_references(
                        reference, strict_reference_boundaries=True
                    )
                    if item.number == number and item.article == article
                ),
                None,
            )
            formal_name = _reference_name(parsed) if parsed is not None else None
            if formal_name is None:
                name = re.match(r"(.+?\bkanun(?:u|un|unun)?\b)", folded(reference))
                formal_name = _formal_law_name(name[0]) if name is not None else None
            if not number and not formal_name:
                raise ValueError("An authority requirement needs a named instrument")
            record = _Requirement(
                requirement_id="authority_" + "0" * 64,
                owner=owner,
                instrument_number=number,
                formal_name=formal_name,
                article=article,
                qualifier=(
                    parsed.qualifier
                    if parsed is not None
                    else next(
                        (
                            locator.qualifier
                            for _, locator in extract_regulatory_provision_reference_occurrences(
                                reference
                            )
                            if locator.article_no == article
                        ),
                        None,
                    )
                ),
                reference_text=reference,
                origin_unit_id=unit,
                outcome_ids=outcome_ids,
            )
            key = "authority_" + _digest(self._identity(record))
            if key not in self._records:
                self._records[key] = record.model_copy(update={"requirement_id": key})

    @staticmethod
    def _probe(record: _Requirement) -> str:
        if record.instrument_number:
            identity = (
                f"{record.instrument_number} sayılı {record.formal_name or 'Kanun'}"
            )
            locator = (
                f" {(record.qualifier + ' ') if record.qualifier else ''}MADDE {record.article}"
                if record.article is not None
                else ""
            )
            return identity + locator + " uygulanır."
        return record.reference_text

    @staticmethod
    def _formal_reference_name(reference: object) -> str | None:
        if not isinstance(reference, str):
            return None
        parsed = statute_references(reference, strict_reference_boundaries=True)
        if parsed:
            return _reference_name(parsed[0])
        name = re.match(r"(.+?\bkanun(?:u|un|unun)?\b)", folded(reference))
        return _formal_law_name(name[0]) if name is not None else None

    @staticmethod
    def _matching_originals(record: _Requirement, ledger: EvidenceLedger) -> set[int]:
        gap = native_named_authority_gap(
            AuthorityRequirements._probe(record),
            ledger,
            strict_reference_boundaries=True,
        )
        if gap is None:
            return set()
        entries = gap.get("named_authority_gaps")
        if not isinstance(entries, list):
            return set()
        return {
            citation
            for row in entries
            if isinstance(row, dict)
            and (
                row.get("instrument_number") == record.instrument_number
                if record.instrument_number is not None
                else AuthorityRequirements._formal_reference_name(
                    row.get("reference_text")
                )
                == record.formal_name
            )
            and row.get("article") == record.article
            for citation in (
                row["matching_original_evidence"]
                if isinstance(row.get("matching_original_evidence"), list)
                else []
            )
            if isinstance(citation, int) and not isinstance(citation, bool)
        }

    @staticmethod
    def _disclosed(record: _Requirement, answer: str) -> bool:
        aliases = (
            {record.formal_name: {record.instrument_number or ""}}
            if record.formal_name is not None
            else {}
        )
        for unit in assertion_inventory(answer):
            if unit["evidence_numbers"] or not _precise_original_gap(
                unit["text"], aliases
            ):
                continue
            numbered = statute_references(
                unit["text"], strict_reference_boundaries=True
            )
            if (
                record.instrument_number is not None
                and numbered
                and not any(
                    reference.number == record.instrument_number
                    for reference in numbered
                )
            ):
                continue
            for reference, name in _named_native_references(
                unit["text"], aliases, strict_reference_boundaries=True
            ):
                if (
                    (
                        reference.number == record.instrument_number
                        if record.instrument_number is not None
                        else name == record.formal_name
                    )
                    and reference.article == record.article
                    and reference.qualifier == record.qualifier
                ):
                    return True
        return False

    def publication_gap(
        self,
        answer: str,
        model_call_id: str | None,
        context: RunContext,
        ledger: EvidenceLedger,
        *,
        native_gap: dict[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue] | None:
        """Require acquired support or a precise disclosure, without judging entailment."""
        self._fence(context)
        with self._lock:
            if native_gap is None:
                native_gap = native_named_authority_gap(
                    answer, ledger, strict_reference_boundaries=True
                )
            self._remember(answer, native_gap, context)
            delivered = (
                ledger.completely_delivered(model_call_id)
                if model_call_id is not None
                else set()
            )
            inline = {
                number
                for unit in assertion_inventory(answer)
                if not unit["presentation_only"]
                for number in unit["evidence_numbers"]
            }
            missing: list[JsonValue] = []
            for record in self._records.values():
                if record.owner != self._owner(context):
                    continue
                matching = self._matching_originals(record, ledger)
                if matching.intersection(inline, delivered) or self._disclosed(
                    record, answer
                ):
                    continue
                missing.append(
                    {
                        **record.model_dump(mode="json"),
                        "matching_original_evidence": sorted(matching),
                    }
                )
            if not missing:
                return native_gap
            return {
                **(native_gap or {}),
                "retained_authority_requirements": missing,
                "instruction": (
                    "A governing-original requirement from an actually rejected legal claim "
                    "remains open after deleting its name or rewriting the claim. Reuse or read "
                    "its matching canonical original and cite it in the current substantive answer "
                    "with complete delivery. If the original remains unavailable or you withdraw "
                    "the unsupported result, disclose that exact named instrument/provision's "
                    "unexamined original in its own uncited paragraph without asserting its legal "
                    "effect. Preserve independently supported implementation and other outcomes. "
                    "Do not invent absence of law or research unrelated legislative tiers. "
                    "This is source-dependency retention, not semantic approval of a result."
                ),
            }

    def view(self, context: RunContext) -> dict[str, JsonValue]:
        self._fence(context)
        with self._lock:
            return {
                "retained_authority_requirements": [
                    record.model_dump(mode="json")
                    for record in self._records.values()
                    if record.owner == self._owner(context)
                ]
            }

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return _Checkpoint(
                run_id=self.run_id,
                scope_hash=self.scope_hash,
                request_hash=self.request_hash,
                records=list(self._records.values()),
                record_integrity={
                    key: _digest(record.model_dump(mode="json"))
                    for key, record in self._records.items()
                },
            ).model_dump(mode="json")

    def restore(
        self, snapshot: dict[str, JsonValue], context: RunContext, request: str
    ) -> None:
        self._fence(context)
        saved = _Checkpoint.model_validate(snapshot)
        if (saved.run_id, saved.scope_hash, saved.request_hash) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
        ) or _digest(request) != self.request_hash:
            raise ValueError("Authority requirement checkpoint changed")
        records: dict[str, _Requirement] = {}
        for record in saved.records:
            if (
                record.requirement_id != "authority_" + _digest(self._identity(record))
                or record.requirement_id in records
                or saved.record_integrity.get(record.requirement_id)
                != _digest(record.model_dump(mode="json"))
            ):
                raise ValueError("Invalid retained authority requirement identity")
            records[record.requirement_id] = record
        if set(saved.record_integrity) != set(records):
            raise ValueError("Invalid retained authority requirement integrity")
        with self._lock:
            self._records = records
