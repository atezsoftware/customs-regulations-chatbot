"""Retain actual rejected governing-source dependencies across answer revisions."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import (
    _abbreviated_statute_references,
    _authority_aliases,
    _defined_statute_abbreviations,
    _formal_law_name,
    _gap_masking_aliases,
    _lowercase_slash_clause,
    _matching_native_rows,
    _named_native_references,
    _native_original_rows,
    _normalized_native_reference,
    _original_gap_references,
    _precise_original_gap,
    _reference_name,
    _without_verified_quotes,
    folded,
    native_named_authority_gap,
    statute_references,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.asv3.parallel_execution import parallel_execution_enabled
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
    reference_policy: Literal["legacy", "syntactic-v1"] = "legacy"
    records: list[_Requirement]
    record_integrity: dict[str, str]


class AuthorityRequirements:
    def __init__(
        self,
        context: RunContext,
        request: str,
        *,
        syntactic_reference_binding: bool = False,
    ) -> None:
        self.run_id = context.run_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self.syntactic_reference_binding = syntactic_reference_binding
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
        ledger: EvidenceLedger,
    ) -> None:
        if gap is None:
            return
        missing = gap.get("named_authority_gaps")
        if not isinstance(missing, list):
            return
        units = {unit["unit_id"]: unit for unit in assertion_inventory(answer)}
        owner = self._owner(context)
        task_outcomes = context.services.get("task_outcome_ids")
        outcome_ids = (
            [item for item in task_outcomes if isinstance(item, str)]
            if isinstance(task_outcomes, list)
            else []
        )
        native_rows = (
            _native_original_rows(ledger) if self.syntactic_reference_binding else []
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
            entry_clause = entry.get("clause")
            parsed_references = [
                _normalized_native_reference(item, _reference_name(item), native_rows)
                if self.syntactic_reference_binding
                else item
                for item in statute_references(
                    reference,
                    strict_reference_boundaries=True,
                    syntactic_reference_binding=self.syntactic_reference_binding,
                )
            ]
            parsed = next(
                (
                    item
                    for item in parsed_references
                    if item.number == number
                    and item.article == article
                    and (entry_clause is None or item.clause == entry_clause)
                ),
                None,
            )
            formal_name = _reference_name(parsed) if parsed is not None else None
            if formal_name is None:
                name = re.match(r"(.+?\bkanun(?:u|un|unun)?\b)", folded(reference))
                formal_name = _formal_law_name(name[0]) if name is not None else None
            bound_reference = parsed
            if self.syntactic_reference_binding and article is not None:
                bound = (
                    [(parsed, formal_name)]
                    if parsed is not None
                    else _named_native_references(
                        reference,
                        {formal_name: {number or ""}} if formal_name else {},
                        strict_reference_boundaries=True,
                        syntactic_reference_binding=True,
                    )
                )
                bound_reference = next(
                    (
                        _normalized_native_reference(item, name, native_rows)
                        for item, name in bound
                        if _normalized_native_reference(item, name, native_rows).article
                        == article
                        and (
                            entry_clause is None
                            or _normalized_native_reference(
                                item, name, native_rows
                            ).clause
                            == entry_clause
                        )
                        and (
                            item.number == number
                            if number is not None
                            else name == formal_name
                        )
                    ),
                    None,
                )
                if bound_reference is None:
                    aliases = _authority_aliases(
                        ledger,
                        native_rows,
                        strict_reference_boundaries=True,
                        syntactic_reference_binding=True,
                    )
                    declared = _defined_statute_abbreviations(
                        "\n".join(
                            _without_verified_quotes(
                                item["text"], set(item["evidence_numbers"]), ledger
                            )
                            for item in units.values()
                        ),
                        aliases,
                        syntactic_reference_binding=True,
                    )
                    abbreviated_references = [
                        (_normalized_native_reference(item, name, native_rows), name)
                        for item, name in _abbreviated_statute_references(
                            units[unit]["text"],
                            declared,
                            syntactic_reference_binding=True,
                        )
                    ]
                    abbreviated = next(
                        (
                            (item, name)
                            for item, name in abbreviated_references
                            if item.article == article
                            and (entry_clause is None or item.clause == entry_clause)
                            and (
                                item.number == number
                                if number is not None
                                else name is not None
                                and (formal_name is None or name == formal_name)
                            )
                        ),
                        None,
                    )
                    if abbreviated is not None:
                        bound_reference, declared_name = abbreviated
                        formal_name = declared_name or formal_name
                if bound_reference is None:
                    raise ValueError(
                        "An authority requirement needs an explicitly bound article"
                    )
            if not number and not formal_name:
                raise ValueError("An authority requirement needs a named instrument")
            record = _Requirement(
                requirement_id="authority_" + "0" * 64,
                owner=owner,
                instrument_number=number,
                formal_name=formal_name,
                article=(
                    f"{bound_reference.article}/{bound_reference.clause.upper()}"
                    if self.syntactic_reference_binding
                    and bound_reference is not None
                    and bound_reference.clause_shorthand
                    and bound_reference.clause
                    else article
                ),
                qualifier=(
                    bound_reference.qualifier
                    if bound_reference is not None
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
            elif (
                self.syntactic_reference_binding
                and bound_reference is not None
                and not bound_reference.clause_shorthand
                and _lowercase_slash_clause(
                    self._records[key].reference_text, record.article
                )
            ):
                # Explicit inserted identity remains required after a shorthand rejection.
                existing = self._records[key]
                self._records[key] = existing.model_copy(
                    update={"reference_text": reference}
                )

    def _probe(self, record: _Requirement) -> str:
        if record.instrument_number or (
            self.syntactic_reference_binding and record.formal_name
        ):
            identity = (
                f"{record.instrument_number} sayılı {record.formal_name or 'Kanun'}"
                if record.instrument_number
                else record.formal_name or ""
            )
            article = record.article
            if self.syntactic_reference_binding and _lowercase_slash_clause(
                record.reference_text, article
            ):
                assert article is not None
                base, letter = article.split("/")
                if not re.search(
                    rf"\b{re.escape(base)}\s*/\s*{re.escape(letter)}\b",
                    record.reference_text,
                ):
                    article = base + "/" + letter.lower().replace("i̇", "i")
            locator = (
                f" {(record.qualifier + ' ') if record.qualifier else ''}MADDE {article}"
                if record.article is not None
                else ""
            )
            return identity + locator + " uygulanır."
        return record.reference_text

    def _formal_reference_name(self, reference: object) -> str | None:
        if not isinstance(reference, str):
            return None
        parsed = statute_references(
            reference,
            strict_reference_boundaries=True,
            syntactic_reference_binding=self.syntactic_reference_binding,
        )
        if parsed:
            return _reference_name(parsed[0])
        name = re.match(r"(.+?\bkanun(?:u|un|unun)?\b)", folded(reference))
        return _formal_law_name(name[0]) if name is not None else None

    def _matching_originals(
        self, record: _Requirement, ledger: EvidenceLedger
    ) -> set[int]:
        probe = self._probe(record)
        if self.syntactic_reference_binding:
            rows = _native_original_rows(ledger)
            references = [
                (reference, name)
                for reference, name in _named_native_references(
                    probe,
                    {record.formal_name: {record.instrument_number or ""}}
                    if record.formal_name
                    else {},
                    strict_reference_boundaries=True,
                    syntactic_reference_binding=True,
                )
                if (
                    reference.number == record.instrument_number
                    if record.instrument_number
                    else name == record.formal_name
                )
                and reference.article == record.article
                and reference.qualifier == record.qualifier
            ]
            references.sort(key=lambda value: value[0].clause_shorthand)
            if references:
                normalized = _normalized_native_reference(*references[0], rows)
                return set(
                    _matching_native_rows(
                        normalized, references[0][1], rows, clause_specific=True
                    )
                )
            return set()
        gap = native_named_authority_gap(
            probe,
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=self.syntactic_reference_binding,
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
                else self._formal_reference_name(row.get("reference_text"))
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

    def _disclosed(
        self, record: _Requirement, answer: str, ledger: EvidenceLedger
    ) -> bool:
        aliases = (
            {record.formal_name: {record.instrument_number or ""}}
            if record.formal_name is not None
            else {}
        )
        if self.syntactic_reference_binding:
            aliases = _authority_aliases(
                ledger,
                _native_original_rows(ledger),
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            )
            for retained in self._records.values():
                if retained.owner == record.owner and retained.formal_name is not None:
                    numbers = aliases.setdefault(retained.formal_name, set())
                    if retained.instrument_number is not None:
                        numbers.add(retained.instrument_number)
        declared = (
            _defined_statute_abbreviations(
                "\n".join(
                    _without_verified_quotes(
                        unit["text"], set(unit["evidence_numbers"]), ledger
                    )
                    for unit in assertion_inventory(answer)
                ),
                aliases,
                syntactic_reference_binding=True,
            )
            if self.syntactic_reference_binding
            else {}
        )
        gap_aliases = {
            **aliases,
            **{
                label: {identity[0]}
                for label, identity in declared.items()
                if identity is not None
            },
        }
        masking_aliases = (
            _gap_masking_aliases(_native_original_rows(ledger), declared)
            if self.syntactic_reference_binding
            else {}
        )
        for unit in assertion_inventory(answer):
            if unit["evidence_numbers"]:
                continue
            disclosed = (
                _original_gap_references(
                    unit["text"],
                    gap_aliases,
                    masking_aliases=masking_aliases,
                )
                if self.syntactic_reference_binding
                else None
            )
            if (
                disclosed is None
                if self.syntactic_reference_binding
                else not _precise_original_gap(unit["text"], gap_aliases)
            ):
                continue
            numbered = statute_references(
                unit["text"],
                strict_reference_boundaries=True,
                syntactic_reference_binding=self.syntactic_reference_binding,
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
            references = (
                disclosed
                if disclosed is not None
                else _named_native_references(
                    unit["text"],
                    aliases,
                    strict_reference_boundaries=True,
                    syntactic_reference_binding=self.syntactic_reference_binding,
                )
            )
            for reference, name in references:
                if self.syntactic_reference_binding and name in declared:
                    identity = declared[name]
                    if identity is not None:
                        name = identity[1]
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
        validated_delivered: set[int] | None = None,
    ) -> dict[str, JsonValue] | None:
        """Require acquired support or a precise disclosure, without judging entailment."""
        self._fence(context)
        with self._lock:
            if native_gap is None:
                native_gap = native_named_authority_gap(
                    answer,
                    ledger,
                    strict_reference_boundaries=True,
                    syntactic_reference_binding=self.syntactic_reference_binding,
                    resolve_defined_abbreviations=self.syntactic_reference_binding,
                )
            self._remember(answer, native_gap, context, ledger)
            delivered = (
                set(validated_delivered)
                if validated_delivered is not None
                else ledger.completely_delivered(model_call_id)
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
                    record, answer, ledger
                ):
                    continue
                missing.append(
                    {
                        **record.model_dump(mode="json"),
                        "matching_original_evidence": sorted(matching),
                    }
                )
            if not missing:
                if native_gap is not None and parallel_execution_enabled(context):
                    from onyx.asv3.authority_reference_diagnostics import (
                        authority_reference_diagnostics,
                    )

                    return authority_reference_diagnostics(
                        answer, native_gap, ledger, delivered
                    )
                return native_gap
            gap: dict[str, JsonValue] = {
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
            if parallel_execution_enabled(context):
                from onyx.asv3.authority_reference_diagnostics import (
                    authority_reference_diagnostics,
                )

                return authority_reference_diagnostics(answer, gap, ledger, delivered)
            return gap

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
            saved = _Checkpoint(
                run_id=self.run_id,
                scope_hash=self.scope_hash,
                request_hash=self.request_hash,
                reference_policy=(
                    "syntactic-v1" if self.syntactic_reference_binding else "legacy"
                ),
                records=list(self._records.values()),
                record_integrity={
                    key: _digest(record.model_dump(mode="json"))
                    for key, record in self._records.items()
                },
            ).model_dump(mode="json")
            if not self.syntactic_reference_binding:
                saved.pop("reference_policy")
            return saved

    def restore(
        self,
        snapshot: dict[str, JsonValue],
        context: RunContext,
        request: str,
        *,
        allow_legacy_upgrade: bool = False,
        ledger: EvidenceLedger | None = None,
        retained_answer: str | None = None,
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
        expected_policy = (
            "syntactic-v1" if self.syntactic_reference_binding else "legacy"
        )
        if saved.reference_policy != expected_policy:
            if not (
                allow_legacy_upgrade
                and self.syntactic_reference_binding
                and saved.reference_policy == "legacy"
                and parallel_execution_enabled(context)
                and isinstance(ledger, EvidenceLedger)
            ):
                raise ValueError("Authority reference binding policy changed")
            rebuilt = AuthorityRequirements(
                context, request, syntactic_reference_binding=True
            )
            for record in records.values():
                owner_context = copy.copy(context)
                owner_context.services = dict(context.services)
                if record.owner == "coordinator":
                    owner_context.services.pop("task_id", None)
                else:
                    owner_context.services["task_id"] = record.owner
                owner_context.services["task_outcome_ids"] = list(record.outcome_ids)
                legacy = AuthorityRequirements(owner_context, request)
                legacy_gap = native_named_authority_gap(
                    record.reference_text, ledger, strict_reference_boundaries=True
                )
                legacy._remember(
                    record.reference_text, legacy_gap, owner_context, ledger
                )
                if record.requirement_id not in legacy._records:
                    raise ValueError(
                        "Legacy authority reference identity cannot be recomputed"
                    )
                gap = native_named_authority_gap(
                    record.reference_text,
                    ledger,
                    strict_reference_boundaries=True,
                    syntactic_reference_binding=True,
                )
                if gap is None or not gap.get("named_authority_gaps"):
                    raise ValueError("Legacy authority reference cannot be recomputed")
                prior_ids = set(rebuilt._records)
                rebuilt._remember(record.reference_text, gap, owner_context, ledger)
                for identifier in rebuilt._records.keys() - prior_ids:
                    rebuilt._records[identifier] = rebuilt._records[
                        identifier
                    ].model_copy(update={"origin_unit_id": record.origin_unit_id})
            if retained_answer:
                rebuilt._remember(
                    retained_answer,
                    native_named_authority_gap(
                        retained_answer,
                        ledger,
                        strict_reference_boundaries=True,
                        syntactic_reference_binding=True,
                    ),
                    context,
                    ledger,
                )
            records = rebuilt._records
        with self._lock:
            self._records = records
