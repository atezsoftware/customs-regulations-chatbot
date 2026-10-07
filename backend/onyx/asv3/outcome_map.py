"""Compact model-selected outcomes and original-bound completion records."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext

RecordId = Annotated[str, Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")]
CitationNumber = Annotated[int, Field(strict=True, ge=1)]


class RequestedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome_id: RecordId
    question_ids: list[str] = Field(min_length=1)
    detail: str = Field(min_length=1)
    decisive_facts: list[str] = Field(default_factory=list)


class OutcomeWitness(BaseModel):
    model_config = ConfigDict(extra="forbid")
    citation: CitationNumber
    start_char: int = Field(default=0, strict=True, ge=0)
    end_char: int = Field(strict=True, ge=1)


class OutcomeCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition_id: RecordId
    outcome_ids: list[RecordId] = Field(min_length=1)
    detail: str = Field(min_length=1)
    witnesses: list[OutcomeWitness] = Field(min_length=1)


class OutcomeResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcome_id: RecordId
    status: Literal["supported", "conditional", "unresolved"]
    condition_ids: list[RecordId] = Field(default_factory=list)
    evidence_numbers: list[CitationNumber] = Field(default_factory=list)
    gap: str = ""


class OutcomeUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    outcomes: list[RequestedOutcome] = Field(default_factory=list)
    conditions: list[OutcomeCondition] = Field(default_factory=list)
    resolutions: list[OutcomeResolution] = Field(default_factory=list)


class _ConditionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    condition: OutcomeCondition
    source_hashes: dict[str, str]


class _OutcomeCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    run_id: str
    scope_hash: str
    request_hash: str
    questions: list[str]
    revision: int = Field(strict=True, ge=0)
    outcomes: list[RequestedOutcome]
    conditions: list[_ConditionRecord]
    resolutions: list[OutcomeResolution]


class OutcomeMap:
    def __init__(
        self,
        questions: list[str],
        context: RunContext,
        *,
        factual_context: str | None = None,
        detailed_fact_errors: bool = False,
        reuse_retained_conditions: bool = False,
    ) -> None:
        self.run_id = context.run_id
        self.scope_hash = hashlib.sha256(
            json.dumps(context.scope, sort_keys=True).encode()
        ).hexdigest()
        self.questions = tuple(questions)
        self._factual_context = (
            "\n".join(questions) if factual_context is None else factual_context
        )
        self.request_hash = hashlib.sha256(self._factual_context.encode()).hexdigest()
        self._detailed_fact_errors = detailed_fact_errors
        self._reuse_retained_conditions = reuse_retained_conditions
        self._outcomes: dict[str, RequestedOutcome] = {}
        self._conditions: dict[str, _ConditionRecord] = {}
        self._resolutions: dict[str, OutcomeResolution] = {}
        self._lock = threading.RLock()
        self.revision = 0

    def factual_context(self) -> str:
        """Return the immutable literal context used to validate decisive facts."""
        return self._factual_context

    def _condition_record(
        self, condition: OutcomeCondition, ledger: EvidenceLedger
    ) -> _ConditionRecord:
        hashes: dict[str, str] = {}
        for witness in condition.witnesses:
            original = ledger.get(witness.citation)
            if original is None or not (
                witness.start_char < witness.end_char <= len(original.text)
            ):
                raise ValueError("Outcome condition needs a recorded original range")
            hashes[str(witness.citation)] = original.text_hash
        return _ConditionRecord(
            condition=condition.model_copy(deep=True), source_hashes=hashes
        )

    def update(self, update: OutcomeUpdate, ledger: EvidenceLedger) -> None:
        """Validate one atomic metadata update without source or model calls."""
        with self._lock:
            outcomes = dict(self._outcomes)
            conditions = dict(self._conditions)
            resolutions = dict(self._resolutions)
            known_questions = {f"q{index}" for index in range(len(self.questions))}
            changed: set[str] = set()
            for rows, field in (
                (update.outcomes, "outcome_id"),
                (update.conditions, "condition_id"),
                (update.resolutions, "outcome_id"),
            ):
                identities = [getattr(row, field) for row in rows]
                if len(identities) != len(set(identities)):
                    raise ValueError("Duplicate outcome metadata record IDs")
            for outcome_index, outcome in enumerate(update.outcomes):
                if set(outcome.question_ids) - known_questions or len(
                    outcome.question_ids
                ) != len(set(outcome.question_ids)):
                    raise ValueError("Outcome must bind to original question IDs")
                if not outcome.detail.strip():
                    raise ValueError("Outcome facts must quote supplied scenario text")
                for fact_index, fact in enumerate(outcome.decisive_facts):
                    if fact.strip() and fact in self._factual_context:
                        continue
                    message = "Outcome facts must quote supplied scenario text"
                    if self._detailed_fact_errors:
                        message = (
                            f"_outcomes[{outcome_index}].decisive_facts[{fact_index}] "
                            f"(outcome_id={outcome.outcome_id}): {message}. "
                            "Replace only this fact with an exact contiguous quote from the "
                            "supplied request or conversation, or omit an unsupported fact. "
                            "Do not normalize or paraphrase it, or rewrite the answer."
                        )
                    raise ValueError(message)
                previous = outcomes.get(outcome.outcome_id)
                if previous is not None and set(previous.question_ids) != set(
                    outcome.question_ids
                ):
                    raise ValueError("An outcome cannot move to other questions")
                if previous != outcome:
                    changed.add(outcome.outcome_id)
                outcomes[outcome.outcome_id] = outcome.model_copy(deep=True)
            for condition_index, condition in enumerate(update.conditions):
                if (
                    not condition.detail.strip()
                    or set(condition.outcome_ids) - outcomes.keys()
                    or len(condition.outcome_ids) != len(set(condition.outcome_ids))
                ):
                    raise ValueError("Condition must bind to known outcomes")
                record = self._condition_record(condition, ledger)
                previous_record = conditions.get(condition.condition_id)
                if previous_record is not None:
                    previous = previous_record.condition
                    if (
                        self._reuse_retained_conditions
                        and previous.detail == condition.detail
                        and all(
                            witness in condition.witnesses
                            for witness in previous.witnesses
                        )
                        and all(
                            record.source_hashes.get(citation) == text_hash
                            for citation, text_hash in previous_record.source_hashes.items()
                        )
                    ):
                        # Repeated proof additions do not redefine a retained requirement.
                        record = previous_record.model_copy(deep=True)
                    if (
                        previous.detail != record.condition.detail
                        or previous.witnesses != record.condition.witnesses
                        or previous_record.source_hashes != record.source_hashes
                    ):
                        message = "Retained source conditions cannot be replaced"
                        if self._reuse_retained_conditions:
                            fields = [
                                name
                                for name in ("detail", "witnesses")
                                if getattr(previous, name) != getattr(condition, name)
                            ]
                            if previous_record.source_hashes != record.source_hashes:
                                fields.append("source_hashes")
                            message = (
                                f"_coverage.conditions[{condition_index}] "
                                f"(condition_id={condition.condition_id}): {message}; "
                                f"changed fields: {', '.join(fields)}. "
                                "Omit this retained condition from conditions rather than "
                                "recopying or modifying it. Bind new operative requirements "
                                "with new condition IDs and actual witnesses; update "
                                "resolutions independently. Preserve the answer edits."
                            )
                        raise ValueError(message)
                    changed.update(
                        set(condition.outcome_ids) - set(previous.outcome_ids)
                    )
                    record.condition.outcome_ids = list(
                        dict.fromkeys([*previous.outcome_ids, *condition.outcome_ids])
                    )
                else:
                    changed.update(condition.outcome_ids)
                conditions[condition.condition_id] = record
            # A changed outcome or newly retained condition needs a fresh assessment.
            for identity in changed:
                resolutions.pop(identity, None)
            for resolution in update.resolutions:
                if resolution.outcome_id not in outcomes:
                    raise ValueError("Resolution must bind to a known outcome")
                if len(resolution.condition_ids) != len(set(resolution.condition_ids)):
                    raise ValueError("Duplicate resolved condition IDs")
                applicable = {
                    identity
                    for identity, record in conditions.items()
                    if resolution.outcome_id in record.condition.outcome_ids
                }
                if set(resolution.condition_ids) - applicable:
                    raise ValueError("Resolution condition belongs to another outcome")
                if any(
                    ledger.get(number) is None for number in resolution.evidence_numbers
                ):
                    raise ValueError("Resolution references unknown original evidence")
                if resolution.status == "unresolved":
                    if not resolution.gap.strip():
                        raise ValueError("Unresolved outcome needs a precise gap")
                else:
                    if resolution.gap.strip() or not resolution.evidence_numbers:
                        raise ValueError(
                            "Supported outcome needs originals and no open gap"
                        )
                    required_citations = {
                        witness.citation
                        for identity in applicable
                        for witness in conditions[identity].condition.witnesses
                    }
                    if applicable != set(
                        resolution.condition_ids
                    ) or required_citations - set(resolution.evidence_numbers):
                        raise ValueError(
                            "Supported outcome must retain its source conditions"
                        )
                resolutions[resolution.outcome_id] = resolution.model_copy(deep=True)
            if (
                outcomes != self._outcomes
                or conditions != self._conditions
                or resolutions != self._resolutions
            ):
                self._outcomes, self._conditions, self._resolutions = (
                    outcomes,
                    conditions,
                    resolutions,
                )
                self.revision += 1

    def outcome_ids(self, question_ids: list[str] | None = None) -> list[str]:
        with self._lock:
            selected = set(question_ids) if question_ids is not None else None
            return [
                identity
                for identity, outcome in self._outcomes.items()
                if selected is None or selected.intersection(outcome.question_ids)
            ]

    def _selected_ids(
        self, question_ids: list[str] | None, outcome_ids: list[str] | None
    ) -> set[str]:
        selected = set(self.outcome_ids(question_ids))
        if outcome_ids is not None:
            if set(outcome_ids) - self._outcomes.keys():
                raise ValueError("Unknown outcome subset")
            selected.intersection_update(outcome_ids)
        return selected

    def preferred_citations(
        self,
        question_ids: list[str] | None = None,
        *,
        outcome_ids: list[str] | None = None,
    ) -> list[int]:
        with self._lock:
            selected = self._selected_ids(question_ids, outcome_ids)
            return list(
                dict.fromkeys(
                    [
                        *(
                            witness.citation
                            for record in self._conditions.values()
                            if selected.intersection(record.condition.outcome_ids)
                            for witness in record.condition.witnesses
                        ),
                        *(
                            citation
                            for identity, resolution in self._resolutions.items()
                            if identity in selected
                            for citation in resolution.evidence_numbers
                        ),
                    ]
                )
            )

    def view(
        self,
        *,
        question_ids: list[str] | None = None,
        outcome_ids: list[str] | None = None,
        delivered_citations: set[int] | None = None,
    ) -> dict[str, JsonValue]:
        with self._lock:
            selected = self._selected_ids(question_ids, outcome_ids)
            citations = self.preferred_citations(question_ids, outcome_ids=outcome_ids)
            result: dict[str, JsonValue] = {
                "revision": self.revision,
                "outcomes": [
                    outcome.model_dump(mode="json")
                    for identity, outcome in self._outcomes.items()
                    if identity in selected
                ],
                "conditions": [
                    {
                        **record.condition.model_dump(mode="json"),
                        "outcome_ids": [
                            identity
                            for identity in record.condition.outcome_ids
                            if identity in selected
                        ],
                    }
                    for record in self._conditions.values()
                    if selected.intersection(record.condition.outcome_ids)
                ],
                "resolutions": [
                    resolution.model_dump(mode="json")
                    for identity, resolution in self._resolutions.items()
                    if identity in selected
                ],
                "unassessed_outcome_ids": [
                    identity
                    for identity in self._outcomes
                    if identity in selected and identity not in self._resolutions
                ],
                "notice": "Model-selected outcomes and source-bound candidate assessments; original text and actual applicability must be evaluated before publication. No recorded map implies no completed assessment.",
            }
            if delivered_citations is not None:
                result["undelivered_evidence_numbers"] = [
                    citation
                    for citation in citations
                    if citation not in delivered_citations
                ]
            return result

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return _OutcomeCheckpoint(
                run_id=self.run_id,
                scope_hash=self.scope_hash,
                request_hash=self.request_hash,
                questions=list(self.questions),
                revision=self.revision,
                outcomes=list(self._outcomes.values()),
                conditions=list(self._conditions.values()),
                resolutions=list(self._resolutions.values()),
            ).model_dump(mode="json")

    def restore(self, data: dict[str, JsonValue], ledger: EvidenceLedger) -> None:
        saved = _OutcomeCheckpoint.model_validate(data)
        if (
            saved.run_id,
            saved.scope_hash,
            saved.request_hash,
            saved.questions,
        ) != (
            self.run_id,
            self.scope_hash,
            self.request_hash,
            list(self.questions),
        ):
            raise ValueError("Outcome checkpoint request or scope mismatch")
        restored = OutcomeMap.__new__(OutcomeMap)
        restored.run_id, restored.scope_hash = self.run_id, self.scope_hash
        restored.request_hash, restored.questions = self.request_hash, self.questions
        restored._factual_context = self._factual_context
        restored._detailed_fact_errors = self._detailed_fact_errors
        restored._reuse_retained_conditions = self._reuse_retained_conditions
        restored._lock = threading.RLock()
        restored._outcomes, restored._conditions, restored._resolutions = {}, {}, {}
        restored.revision = 0
        restored.update(
            OutcomeUpdate(
                outcomes=saved.outcomes,
                conditions=[record.condition for record in saved.conditions],
                resolutions=saved.resolutions,
            ),
            ledger,
        )
        if list(restored._conditions.values()) != saved.conditions:
            raise ValueError("Outcome checkpoint original identity changed")
        with self._lock:
            self._outcomes, self._conditions, self._resolutions = (
                restored._outcomes,
                restored._conditions,
                restored._resolutions,
            )
            self.revision = saved.revision
