"""Retain explicit source-lead assessments without treating them as legal approval."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_navigation import (
    RelatedSourceRole,
    derive_provision_navigation_anchor,
)
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.outcome_map import OutcomeWitness


class RelatedSourceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(pattern=r"^lead_[a-f0-9]{64}$")
    status: Literal["examined", "not_material", "unresolved"]
    source_role: Literal["operative_text", "argument_only", "unknown"]
    effect: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    witnesses: list[OutcomeWitness] = Field(default_factory=list)
    gap: str = ""


class _LeadRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner: str = Field(min_length=1)
    lead_id: str = Field(pattern=r"^lead_[a-f0-9]{64}$")
    anchor_source_id: str = Field(min_length=1)
    article_no: str = Field(min_length=1)
    qualifier: str | None
    source_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    candidate_role: RelatedSourceRole
    anchor_witnesses: list[OutcomeWitness] = Field(min_length=1)
    anchor_hashes: dict[str, str]
    first_call_id: str = Field(min_length=1)
    review: RelatedSourceReview | None = None
    review_hashes: dict[str, str] = Field(default_factory=dict)


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    run_id: str
    scope_hash: str
    request_hash: str
    records: list[_LeadRecord]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _hashes(witnesses: list[OutcomeWitness], ledger: EvidenceLedger) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for witness in witnesses:
        item = ledger.get(witness.citation)
        if item is None:
            raise ValueError("Related-source witness has no recorded original")
        hashes[str(witness.citation)] = item.text_hash
    return hashes


def _lead_id(navigation: dict[str, JsonValue], source_id: str) -> str:
    return "lead_" + _digest(
        [
            navigation.get("anchor_source_id"),
            navigation.get("article_no"),
            navigation.get("qualifier"),
            source_id,
        ]
    )


def annotate_navigation(
    navigation: list[dict[str, JsonValue]],
) -> list[dict[str, JsonValue]]:
    """Expose stable identities without recording an unsubmitted model context."""
    annotated = copy.deepcopy(navigation)
    for entry in annotated:
        candidates = entry.get("candidates")
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    source_id = candidate.get("source_id")
                    if isinstance(source_id, str):
                        candidate["lead_id"] = _lead_id(entry, source_id)
    return annotated


class LegalSourceReviews:
    annotate_navigation = staticmethod(annotate_navigation)

    def __init__(self, context: RunContext, request: str) -> None:
        self.run_id = context.run_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self._records: dict[tuple[str, str], _LeadRecord] = {}
        self._lock = threading.RLock()

    def _fence(self, context: RunContext) -> None:
        if (context.run_id, _digest(context.scope)) != (
            self.run_id,
            self.scope_hash,
        ):
            raise ValueError("Related-source review run or scope changed")

    @staticmethod
    def _owner(context: RunContext) -> str:
        task = context.services.get("task_id")
        return task if isinstance(task, str) and task else "coordinator"

    def record_delivery(
        self,
        call_id: str,
        context: RunContext,
        navigation: list[dict[str, JsonValue]],
        ledger: EvidenceLedger,
    ) -> None:
        """Register only candidates accompanying a delivered genuine provision."""
        self._fence(context)
        if not call_id.strip():
            raise ValueError("Related-source delivery needs its actual model call")
        anchors: dict[tuple[str, str, str | None], list[OutcomeWitness]] = {}
        for citation in ledger.completely_delivered(call_id):
            item = ledger.get(citation)
            if item is None:
                continue
            anchor = derive_provision_navigation_anchor(item.source_id, [item])
            if anchor is not None:
                anchors.setdefault(
                    (anchor.source_id, anchor.article_no, anchor.qualifier), []
                ).append(OutcomeWitness(citation=citation, end_char=len(item.text)))
        owner = self._owner(context)
        with self._lock:
            records = dict(self._records)
            for entry in annotate_navigation(navigation):
                source = entry.get("anchor_source_id")
                article = entry.get("article_no")
                qualifier = entry.get("qualifier")
                if (
                    not isinstance(source, str)
                    or not isinstance(article, str)
                    or (qualifier is not None and not isinstance(qualifier, str))
                ):
                    continue
                witnesses = anchors.get((source, article, qualifier), [])
                candidates = entry.get("candidates")
                if not witnesses or not isinstance(candidates, list):
                    continue
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    identifier = candidate.get("source_id")
                    name = candidate.get("name")
                    role = candidate.get("candidate_role")
                    if (
                        not isinstance(identifier, str)
                        or not identifier.strip()
                        or identifier == source
                        or not isinstance(name, str)
                        or not name.strip()
                        or role
                        not in {
                            "judicial_candidate",
                            "referral_candidate",
                            "executive_candidate",
                            "amendment_candidate",
                        }
                    ):
                        continue
                    record = _LeadRecord.model_validate(
                        {
                            "owner": owner,
                            "lead_id": _lead_id(entry, identifier),
                            "anchor_source_id": source,
                            "article_no": article,
                            "qualifier": qualifier,
                            "source_id": identifier,
                            "name": name,
                            "candidate_role": role,
                            "anchor_witnesses": witnesses,
                            "anchor_hashes": _hashes(witnesses, ledger),
                            "first_call_id": call_id,
                        }
                    )
                    key = (owner, record.lead_id)
                    previous = records.get(key)
                    if previous is None:
                        records[key] = record
                    elif previous.anchor_hashes != record.anchor_hashes:
                        records[key] = record
            self._records = records

    @staticmethod
    def _validate_saved(record: _LeadRecord, ledger: EvidenceLedger) -> None:
        if record.lead_id != _lead_id(
            {
                "anchor_source_id": record.anchor_source_id,
                "article_no": record.article_no,
                "qualifier": record.qualifier,
            },
            record.source_id,
        ):
            raise ValueError("Related-source lead identity changed")
        for witness in record.anchor_witnesses:
            item = ledger.get(witness.citation)
            anchor = (
                derive_provision_navigation_anchor(item.source_id, [item])
                if item is not None
                else None
            )
            if (
                item is None
                or record.anchor_hashes.get(str(witness.citation)) != item.text_hash
                or witness.start_char != 0
                or witness.end_char != len(item.text)
                or anchor is None
                or (anchor.source_id, anchor.article_no, anchor.qualifier)
                != (record.anchor_source_id, record.article_no, record.qualifier)
            ):
                raise ValueError("Related-source anchor original changed")
        if set(record.anchor_hashes) != {
            str(w.citation) for w in record.anchor_witnesses
        }:
            raise ValueError("Related-source anchor hash inventory changed")
        if record.review is not None:
            LegalSourceReviews._validate_review(
                record.review, record, ledger, delivered=None
            )
            if record.review_hashes != _hashes(record.review.witnesses, ledger):
                raise ValueError("Related-source review original changed")
        elif record.review_hashes:
            raise ValueError("Related-source hashes have no review")

    @staticmethod
    def _validate_review(
        review: RelatedSourceReview,
        record: _LeadRecord,
        ledger: EvidenceLedger,
        *,
        delivered: set[int] | None,
    ) -> None:
        if (
            review.lead_id != record.lead_id
            or not review.effect.strip()
            or not review.limitations.strip()
        ):
            raise ValueError("A related-source review needs its effect and limitations")
        if review.status == "unresolved":
            if not review.gap.strip():
                raise ValueError(
                    "An unresolved source interaction needs its precise gap"
                )
        else:
            if review.gap.strip() or not review.witnesses:
                raise ValueError("Examined and excluded leads need their own originals")
            if review.status == "examined" and review.source_role != "operative_text":
                raise ValueError("An argument alone cannot close operative examination")
        identities: set[tuple[int, int, int]] = set()
        for witness in review.witnesses:
            identity = (witness.citation, witness.start_char, witness.end_char)
            item = ledger.get(witness.citation)
            if (
                identity in identities
                or item is None
                or item.source_id != record.source_id
                or witness.start_char >= witness.end_char
                or witness.end_char > len(item.text)
                or (delivered is not None and witness.citation not in delivered)
            ):
                raise ValueError(
                    "A review needs fully delivered candidate-source ranges"
                )
            identities.add(identity)

    def _preview(
        self,
        raw: list[JsonValue],
        call_id: str,
        context: RunContext,
        ledger: EvidenceLedger,
    ) -> dict[tuple[str, str], _LeadRecord]:
        if not isinstance(raw, list):
            raise ValueError("Related-source reviews must be a list")
        if not call_id.strip():
            raise ValueError("A related-source assessment needs its actual model call")
        owner = self._owner(context)
        records = dict(self._records)
        seen: set[str] = set()
        for row in raw:
            review = RelatedSourceReview.model_validate(row)
            key = (owner, review.lead_id)
            if review.lead_id in seen or key not in records:
                raise ValueError(
                    "Review only this task's actually delivered leads once"
                )
            seen.add(review.lead_id)
            record = records[key]
            self._validate_saved(record, ledger)
            self._validate_review(
                review, record, ledger, delivered=ledger.completely_delivered(call_id)
            )
            records[key] = record.model_copy(
                deep=True,
                update={
                    "review": review,
                    "review_hashes": _hashes(review.witnesses, ledger),
                },
            )
        return records

    def apply(
        self,
        raw: list[JsonValue],
        call_id: str,
        context: RunContext,
        ledger: EvidenceLedger,
    ) -> None:
        self._fence(context)
        with self._lock:
            self._records = self._preview(raw, call_id, context, ledger)

    def view(
        self,
        context: RunContext,
        ledger: EvidenceLedger,
        delivered_citations: set[int],
    ) -> dict[str, JsonValue]:
        self._fence(context)
        owner = self._owner(context)
        rows: list[JsonValue] = []
        pending: list[JsonValue] = []
        with self._lock:
            for (record_owner, _), record in self._records.items():
                if record_owner != owner:
                    continue
                self._validate_saved(record, ledger)
                originals = [
                    number
                    for number in sorted(delivered_citations)
                    if (item := ledger.get(number)) is not None
                    and item.source_id == record.source_id
                ]
                rows.append(
                    {
                        "lead_id": record.lead_id,
                        "anchor_source_id": record.anchor_source_id,
                        "article_no": record.article_no,
                        "qualifier": record.qualifier,
                        "source_id": record.source_id,
                        "name": record.name,
                        "candidate_role": record.candidate_role,
                        "status": record.review.status if record.review else "pending",
                        "available_original_citations": originals,
                        "review": record.review.model_dump(mode="json")
                        if record.review
                        else None,
                    }
                )
                if record.review is None:
                    pending.append(record.lead_id)
        return {
            "reviews": rows,
            "pending_lead_ids": pending,
            "notice": "Acquired citations are available passages, not examined holdings. These are model assessments with validated provenance, not legal approval.",
        }

    def publication_gap(
        self,
        answer: str,
        call_id: str,
        context: RunContext,
        ledger: EvidenceLedger,
        raw_reviews: list[JsonValue] | None = None,
    ) -> ToolOutcome | None:
        self._fence(context)
        if not extract_citation_numbers(answer):
            return None
        owner = self._owner(context)
        with self._lock:
            records = self._preview(
                [] if raw_reviews is None else raw_reviews, call_id, context, ledger
            )
            pending: list[JsonValue] = []
            undisclosed: list[JsonValue] = []
            paragraphs = {p.strip() for p in re.split(r"\n\s*\n", answer)}
            for (record_owner, _), record in records.items():
                if record_owner != owner:
                    continue
                self._validate_saved(record, ledger)
                if record.review is None:
                    pending.append(
                        {
                            "lead_id": record.lead_id,
                            "source_id": record.source_id,
                            "name": record.name,
                            "anchor_source_id": record.anchor_source_id,
                            "article_no": record.article_no,
                            "qualifier": record.qualifier,
                        }
                    )
                elif record.review.status == "unresolved" and (
                    record.review.gap.strip() not in paragraphs
                    or extract_citation_numbers(record.review.gap)
                ):
                    undisclosed.append(
                        {"lead_id": record.lead_id, "gap": record.review.gap}
                    )
            if pending or undisclosed:
                return ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="Examine the delivered related-source candidates with their own operative originals, or disclose the precise unresolved interaction separately. Available passages and titles do not approve an effect.",
                    data={
                        "pending_related_source_review": True,
                        "unread_related_sources": pending,
                        "undisclosed_related_source_gaps": undisclosed,
                    },
                )
        return None

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return _Checkpoint(
                run_id=self.run_id,
                scope_hash=self.scope_hash,
                request_hash=self.request_hash,
                records=list(self._records.values()),
            ).model_dump(mode="json")

    def restore(
        self,
        snapshot: dict[str, JsonValue],
        context: RunContext,
        request: str,
        ledger: EvidenceLedger,
    ) -> None:
        self._fence(context)
        saved = _Checkpoint.model_validate(snapshot)
        if (saved.run_id, saved.scope_hash, saved.request_hash) != (
            context.run_id,
            _digest(context.scope),
            _digest(request),
        ) or saved.request_hash != self.request_hash:
            raise ValueError("Related-source checkpoint request, run or scope changed")
        records: dict[tuple[str, str], _LeadRecord] = {}
        for record in saved.records:
            self._validate_saved(record, ledger)
            key = (record.owner, record.lead_id)
            if key in records:
                raise ValueError("Duplicate related-source checkpoint records")
            records[key] = record.model_copy(deep=True)
        with self._lock:
            if self._records and self._records != records:
                raise ValueError(
                    "Related-source checkpoint cannot replace live records"
                )
            self._records = records
