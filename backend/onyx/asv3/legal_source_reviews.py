"""Retain explicit source-lead assessments without treating them as legal approval."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Literal, NoReturn

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_navigation import (
    RelatedSourceRole,
    derive_provision_navigation_anchor,
)
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.outcome_map import OutcomeWitness


def serial_session_diagnostics_enabled(context: RunContext) -> bool:
    return (
        context.services.get("serial_session_diagnostics") is True
        and context.services.get("lean_native_mode") is True
        and context.services.get("research_profile") == "experimental"
        and context.services.get("experimental_parallel") is False
        and context.depth == 0
    )


class RelatedSourceReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lead_id: str = Field(pattern=r"^lead_[a-f0-9]{64}$")
    status: Literal["examined", "not_material", "unresolved"]
    source_role: Literal["operative_text", "argument_only", "unknown"]
    effect: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    witnesses: list[OutcomeWitness] = Field(default_factory=list)
    gap: str = Field(
        default="",
        description="Precise open interaction for unresolved only; otherwise empty.",
    )


class RelatedSourceReviewValidationError(ValueError):
    """Carry a provenance repair target without changing the rejection message."""

    def __init__(self, message: str, diagnostic: dict[str, JsonValue]) -> None:
        super().__init__(message)
        self.diagnostic = diagnostic


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
                    else:
                        self._validate_saved(previous, ledger)
                        previous_citations = {
                            witness.citation for witness in previous.anchor_witnesses
                        }
                        records[key] = previous.model_copy(
                            deep=True,
                            update={
                                "anchor_witnesses": [
                                    *previous.anchor_witnesses,
                                    *(
                                        witness
                                        for witness in record.anchor_witnesses
                                        if witness.citation not in previous_citations
                                    ),
                                ],
                                "anchor_hashes": {
                                    **previous.anchor_hashes,
                                    **record.anchor_hashes,
                                },
                            },
                        )
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
        diagnostic_index: int | None = None,
    ) -> None:
        def reject(
            message: str,
            code: str,
            field: str,
            witness_index: int | None = None,
        ) -> NoReturn:
            if diagnostic_index is None:
                raise ValueError(message)
            available: list[JsonValue] = [
                {
                    "citation": citation,
                    "start_char": 0,
                    "end_char": len(item.text),
                }
                for citation in sorted(delivered or set())
                if (item := ledger.get(citation)) is not None
                and item.source_id == record.source_id
            ]
            diagnostic: dict[str, JsonValue] = {
                "review_index": diagnostic_index,
                "lead_id": record.lead_id,
                "source_id": record.source_id,
                "status": review.status,
                "code": code,
                "field": f"_related_source_reviews[{diagnostic_index}].{field}",
                "available_original_witnesses": available,
                "instruction": (
                    "Correct only this assessment metadata. Its witnesses must belong "
                    "to the candidate source and be fully delivered to this decision. "
                    "Keep other supporting sources and the answer's supported detail; "
                    "they may remain answer citations but cannot be this candidate's "
                    "own witnesses. Available ranges identify passages, not legal approval."
                ),
            }
            if witness_index is not None:
                witness = review.witnesses[witness_index]
                diagnostic["witness_index"] = witness_index
                diagnostic["witness"] = witness.model_dump(mode="json")
                if (item := ledger.get(witness.citation)) is not None:
                    diagnostic["actual_source_id"] = item.source_id
            if code == "missing_own_originals" or code == "not_fully_delivered":
                diagnostic["instruction"] = (
                    "Use this candidate's actually supporting, fully delivered original "
                    "ranges if available. If the necessary original has not reached this "
                    "decision, obtain or deliver it, or report the precise unresolved "
                    "interaction. Do not mark an unread candidate not_material. Preserve "
                    "the answer's independently supported detail."
                )
            elif code in {"missing_unresolved_gap", "closed_review_has_gap"}:
                diagnostic["instruction"] = (
                    "Correct only this status/gap metadata to reflect the actual examined "
                    "evidence. An unresolved interaction needs its precise gap; an examined "
                    "or not_material review leaves gap empty. Do not change status merely "
                    "to pass validation, or rewrite supported answer detail."
                )
            raise RelatedSourceReviewValidationError(message, diagnostic)

        if (
            review.lead_id != record.lead_id
            or not review.effect.strip()
            or not review.limitations.strip()
        ):
            field = (
                "lead_id"
                if review.lead_id != record.lead_id
                else "effect"
                if not review.effect.strip()
                else "limitations"
            )
            reject(
                "A related-source review needs its effect and limitations",
                "missing_assessment_field",
                field,
            )
        if review.status == "unresolved":
            if not review.gap.strip():
                reject(
                    "An unresolved source interaction needs its precise gap",
                    "missing_unresolved_gap",
                    "gap",
                )
        else:
            if review.gap.strip():
                reject(
                    "Examined and not-material reviews must leave gap empty; "
                    "use unresolved for an open interaction",
                    "closed_review_has_gap",
                    "gap",
                )
            if not review.witnesses:
                reject(
                    "Examined and excluded leads need their own originals",
                    "missing_own_originals",
                    "witnesses",
                )
            if review.status == "examined" and review.source_role != "operative_text":
                reject(
                    "An argument alone cannot close operative examination",
                    "argument_cannot_close_operative",
                    "source_role",
                )
        identities: set[tuple[int, int, int]] = set()
        for witness_index, witness in enumerate(review.witnesses):
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
                code = (
                    "duplicate_witness"
                    if identity in identities
                    else "unknown_citation"
                    if item is None
                    else "wrong_source"
                    if item.source_id != record.source_id
                    else "invalid_range"
                    if witness.start_char >= witness.end_char
                    or witness.end_char > len(item.text)
                    else "not_fully_delivered"
                )
                reject(
                    "A review needs fully delivered candidate-source ranges",
                    code,
                    f"witnesses[{witness_index}]",
                    witness_index,
                )
            identities.add(identity)

    def _preview(
        self,
        raw: list[JsonValue],
        call_id: str,
        context: RunContext,
        ledger: EvidenceLedger,
        *,
        detailed_errors: bool = False,
    ) -> dict[tuple[str, str], _LeadRecord]:
        if not isinstance(raw, list):
            raise ValueError("Related-source reviews must be a list")
        if not call_id.strip():
            raise ValueError("A related-source assessment needs its actual model call")
        owner = self._owner(context)
        records = dict(self._records)
        seen: set[str] = set()
        for review_index, row in enumerate(raw):
            review = RelatedSourceReview.model_validate(row)
            key = (owner, review.lead_id)
            if review.lead_id in seen or key not in records:
                message = "Review only this task's actually delivered leads once"
                if detailed_errors:
                    raise RelatedSourceReviewValidationError(
                        message,
                        {
                            "review_index": review_index,
                            "lead_id": review.lead_id,
                            "status": review.status,
                            "code": "duplicate_or_foreign_lead",
                            "field": f"_related_source_reviews[{review_index}].lead_id",
                            "instruction": "Assess each of this task's actually delivered leads once; do not use another task's lead inventory or rewrite the answer.",
                        },
                    )
                raise ValueError(message)
            seen.add(review.lead_id)
            record = records[key]
            self._validate_saved(record, ledger)
            self._validate_review(
                review,
                record,
                ledger,
                delivered=ledger.completely_delivered(call_id),
                diagnostic_index=review_index if detailed_errors else None,
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
        *,
        detailed_errors: bool = False,
    ) -> None:
        self._fence(context)
        with self._lock:
            self._records = self._preview(
                raw,
                call_id,
                context,
                ledger,
                detailed_errors=detailed_errors
                and context.services.get("research_profile") == "experimental"
                and (
                    context.services.get("experimental_parallel") is True
                    or serial_session_diagnostics_enabled(context)
                ),
            )

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

    def publication_gap_for_assembly(
        self,
        answer: str,
        context: RunContext,
        ledger: EvidenceLedger,
        *,
        accepted_owners: set[str],
    ) -> ToolOutcome | None:
        """Check root leads against already validated, immutable child bodies."""
        self._fence(context)
        if self._owner(context) != "coordinator" or any(
            not owner.strip() or owner == "coordinator" for owner in accepted_owners
        ):
            raise ValueError("Assembly reviews require accepted child owners")
        if not extract_citation_numbers(answer):
            return None
        paragraphs = {p.strip() for p in re.split(r"\n\s*\n", answer)}

        def disclosed(review: RelatedSourceReview) -> bool:
            return (
                review.status == "unresolved"
                and review.gap.strip() in paragraphs
                and not extract_citation_numbers(review.gap)
            )

        pending: list[JsonValue] = []
        undisclosed: list[JsonValue] = []
        with self._lock:
            for (owner, lead), record in self._records.items():
                if owner != "coordinator":
                    continue
                self._validate_saved(record, ledger)
                if record.review is not None and (
                    record.review.status != "unresolved" or disclosed(record.review)
                ):
                    continue
                reviews: dict[str, RelatedSourceReview] = {}
                for child_owner in accepted_owners:
                    child = self._records.get((child_owner, lead))
                    if child is None:
                        continue
                    self._validate_saved(child, ledger)
                    if (
                        child.anchor_source_id,
                        child.article_no,
                        child.qualifier,
                        child.source_id,
                    ) != (
                        record.anchor_source_id,
                        record.article_no,
                        record.qualifier,
                        record.source_id,
                    ):
                        raise ValueError("Assembly related-source identity changed")
                    if child.review is not None:
                        reviews[child_owner] = child.review
                if any(
                    review.status == "examined"
                    and review.source_role == "operative_text"
                    for review in reviews.values()
                ) or any(disclosed(review) for review in reviews.values()):
                    continue
                if (
                    accepted_owners
                    and set(reviews) == accepted_owners
                    and all(
                        review.status == "not_material" for review in reviews.values()
                    )
                ):
                    continue
                if record.review is not None:
                    undisclosed.append({"lead_id": lead, "gap": record.review.gap})
                else:
                    pending.append(
                        {
                            "lead_id": lead,
                            "source_id": record.source_id,
                            "name": record.name,
                            "anchor_source_id": record.anchor_source_id,
                            "article_no": record.article_no,
                            "qualifier": record.qualifier,
                        }
                    )
        if not pending and not undisclosed:
            return None
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="The accepted child bodies must retain an examined related original or its precise unresolved interaction. One task's exclusion cannot resolve the whole request.",
            data={
                "pending_related_source_review": True,
                "unread_related_sources": pending,
                "undisclosed_related_source_gaps": undisclosed,
            },
        )

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
