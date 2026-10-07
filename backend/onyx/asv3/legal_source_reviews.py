"""Retain explicit source-lead assessments without treating them as legal approval."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import threading
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.assertions import assertion_inventory, presentation_block
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.judicial_sections import (
    JudicialSectionRole,
    canonical_disposition_witness,
    judicial_disposition_gap,
    judicial_disposition_missing,
    nonoperative_judicial_witness_role,
)
from onyx.asv3.legal_source_navigation import (
    RelatedSourceRole,
    derive_provision_navigation_anchor,
)
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.outcome_map import OutcomeWitness
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT


def related_source_reviews_enabled(context: RunContext) -> bool:
    return (
        context.services.get("research_profile") == "experimental"
        or context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
    )


def serial_session_diagnostics_enabled(context: RunContext) -> bool:
    return (
        context.services.get("serial_session_diagnostics") is True
        and context.services.get("lean_native_mode") is True
        and context.services.get("research_profile") == "experimental"
        and context.services.get("experimental_parallel") is False
        and context.depth == 0
    )


def operative_review_retention_enabled(context: RunContext) -> bool:
    if context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT:
        return True
    return context.services.get("research_profile") == "experimental" and (
        context.services.get("experimental_parallel") is True
        or (
            serial_session_diagnostics_enabled(context)
            and isinstance(owner := context.services.get("task_id"), str)
            and bool(owner.strip())
        )
    )


_CITATION_MARKER = re.compile(r"[\[【［]{1,2}\d+(?:, ?\d+)*[\]】］]{1,2}")


def _substantive_inline_citations(text: str) -> set[int]:
    return {
        citation
        for line in text.splitlines()
        if not presentation_block(line)
        and any(character.isalnum() for character in _CITATION_MARKER.sub("", line))
        for citation in extract_citation_numbers(line)
    }


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


def _combined_validation_error(
    errors: list[RelatedSourceReviewValidationError],
) -> RelatedSourceReviewValidationError:
    """Keep the first repair target and share repeated candidate inventories."""
    first = errors[0]
    diagnostic = copy.deepcopy(first.diagnostic)
    inventories: list[JsonValue] = []
    instructions: list[JsonValue] = []
    inventory_refs: dict[tuple[str, str], str] = {}
    instruction_refs: dict[str, str] = {}

    def inventory_key(details: dict[str, JsonValue]) -> tuple[str, str]:
        return str(details["source_id"]), json.dumps(
            details["available_original_witnesses"], sort_keys=True, ensure_ascii=False
        )

    initial = first.diagnostic
    if "available_original_witnesses" in initial:
        inventory_refs[inventory_key(initial)] = "available_original_witnesses"
    if isinstance(instruction := initial.get("instruction"), str):
        instruction_refs[instruction] = "instruction"
    rows: list[JsonValue] = []
    for error in errors:
        details = error.diagnostic
        row = {
            key: value
            for key, value in details.items()
            if key not in {"available_original_witnesses", "instruction"}
        }
        if "available_original_witnesses" in details:
            key = inventory_key(details)
            if key not in inventory_refs:
                inventory_refs[key] = (
                    f"additional_original_witnesses[{len(inventories)}]"
                )
                inventories.append(
                    {
                        "source_id": details["source_id"],
                        "available_original_witnesses": details[
                            "available_original_witnesses"
                        ],
                    }
                )
            row["available_original_witnesses_ref"] = inventory_refs[key]
        if isinstance(instruction := details.get("instruction"), str):
            if instruction not in instruction_refs:
                instruction_refs[instruction] = (
                    f"additional_instructions[{len(instructions)}]"
                )
                instructions.append(instruction)
            row["instruction_ref"] = instruction_refs[instruction]
        rows.append(row)
    diagnostic["validation_errors"] = rows
    if inventories:
        diagnostic["additional_original_witnesses"] = inventories
    if instructions:
        diagnostic["additional_instructions"] = instructions
    return RelatedSourceReviewValidationError(str(first), diagnostic)


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


def _only_nonoperative_judicial_witnesses(
    record: _LeadRecord, review: RelatedSourceReview, ledger: EvidenceLedger
) -> bool:
    return (
        record.candidate_role == "judicial_candidate"
        and review.status in {"examined", "not_material"}
        and bool(review.witnesses)
        and all(
            (item := ledger.get(witness.citation)) is not None
            and _judicial_witness_role(
                item, witness.start_char, witness.end_char, ledger
            )
            in {"preliminary", "argument_only"}
            for witness in review.witnesses
        )
    )


def _judicial_witness_role(
    item: EvidenceItem, start_char: int, end_char: int, ledger: EvidenceLedger
) -> JudicialSectionRole:
    originals: list[EvidenceItem] = []
    for number, doc in ledger.citation_mapping().items():
        if doc.document_id == item.source_id:
            original = ledger.get(number)
            if original is not None:
                originals.append(original)
    return nonoperative_judicial_witness_role(
        item, start_char, end_char, source_context=originals
    )


def _argument_role_contains_disposition(
    record: _LeadRecord, review: RelatedSourceReview, ledger: EvidenceLedger
) -> bool:
    if (
        record.candidate_role != "judicial_candidate"
        or review.status not in {"examined", "not_material"}
        or review.source_role != "argument_only"
    ):
        return False
    originals = [
        original
        for number, doc in ledger.citation_mapping().items()
        if doc.document_id == record.source_id
        for original in (ledger.get(number),)
        if original is not None
    ]
    return any(
        (original := ledger.get(witness.citation)) is not None
        and canonical_disposition_witness(
            original,
            witness.start_char,
            witness.end_char,
            source_context=originals,
        )
        for witness in review.witnesses
    )


def _missing_judicial_disposition(
    record: _LeadRecord,
    review: RelatedSourceReview | None,
    ledger: EvidenceLedger,
    delivered: set[int] | None = None,
) -> bool:
    if (
        record.candidate_role != "judicial_candidate"
        or review is None
        or review.status not in {"examined", "not_material"}
        or review.source_role != "operative_text"
    ):
        return False
    originals = [
        item
        for number, doc in ledger.citation_mapping().items()
        if doc.document_id == record.source_id
        and (delivered is None or number in delivered)
        for item in (ledger.get(number),)
        if item is not None
    ]
    return (
        judicial_disposition_missing(originals)
        or judicial_disposition_gap(originals) is not None
    )


def _retention_text(value: str) -> str:
    """Normalize presentation only; validate inline witnesses on the original block."""
    value = re.sub(
        r"(?<![\w*])\*\*(?=\S)(.+?)(?<=\S)\*\*(?![\w*])",
        r"\1",
        value,
    )
    value = re.sub(r"(?:\s*" + _CITATION_MARKER.pattern + r")+(?=[.,;:!?])", "", value)
    value = _CITATION_MARKER.sub("", value)
    return " ".join(
        value.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})).split()
    )


def _examined_answer_omission(
    record: _LeadRecord,
    answer: str,
    *,
    delivered: set[int] | None,
    include_repair_text: bool = False,
    include_exclusions: bool = False,
    normalize_presentation: bool = False,
) -> dict[str, JsonValue] | None:
    """Retain copied passages and their provenance, without assessing legal entailment."""
    review = record.review
    if (
        review is None
        or review.status
        not in ({"examined", "not_material"} if include_exclusions else {"examined"})
        or review.source_role != "operative_text"
    ):
        return None
    numbers = {witness.citation for witness in review.witnesses}
    units = [
        unit for unit in assertion_inventory(answer) if not unit["presentation_only"]
    ]
    omitted: list[JsonValue] = []
    bound: set[int] = set()
    normalize = _retention_text if normalize_presentation else str.strip
    for field, passage in (
        ("effect", review.effect),
        ("limitations", review.limitations),
    ):
        exact = normalize(passage)
        if (
            presentation_block(exact)
            or not any(
                character.isalnum() for character in _CITATION_MARKER.sub("", exact)
            )
            or not exact
        ):
            omitted.append(field)
            continue
        matching = [
            numbers & _substantive_inline_citations(unit["text"])
            for unit in units
            if exact in normalize(unit["text"])
        ]
        if not any(matching):
            omitted.append(field)
        for citations in matching:
            bound.update(citations)
    undelivered = sorted(numbers - delivered) if delivered is not None else []
    unbound = sorted(numbers - bound)
    if not omitted and not undelivered and not unbound:
        return None
    return {
        "lead_id": record.lead_id,
        "source_id": record.source_id,
        **({"assessment_status": review.status} if include_exclusions else {}),
        "missing_answer_passages": omitted,
        "required_inline_citations": sorted(numbers),
        "undelivered_evidence_numbers": undelivered,
        **({"unbound_evidence_numbers": unbound} if unbound else {}),
        **(
            {
                "required_answer_passages": {
                    field: getattr(review, field)
                    for field in ("effect", "limitations")
                    if field in omitted
                },
                "repair_instruction": (
                    "Retain these exact declared passages with their original citations, "
                    "or correct the review and answer together from the delivered originals. "
                    "This is text retention, not legal approval; do not reread supplied text."
                ),
            }
            if include_repair_text
            else {}
        ),
    }


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
        errors: list[RelatedSourceReviewValidationError] | None = None,
        check_judicial_sections: bool = False,
    ) -> None:
        available: list[JsonValue] | None = None

        def reject(
            message: str,
            code: str,
            field: str,
            witness_index: int | None = None,
        ) -> None:
            nonlocal available
            if diagnostic_index is None:
                raise ValueError(message)
            if available is None:
                available = [
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
            elif code in {
                "nonoperative_judicial_witnesses",
                "missing_judicial_disposition",
            }:
                diagnostic["instruction"] = (
                    "The candidate's connected disposition is incomplete. Locate and read "
                    "this source's actual disposition and relevant "
                    "qualifications through source-local search or headings; preserve already "
                    "supported answer detail. Do not relabel or reread the same introduction, "
                    "scan unrelated sources, or claim its operative effect from a title."
                )
                originals = [
                    item
                    for citation in sorted(delivered or set())
                    if (item := ledger.get(citation)) is not None
                    and item.source_id == record.source_id
                ]
                gap = judicial_disposition_gap(originals)
                if gap is not None:
                    diagnostic["suggested_acquisition"] = {
                        "name": "read_source_range",
                        "arguments": {
                            "source_id": record.source_id,
                            "start": gap,
                            "limit": 2,
                        },
                    }
            if (
                check_judicial_sections
                and not available
                and code
                in {
                    "wrong_source",
                    "missing_own_originals",
                    "not_fully_delivered",
                    "argument_cannot_close_operative",
                }
            ):
                retained = [
                    citation
                    for citation, doc in ledger.citation_mapping().items()
                    if doc.document_id == record.source_id
                ]
                diagnostic["suggested_acquisition"] = (
                    {"name": "read_evidence", "arguments": {"citation": retained[0]}}
                    if retained
                    else {
                        "name": "read_source_range",
                        "arguments": {"source_id": record.source_id, "start": 0},
                    }
                )
                diagnostic["instruction"] = (
                    "No own original for this candidate was delivered. Metadata edits cannot "
                    "supply it. Reopen its retained original or read this exact source, then "
                    "locate the operative section and qualifications before assessment. Do not "
                    "reuse a different source's citation or repeat the anchor statute search. "
                    "The suggested acquisition is a navigation lead, not legal approval; "
                    "follow any required continuation. Preserve the supported answer detail."
                )
            error = RelatedSourceReviewValidationError(message, diagnostic)
            if errors is not None:
                errors.append(error)
                return
            raise error

        for field, missing in (
            ("lead_id", review.lead_id != record.lead_id),
            ("effect", not review.effect.strip()),
            ("limitations", not review.limitations.strip()),
        ):
            if missing:
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
        if check_judicial_sections and _only_nonoperative_judicial_witnesses(
            record, review, ledger
        ):
            reject(
                "The candidate's supplied witnesses contain only a recognized application "
                "subject, procedural introduction or party argument. Read its own disposition "
                "and connected qualifications using source-local search or headings, or "
                "disclose that precise unresolved effect; do not reread the introduction.",
                "nonoperative_judicial_witnesses",
                "witnesses",
            )
        if check_judicial_sections and _argument_role_contains_disposition(
            record, review, ledger
        ):
            reject(
                "The selected original witness is in the judicial disposition body, "
                "not merely a party argument. Assess that operative text and its connected "
                "qualifications separately from applicability; a reasoned not-material "
                "assessment may still use operative_text. Do not reread delivered originals.",
                "disposition_cannot_be_argument",
                "source_role",
            )
        if check_judicial_sections and _missing_judicial_disposition(
            record, review, ledger, delivered
        ):
            reject(
                "The supplied candidate includes judicial reasoning but lacks a complete "
                "connected disposition body. Read that source's actual continuation or locate its "
                "disposition before claiming the effect; a heading is not its operative text.",
                "missing_judicial_disposition",
                "witnesses",
            )

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
        errors: list[RelatedSourceReviewValidationError] = []
        delivered = ledger.completely_delivered(call_id)
        for review_index, row in enumerate(raw):
            review = RelatedSourceReview.model_validate(row)
            key = (owner, review.lead_id)
            if review.lead_id in seen or key not in records:
                message = "Review only this task's actually delivered leads once"
                if detailed_errors:
                    errors.append(
                        RelatedSourceReviewValidationError(
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
                    )
                    continue
                raise ValueError(message)
            seen.add(review.lead_id)
            record = records[key]
            self._validate_saved(record, ledger)
            previous_errors = len(errors)
            self._validate_review(
                review,
                record,
                ledger,
                delivered=delivered,
                diagnostic_index=review_index if detailed_errors else None,
                errors=errors if detailed_errors else None,
                check_judicial_sections=context.services.get("asv3_workflow_variant")
                == ASV3_TUNED_VARIANT,
            )
            if len(errors) != previous_errors:
                continue
            records[key] = record.model_copy(
                deep=True,
                update={
                    "review": review,
                    "review_hashes": _hashes(review.witnesses, ledger),
                },
            )
        if errors:
            raise _combined_validation_error(errors)
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
                and (
                    context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
                    or (
                        context.services.get("research_profile") == "experimental"
                        and (
                            context.services.get("experimental_parallel") is True
                            or serial_session_diagnostics_enabled(context)
                        )
                    )
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
                missing_disposition = context.services.get(
                    "asv3_workflow_variant"
                ) == ASV3_TUNED_VARIANT and (
                    _missing_judicial_disposition(
                        record, record.review, ledger, delivered_citations
                    )
                    or (
                        record.review is not None
                        and _argument_role_contains_disposition(
                            record, record.review, ledger
                        )
                    )
                )
                originals = [
                    number
                    for number in sorted(delivered_citations)
                    if (item := ledger.get(number)) is not None
                    and item.source_id == record.source_id
                ]
                disposition_gap = (
                    judicial_disposition_gap(
                        [
                            item
                            for number in originals
                            if (item := ledger.get(number)) is not None
                        ]
                    )
                    if context.services.get("asv3_workflow_variant")
                    == ASV3_TUNED_VARIANT
                    and record.candidate_role == "judicial_candidate"
                    else None
                )
                rows.append(
                    {
                        "lead_id": record.lead_id,
                        "anchor_source_id": record.anchor_source_id,
                        "article_no": record.article_no,
                        "qualifier": record.qualifier,
                        "source_id": record.source_id,
                        "name": record.name,
                        "candidate_role": record.candidate_role,
                        "status": record.review.status
                        if record.review and not missing_disposition
                        else "pending",
                        "available_original_citations": originals,
                        **(
                            {"disposition_gap_position": disposition_gap}
                            if disposition_gap is not None
                            else {}
                        ),
                        **(
                            {
                                "nonoperative_original_citations": [
                                    number
                                    for number in originals
                                    if (item := ledger.get(number)) is not None
                                    and _judicial_witness_role(
                                        item, 0, len(item.text), ledger
                                    )
                                    in {"preliminary", "argument_only"}
                                ]
                            }
                            if context.services.get("asv3_workflow_variant")
                            == ASV3_TUNED_VARIANT
                            and record.candidate_role == "judicial_candidate"
                            else {}
                        ),
                        **(
                            {
                                "anchor_evidence_numbers": [
                                    witness.citation
                                    for witness in record.anchor_witnesses
                                ]
                            }
                            if context.services.get("asv3_workflow_variant")
                            == ASV3_TUNED_VARIANT
                            else {}
                        ),
                        "review": record.review.model_dump(mode="json")
                        if record.review and not missing_disposition
                        else None,
                    }
                )
                if record.review is None or missing_disposition:
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
        tuned = context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
        with self._lock:
            records = self._preview(
                [] if raw_reviews is None else raw_reviews, call_id, context, ledger
            )
            pending: list[JsonValue] = []
            undisclosed: list[JsonValue] = []
            unretained: list[JsonValue] = []
            retain_examined = operative_review_retention_enabled(context)
            delivered = (
                ledger.completely_delivered(call_id) if retain_examined else None
            )
            paragraphs = {p.strip() for p in re.split(r"\n\s*\n", answer)}
            for (record_owner, _), record in records.items():
                if record_owner != owner:
                    continue
                self._validate_saved(record, ledger)
                if (
                    tuned
                    and record.review is not None
                    and (
                        _only_nonoperative_judicial_witnesses(
                            record, record.review, ledger
                        )
                        or _argument_role_contains_disposition(
                            record, record.review, ledger
                        )
                        or _missing_judicial_disposition(
                            record, record.review, ledger, delivered
                        )
                    )
                ):
                    return ToolOutcome(
                        status=OutcomeStatus.PARTIAL,
                        summary="The saved assessment lacks connected disposition text or misclassifies its operative body. Reuse delivered originals, complete the actual connected gap and correct the source role; otherwise disclose the precise unresolved effect.",
                        data={
                            "pending_related_source_review": True,
                            "nonoperative_judicial_source_id": record.source_id,
                            "lead_id": record.lead_id,
                        },
                    )
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
                elif retain_examined:
                    omission = _examined_answer_omission(
                        record,
                        answer,
                        delivered=delivered,
                        include_repair_text=tuned,
                        include_exclusions=tuned,
                        normalize_presentation=tuned,
                    )
                    if omission is not None:
                        unretained.append(omission)
            if pending or undisclosed or unretained:
                return ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary=(
                        "Use the examined candidate's operative effect and limitations in substantive answer blocks with its own original citations; do not acquire the same passages again."
                        if tuned and unretained and not pending and not undisclosed
                        else "Retain the declared operative effect and limitations in substantive answer blocks with their own witness citations. This checks copied passage retention, not legal entailment."
                        if unretained and not pending and not undisclosed
                        else "Examine the delivered related-source candidates with their own operative originals, or disclose the precise unresolved interaction separately. Available passages and titles do not approve an effect."
                    ),
                    data={
                        "pending_related_source_review": True,
                        "unread_related_sources": pending,
                        "undisclosed_related_source_gaps": undisclosed,
                        **(
                            {"unretained_examined_source_effects": unretained}
                            if unretained
                            else {}
                        ),
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
        unretained: list[JsonValue] = []
        retain_examined = operative_review_retention_enabled(context)
        tuned = context.services.get("asv3_workflow_variant") == ASV3_TUNED_VARIANT
        with self._lock:
            for (owner, lead), record in self._records.items():
                if owner != "coordinator":
                    continue
                self._validate_saved(record, ledger)
                if record.review is not None and (
                    record.review.status != "unresolved" or disclosed(record.review)
                ):
                    if retain_examined:
                        omission = _examined_answer_omission(
                            record,
                            answer,
                            delivered=None,
                            include_exclusions=tuned,
                            normalize_presentation=tuned,
                        )
                        if omission is not None:
                            unretained.append(omission)
                    continue
                reviews: dict[str, RelatedSourceReview] = {}
                child_records: dict[str, _LeadRecord] = {}
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
                        child_records[child_owner] = child
                if retain_examined:
                    child_omissions = [
                        {**omission, "owner": child_owner}
                        for child_owner, child_record in child_records.items()
                        if (
                            omission := _examined_answer_omission(
                                child_record,
                                answer,
                                delivered=None,
                                include_exclusions=tuned,
                                normalize_presentation=tuned,
                            )
                        )
                        is not None
                    ]
                    unretained.extend(child_omissions)
                    if child_omissions:
                        continue
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
        if not pending and not undisclosed and not unretained:
            return None
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="The accepted child bodies must retain an examined related original or its precise unresolved interaction. One task's exclusion cannot resolve the whole request.",
            data={
                "pending_related_source_review": True,
                "unread_related_sources": pending,
                "undisclosed_related_source_gaps": undisclosed,
                **(
                    {"unretained_examined_source_effects": unretained}
                    if unretained
                    else {}
                ),
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
