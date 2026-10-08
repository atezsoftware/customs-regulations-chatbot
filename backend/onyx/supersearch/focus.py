"""Grounded expansion decisions preserve every raw original for independent review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, cast

from pydantic import JsonValue

from onyx.asv3.authority import _named_native_references, explicit_reference_leads
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_navigation import derive_provision_navigation_anchor
from onyx.asv3.models import EvidenceItem, RunStopped, model_evidence_metadata
from onyx.legal_composite.dependencies import _entity_key, _literal_names
from onyx.legal_composite.models import PassageSupport, ResearchPlan
from onyx.regulatory.heading_path import parse_regulatory_article_heading
from onyx.supersearch.models import FocusAnswerReview, NeedFocusDecision


def provision_article(item: EvidenceItem) -> str | None:
    metadata = model_evidence_metadata(item.metadata)
    headings = metadata.get("heading_path")
    if isinstance(headings, list):
        for heading in headings:
            parsed = (
                parse_regulatory_article_heading(heading)
                if isinstance(heading, str)
                else None
            )
            if parsed is not None:
                return " ".join(
                    part for part in (parsed.qualifier, parsed.article_no) if part
                )
    return (
        str(metadata["article_no"]) if metadata.get("article_no") is not None else None
    )


def _owned_original(item: EvidenceItem) -> bool:
    doc = item.search_doc
    metadata = model_evidence_metadata(item.metadata)
    return bool(
        item.chunk_id
        and doc is not None
        and doc.document_id == item.source_id
        and doc.metadata.get("regulatory_chunk_id") == item.chunk_id
        and not any(
            metadata.get(flag) or item.metadata.get(flag)
            for flag in ("derived", "external", "untrusted", "truncated")
        )
    )


@dataclass(frozen=True)
class FocusSubject:
    subject_id: str
    kind: Literal["provision", "reference"]
    citations: tuple[int, ...]
    complete: bool
    source_id: str
    article: str | None
    edge_key: str | None = None
    instrument_name: str | None = None
    identity_key: str = ""

    def payload(self) -> dict[str, JsonValue]:
        return {
            "subject_id": self.subject_id,
            "kind": self.kind,
            "citations": list(self.citations),
            "complete_provision": self.complete,
            "source_id": self.source_id,
            "article": self.article,
            "reference_edge_key": self.edge_key,
            "instrument_name": self.instrument_name,
        }


def focus_subjects(ledger: EvidenceLedger, frontier: set[int]) -> list[FocusSubject]:
    groups: dict[tuple[str, str | None, str | None], list[int]] = {}
    for number in sorted(frontier):
        item = ledger.get(number)
        if item is None:
            continue
        article = provision_article(item)
        key = (item.source_id, article, item.chunk_id if article is None else None)
        groups.setdefault(key, []).append(number)
    subjects: list[FocusSubject] = []
    units: dict[int, FocusSubject] = {}
    for (source_id, article, chunk_id), numbers in groups.items():
        items = [ledger.get(number) for number in numbers]
        complete = article is not None and all(
            item is not None
            and _owned_original(item)
            and model_evidence_metadata(item.metadata).get("article_closure_complete")
            is True
            and str(model_evidence_metadata(item.metadata).get("document_type", ""))
            not in {"judicial_decision", "court_decision", "judgment"}
            for item in items
        )
        identity_key = sha256(
            json.dumps([source_id, article, chunk_id]).encode()
        ).hexdigest()
        subject = FocusSubject(
            "provision-" + identity_key[:12],
            "provision",
            tuple(numbers),
            complete,
            source_id,
            article,
            identity_key=identity_key,
        )
        subjects.append(subject)
        units.update({number: subject for number in numbers})
    records = cast(
        list[dict[str, JsonValue]],
        json.loads(ledger.serialize_records(frontier, max_chars=None)),
    )
    leads = explicit_reference_leads(ledger, records, syntactic_reference_binding=True)
    for number in sorted(frontier):
        item = ledger.get(number)
        if item is None or not _owned_original(item):
            continue
        aliases = _literal_names(item.text)
        for reference, name in _named_native_references(
            item.text,
            {alias: set() for alias in aliases},
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
            unicode_ordinals=True,
        ):
            if reference.article is not None and name in aliases:
                leads.append(
                    {
                        "formal_name": aliases[name],
                        "instrument_number": reference.number or None,
                        "article": reference.article,
                        "qualifier": reference.qualifier,
                        "origin_citations": [number],
                    }
                )
        anchor = derive_provision_navigation_anchor(item.source_id, [item])
        if anchor is not None:
            leads.append(
                {
                    "formal_name": anchor.instrument_name,
                    "instrument_number": anchor.instrument_number,
                    "article": anchor.article_no,
                    "qualifier": anchor.qualifier,
                    "origin_citations": [number],
                }
            )
    known: dict[str, str] = {
        subject.subject_id: subject.identity_key for subject in subjects
    }
    if len(known) != len(subjects):
        raise RunStopped("Expansion focus subject identities collided")
    for lead in leads:
        name, article = lead.get("formal_name"), lead.get("article")
        if not isinstance(name, str) or not isinstance(article, str):
            continue
        number, qualifier = lead.get("instrument_number"), lead.get("qualifier")
        edge_key = _entity_key(
            name,
            number if isinstance(number, str) else None,
            article,
            qualifier if isinstance(qualifier, str) else None,
        )
        origins = lead.get("origin_citations", [])
        if not isinstance(origins, list):
            continue
        for citation in origins:
            if type(citation) is not int or citation not in units:
                continue
            unit = units[citation]
            identity_key = f"{citation}:{edge_key}"
            subject_id = f"reference-{citation}-{edge_key[:12]}"
            if subject_id in known and known[subject_id] != identity_key:
                raise RunStopped("Expansion focus reference identities collided")
            if subject_id not in known:
                subjects.append(
                    FocusSubject(
                        subject_id,
                        "reference",
                        unit.citations,
                        unit.complete,
                        unit.source_id,
                        article,
                        edge_key,
                        name,
                        identity_key,
                    )
                )
                known[subject_id] = identity_key
    return subjects


class FocusSelection:
    def __init__(self, ledger: EvidenceLedger) -> None:
        self.ledger = ledger
        self.subjects: dict[str, FocusSubject] = {}
        self.assessments: dict[tuple[str, str], dict[str, JsonValue]] = {}
        self._protected: set[int] = set()

    def protect(self, citations: set[int]) -> None:
        self._protected.update(citations)

    def covers(self, citation: int) -> bool:
        return any(citation in subject.citations for subject in self.subjects.values())

    def _witnesses_valid(
        self,
        subject: FocusSubject,
        witnesses: list[PassageSupport],
        delivered: set[int],
    ) -> bool:
        return bool(witnesses) and all(
            witness.citation in subject.citations
            and witness.citation in delivered
            and (item := self.ledger.get(witness.citation)) is not None
            and _owned_original(item)
            and bool(witness.quotation.strip())
            and witness.quotation in item.text
            for witness in witnesses
        )

    def apply(
        self,
        subjects: list[FocusSubject],
        plan: ResearchPlan,
        decision: NeedFocusDecision,
        delivered: set[int],
        protected: set[int],
    ) -> None:
        self.protect(protected)
        rows: dict[tuple[str, str], list[dict[str, JsonValue]]] = {}
        for assessment in decision.assessments:
            rows.setdefault((assessment.subject_id, assessment.need_id), []).append(
                assessment.model_dump(mode="json")
            )
        for subject in subjects:
            previous = self.subjects.get(subject.subject_id)
            if previous is not None and previous.identity_key != subject.identity_key:
                raise RunStopped("Expansion focus subject identity changed")
            self.subjects[subject.subject_id] = subject
            for need in plan.needs:
                key = subject.subject_id, need.need_id
                choices = rows.get(key, [])
                row: dict[str, JsonValue] = {
                    "subject_id": subject.subject_id,
                    "need_id": need.need_id,
                    "status": "pending",
                    "explanation": "No unique grounded relevance decision was delivered; retain this expansion lead.",
                    "witnesses": [],
                }
                if len(choices) == 1:
                    candidate = choices[0]
                    witnesses = [
                        PassageSupport.model_validate(value)
                        for value in cast(list[JsonValue], candidate["witnesses"])
                    ]
                    if candidate["status"] != "incidental" or (
                        subject.complete
                        and self._witnesses_valid(subject, witnesses, delivered)
                    ):
                        row = candidate
                    else:
                        row["explanation"] = (
                            "The proposed exclusion lacked a complete source-owned provision or exact delivered witnesses; retain the lead."
                        )
                self.assessments[key] = row

    def _incidental(self, subject: FocusSubject, need_id: str) -> bool:
        return (
            self.assessments.get((subject.subject_id, need_id), {}).get("status")
            == "incidental"
        )

    def allows(self, citation: int, need_id: str, edge_key: str | None = None) -> bool:
        references = [
            subject
            for subject in self.subjects.values()
            if subject.kind == "reference"
            and subject.subject_id.startswith(f"reference-{citation}-")
        ]
        for subject in self.subjects.values():
            if citation not in subject.citations:
                continue
            if (
                subject.kind == "provision"
                and citation not in self._protected
                and self._incidental(subject, need_id)
                and all(
                    self._incidental(reference, need_id) for reference in references
                )
            ):
                return False
            if (
                subject.kind == "reference"
                and subject.edge_key == edge_key
                and self._incidental(subject, need_id)
            ):
                # Reference decisions concern the exact origin, not every source member.
                if subject in references:
                    item = self.ledger.get(citation)
                    anchor = (
                        derive_provision_navigation_anchor(item.source_id, [item])
                        if item is not None
                        else None
                    )
                    if (
                        anchor is not None
                        and _entity_key(
                            anchor.instrument_name,
                            anchor.instrument_number,
                            anchor.article_no,
                            anchor.qualifier,
                        )
                        == edge_key
                        and (
                            citation in self._protected
                            or any(
                                unit.kind == "provision"
                                and citation in unit.citations
                                and not self._incidental(unit, need_id)
                                for unit in self.subjects.values()
                            )
                        )
                    ):
                        return True
                    return False
        return True

    def frontier(self, numbers: set[int], plan: ResearchPlan) -> set[int]:
        return {
            number
            for number in numbers
            if any(self.allows(number, need.need_id) for need in plan.needs)
        }

    def audit(self) -> list[dict[str, JsonValue]]:
        return [
            {
                **subject.payload(),
                "assessments": [
                    row
                    for (subject_id, _), row in self.assessments.items()
                    if subject_id == subject.subject_id
                ],
            }
            for subject in self.subjects.values()
        ]

    def review(
        self, review: FocusAnswerReview, delivered: set[int]
    ) -> tuple[list[str], set[int]]:
        counts: dict[tuple[str, str], int] = {}
        rows = {}
        for row in review.focus_reviews:
            key = row.subject_id, row.need_id
            counts[key] = counts.get(key, 0) + 1
            rows[key] = row
        defects: list[str] = []
        reopened: set[int] = set()
        for key, assessment in list(self.assessments.items()):
            if assessment.get("status") != "incidental":
                continue
            subject = self.subjects[key[0]]
            row = rows.get(key)
            if (
                row is not None
                and counts[key] == 1
                and row.status == "nonmaterial"
                and self._witnesses_valid(subject, row.witnesses, delivered)
            ):
                continue
            defects.append(
                f"Expansion exclusion {key[0]} for need {key[1]} was not independently established as nonmaterial."
            )
            reopened.update(subject.citations)
            self.assessments[key] = {
                **assessment,
                "status": "pending",
                "explanation": "Independent review reopened this expansion lead.",
            }
        return defects, reopened
