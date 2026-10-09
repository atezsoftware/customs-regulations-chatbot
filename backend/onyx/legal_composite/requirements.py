"""Bind source-derived requirements and answer claims to delivered originals."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable
from typing import TypeVar

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    DraftPatch,
    PassageSupport,
    ResearchPlan,
    SourceRequirement,
    SpanSupport,
)
from onyx.legal_composite.models import (
    StructuredDraftAnswer as DraftAnswer,
)

DraftWithSupports = TypeVar("DraftWithSupports", DraftAnswer, DraftPatch)


def support_is_original(
    support: PassageSupport, ledger: EvidenceLedger, delivered: set[int]
) -> bool:
    item = ledger.get(support.citation)
    if item is None or item.search_doc is None or support.citation not in delivered:
        return False
    canonical = item.metadata.get("canonical_metadata")
    layers = [
        item.metadata,
        item.search_doc.metadata,
        canonical if isinstance(canonical, dict) else {},
    ]
    if isinstance(support, SpanSupport) and support.span_id is not None:
        matches = [
            span
            for span in original_witness_spans(support.citation, item.text)
            if span["witness_id"] == support.span_id
        ]
        if (
            len(matches) != 1
            or support.quotation
            != item.text[matches[0]["start_char"] : matches[0]["end_char"]]
        ):
            return False
    return (
        bool(support.quotation.strip())
        and item.search_doc.document_id == item.source_id
        and hashlib.sha256(item.text.encode()).hexdigest() == item.text_hash
        and item.search_doc.metadata.get("regulatory_chunk_id") == item.chunk_id
        and support.quotation in item.text
        and not any(
            layer.get(key) is True
            for layer in layers
            for key in ("external", "derived", "untrusted", "truncated")
        )
    )


def canonicalize_source_support(
    support: PassageSupport, ledger: EvidenceLedger, delivered: set[int]
) -> SpanSupport:
    """Resolve only host-provided selectors against the delivered immutable original."""
    item = ledger.get(support.citation)
    if item is None or not support_is_original(
        PassageSupport(citation=support.citation, quotation=item.text),
        ledger,
        delivered,
    ):
        raise InvalidSourceAction(
            "Support is not an exact delivered original: canonical original binding failed"
        )
    span_id = support.span_id if isinstance(support, SpanSupport) else None
    quotation = support.quotation
    if span_id is not None:
        matches = [
            span
            for span in original_witness_spans(support.citation, item.text)
            if span["witness_id"] == span_id
        ]
        if len(matches) != 1:
            raise InvalidSourceAction("Support span ID is not in this exact original")
        span = matches[0]
        original = item.text[span["start_char"] : span["end_char"]]
        if quotation and quotation != original:
            raise InvalidSourceAction(
                "Support quotation conflicts with its original span"
            )
        quotation = original
    if not quotation.strip() or quotation not in item.text:
        raise InvalidSourceAction(
            "Support quotation is not an exact delivered original passage"
        )
    return SpanSupport(citation=support.citation, quotation=quotation, span_id=span_id)


def canonicalize_source_requirement(
    requirement: SourceRequirement, ledger: EvidenceLedger, delivered: set[int]
) -> SourceRequirement:
    return requirement.model_copy(
        deep=True,
        update={
            "supports": [
                canonicalize_source_support(support, ledger, delivered)
                for support in requirement.supports
            ]
        },
    )


def canonicalize_draft_supports(
    draft: DraftWithSupports, ledger: EvidenceLedger, delivered: set[int]
) -> DraftWithSupports:
    """Bind writer and repair references without altering their claims or answer text."""
    claims = [
        claim.model_copy(
            deep=True,
            update={
                "supports": [
                    canonicalize_source_support(support, ledger, delivered)
                    for support in claim.supports
                ]
            },
        )
        for claim in draft.claims
    ]
    requirements = [
        canonicalize_source_requirement(requirement, ledger, delivered)
        for requirement in draft.requirements
    ]
    return draft.model_copy(
        deep=True, update={"claims": claims, "requirements": requirements}
    )


def _original_binding(
    support: SpanSupport, evidence: EvidenceLedger
) -> dict[str, JsonValue] | None:
    item = evidence.get(support.citation)
    if (
        item is None
        or not support_is_original(support, evidence, {support.citation})
        or support.quotation not in item.text
    ):
        return None
    start = item.text.index(support.quotation)
    if support.span_id is not None:
        matches = [
            span
            for span in original_witness_spans(support.citation, item.text)
            if span["witness_id"] == support.span_id
        ]
        if len(matches) != 1:
            return None
        start = matches[0]["start_char"]
        if item.text[start : matches[0]["end_char"]] != support.quotation:
            return None
    return {
        "citation": support.citation,
        "source_id": item.source_id,
        "chunk_id": item.chunk_id,
        "text_hash": item.text_hash,
        "span_id": support.span_id,
        "start": start,
        "end": start + len(support.quotation),
    }


class RequirementLedger:
    def __init__(self, evidence: EvidenceLedger) -> None:
        self.evidence = evidence
        self._requirements: dict[str, SourceRequirement] = {}
        self._superseded_by: dict[str, set[str]] = {}

    def update(
        self,
        requirements: Iterable[SourceRequirement],
        plan: ResearchPlan,
        delivered: set[int],
    ) -> None:
        known = {need.need_id for need in plan.needs}
        pending = dict(self._requirements)
        seen: set[str] = set()
        for requirement in requirements:
            requirement = canonicalize_source_requirement(
                requirement, self.evidence, delivered
            )
            if requirement.requirement_id in seen:
                raise InvalidSourceAction("Duplicate source requirement identity")
            seen.add(requirement.requirement_id)
            if requirement.need_id not in known:
                raise InvalidSourceAction(
                    "Source requirement refers to an unknown issue"
                )
            if not all(
                support_is_original(support, self.evidence, delivered)
                for support in requirement.supports
            ):
                raise InvalidSourceAction(
                    f"Requirement {requirement.requirement_id}: support is not an exact delivered original"
                )
            previous = pending.get(requirement.requirement_id)
            if previous is not None and previous != requirement:
                raise InvalidSourceAction("Source requirement identities are immutable")
            pending[requirement.requirement_id] = requirement.model_copy(deep=True)
        superseded_by: dict[str, set[str]] = {}
        inbound = dict.fromkeys(pending, 0)
        for requirement in pending.values():
            targets = requirement.supersedes_requirement_ids
            if len(targets) != len(set(targets)):
                raise InvalidSourceAction("Duplicate requirement supersession target")
            for identity in targets:
                previous = pending.get(identity)
                if previous is None:
                    raise InvalidSourceAction("Unknown requirement supersession target")
                if identity == requirement.requirement_id:
                    raise InvalidSourceAction("Requirement cannot supersede itself")
                if previous.need_id != requirement.need_id:
                    raise InvalidSourceAction(
                        "Requirement supersession cannot cross issues"
                    )
                superseded_by.setdefault(identity, set()).add(
                    requirement.requirement_id
                )
                inbound[identity] += 1
        ready = deque(identity for identity, count in inbound.items() if not count)
        visited = 0
        while ready:
            identity = ready.popleft()
            visited += 1
            for target in pending[identity].supersedes_requirement_ids:
                inbound[target] -= 1
                if not inbound[target]:
                    ready.append(target)
        if visited != len(pending):
            raise InvalidSourceAction("Requirement supersession cannot form a cycle")
        self._requirements = pending
        self._superseded_by = superseded_by

    def records(self, need_ids: set[str] | None = None) -> list[SourceRequirement]:
        return [
            requirement.model_copy(deep=True)
            for requirement in self._requirements.values()
            if requirement.requirement_id not in self._superseded_by
            and (need_ids is None or requirement.need_id in need_ids)
        ]

    def citations(self, need_ids: set[str] | None = None) -> set[int]:
        return {
            support.citation
            for requirement in self.records(need_ids)
            for support in requirement.supports
        }

    def export(self) -> list[dict[str, JsonValue]]:
        records: list[dict[str, JsonValue]] = []
        for requirement in self._requirements.values():
            record = requirement.model_dump(mode="json")
            record["active"] = requirement.requirement_id not in self._superseded_by
            record["superseded_by_requirement_ids"] = sorted(
                self._superseded_by.get(requirement.requirement_id, set())
            )
            bindings: list[dict[str, JsonValue]] = []
            binding_errors: list[dict[str, JsonValue]] = []
            for support in requirement.supports:
                binding = _original_binding(support, self.evidence)
                if binding is None:
                    binding_errors.append(
                        {
                            "citation": support.citation,
                            "span_id": support.span_id,
                            "status": "not_canonical_original",
                        }
                    )
                else:
                    bindings.append(binding)
            record["original_bindings"] = bindings
            if binding_errors:
                record["original_binding_errors"] = binding_errors
                record["binding_status"] = "invalid"
            records.append(record)
        return records


def draft_binding_gaps(
    draft: DraftAnswer,
    plan: ResearchPlan,
    requirements: list[SourceRequirement],
    ledger: EvidenceLedger,
    delivered: set[int],
) -> list[str]:
    gaps: list[str] = []
    known = {need.need_id for need in plan.needs}
    requirement_map = {row.requirement_id: row for row in requirements}
    sections = {row.section_id: row for row in draft.sections}
    if not sections:
        gaps.append("draft:no_stable_sections")
    if set(draft.unresolved_need_ids) - known:
        gaps.append("draft:unknown_unresolved_issue")
    if any(set(section.need_ids) - known for section in draft.sections):
        gaps.append("draft:unknown_section_issue")
    if set().union(*(set(section.need_ids) for section in draft.sections)) != known:
        gaps.append("draft:issue_without_section")
    cited = set(extract_citation_numbers(draft.answer))
    if plan.requires_sources and (not cited or not draft.claims):
        gaps.append("draft:missing_original_claim_bindings")
    for citation in cited:
        item = ledger.get(citation)
        if item is None or not support_is_original(
            PassageSupport(citation=citation, quotation=item.text), ledger, delivered
        ):
            gaps.append(f"citation:{citation}:not_delivered_original")
    for claim in draft.claims:
        section = sections.get(claim.section_id)
        if (
            section is None
            or claim.answer_excerpt not in section.text
            or set(claim.need_ids) - set(section.need_ids)
            or not claim.need_ids
        ):
            gaps.append(f"claim:{claim.claim_id}:invalid_answer_binding")
        requirement_supports = [
            support
            for requirement_id in claim.requirement_ids
            if (requirement := requirement_map.get(requirement_id)) is not None
            for support in requirement.supports
        ]
        supports = [*claim.supports, *requirement_supports]
        if not supports or any(
            support.citation not in cited
            or not support_is_original(support, ledger, delivered)
            for support in supports
        ):
            gaps.append(f"claim:{claim.claim_id}:invalid_original_binding")
        for requirement_id in claim.requirement_ids:
            requirement = requirement_map.get(requirement_id)
            if requirement is None or requirement.need_id not in claim.need_ids:
                gaps.append(f"claim:{claim.claim_id}:invalid_requirement_binding")
    return list(dict.fromkeys(gaps))


def apply_patch(
    draft: DraftAnswer, patch: DraftPatch, affected_sections: set[str]
) -> DraftAnswer:
    replaced = {section.section_id for section in patch.sections}
    existing = {section.section_id for section in draft.sections}
    if replaced != affected_sections or not replaced <= existing:
        raise InvalidSourceAction("Repair must replace exactly the affected sections")
    if any(claim.section_id not in replaced for claim in patch.claims):
        raise InvalidSourceAction(
            "Repair cannot replace claims in an unchanged section"
        )
    replacements = {section.section_id: section for section in patch.sections}
    sections = [
        replacements.get(section.section_id, section) for section in draft.sections
    ]
    claims = [
        claim for claim in draft.claims if claim.section_id not in replaced
    ] + patch.claims
    identities = [claim.claim_id for claim in claims]
    if len(identities) != len(set(identities)):
        raise InvalidSourceAction(
            "Repair claim identity conflicts with an unchanged section"
        )
    return DraftAnswer(
        answer="\n\n".join(section.text for section in sections),
        sections=sections,
        claims=claims,
        unresolved_need_ids=patch.unresolved_need_ids,
    )
