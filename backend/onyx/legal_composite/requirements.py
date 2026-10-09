"""Bind source-derived requirements and answer claims to delivered originals."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.models import (
    DraftPatch,
    PassageSupport,
    ResearchPlan,
    SourceRequirement,
)
from onyx.legal_composite.models import (
    StructuredDraftAnswer as DraftAnswer,
)


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
            record["original_bindings"] = [
                {
                    "citation": support.citation,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                    "start": item.text.index(support.quotation),
                    "end": item.text.index(support.quotation) + len(support.quotation),
                }
                for support in requirement.supports
                if (item := self.evidence.get(support.citation)) is not None
            ]
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
    return DraftAnswer(
        answer="\n\n".join(section.text for section in sections),
        sections=sections,
        claims=[claim for claim in draft.claims if claim.section_id not in replaced]
        + patch.claims,
        unresolved_need_ids=patch.unresolved_need_ids,
    )
