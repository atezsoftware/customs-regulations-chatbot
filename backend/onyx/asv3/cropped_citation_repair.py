"""Bound a final citation correction to unchanged owned assertions and originals."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.citation_numbers import strip_citation_markers
from onyx.asv3.cropped_repair_state import CroppedRepairTarget
from onyx.asv3.models import RunStopped


@dataclass(frozen=True)
class CitationRepairPlan:
    requirements: dict[str, tuple[frozenset[int], ...]]

    def metadata(self, draft: str) -> dict[str, JsonValue]:
        return {
            "mode": "add_matching_original_citations_only",
            "unit_original_requirements": {
                identifier: [sorted(numbers) for numbers in groups]
                for identifier, groups in self.requirements.items()
            },
            "unit_texts": {
                unit["unit_id"]: unit["text"]
                for unit in assertion_inventory(draft)
                if unit["unit_id"] in self.requirements
            },
        }

    def validate(self, before: str, after: str) -> None:
        previous, current = assertion_inventory(before), assertion_inventory(after)
        if len(previous) != len(current):
            raise RunStopped(
                "Citation correction changed the owned assertion structure"
            )
        changed = False
        for old, new in zip(previous, current, strict=True):
            groups = self.requirements.get(old["unit_id"])
            if groups is None:
                if old["text"] != new["text"]:
                    raise RunStopped(
                        "Citation correction edited an unrelated assertion"
                    )
                continue
            if " ".join(strip_citation_markers(old["text"]).split()) != " ".join(
                strip_citation_markers(new["text"]).split()
            ):
                raise RunStopped(
                    "Citation correction changed substantive answer wording"
                )
            existing, supplied = (
                set(old["evidence_numbers"]),
                set(new["evidence_numbers"]),
            )
            allowed = set().union(*groups)
            if not existing.issubset(supplied) or (supplied - existing) - allowed:
                raise RunStopped(
                    "Citation correction changed citations outside its originals"
                )
            if any(not supplied.intersection(numbers) for numbers in groups):
                raise RunStopped(
                    "Citation correction omitted a matching governing original"
                )
            changed = changed or bool(supplied - existing)
        if not changed:
            raise RunStopped(
                "Citation correction did not add the missing original citation"
            )


def citation_repair_plan(
    draft: str,
    gap: dict[str, JsonValue],
    targets: list[CroppedRepairTarget],
    records: list[dict[str, JsonValue]],
) -> CitationRepairPlan | None:
    """A found source permits citation repair, never a fabricated absence notice."""
    if (
        not gap.get("named_authority_gaps")
        or set(gap)
        - {"named_authority_gaps", "citation_format", "instruction", "summary"}
        or not targets
    ):
        return None
    units = {unit["unit_id"] for unit in assertion_inventory(draft)}
    by_key = {row.get("key"): row for row in records}
    requirements: dict[str, tuple[frozenset[int], ...]] = {}
    for target in targets:
        record = by_key.get(target.key)
        numbers = record.get("evidence_numbers") if record is not None else None
        if (
            target.check_kind != "governing_original"
            or not target.article
            or not target.unit_ids
            or not set(target.unit_ids).issubset(units)
            or record is None
            or record.get("status") != "found"
            or not isinstance(numbers, list)
            or not numbers
            or any(type(number) is not int or number < 1 for number in numbers)
        ):
            return None
        originals = frozenset(number for number in numbers if isinstance(number, int))
        for identifier in target.unit_ids:
            requirements[identifier] = (*requirements.get(identifier, ()), originals)
    return CitationRepairPlan(requirements)
