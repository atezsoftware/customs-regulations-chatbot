"""Account for material research findings without forcing a citation per source."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field, JsonValue, create_model

from onyx.legal_review.drafting import GeneratedDraft
from onyx.legal_review.models import RequirementRecord, StrictModel


class FindingUse(StrictModel):
    disposition: Literal["covered", "qualified", "not_applicable", "unresolved"]
    claim_ids: list[str]
    reason: str = Field(min_length=1)


class WriterCoverageContract:
    def __init__(self, requirements: Mapping[str, RequirementRecord]) -> None:
        self.slots = {
            f"f{index:04d}": row for index, row in enumerate(requirements.values(), 1)
        }
        fields: dict[str, Any] = {slot: (FindingUse, Field()) for slot in self.slots}
        inventory = create_model("FindingCoverage", __base__=StrictModel, **fields)
        self.response_model: type[GeneratedDraft] = create_model(
            "AccountedLegalDraft",
            __base__=GeneratedDraft,
            finding_coverage=(inventory, Field()),
        )

    def state(self) -> dict[str, JsonValue]:
        return {
            "finding_coverage_contract": {
                slot: {
                    "requirement_id": row.requirement_id,
                    "supports": [
                        support.model_dump(mode="json") for support in row.supports
                    ],
                }
                for slot, row in self.slots.items()
            }
        }

    def compile(
        self, response: GeneratedDraft
    ) -> tuple[GeneratedDraft, list[dict[str, JsonValue]]]:
        payload = response.model_dump(mode="json")
        coverage = payload.pop("finding_coverage")
        generated = GeneratedDraft.model_validate(payload)
        known = {claim.claim_id for block in generated.blocks for claim in block.claims}
        if coverage.keys() != self.slots.keys():
            raise ValueError("Writer must account for the complete finding inventory")
        rows: list[dict[str, JsonValue]] = []
        for slot, requirement in self.slots.items():
            use = FindingUse.model_validate(coverage[slot])
            if set(use.claim_ids) - known or len(use.claim_ids) != len(
                set(use.claim_ids)
            ):
                raise ValueError("Finding coverage selects unknown or duplicate claims")
            if use.disposition in {"covered", "qualified"} and not use.claim_ids:
                raise ValueError("A covered finding needs an actual published claim")
            if not use.reason.strip():
                raise ValueError("Finding coverage requires a substantive reason")
            rows.append(
                {
                    "requirement_id": requirement.requirement_id,
                    **use.model_dump(mode="json"),
                }
            )
        return generated, rows
