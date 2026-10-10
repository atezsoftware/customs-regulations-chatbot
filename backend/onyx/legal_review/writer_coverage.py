"""Account for material research findings without forcing a citation per source."""

from collections.abc import Mapping
from types import GenericAlias
from typing import Any, Literal

from pydantic import BaseModel, Field, JsonValue, create_model

from onyx.legal_review.drafting import GeneratedDraft
from onyx.legal_review.models import RequirementRecord, StrictModel


class FindingUse(StrictModel):
    disposition: Literal["covered", "qualified", "not_applicable", "unresolved"]
    claim_ids: list[str]
    reason: str = Field(min_length=1)


class FindingBindingError(ValueError):
    def __init__(self, invalid_slots: list[str]) -> None:
        self.invalid_slots = invalid_slots
        super().__init__(
            "Finding coverage has invalid claim bindings: " + ", ".join(invalid_slots)
        )


def _validated_use(use: FindingUse, known: set[str]) -> FindingUse:
    if set(use.claim_ids) - known:
        raise ValueError("Finding coverage selects an unknown claim")
    # Selecting a known claim twice does not change its coverage.
    normalized = use.model_copy(
        update={"claim_ids": list(dict.fromkeys(use.claim_ids))}
    )
    if normalized.disposition in {"covered", "qualified"} and not normalized.claim_ids:
        raise ValueError("A covered finding needs an actual published claim")
    if not normalized.reason.strip():
        raise ValueError("Finding coverage requires a substantive reason")
    return normalized


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
        invalid: list[str] = []
        for slot, requirement in self.slots.items():
            try:
                use = _validated_use(FindingUse.model_validate(coverage[slot]), known)
            except ValueError:
                invalid.append(slot)
                continue
            rows.append(
                {
                    "requirement_id": requirement.requirement_id,
                    **use.model_dump(mode="json"),
                }
            )
        if invalid:
            raise FindingBindingError(invalid)
        return generated, rows

    def unresolved_audit(self, response: GeneratedDraft) -> list[dict[str, JsonValue]]:
        payload = response.model_dump(mode="json")
        known = {claim.claim_id for block in response.blocks for claim in block.claims}
        rows: list[dict[str, JsonValue]] = []
        for slot, requirement in self.slots.items():
            proposed = FindingUse.model_validate(payload["finding_coverage"][slot])
            try:
                use = _validated_use(proposed, known)
                status = "valid"
            except ValueError:
                use = FindingUse(
                    disposition="unresolved",
                    claim_ids=[],
                    reason="No validated finding-to-claim binding was established.",
                )
                status = "unresolved"
            rows.append(
                {
                    "requirement_id": requirement.requirement_id,
                    **use.model_dump(mode="json"),
                    "binding_status": status,
                    "proposed_use": proposed.model_dump(mode="json"),
                }
            )
        return rows


class FindingBindingContract:
    """Repair only metadata against claim identities in an immutable completed draft."""

    def __init__(
        self,
        writer: WriterCoverageContract,
        response: GeneratedDraft,
        invalid_slots: list[str],
    ) -> None:
        self.writer = writer
        self.response = response
        self.invalid_slots = invalid_slots
        known = tuple(
            claim.claim_id for block in response.blocks for claim in block.claims
        )
        selector: Any = (
            GenericAlias(list, Literal.__getitem__(known)) if known else list[str]
        )
        use_model = create_model(
            "KnownFindingUse",
            __base__=FindingUse,
            claim_ids=(selector, Field() if known else Field(max_length=0)),
        )
        fields: dict[str, Any] = {slot: (use_model, Field()) for slot in invalid_slots}
        self.response_model: type[BaseModel] = create_model(
            "FindingBindingRepair",
            __base__=StrictModel,
            **fields,
        )

    def state(self) -> dict[str, JsonValue]:
        payload = self.response.model_dump(mode="json")
        return {
            "frozen_blocks": payload["blocks"],
            "affected_findings": {
                slot: {
                    "requirement": self.writer.slots[slot].model_dump(mode="json"),
                    "proposed_use": payload["finding_coverage"][slot],
                }
                for slot in self.invalid_slots
            },
        }

    def compile(self, repair: BaseModel) -> list[dict[str, JsonValue]]:
        replacements = repair.model_dump(mode="json")
        if set(replacements) != set(self.invalid_slots):
            raise ValueError("Finding repair must cover exactly the affected slots")
        payload = self.response.model_dump(mode="json")
        payload["finding_coverage"].update(replacements)
        corrected = self.writer.response_model.model_validate(payload)
        _, rows = self.writer.compile(corrected)
        return rows
