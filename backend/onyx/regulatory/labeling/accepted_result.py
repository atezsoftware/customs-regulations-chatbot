"""A validated intake repair carries its durable evidence into normal DB apply."""

from dataclasses import dataclass

from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.provider import LabelingOutcome


@dataclass(frozen=True, slots=True)
class AcceptedIntakeRepair:
    outcome: LabelingOutcome
    audit: AbstentionRepairAudit


LabelingApplyOutcome = (
    tuple[list[str], list[dict[str, object]]] | AcceptedIntakeRepair | str
)
