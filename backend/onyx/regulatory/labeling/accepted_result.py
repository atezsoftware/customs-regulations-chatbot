"""A validated intake repair carries its durable evidence into normal DB apply."""

from dataclasses import dataclass

from onyx.regulatory.labeling.article_lead_in import SourceOnlyArticleLeadInReceipt
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.provider import LabelingOutcome
from onyx.regulatory.labeling.self_repeal import SourceOnlySelfRepealReceipt


@dataclass(frozen=True, slots=True)
class AcceptedIntakeRepair:
    outcome: LabelingOutcome
    audit: AbstentionRepairAudit


@dataclass(frozen=True, slots=True)
class SourceOnlySelfRepeal:
    receipt: SourceOnlySelfRepealReceipt


@dataclass(frozen=True, slots=True)
class SourceOnlyArticleLeadIn:
    receipt: SourceOnlyArticleLeadInReceipt


LabelingApplyOutcome = (
    tuple[list[str], list[dict[str, object]]]
    | AcceptedIntakeRepair
    | SourceOnlySelfRepeal
    | SourceOnlyArticleLeadIn
    | str
)
