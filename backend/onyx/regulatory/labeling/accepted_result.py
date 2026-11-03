"""A validated intake repair carries its durable evidence into normal DB apply."""

from dataclasses import dataclass

from onyx.regulatory.labeling.article_lead_in import SourceOnlyArticleLeadInReceipt
from onyx.regulatory.labeling.chapter_caption import SourceOnlyChapterCaptionReceipt
from onyx.regulatory.labeling.intake_abstention_repair import AbstentionRepairAudit
from onyx.regulatory.labeling.intake_duplicate_label_repair import (
    DuplicateLabelRepairAudit,
)
from onyx.regulatory.labeling.provider import LabelingOutcome
from onyx.regulatory.labeling.reviewed_fragment import SourceOnlyReviewedFragmentReceipt
from onyx.regulatory.labeling.self_repeal import SourceOnlySelfRepealReceipt


@dataclass(frozen=True, slots=True)
class AcceptedIntakeRepair:
    outcome: LabelingOutcome
    audit: AbstentionRepairAudit


@dataclass(frozen=True, slots=True)
class AcceptedDuplicateLabelRepair:
    outcome: LabelingOutcome
    audit: DuplicateLabelRepairAudit


@dataclass(frozen=True, slots=True)
class SourceOnlySelfRepeal:
    receipt: SourceOnlySelfRepealReceipt


@dataclass(frozen=True, slots=True)
class SourceOnlyArticleLeadIn:
    receipt: SourceOnlyArticleLeadInReceipt


@dataclass(frozen=True, slots=True)
class SourceOnlyChapterCaption:
    receipt: SourceOnlyChapterCaptionReceipt


@dataclass(frozen=True, slots=True)
class SourceOnlyReviewedFragment:
    receipt: SourceOnlyReviewedFragmentReceipt


LabelingApplyOutcome = (
    tuple[list[str], list[dict[str, object]]]
    | AcceptedIntakeRepair
    | AcceptedDuplicateLabelRepair
    | SourceOnlySelfRepeal
    | SourceOnlyArticleLeadIn
    | SourceOnlyChapterCaption
    | SourceOnlyReviewedFragment
    | str
)
