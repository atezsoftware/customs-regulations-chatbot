"""Deterministic safety policy for brand-new top-level provisions."""

import re
from typing import Literal

from onyx.regulatory.amendments.structural_target import amended_body

# Official drafting routinely conjoins the addition with the renumbering it
# causes ("eklenmiş ve diğer bentler buna göre teselsül ettirilmiştir"), which
# leaves the participle rather than the finite verb. Matching only the finite
# form silently drops those instructions from both classifications.
_ADDITION_VERB = r"(?:eklenmiş|eklenmis|ilave\s+edilmiş|ilave\s+edilmis)(?:tir|ti)?\b"
_SUBORDINATE_UNIT = r"(?:fıkra|fikra|bent|cümle|cumle|ibare|paragraf)"
_SUBORDINATE_ADDITION_RE = re.compile(
    rf"maddes(?:ine|inin|inde|inden).*?{_SUBORDINATE_UNIT}.*?{_ADDITION_VERB}",
    re.IGNORECASE | re.DOTALL,
)
_TOP_LEVEL_ADDITION_PATTERNS = (
    re.compile(
        rf"(?:geçici\s+|gecici\s+)?madde\s+\d+[a-zçğıöşü]*\b.{{0,160}}?{_ADDITION_VERB}",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf"aşağıdaki\s+(?:yeni\s+)?(?:geçici\s+|gecici\s+)?madde\b.{{0,200}}?"
        rf"{_ADDITION_VERB}",
        re.IGNORECASE | re.DOTALL,
    ),
)
_ADDED_PARAGRAPH_RE = re.compile(
    rf"aşağıdaki\s+(?:yeni\s+)?(?:fıkra|fikra|paragraf).*?{_ADDITION_VERB}",
    re.IGNORECASE | re.DOTALL,
)
_ADDED_CLAUSE_RE = re.compile(
    rf"aşağıdaki\s+(?:yeni\s+)?(?:bent|bend).*?{_ADDITION_VERB}",
    re.IGNORECASE | re.DOTALL,
)

SubordinateUnitKind = Literal["paragraph", "clause"]


def explicitly_adds_top_level_provision(instruction_text: str) -> bool:
    """Return true only when the text itself adds a top-level article.

    Adding a paragraph, clause, sentence, or phrase to an existing article is
    an update to that article and must never enter the new-article path.
    """

    # The instruction's own "MADDE N-" designator is not a provision it adds.
    normalized = " ".join(amended_body(instruction_text).split())
    if _SUBORDINATE_ADDITION_RE.search(normalized):
        return False
    return any(pattern.search(normalized) for pattern in _TOP_LEVEL_ADDITION_PATTERNS)


def added_subordinate_unit_kind(instruction_text: str) -> SubordinateUnitKind | None:
    """Classify an addition that creates a new unit inside an existing article.

    Such an instruction has no chunk to replace — the amended article gains a
    unit — so it belongs to neither the replacement nor the new-article path.
    Naming the unit lets retrieval anchor on the article and drafting position
    the new chunk without inventing a top-level provision.
    """

    normalized = " ".join(amended_body(instruction_text).split())
    if not _SUBORDINATE_ADDITION_RE.search(normalized):
        return None
    if _ADDED_CLAUSE_RE.search(normalized):
        return "clause"
    if _ADDED_PARAGRAPH_RE.search(normalized):
        return "paragraph"
    return None
