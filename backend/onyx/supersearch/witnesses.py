"""Bind uniquely equivalent display quotations back to immutable exact spans."""

from __future__ import annotations

import re

from onyx.asv3.evidence import EvidenceLedger
from onyx.legal_composite.models import AnswerReview, DraftAnswer, PassageSupport

_DECORATION = re.compile(r"(\*\*|__|`)(\S(?:.*?\S)?)\1", re.DOTALL)


def _display_index(text: str) -> tuple[str, list[tuple[int, int]]]:
    ignored: set[int] = set()
    for match in _DECORATION.finditer(text):
        ignored.update(range(match.start(1), match.end(1)))
        ignored.update(range(match.end(2), match.end()))
    letters: list[str] = []
    positions: list[tuple[int, int]] = []
    for position, char in enumerate(text):
        if position in ignored:
            continue
        if char.isspace():
            if letters and letters[-1] == " ":
                positions[-1] = positions[-1][0], position + 1
            else:
                letters.append(" ")
                positions.append((position, position + 1))
        else:
            for folded in char.casefold():
                letters.append(folded)
                positions.append((position, position + 1))
    return "".join(letters), positions


def exact_display_witness(quotation: str, original: str) -> str:
    """Retain exact text or recover one contiguous span without changing words."""
    if not quotation.strip() or quotation in original:
        return quotation
    needle = _display_index(quotation)[0].strip()
    haystack, positions = _display_index(original)
    if not needle:
        return quotation
    start = haystack.find(needle)
    if start < 0 or haystack.find(needle, start + 1) >= 0:
        return quotation
    left = positions[start][0]
    right = positions[start + len(needle) - 1][1]
    # Include boundary decorations when the quotation spans their complete body.
    for match in _DECORATION.finditer(original):
        if left == match.start(2) and right >= match.end(2):
            left = match.start()
        if right == match.end(2) and left <= match.start(2):
            right = match.end()
    return original[left:right]


def bind_review_witnesses(
    review: AnswerReview, draft: DraftAnswer, ledger: EvidenceLedger
) -> AnswerReview:
    """Only presentation differs; IDs, conclusions and original texts stay fixed."""
    bound = review.model_copy(deep=True)

    def bind_support(support: PassageSupport) -> None:
        item = ledger.get(support.citation)
        if item is not None:
            support.quotation = exact_display_witness(support.quotation, item.text)

    for need in bound.needs:
        for support in need.supports:
            bind_support(support)
        for condition in need.condition_reviews:
            condition.answer_excerpt = exact_display_witness(
                condition.answer_excerpt, draft.answer
            )
        if need.gap_disclosure is not None:
            need.gap_disclosure = exact_display_witness(
                need.gap_disclosure, draft.answer
            )
    for dependency in bound.dependency_assessments or []:
        for witness in dependency.witnesses:
            bind_support(witness)
        if dependency.conditional_excerpt is not None:
            dependency.conditional_excerpt = exact_display_witness(
                dependency.conditional_excerpt, draft.answer
            )
        if dependency.gap_disclosure is not None:
            dependency.gap_disclosure = exact_display_witness(
                dependency.gap_disclosure, draft.answer
            )
    return bound
