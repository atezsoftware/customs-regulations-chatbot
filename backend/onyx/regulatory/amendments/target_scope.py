"""Resolve legacy source context without rewriting canonical chunk metadata."""

import re
from collections.abc import Sequence
from dataclasses import replace

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    article_identity,
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
    unquoted_text,
)

_ARTICLE_HEADING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?\s*"
    r"(?:(?:ek|geçici|gecici|mükerrer|mukerrer)\s+)?madde\s+"
    r"\d+[a-z]?(?=\s*[-–—:.])",
    re.IGNORECASE | re.MULTILINE,
)
_MAX_ARTICLE_SCOPE = 64


def article_scope_candidates(
    ordered_rows: Sequence[CandidateChunk],
    article_no: str,
) -> list[CandidateChunk]:
    """Use real opening headings, stopping at the next article or source.

    A citation in body text or a quoted new article is not a boundary. Multiple
    occurrences are ambiguous, so no legacy inference is made in that case.
    """
    found: list[CandidateChunk] = []
    anchor: CandidateChunk | None = None
    occurrences = 0
    for row in ordered_rows:
        if re.search(
            r"kanuna\s+[iİı]şlenemeyen\s+hükümler", row.text[:200], re.IGNORECASE
        ):
            break
        if anchor is not None and row.user_file_id != anchor.user_file_id:
            anchor = None
        headings = list(_ARTICLE_HEADING.finditer(unquoted_text(row.text)))
        if len(headings) > 1 or (headings and headings[0].start() > 0):
            if anchor is not None or any(
                article_identity(item.group()) == article_no for item in headings
            ):
                return []
        heading = _ARTICLE_HEADING.match(row.text)
        if heading:
            identity = article_identity(heading.group())
            anchor = row if identity == article_no else None
            if anchor is not None:
                occurrences += 1
        if anchor is not None:
            found.append(
                replace(
                    row,
                    resolved_article_no=article_no,
                    scope_evidence=f"Article {article_no}; opening chunk {anchor.chunk_id}: {anchor.text[:1200]}",
                )
            )
    return found if occurrences == 1 and len(found) <= _MAX_ARTICLE_SCOPE else []


def validated_addition_anchor(
    instruction: AmendmentInstruction,
    candidates: Sequence[CandidateChunk],
) -> CandidateChunk | None:
    """Require one identified source and, for a subunit, its actual parent."""
    target = parse_amendment_structural_target(instruction)
    subordinate = added_subordinate_unit_kind(instruction.instruction_text)
    if subordinate is None and not explicitly_adds_top_level_provision(
        instruction.instruction_text
    ):
        return None
    identified = [
        candidate
        for candidate in candidates
        if (
            candidate.source_verified
            or (
                source_identity_distinguishing_tokens(instruction.target_source)
                and source_identity_matches(
                    instruction.target_source, candidate.source_name
                )
            )
        )
    ]
    if len({candidate.user_file_id for candidate in identified}) != 1:
        return None
    if subordinate is not None:
        if target is None or target.article_no is None:
            return None
        identified = [
            candidate
            for candidate in identified
            if (candidate.resolved_article_no or candidate.metadata.get("article_no"))
            == target.article_no
        ]
    return next(
        (candidate for candidate in identified if candidate.structured_match),
        identified[0] if identified else None,
    )
