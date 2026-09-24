"""Resolve legacy source context without rewriting canonical chunk metadata."""

from collections.abc import Sequence
from dataclasses import replace

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)
from onyx.regulatory.article_scope import SourceFragment, article_scope_indices


def article_scope_candidates(
    ordered_rows: Sequence[CandidateChunk],
    article_no: str,
) -> list[CandidateChunk]:
    indices = article_scope_indices(
        [SourceFragment(str(row.user_file_id), row.text) for row in ordered_rows],
        article_no,
    )
    if not indices:
        return []
    anchor = ordered_rows[indices[0]]
    return [
        replace(
            ordered_rows[index],
            resolved_article_no=article_no,
            scope_evidence=f"Article {article_no}; opening chunk {anchor.chunk_id}: {anchor.text[:1200]}",
        )
        for index in indices
    ]


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
