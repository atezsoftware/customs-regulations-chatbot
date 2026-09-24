"""Resolve legacy source context without rewriting canonical chunk metadata."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import replace

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
    explicitly_adds_top_level_provision,
)
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    amendment_operation_text,
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)
from onyx.regulatory.article_scope import SourceFragment, article_scope_indices


def _fold_identity(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize(
            "NFKD", value.casefold().replace("ı", "i")
        )
        if not unicodedata.combining(character)
    )


def addition_source_identity_is_compatible(
    instruction: AmendmentInstruction, candidate: CandidateChunk
) -> bool:
    """Title overlap cannot override a contradictory instrument identity."""
    title = _fold_identity(instruction.target_source or "").strip()
    kind = re.search(r"\b(karar|kanun|teblig|yonetmelik)(?:i|u)?$", title)
    candidate_kind = candidate.metadata.get("document_type")
    known_kinds = {"karar", "kanun", "teblig", "yonetmelik"}
    if kind and isinstance(candidate_kind, str):
        normalized_kind = _fold_identity(candidate_kind).strip()
        if normalized_kind in known_kinds and kind.group(1) != normalized_kind:
            return False
    # Only the command names the target: the inserted body may cite other acts.
    numbers = set(re.findall(r"(?<![\d/])\d{4}/\d+(?![\d/])", title))
    if not numbers and kind and kind.group(1) == "karar":
        numbers = set(
            re.findall(
                r"(?<![\d/])(\d{4}/\d+)\s+sayili\b",
                _fold_identity(amendment_operation_text(instruction.instruction_text)),
            )
        )
    candidate_number = candidate.metadata.get("document_number")
    if len(numbers) == 1 and isinstance(candidate_number, str) and candidate_number:
        normalized_number = candidate_number.strip().replace("-", "/")
        if (
            re.fullmatch(r"\d{4}/\d+", normalized_number)
            and normalized_number not in numbers
        ):
            return False
    return True


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
        and addition_source_identity_is_compatible(instruction, candidate)
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
        if subordinate == "clause" and target.paragraph_no is not None:
            identified = [
                candidate
                for candidate in identified
                if candidate.metadata.get("paragraph_no") == target.paragraph_no
            ]
    return next(
        (candidate for candidate in identified if candidate.structured_match),
        identified[0] if identified else None,
    )
