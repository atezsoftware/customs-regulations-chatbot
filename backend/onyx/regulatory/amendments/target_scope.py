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
    AmendmentStructuralTarget,
    amendment_operation_text,
    parse_amendment_structural_target,
    source_identity_distinguishing_tokens,
    source_identity_matches,
)
from onyx.regulatory.article_scope import SourceFragment, article_scope_indices
from onyx.regulatory.provision_identity import canonical_clause_label

_UNIT_OPENING = re.compile(
    r"^(?:(?:EK|GEÇİCİ|GECICI|MÜKERRER|MUKERRER)\s+)?MADDE\s+\d+[a-z]?\s*[-–—:.]\s*",
    re.IGNORECASE,
)
_PARAGRAPH_OPENING = re.compile(r"^\((\d+)\)")
_CLAUSE_OPENING = re.compile(r"^(?:\(([a-zçğıöşü])\)|([a-zçğıöşü])\))", re.IGNORECASE)


def reconcile_structural_candidates(
    candidates: Sequence[CandidateChunk],
    ordered_rows: Sequence[CandidateChunk],
    target: AmendmentStructuralTarget,
) -> list[CandidateChunk]:
    """Keep contradictory source evidence ahead of a bounded matching shortlist.

    Parent context is evidence only; stored metadata is never repaired by a
    retrieval. Exact lookup cannot hide legacy null parents or appended units.
    """
    if target.article_no is None or target.appendix_label is not None:
        return list(candidates)
    resolved = article_scope_candidates(ordered_rows, target.article_no)
    by_id = {item.chunk_id: item for item in resolved}
    units: dict[str, tuple[str | None, str | None]] = {}
    paragraph: str | None = None
    for item in resolved:
        if item.metadata.get(
            "chunk_variant"
        ) == "hierarchical_aggregate" or item.metadata.get(
            "bound_to_regulatory_chunk_id"
        ):
            continue
        clean = re.sub(r"(?:\*\*|__|`|<[^>]+>)", "", item.text).strip()
        clean = _UNIT_OPENING.sub("", clean)
        paragraph_match = _PARAGRAPH_OPENING.match(clean)
        if paragraph_match:
            paragraph = paragraph_match.group(1)
        clause_match = _CLAUSE_OPENING.match(clean)
        clause = (
            canonical_clause_label(clause_match.group(1) or clause_match.group(2))
            if clause_match
            else None
        )
        # A continuation without an opening still belongs to the same named
        # unit when its metadata says so; it must not vanish from exact matching.
        if clause is None and not paragraph_match:
            stored_clause = item.metadata.get("clause_label")
            clause = (
                canonical_clause_label(stored_clause)
                if isinstance(stored_clause, str)
                else None
            )
        units[item.chunk_id] = (paragraph, clause)
        by_id[item.chunk_id] = replace(
            item,
            scope_evidence=f"{item.scope_evidence}; paragraph {paragraph or 'unresolved'}; clause {clause or 'none'}",
        )

    def named(paragraph_no: object, clause_label: object) -> bool:
        if target.paragraph_no is not None and paragraph_no != target.paragraph_no:
            return False
        if target.clause_label is not None:
            return clause_label == target.clause_label
        return clause_label is None

    relevant: list[CandidateChunk] = []
    conflict: str | None = None
    for item in ordered_rows:
        if item.metadata.get(
            "chunk_variant"
        ) == "hierarchical_aggregate" or item.metadata.get(
            "bound_to_regulatory_chunk_id"
        ):
            continue
        actual = units.get(item.chunk_id)
        stored = (item.metadata.get("paragraph_no"), item.metadata.get("clause_label"))
        actual_matches = actual is not None and named(*actual)
        stored_matches = item.metadata.get("article_no") == target.article_no and named(
            *stored
        )
        if not (actual_matches or stored_matches):
            continue
        relevant.append(by_id.get(item.chunk_id, item))
        if actual is None or actual_matches != stored_matches:
            conflict = "The named unit's stored metadata and bounded source structure disagree."
    if len(relevant) > 1:
        conflict = "Several canonical pieces claim the named unit; compare their bodies and source boundaries."
    if not resolved and candidates:
        conflict = "The article's complete source boundaries could not be verified."
    result: dict[str, CandidateChunk] = {}
    originals = {item.chunk_id: item for item in candidates}
    # Actual source matches precede stale scalar matches and unrelated siblings.
    relevant.sort(
        key=lambda item: not (item.chunk_id in units and named(*units[item.chunk_id]))
    )
    for item in [*relevant, *candidates, *resolved]:
        original = originals.get(item.chunk_id, item)
        evidence = by_id.get(item.chunk_id, item)
        result.setdefault(
            item.chunk_id,
            replace(
                original,
                resolved_article_no=evidence.resolved_article_no,
                scope_evidence=evidence.scope_evidence,
                structure_conflict=conflict or original.structure_conflict,
            ),
        )
    return list(result.values())


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
