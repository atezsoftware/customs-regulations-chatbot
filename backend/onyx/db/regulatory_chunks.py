"""DB operations for regulatory chunks.

Postgres rows are the source of truth for chunk text/metadata/validity; the
Elasticsearch index is a projection. Callers that mutate rows here are
responsible for re-projecting the affected chunks into Elasticsearch (see
onyx/regulatory/indexing.py).
"""

import datetime
import hashlib
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast
from uuid import UUID

from sqlalchemy import delete, exists, func, or_, select
from sqlalchemy.orm import Session

from onyx.db.enums import RegulatoryChunkSource, RegulatoryChunkStatus
from onyx.db.models import RegulatoryChunk, UserFile
from onyx.regulatory.chunker import (
    ATOMIC_CHUNK_VARIANT,
    HIERARCHICAL_AGGREGATE_CHUNK_VARIANT,
)
from onyx.regulatory.chunker import RegulatoryChunk as ChunkerChunk
from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_regulatory_provision_references,
    normalize_regulatory_heading_path,
    parse_regulatory_article_heading,
)

DEFAULT_PROVISION_MAX_CHUNKS = 24
DEFAULT_PROVISION_MAX_CHARS = 8_000
DEFAULT_NAVIGATION_MAX_HEADINGS = 12
DEFAULT_REFERENCE_MAX_PROVISIONS = 3
DEFAULT_REFERENCE_MAX_CHUNKS_PER_PROVISION = 4
DEFAULT_REFERENCE_MAX_CHARS_PER_PROVISION = 4_000
DEFAULT_ADJACENT_MAX_PROVISIONS = 2
DEFAULT_ADJACENT_MAX_CHUNKS_PER_PROVISION = 2
DEFAULT_ADJACENT_MAX_TOTAL_CHARS = 4_000

_LEXICAL_TERM_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NUMBERED_PEER_HEADING_RE = re.compile(
    r"^(?:\((?P<parenthesized>\d{1,3})\)|(?P<plain>\d{1,3})[.)])\s+\S"
)
_TERSE_LIST_ITEM_MAX_CHARS = 100
_TERSE_LIST_ITEM_MAX_TERMS = 8
_TERSE_STRUCTURAL_INTRO_MAX_CHARS = 240
_MAX_LOCAL_PARAGRAPH_REFERENCES = 8
_TURKISH_ORDINAL_SUFFIX_RE = r"(?:[iı]nc[iı]|nc[iı]|uncu)"
_TURKISH_LOCAL_REFERENCE_ITEM_RE = (
    rf"\d{{1,3}}(?:\s*['’]?\s*{_TURKISH_ORDINAL_SUFFIX_RE})?"
)
_TURKISH_PARAGRAPH_CASE_SUFFIX_RE = (
    r"(?:da|de|dan|den|a|e|ya|ye|y[iı]|[iı]|[iı]n|nin|nın|"
    r"s[iı](?:n(?:da|de|dan|den|[iı]n|a|e|[iı]))?)?"
)
_TURKISH_LOCAL_PARAGRAPH_REFERENCE_RE = re.compile(
    rf"(?<![\d\-–—])(?P<numbers>"
    rf"(?:{_TURKISH_LOCAL_REFERENCE_ITEM_RE}"
    rf"(?:\s*,\s*|\s+(?:ve|ile)\s+))*"
    rf"\d{{1,3}}\s*['’]?\s*{_TURKISH_ORDINAL_SUFFIX_RE})"
    rf"\s+f[iı]kra(?:lar)?{_TURKISH_PARAGRAPH_CASE_SUFFIX_RE}\b"
)
_ENGLISH_LOCAL_REFERENCE_ITEM_RE = r"\(?\d{1,3}\)?"
_ENGLISH_LOCAL_PARAGRAPH_REFERENCE_RE = re.compile(
    rf"\bparagraphs?\s+(?P<numbers>{_ENGLISH_LOCAL_REFERENCE_ITEM_RE}"
    rf"(?:\s*,\s*{_ENGLISH_LOCAL_REFERENCE_ITEM_RE})*"
    rf"(?:\s+(?:and|or)\s+{_ENGLISH_LOCAL_REFERENCE_ITEM_RE})?)"
    r"(?=$|[\s,.;:])"
)
_STRUCTURAL_SCOPE_DESIGNATOR = r"(?:\d+[a-z]?|[ivxlcdm]+|[a-z])"
_STRUCTURAL_SCOPE_ORDINAL = (
    r"(?:birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|"
    r"dokuzuncu|onuncu)"
)
_STRUCTURAL_SCOPE_BOUNDARY_RE = re.compile(
    rf"^(?:(?:ana metin|main text)(?:$|\s*[-–—:])|"
    rf"(?:ek|annex|appendix|schedule|cetvel|liste|ilave|supplement|"
    rf"protokol|protocol)(?:$|\s*[-–—:]?\s*{_STRUCTURAL_SCOPE_DESIGNATOR}\b)|"
    rf"(?:bolum|chapter|kisim|part|fasil|cilt|baslik|title|section|subsection)"
    rf"(?:$|\s+{_STRUCTURAL_SCOPE_DESIGNATOR}\b)|"
    rf"{_STRUCTURAL_SCOPE_ORDINAL}\s+(?:bolum|kisim|fasil)\b)"
)


@dataclass(frozen=True, slots=True)
class RegulatoryChunkSiblingCandidate:
    """Minimal, immutable source row used by the pure sibling selector."""

    regulatory_chunk_id: str
    user_file_id: UUID
    position: int
    text: str
    status: str
    heading_path: tuple[str, ...] = ()
    article_no: str | None = None
    article_title: str | None = None
    chunk_type: str | None = None
    paragraph_no: str | None = None
    clause_label: str | None = None
    validity_start_date: datetime.date | None = None
    validity_end_date: datetime.date | None = None


@dataclass(frozen=True, slots=True)
class RegulatoryChunkProjection:
    """A selected regulatory chunk plus its stable Elasticsearch projection index.

    ``projection_index`` is calculated across every row in a user file before
    applying the temporal filter. This mirrors the projection enumerator and
    lets the search layer retrieve the exact Elasticsearch chunk without exposing
    mutable SQLAlchemy models.
    """

    regulatory_chunk_id: str
    user_file_id: UUID
    projection_index: int
    position: int
    text: str
    heading_path: tuple[str, ...]
    article_no: str | None
    status: str
    validity_start_date: datetime.date | None
    validity_end_date: datetime.date | None
    chunk_type: str | None = None
    paragraph_no: str | None = None
    clause_label: str | None = None
    # Lower values are consumed first when the caller has a tighter global
    # evidence budget than this provision's local selection budget.
    expansion_priority: int = 1


@dataclass(frozen=True, slots=True)
class RegulatoryNavigationSeed:
    """A selected regulatory result used only to choose a navigation source."""

    regulatory_chunk_id: str
    user_file_id: UUID
    position: int


@dataclass(frozen=True, slots=True)
class RegulatoryProvisionHeadingCandidate:
    """Structural metadata for one row; intentionally excludes legal text."""

    user_file_id: UUID
    position: int
    heading_path: tuple[str, ...]
    status: str
    validity_start_date: datetime.date | None
    validity_end_date: datetime.date | None
    article_title: str | None = None
    regulatory_chunk_id: str | None = None


@dataclass(frozen=True, slots=True)
class RegulatoryProvisionHeadingSource:
    """One dominant document and its bounded-navigation input metadata."""

    user_file_id: UUID
    document_title: str
    seed_positions: tuple[int, ...]
    candidates: tuple[RegulatoryProvisionHeadingCandidate, ...]


ValidityDateUpdate = datetime.date | None | Literal["unset"]


@dataclass(frozen=True, slots=True)
class RegulatoryFileValidityWindow:
    """A uniform, explicitly persisted source-snapshot window."""

    start: datetime.date | None
    end: datetime.date | None


@dataclass(frozen=True, slots=True)
class RegulatoryFileValidityUpdateResult:
    """Counts and uniform windows from a file-level temporal update."""

    updated_chunk_count: int
    skipped_versioned_chunk_count: int
    previous_window: RegulatoryFileValidityWindow | None = None
    updated_window: RegulatoryFileValidityWindow | None = None


@dataclass(frozen=True, slots=True)
class RegulatoryChunkValidityState:
    """Temporal/version fields read before destructive re-chunking."""

    source: str
    status: str
    validity_start_date: datetime.date | None
    validity_end_date: datetime.date | None
    supersedes_chunk_id: str | None
    superseded_by_chunk_id: str | None


@dataclass(frozen=True, slots=True)
class _ArticleAnchor:
    key: tuple[str, ...]
    article_no: str
    qualifier: str | None


@dataclass(frozen=True, slots=True)
class _ContiguousArticleLineage:
    identity: tuple[str, str | None]
    heading_path: tuple[str, ...]
    article_title: str
    position: int


@dataclass(slots=True)
class _ProvisionSelectionGroup:
    user_file_id: UUID
    projection_indices: set[int]
    seed_ids: list[str]


def _fold_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    # Turkish dotless i does not decompose under NFKD. Treat it as its ASCII
    # search equivalent so bilingual query concepts such as limitation periods
    # match Turkish source text consistently.
    return " ".join(without_marks.replace("ı", "i").split())


def _normalize_article_no(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = "".join(_fold_text(value).split())
    return normalized or None


def _article_identity_from_metadata(
    article_no: str | None,
) -> tuple[str, str | None] | None:
    if article_no is None:
        return None
    folded = " ".join(_fold_text(article_no).split())
    qualifier: str | None = None
    for prefix in ("gecici ", "mukerrer "):
        if folded.startswith(prefix):
            qualifier = prefix.strip()
            folded = folded.removeprefix(prefix).strip()
            break
    normalized_article_no = _normalize_article_no(folded)
    if normalized_article_no is None:
        return None
    return normalized_article_no, qualifier


def _article_identity_from_heading(
    heading: str,
) -> tuple[str, str | None] | None:
    parsed = parse_regulatory_article_heading(heading)
    if parsed is None:
        return None
    article_no = _normalize_article_no(parsed.article_no)
    if article_no is None:
        return None
    qualifier = _fold_text(parsed.qualifier) if parsed.qualifier else None
    return article_no, qualifier


def _last_article_index(
    heading_path: Sequence[str],
    *,
    identity: tuple[str, str | None],
) -> int | None:
    for index in range(len(heading_path) - 1, -1, -1):
        if _article_identity_from_heading(heading_path[index]) == identity:
            return index
    return None


def _shared_heading_scope_length(
    left: Sequence[str],
    right: Sequence[str],
) -> int:
    shared = 0
    for left_heading, right_heading in zip(left, right, strict=False):
        if _fold_text(left_heading) != _fold_text(right_heading):
            break
        shared += 1
    return shared


def _is_structural_scope_boundary(heading: str) -> bool:
    return _STRUCTURAL_SCOPE_BOUNDARY_RE.match(_fold_text(heading)) is not None


def _crosses_structural_scope_boundary(
    preceding_article_path: Sequence[str],
    candidate_path: Sequence[str],
) -> bool:
    preceding_scope = preceding_article_path[:-1]
    shared_scope_length = _shared_heading_scope_length(
        preceding_scope,
        candidate_path,
    )
    return any(
        _is_structural_scope_boundary(heading)
        for heading in candidate_path[shared_scope_length:]
    )


def _inherit_contiguous_article_lineage(
    heading_path: tuple[str, ...],
    *,
    article_identity: tuple[str, str | None] | None,
    article_title: str | None,
    preceding_lineage: _ContiguousArticleLineage | None,
) -> tuple[str, ...]:
    """Move a legacy descendant beneath its nearest explicit article boundary."""

    if (
        article_identity is None
        or preceding_lineage is None
        or article_identity != preceding_lineage.identity
    ):
        return heading_path
    normalized_title = _fold_text(article_title or "")
    if not normalized_title or normalized_title != preceding_lineage.article_title:
        return heading_path
    article_index = _last_article_index(
        heading_path,
        identity=article_identity,
    )
    if article_index is None:
        return heading_path

    preceding_scope = preceding_lineage.heading_path[:-1]
    candidate_scope = heading_path[:article_index]
    shared_scope_length = _shared_heading_scope_length(
        preceding_scope,
        candidate_scope,
    )
    # A document title alone is not enough to collapse two real structural
    # scopes such as a main text and an annex with the same article number.
    if shared_scope_length < 2:
        return heading_path
    displaced_descendants = candidate_scope[shared_scope_length:]
    if any(_is_structural_scope_boundary(heading) for heading in displaced_descendants):
        return heading_path
    return (
        *preceding_lineage.heading_path,
        *displaced_descendants,
        *heading_path[article_index + 1 :],
    )


def _lineage_from_candidate(
    heading_path: tuple[str, ...],
    *,
    article_identity: tuple[str, str | None] | None,
    article_title: str | None,
    position: int,
) -> _ContiguousArticleLineage | None:
    if article_identity is None:
        return None
    article_index = _last_article_index(heading_path, identity=article_identity)
    if article_index is None:
        return None
    return _ContiguousArticleLineage(
        identity=article_identity,
        heading_path=heading_path[: article_index + 1],
        article_title=_fold_text(article_title or ""),
        position=position,
    )


def _is_snapshot_visible(
    *,
    status: str,
    validity_start_date: datetime.date | None,
    validity_end_date: datetime.date | None,
    as_of_date: datetime.date | None,
) -> bool:
    if as_of_date is None:
        return status == RegulatoryChunkStatus.ACTIVE.value
    if validity_start_date is not None and validity_start_date > as_of_date:
        return False
    if validity_end_date is not None and validity_end_date <= as_of_date:
        return False
    return True


def _last_explicit_article_anchor(
    heading_path: tuple[str, ...],
) -> _ArticleAnchor | None:
    last_anchor: _ArticleAnchor | None = None
    normalized_prefix: list[str] = []
    for heading in heading_path:
        folded_heading = _fold_text(heading)
        parsed_heading = parse_regulatory_article_heading(heading)
        if parsed_heading is None:
            normalized_prefix.append(folded_heading)
            continue

        article_no = _normalize_article_no(parsed_heading.article_no)
        if article_no is None:
            continue
        qualifier = parsed_heading.qualifier or ""
        canonical_heading = f"{qualifier}:madde:{article_no}"
        last_anchor = _ArticleAnchor(
            key=tuple([*normalized_prefix, canonical_heading]),
            article_no=article_no,
            qualifier=parsed_heading.qualifier,
        )
        normalized_prefix.append(canonical_heading)
    return last_anchor


def _candidate_article_no(
    candidate: RegulatoryChunkProjection,
) -> str | None:
    anchor = _last_explicit_article_anchor(candidate.heading_path)
    if anchor is not None:
        return anchor.article_no
    return _normalize_article_no(candidate.article_no)


def _numbered_peer_identity(
    candidate: RegulatoryChunkProjection,
) -> tuple[tuple[str, ...], int] | None:
    """Identify an article-less numbered unit within its exact parent scope."""

    if (
        _candidate_article_no(candidate) is not None
        or candidate.chunk_type not in {"paragraph", "numbered_section"}
        or not candidate.heading_path
    ):
        return None

    match = _NUMBERED_PEER_HEADING_RE.match(candidate.heading_path[-1].strip())
    if match is None:
        return None
    raw_number = match.group("parenthesized") or match.group("plain")
    if raw_number is None:
        return None
    number = int(raw_number)
    if number <= 0:
        return None

    metadata_number = _positive_decimal(candidate.paragraph_no)
    if metadata_number is not None and metadata_number != number:
        return None
    parent_scope = tuple(_fold_text(heading) for heading in candidate.heading_path[:-1])
    return parent_scope, number


def _articleless_numbered_family_anchor(
    candidate: RegulatoryChunkProjection,
) -> tuple[str, ...] | None:
    """Return the exact numbered parent path for a non-MADDE structural family."""

    if _candidate_article_no(candidate) is not None:
        return None
    for heading_index in range(len(candidate.heading_path) - 1, -1, -1):
        heading = candidate.heading_path[heading_index].strip()
        if _NUMBERED_PEER_HEADING_RE.match(heading) is None:
            continue
        return tuple(
            _fold_text(path_part)
            for path_part in candidate.heading_path[: heading_index + 1]
        )
    return None


def _positive_decimal(value: str | None) -> int | None:
    if value is None or not value.isdecimal():
        return None
    number = int(value)
    return number if number > 0 else None


def _is_valid_on(
    candidate: RegulatoryChunkProjection, as_of_date: datetime.date | None
) -> bool:
    return _is_snapshot_visible(
        status=candidate.status,
        validity_start_date=candidate.validity_start_date,
        validity_end_date=candidate.validity_end_date,
        as_of_date=as_of_date,
    )


def _project_candidates(
    candidates: Iterable[RegulatoryChunkSiblingCandidate],
    *,
    as_of_date: datetime.date | None,
) -> tuple[
    dict[UUID, list[RegulatoryChunkProjection]],
    dict[str, RegulatoryChunkProjection],
]:
    candidates_by_file: dict[UUID, list[RegulatoryChunkSiblingCandidate]] = {}
    seen_ids: set[str] = set()
    for candidate in candidates:
        if candidate.regulatory_chunk_id in seen_ids:
            raise ValueError(
                "regulatory chunk candidates must have unique regulatory_chunk_id values"
            )
        seen_ids.add(candidate.regulatory_chunk_id)
        candidates_by_file.setdefault(candidate.user_file_id, []).append(candidate)

    projected_by_file: dict[UUID, list[RegulatoryChunkProjection]] = {}
    projected_by_id: dict[str, RegulatoryChunkProjection] = {}
    for user_file_id, file_candidates in candidates_by_file.items():
        ordered_candidates = sorted(
            file_candidates,
            key=lambda candidate: (
                candidate.position,
                candidate.regulatory_chunk_id,
            ),
        )
        projected_rows: list[RegulatoryChunkProjection] = []
        preceding_lineage: _ContiguousArticleLineage | None = None
        blocked_lineage_identity: tuple[str, str | None] | None = None
        for projection_index, candidate in enumerate(ordered_candidates):
            normalized_heading_path = (
                tuple(
                    normalize_regulatory_heading_path(
                        candidate.heading_path,
                        article_no=candidate.article_no,
                        chunk_type=candidate.chunk_type,
                        paragraph_no=candidate.paragraph_no,
                        clause_label=candidate.clause_label,
                    )
                )
                if candidate.chunk_type is not None or not candidate.heading_path
                else candidate.heading_path
            )
            raw_article_identities = [
                identity
                for heading in candidate.heading_path
                if (identity := _article_identity_from_heading(heading)) is not None
            ]
            snapshot_visible = _is_snapshot_visible(
                status=candidate.status,
                validity_start_date=candidate.validity_start_date,
                validity_end_date=candidate.validity_end_date,
                as_of_date=as_of_date,
            )
            if snapshot_visible:
                if raw_article_identities:
                    current_lineage = _lineage_from_candidate(
                        normalized_heading_path,
                        article_identity=raw_article_identities[-1],
                        article_title=candidate.article_title,
                        position=candidate.position,
                    )
                    if (
                        preceding_lineage is not None
                        and preceding_lineage.position == candidate.position
                        and current_lineage is not None
                        and current_lineage != preceding_lineage
                    ):
                        blocked_lineage_identity = current_lineage.identity
                        preceding_lineage = None
                    elif (
                        blocked_lineage_identity is None
                        or current_lineage is None
                        or current_lineage.identity != blocked_lineage_identity
                    ):
                        preceding_lineage = current_lineage
                        blocked_lineage_identity = None
                elif candidate.article_no is not None:
                    article_identity = _article_identity_from_metadata(
                        candidate.article_no
                    )
                    if article_identity != blocked_lineage_identity:
                        inherited_heading_path = _inherit_contiguous_article_lineage(
                            normalized_heading_path,
                            article_identity=article_identity,
                            article_title=candidate.article_title,
                            preceding_lineage=preceding_lineage,
                        )
                        if inherited_heading_path != normalized_heading_path:
                            normalized_heading_path = inherited_heading_path
                        else:
                            preceding_lineage = _lineage_from_candidate(
                                normalized_heading_path,
                                article_identity=article_identity,
                                article_title=candidate.article_title,
                                position=candidate.position,
                            )
                            blocked_lineage_identity = None
                elif preceding_lineage is not None and (
                    _crosses_structural_scope_boundary(
                        preceding_lineage.heading_path,
                        normalized_heading_path,
                    )
                ):
                    preceding_lineage = None
                    blocked_lineage_identity = None
            projected = RegulatoryChunkProjection(
                regulatory_chunk_id=candidate.regulatory_chunk_id,
                user_file_id=user_file_id,
                projection_index=projection_index,
                position=candidate.position,
                text=candidate.text,
                heading_path=normalized_heading_path,
                article_no=candidate.article_no,
                chunk_type=candidate.chunk_type,
                paragraph_no=candidate.paragraph_no,
                clause_label=candidate.clause_label,
                validity_start_date=candidate.validity_start_date,
                validity_end_date=candidate.validity_end_date,
                status=candidate.status,
            )
            projected_rows.append(projected)
            projected_by_id[projected.regulatory_chunk_id] = projected
        projected_by_file[user_file_id] = projected_rows
    return projected_by_file, projected_by_id


def _relation_to_seed_provision(
    candidate: RegulatoryChunkProjection,
    *,
    seed_article_no: str,
    seed_anchor: _ArticleAnchor | None,
) -> int:
    """Return 1 for same provision, 0 for a metadata bridge, -1 for boundary."""
    candidate_anchor = _last_explicit_article_anchor(candidate.heading_path)
    candidate_article_no = _candidate_article_no(candidate)

    if candidate_anchor is not None:
        if seed_anchor is not None and candidate_anchor.key != seed_anchor.key:
            return -1
        if candidate_anchor.article_no != seed_article_no:
            return -1

    if candidate_article_no is not None:
        return 1 if candidate_article_no == seed_article_no else -1
    if candidate_anchor is not None:
        return 1
    return 0


def _provision_span_for_seed(
    rows: Sequence[RegulatoryChunkProjection],
    seed_index: int,
    *,
    as_of_date: datetime.date | None,
) -> set[int]:
    seed = rows[seed_index]
    seed_anchor = _last_explicit_article_anchor(seed.heading_path)
    seed_article_no = _candidate_article_no(seed)
    if seed_article_no is None:
        numbered_family_anchor = _articleless_numbered_family_anchor(seed)
        if numbered_family_anchor is not None:
            family_indices = {seed_index}
            for direction in (-1, 1):
                cursor = seed_index + direction
                while 0 <= cursor < len(rows):
                    candidate = rows[cursor]
                    cursor += direction
                    if not _is_valid_on(candidate, as_of_date):
                        continue
                    if (
                        _articleless_numbered_family_anchor(candidate)
                        != numbered_family_anchor
                    ):
                        break
                    family_indices.add(candidate.projection_index)
            if len(family_indices) > 1:
                return family_indices

        numbered_peer_identity = _numbered_peer_identity(seed)
        if numbered_peer_identity is not None:
            peer_scope, seed_number = numbered_peer_identity
            peer_indices = {seed_index}
            for direction in (-1, 1):
                cursor = seed_index + direction
                while 0 <= cursor < len(rows):
                    candidate = rows[cursor]
                    cursor += direction
                    if not _is_valid_on(candidate, as_of_date):
                        continue
                    candidate_identity = _numbered_peer_identity(candidate)
                    if (
                        candidate_identity is not None
                        and candidate_identity[0] == peer_scope
                        and candidate_identity[1] == seed_number + direction
                    ):
                        peer_indices.add(candidate.projection_index)
                    break
            return peer_indices

        identified_neighbors: list[RegulatoryChunkProjection] = []
        for direction in (-1, 1):
            cursor = seed_index + direction
            while 0 <= cursor < len(rows):
                candidate = rows[cursor]
                cursor += direction
                if not _is_valid_on(candidate, as_of_date):
                    continue
                if _candidate_article_no(candidate) is None:
                    continue
                identified_neighbors.append(candidate)
                break
        if len(identified_neighbors) != 2:
            return {seed_index}
        left_neighbor, right_neighbor = identified_neighbors
        left_article_no = _candidate_article_no(left_neighbor)
        right_article_no = _candidate_article_no(right_neighbor)
        if left_article_no is None or left_article_no != right_article_no:
            return {seed_index}
        left_anchor = _last_explicit_article_anchor(left_neighbor.heading_path)
        right_anchor = _last_explicit_article_anchor(right_neighbor.heading_path)
        if (
            left_anchor is not None
            and right_anchor is not None
            and left_anchor.key != right_anchor.key
        ):
            return {seed_index}
        seed_article_no = left_article_no
        seed_anchor = left_anchor or right_anchor

    selected_indices = {seed_index}
    for direction in (-1, 1):
        pending_bridge_indices: list[int] = []
        cursor = seed_index + direction
        while 0 <= cursor < len(rows):
            candidate_index = cursor
            candidate = rows[candidate_index]
            cursor += direction
            if not _is_valid_on(candidate, as_of_date):
                continue
            relation = _relation_to_seed_provision(
                candidate,
                seed_article_no=seed_article_no,
                seed_anchor=seed_anchor,
            )
            if relation < 0:
                break
            if relation == 0:
                pending_bridge_indices.append(candidate_index)
            else:
                selected_indices.update(pending_bridge_indices)
                selected_indices.add(candidate_index)
                pending_bridge_indices.clear()
        # Unknown rows at an outer edge are not a bridge and are discarded.
    return selected_indices


def _merge_seed_spans(
    projected_by_file: dict[UUID, list[RegulatoryChunkProjection]],
    projected_by_id: dict[str, RegulatoryChunkProjection],
    seed_chunk_ids: Sequence[str],
    as_of_date: datetime.date | None,
) -> list[_ProvisionSelectionGroup]:
    groups: list[_ProvisionSelectionGroup] = []
    seen_seed_ids: set[str] = set()
    for seed_id in seed_chunk_ids:
        if seed_id in seen_seed_ids:
            continue
        seen_seed_ids.add(seed_id)
        seed = projected_by_id.get(seed_id)
        if seed is None or not _is_valid_on(seed, as_of_date):
            continue

        file_rows = projected_by_file[seed.user_file_id]
        span = _provision_span_for_seed(
            file_rows,
            seed.projection_index,
            as_of_date=as_of_date,
        )
        if _has_overlapping_visible_positions(
            file_rows,
            span,
            as_of_date=as_of_date,
        ):
            continue
        overlapping_group_indices = [
            index
            for index, group in enumerate(groups)
            if group.user_file_id == seed.user_file_id
            and not group.projection_indices.isdisjoint(span)
        ]
        if not overlapping_group_indices:
            groups.append(
                _ProvisionSelectionGroup(
                    user_file_id=seed.user_file_id,
                    projection_indices=set(span),
                    seed_ids=[seed_id],
                )
            )
            continue

        target_index = overlapping_group_indices[0]
        target = groups[target_index]
        target.projection_indices.update(span)
        target.seed_ids.append(seed_id)
        for redundant_index in reversed(overlapping_group_indices[1:]):
            redundant = groups.pop(redundant_index)
            target.projection_indices.update(redundant.projection_indices)
            target.seed_ids.extend(redundant.seed_ids)
    return groups


def _has_overlapping_visible_positions(
    rows: Sequence[RegulatoryChunkProjection],
    indices: Iterable[int],
    *,
    as_of_date: datetime.date | None,
) -> bool:
    visible_chunk_id_by_position: dict[int, str] = {}
    for index in indices:
        row = rows[index]
        if not _is_valid_on(row, as_of_date):
            continue
        existing_chunk_id = visible_chunk_id_by_position.setdefault(
            row.position,
            row.regulatory_chunk_id,
        )
        if existing_chunk_id != row.regulatory_chunk_id:
            return True
    return False


def _lexical_terms(value: str) -> set[str]:
    return set(_LEXICAL_TERM_RE.findall(_fold_text(value)))


def _focused_term_overlap_score(query_terms: set[str], value: str) -> int:
    """Score exact and conservative inflection-tolerant lexical overlap."""

    value_terms = _lexical_terms(value)
    score = 0
    for query_term in query_terms:
        if len(query_term) < 4:
            continue
        if query_term in value_terms:
            score += 5 if query_term.isdigit() else 3
            continue
        if query_term.isdigit():
            continue
        if any(
            len(value_term) >= 4 and value_term[:4] == query_term[:4]
            for value_term in value_terms
        ):
            score += 2
    return score


def _positive_list_number(row: RegulatoryChunkProjection) -> int | None:
    metadata_number = _positive_decimal(row.paragraph_no)
    if metadata_number is not None:
        return metadata_number
    identity = _numbered_peer_identity(row)
    return identity[1] if identity is not None else None


def _local_paragraph_reference_numbers(text: str) -> tuple[int, ...]:
    """Extract a small, explicit list of local paragraph references.

    Article and other scope references intentionally are not interpreted here.
    Unsupported or excessive lists fail closed instead of widening retrieval.
    """

    folded_text = _fold_text(text)
    occurrences: list[tuple[int, tuple[int, ...]]] = []
    for pattern in (
        _TURKISH_LOCAL_PARAGRAPH_REFERENCE_RE,
        _ENGLISH_LOCAL_PARAGRAPH_REFERENCE_RE,
    ):
        for match in pattern.finditer(folded_text):
            numbers = tuple(
                int(value) for value in re.findall(r"\d{1,3}", match.group("numbers"))
            )
            if numbers and all(number > 0 for number in numbers):
                occurrences.append((match.start(), numbers))

    references: list[int] = []
    seen: set[int] = set()
    for _offset, numbers in sorted(occurrences, key=lambda occurrence: occurrence[0]):
        for number in numbers:
            if number in seen:
                continue
            if len(references) >= _MAX_LOCAL_PARAGRAPH_REFERENCES:
                return ()
            seen.add(number)
            references.append(number)
    return tuple(references)


def _local_paragraph_companion_rows(
    rows: Sequence[RegulatoryChunkProjection],
    seed: RegulatoryChunkProjection,
) -> tuple[RegulatoryChunkProjection, ...]:
    """Resolve explicit paragraph references only within the seed's article."""

    seed_anchor = _last_explicit_article_anchor(seed.heading_path)
    if seed_anchor is None:
        return ()
    if extract_regulatory_provision_references(seed.text):
        return ()

    reference_numbers = _local_paragraph_reference_numbers(seed.text)
    if not reference_numbers:
        return ()

    candidates_by_number: dict[int, list[RegulatoryChunkProjection]] = {
        number: [] for number in reference_numbers
    }
    for row in rows:
        if (
            row.regulatory_chunk_id == seed.regulatory_chunk_id
            or row.chunk_type not in {"paragraph", "numbered_section"}
        ):
            continue
        paragraph_number = _positive_decimal(row.paragraph_no)
        if paragraph_number not in candidates_by_number:
            continue
        row_anchor = _last_explicit_article_anchor(row.heading_path)
        if row_anchor is None or row_anchor.key != seed_anchor.key:
            continue
        candidates_by_number[paragraph_number].append(row)

    companions: list[RegulatoryChunkProjection] = []
    for reference_number in reference_numbers:
        candidates = candidates_by_number[reference_number]
        if len(candidates) == 1:
            companions.append(candidates[0])
    return tuple(companions)


def _is_terse_numbered_item(row: RegulatoryChunkProjection) -> bool:
    return (
        row.chunk_type in {"paragraph", "numbered_section"}
        and _positive_list_number(row) is not None
        and len(row.text) <= _TERSE_LIST_ITEM_MAX_CHARS
        and len(_LEXICAL_TERM_RE.findall(row.text)) <= _TERSE_LIST_ITEM_MAX_TERMS
    )


def _is_terse_structural_intro(row: RegulatoryChunkProjection) -> bool:
    """Whether a short row grammatically opens a following child run."""

    stripped = row.text.rstrip()
    return (
        row.chunk_type in {"paragraph", "numbered_section"}
        and len(stripped) <= _TERSE_STRUCTURAL_INTRO_MAX_CHARS
        and stripped.endswith((":", ";"))
    )


def _structural_companion_ids(
    rows: Sequence[RegulatoryChunkProjection],
    seeds: Sequence[RegulatoryChunkProjection],
) -> list[str]:
    """Choose bounded context chunks without rewriting evidentiary text.

    Explicit local paragraph references stay within the current article. A
    clause depends on the nearest paragraph before its run. Numbered lists
    depend on their introduction, and a terse item in a restarted list can be
    ambiguous without the nearest earlier item carrying the same ordinal.
    These are retrieval leads only: each companion remains an independently
    citable source chunk and the caller's existing budgets still apply.
    """

    index_by_id = {row.regulatory_chunk_id: index for index, row in enumerate(rows)}
    companion_ids: list[str] = []
    seen_ids = {seed.regulatory_chunk_id for seed in seeds}

    def add(row: RegulatoryChunkProjection) -> None:
        if row.regulatory_chunk_id in seen_ids:
            return
        seen_ids.add(row.regulatory_chunk_id)
        companion_ids.append(row.regulatory_chunk_id)

    def direct_clause_parent_path(
        row: RegulatoryChunkProjection,
    ) -> tuple[str, ...] | None:
        if row.chunk_type != "clause" and row.clause_label is None:
            return None
        if len(row.heading_path) < 2:
            return None
        return tuple(_fold_text(value) for value in row.heading_path[:-1])

    for seed in seeds:
        seed_index = index_by_id[seed.regulatory_chunk_id]

        # Descendant clauses and paragraphs often carry only the narrow branch,
        # while the article lead carries the operative subject and trigger. Keep
        # that parent independently citable and inside the same bounded packet.
        if seed.chunk_type != "article":
            seed_anchor = _last_explicit_article_anchor(seed.heading_path)
            if seed_anchor is not None:
                article_parents = [
                    row
                    for row in rows
                    if row.chunk_type == "article"
                    and (row_anchor := _last_explicit_article_anchor(row.heading_path))
                    is not None
                    and row_anchor.key == seed_anchor.key
                ]
                if article_parents:
                    nearest_parent = min(
                        article_parents,
                        key=lambda row: (
                            abs(row.projection_index - seed.projection_index),
                            row.projection_index,
                            row.regulatory_chunk_id,
                        ),
                    )
                    add(nearest_parent)
                    # Long provisions can split their operative lead across
                    # consecutive article chunks. Keep the earliest lead as a
                    # second bounded companion while retaining the nearest
                    # parent as the first choice under a two-chunk budget.
                    add(
                        min(
                            article_parents,
                            key=lambda row: (
                                row.projection_index,
                                row.regulatory_chunk_id,
                            ),
                        )
                    )

        for companion in _local_paragraph_companion_rows(rows, seed):
            add(companion)

        numbered_peer_identity = _numbered_peer_identity(seed)
        if numbered_peer_identity is not None:
            peer_scope, seed_number = numbered_peer_identity
            for direction in (-1, 1):
                candidate_index = seed_index + direction
                if not 0 <= candidate_index < len(rows):
                    continue
                candidate = rows[candidate_index]
                candidate_identity = _numbered_peer_identity(candidate)
                if (
                    candidate_identity is not None
                    and candidate_identity[0] == peer_scope
                    and candidate_identity[1] == seed_number + direction
                ):
                    add(candidate)

        if seed.chunk_type == "clause" or seed.clause_label is not None:
            for candidate_index in range(seed_index - 1, -1, -1):
                candidate = rows[candidate_index]
                if candidate.chunk_type == "clause" or candidate.clause_label:
                    continue
                if candidate.chunk_type in {"paragraph", "numbered_section"}:
                    add(candidate)
                break

            # Labelled limbs under one direct grammatical parent form a
            # bounded operative set. Preserve the peer conditions and
            # exceptions without crossing into a restarted list. The parent
            # is admitted first because it may carry the inherited operator.
            clause_parent_path = direct_clause_parent_path(seed)
            if clause_parent_path is not None:
                for candidate in rows:
                    if (
                        candidate.regulatory_chunk_id != seed.regulatory_chunk_id
                        and direct_clause_parent_path(candidate) == clause_parent_path
                    ):
                        add(candidate)

        # A terse lead-in such as a definition/list operator is not useful on
        # its own, while each child can inherit its legal direction from that
        # lead-in. Build a bounded same-article packet by prioritizing the
        # consecutive child run. Children remain independent chunks/citations;
        # this only guarantees that retrieval presents the grammatical unit.
        if _is_terse_structural_intro(seed):
            seed_anchor = _last_explicit_article_anchor(seed.heading_path)
            folded_seed_path = tuple(_fold_text(value) for value in seed.heading_path)
            for candidate_index in range(seed_index + 1, len(rows)):
                candidate = rows[candidate_index]
                candidate_anchor = _last_explicit_article_anchor(candidate.heading_path)
                if (
                    seed_anchor is None
                    or candidate_anchor is None
                    or candidate_anchor.key != seed_anchor.key
                ):
                    break
                folded_candidate_path = tuple(
                    _fold_text(value) for value in candidate.heading_path
                )
                if candidate.chunk_type != "clause" and candidate.clause_label is None:
                    break
                if folded_candidate_path[: len(folded_seed_path)] != folded_seed_path:
                    break
                add(candidate)

        seed_number = _positive_list_number(seed)
        if seed_number is None:
            continue

        run_start_index = seed_index
        expected_number = seed_number
        while expected_number > 1 and run_start_index > 0:
            previous = rows[run_start_index - 1]
            if _positive_list_number(previous) != expected_number - 1:
                break
            run_start_index -= 1
            expected_number -= 1

        if expected_number == 1 and run_start_index > 0:
            possible_intro = rows[run_start_index - 1]
            if _positive_list_number(possible_intro) is None:
                add(possible_intro)

        if not _is_terse_numbered_item(seed):
            continue
        for candidate_index in range(run_start_index - 1, -1, -1):
            candidate = rows[candidate_index]
            if _positive_list_number(candidate) == seed_number:
                add(candidate)
                break

    return companion_ids


def _select_group_within_budget(
    rows: Sequence[RegulatoryChunkProjection],
    group: _ProvisionSelectionGroup,
    *,
    query_terms: set[str],
    as_of_date: datetime.date | None,
    max_chunks: int,
    max_chars: int,
) -> list[RegulatoryChunkProjection]:
    visible_rows = sorted(
        [
            rows[index]
            for index in group.projection_indices
            if _is_valid_on(rows[index], as_of_date)
        ],
        key=lambda row: (row.projection_index, row.regulatory_chunk_id),
    )
    visible_article_numbers = {
        article_no
        for row in visible_rows
        if (article_no := _candidate_article_no(row)) is not None
    }
    if len(visible_article_numbers) == 1:
        provision_article_no = next(iter(visible_article_numbers))
        visible_rows = [
            replace(
                row,
                article_no=provision_article_no,
                heading_path=tuple(
                    normalize_regulatory_heading_path(
                        row.heading_path,
                        article_no=provision_article_no,
                        chunk_type=row.chunk_type,
                        paragraph_no=row.paragraph_no,
                        clause_label=row.clause_label,
                    )
                ),
            )
            if _candidate_article_no(row) is None
            else row
            for row in visible_rows
        ]
    seed_id_set = set(group.seed_ids)
    seeds = [row for row in visible_rows if row.regulatory_chunk_id in seed_id_set]
    seed_projection_indices = [seed.projection_index for seed in seeds]

    # Seed inclusion has priority when caller input itself exceeds a budget;
    # sibling expansion never makes that exceptional overage larger.
    selected_by_id = {seed.regulatory_chunk_id: seed for seed in seeds}
    selected_chars = sum(len(seed.text) for seed in seeds)

    row_by_id = {row.regulatory_chunk_id: row for row in visible_rows}
    structural_companion_ids = _structural_companion_ids(visible_rows, seeds)
    structural_companion_id_set = set(structural_companion_ids)
    preserves_restarted_list_mapping = any(
        _is_terse_numbered_item(seed) for seed in seeds
    )

    # A large grammatical family can otherwise consume the entire local window
    # before a separate sibling that directly matches the focused query is
    # considered. Reserve part of the remaining window only for lexical matches
    # outside that family. Query-matched structural context keeps its ordinary
    # structural priority and cannot displace required parents or list mappings.
    query_matched_rows = sorted(
        [
            row
            for row in visible_rows
            if row.regulatory_chunk_id not in selected_by_id
            and row.regulatory_chunk_id not in structural_companion_id_set
            and not preserves_restarted_list_mapping
            and _focused_term_overlap_score(query_terms, row.text) > 0
        ],
        key=lambda row: (
            -_focused_term_overlap_score(query_terms, row.text),
            min(
                abs(row.projection_index - seed_index)
                for seed_index in seed_projection_indices
            ),
            row.projection_index,
            row.regulatory_chunk_id,
        ),
    )
    available_non_seed_slots = max(0, max_chunks - len(selected_by_id))
    query_match_reserve = min(
        len(query_matched_rows),
        (max(1, available_non_seed_slots // 3) if available_non_seed_slots >= 2 else 0),
    )
    for row in query_matched_rows[:query_match_reserve]:
        if selected_chars + len(row.text) > max_chars:
            continue
        selected_by_id[row.regulatory_chunk_id] = replace(
            row,
            expansion_priority=-1,
        )
        selected_chars += len(row.text)

    for companion_id in structural_companion_ids:
        if len(selected_by_id) >= max_chunks:
            break
        row = row_by_id[companion_id]
        if selected_chars + len(row.text) > max_chars:
            continue
        selected_by_id[row.regulatory_chunk_id] = replace(row, expansion_priority=0)
        selected_chars += len(row.text)

    non_seeds = [
        row for row in visible_rows if row.regulatory_chunk_id not in selected_by_id
    ]
    ranked_non_seeds = sorted(
        non_seeds,
        key=lambda row: (
            -_focused_term_overlap_score(query_terms, row.text),
            min(
                abs(row.projection_index - seed_index)
                for seed_index in seed_projection_indices
            ),
            row.projection_index,
            row.regulatory_chunk_id,
        ),
    )
    for row in ranked_non_seeds:
        if len(selected_by_id) >= max_chunks:
            break
        if selected_chars + len(row.text) > max_chars:
            continue
        selected_by_id[row.regulatory_chunk_id] = row
        selected_chars += len(row.text)

    return sorted(
        selected_by_id.values(),
        key=lambda row: (row.projection_index, row.regulatory_chunk_id),
    )


def select_bounded_same_provision_siblings(
    candidates: Iterable[RegulatoryChunkSiblingCandidate],
    seed_chunk_ids: Sequence[str],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_chunks_per_provision: int = DEFAULT_PROVISION_MAX_CHUNKS,
    max_chars_per_provision: int = DEFAULT_PROVISION_MAX_CHARS,
) -> list[RegulatoryChunkProjection]:
    """Select bounded, date-valid local provision runs around seed chunks.

    Repeated article numbers elsewhere in a file are never globally grouped.
    Exact article-less numbered families and their immediate numbered peers are
    admitted only within the same parent heading scope. Other metadata-free rows
    must bridge two rows structurally compatible with the seed provision.
    """
    if max_chunks_per_provision <= 0:
        raise ValueError("max_chunks_per_provision must be positive")
    if max_chars_per_provision <= 0:
        raise ValueError("max_chars_per_provision must be positive")
    if not seed_chunk_ids:
        return []

    projected_by_file, projected_by_id = _project_candidates(
        candidates,
        as_of_date=as_of_date,
    )
    groups = _merge_seed_spans(
        projected_by_file,
        projected_by_id,
        seed_chunk_ids,
        as_of_date,
    )
    query_terms = _lexical_terms(query)
    selected: list[RegulatoryChunkProjection] = []
    selected_ids: set[str] = set()
    for group in groups:
        group_rows = _select_group_within_budget(
            projected_by_file[group.user_file_id],
            group,
            query_terms=query_terms,
            as_of_date=as_of_date,
            max_chunks=max_chunks_per_provision,
            max_chars=max_chars_per_provision,
        )
        for row in group_rows:
            if row.regulatory_chunk_id not in selected_ids:
                selected.append(row)
                selected_ids.add(row.regulatory_chunk_id)
    return selected


def select_bounded_adjacent_provisions(
    candidates: Iterable[RegulatoryChunkSiblingCandidate],
    seed_chunk_ids: Sequence[str],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_provisions: int = DEFAULT_ADJACENT_MAX_PROVISIONS,
    max_chunks_per_provision: int = DEFAULT_ADJACENT_MAX_CHUNKS_PER_PROVISION,
    max_total_chars: int = DEFAULT_ADJACENT_MAX_TOTAL_CHARS,
) -> list[RegulatoryChunkProjection]:
    """Select the immediately preceding/following provisions in the same scope.

    This is a bounded context window around real search hits, not a numeric
    article guess. It follows document order and refuses cross-annex/part or
    overlapping-version boundaries.
    """

    if max_provisions <= 0 or max_chunks_per_provision <= 0 or max_total_chars <= 0:
        return []
    projected_by_file, projected_by_id = _project_candidates(
        candidates,
        as_of_date=as_of_date,
    )
    visible_seeds = [
        seed
        for seed_id in dict.fromkeys(seed_chunk_ids)
        if (seed := projected_by_id.get(seed_id)) is not None
        and _is_valid_on(seed, as_of_date)
    ]
    query_terms = _lexical_terms(query)
    selected: list[RegulatoryChunkProjection] = []
    selected_ids: set[str] = set(seed_chunk_ids)
    selected_anchor_keys: set[tuple[str, ...]] = set()
    selected_chars = 0

    for seed in visible_seeds:
        if len(selected_anchor_keys) >= max_provisions:
            break
        seed_anchor = _last_explicit_article_anchor(seed.heading_path)
        if seed_anchor is None:
            continue
        rows = projected_by_file[seed.user_file_id]
        ordered_anchor_starts: list[tuple[_ArticleAnchor, int]] = []
        seen_anchor_keys: set[tuple[str, ...]] = set()
        for row_index, row in enumerate(rows):
            if not _is_valid_on(row, as_of_date):
                continue
            anchor = _last_explicit_article_anchor(row.heading_path)
            if anchor is None or anchor.key in seen_anchor_keys:
                continue
            seen_anchor_keys.add(anchor.key)
            ordered_anchor_starts.append((anchor, row_index))
        seed_anchor_index = next(
            (
                index
                for index, (anchor, _) in enumerate(ordered_anchor_starts)
                if anchor.key == seed_anchor.key
            ),
            None,
        )
        if seed_anchor_index is None:
            continue

        for neighbor_index in (seed_anchor_index - 1, seed_anchor_index + 1):
            if (
                neighbor_index < 0
                or neighbor_index >= len(ordered_anchor_starts)
                or len(selected_anchor_keys) >= max_provisions
            ):
                continue
            neighbor_anchor, neighbor_start = ordered_anchor_starts[neighbor_index]
            if neighbor_anchor.key[:-1] != seed_anchor.key[:-1]:
                continue
            if neighbor_anchor.key in selected_anchor_keys:
                continue
            span = _provision_span_for_seed(
                rows,
                neighbor_start,
                as_of_date=as_of_date,
            )
            if _has_overlapping_visible_positions(
                rows,
                span,
                as_of_date=as_of_date,
            ):
                continue
            group = _ProvisionSelectionGroup(
                user_file_id=seed.user_file_id,
                projection_indices=span,
                seed_ids=[rows[neighbor_start].regulatory_chunk_id],
            )
            neighbor_rows = _select_group_within_budget(
                rows,
                group,
                query_terms=query_terms,
                as_of_date=as_of_date,
                max_chunks=max_chunks_per_provision,
                max_chars=max_total_chars - selected_chars,
            )
            admitted = [
                row
                for row in neighbor_rows
                if row.regulatory_chunk_id not in selected_ids
                and selected_chars + len(row.text) <= max_total_chars
            ]
            if not admitted:
                continue
            selected_anchor_keys.add(neighbor_anchor.key)
            for row in admitted:
                selected.append(replace(row, expansion_priority=2))
                selected_ids.add(row.regulatory_chunk_id)
                selected_chars += len(row.text)
    return selected


def select_bounded_source_lexical_matches(
    candidates: Iterable[RegulatoryChunkSiblingCandidate],
    *,
    user_file_id: UUID,
    query: str,
    as_of_date: datetime.date | None,
    excluded_chunk_ids: Sequence[str] = (),
    max_matches: int = 2,
    max_total_chars: int = 4_000,
) -> list[RegulatoryChunkProjection]:
    """Select rare lexical matches inside one already-retrieved source.

    This bounded fallback never chooses a document. It only rescans the single
    source selected by ordinary indexed retrieval and uses within-document term
    rarity to keep boilerplate from outranking a focused request term.
    """

    if max_matches <= 0 or max_total_chars <= 0:
        return []
    projected_by_file, _ = _project_candidates(candidates, as_of_date=as_of_date)
    visible_rows = [
        row
        for row in projected_by_file.get(user_file_id, [])
        if _is_valid_on(row, as_of_date)
        and row.regulatory_chunk_id not in set(excluded_chunk_ids)
    ]
    query_terms = {
        term for term in _lexical_terms(query) if len(term) >= 4 or term.isdigit()
    }
    if not visible_rows or not query_terms:
        return []

    row_terms = {
        row.regulatory_chunk_id: _lexical_terms(" ".join((*row.heading_path, row.text)))
        for row in visible_rows
    }

    def matches(query_term: str, value_term: str) -> tuple[bool, bool]:
        if query_term == value_term:
            return True, True
        if query_term.isdigit() or len(query_term) < 4 or len(value_term) < 4:
            return False, False
        return query_term[:4] == value_term[:4], False

    document_frequency: dict[str, int] = {}
    for query_term in query_terms:
        document_frequency[query_term] = sum(
            1
            for terms in row_terms.values()
            if any(matches(query_term, value_term)[0] for value_term in terms)
        )

    ranked_rows: list[tuple[float, int, int, RegulatoryChunkProjection]] = []
    row_count = len(visible_rows)
    for row in visible_rows:
        terms = row_terms[row.regulatory_chunk_id]
        score = 0.0
        matched_terms = 0
        exact_terms = 0
        for query_term in query_terms:
            term_matches = [matches(query_term, value_term) for value_term in terms]
            if not any(is_match for is_match, _ in term_matches):
                continue
            is_exact = any(is_exact for _, is_exact in term_matches)
            rarity = (
                math.log((row_count + 1) / (document_frequency[query_term] + 1)) + 1.0
            )
            score += rarity * (3.0 if is_exact else 2.0)
            matched_terms += 1
            exact_terms += int(is_exact)
        if matched_terms:
            ranked_rows.append((score, matched_terms, exact_terms, row))

    ranked_rows.sort(
        key=lambda item: (
            -item[0],
            -item[1],
            -item[2],
            item[3].projection_index,
            item[3].regulatory_chunk_id,
        )
    )
    selected: list[RegulatoryChunkProjection] = []
    selected_ids: set[str] = set()
    selected_chars = 0
    selected_provisions: set[tuple[str, ...]] = set()

    def provision_key(row: RegulatoryChunkProjection) -> tuple[str, ...]:
        anchor = _last_explicit_article_anchor(row.heading_path)
        return anchor.key if anchor is not None else (row.regulatory_chunk_id,)

    for require_new_provision in (True, False):
        for _, _, _, row in ranked_rows:
            if row.regulatory_chunk_id in selected_ids:
                continue
            key = provision_key(row)
            if require_new_provision and key in selected_provisions:
                continue
            if selected_chars + len(row.text) > max_total_chars:
                continue
            selected.append(replace(row, expansion_priority=-2))
            selected_ids.add(row.regulatory_chunk_id)
            selected_provisions.add(key)
            selected_chars += len(row.text)
            if len(selected) >= max_matches:
                return selected
    return selected


def get_bounded_source_lexical_matches(
    db_session: Session,
    *,
    user_file_id: UUID,
    query: str,
    as_of_date: datetime.date | None,
    excluded_chunk_ids: Sequence[str] = (),
    max_matches: int = 2,
    max_total_chars: int = 4_000,
) -> list[RegulatoryChunkProjection]:
    """Load one known source and return its bounded lexical fallback matches."""

    all_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == user_file_id)
        ).all()
    )
    return select_bounded_source_lexical_matches(
        [_regulatory_sibling_candidate(row) for row in all_rows],
        user_file_id=user_file_id,
        query=query,
        as_of_date=as_of_date,
        excluded_chunk_ids=excluded_chunk_ids,
        max_matches=max_matches,
        max_total_chars=max_total_chars,
    )


def _rank_regulatory_navigation_seed_files(
    seeds: Sequence[RegulatoryNavigationSeed],
    *,
    minimum_seed_count: int = 2,
) -> list[tuple[UUID, tuple[int, ...]]]:
    """Rank eligible sources by selected-hit count, then first-result order.

    Duplicate chunk ids cannot manufacture dominance. A tie is resolved by
    the first selected result, which preserves the search ranking.
    """
    if minimum_seed_count <= 0:
        raise ValueError("minimum_seed_count must be positive")

    positions_by_file: dict[UUID, list[int]] = {}
    first_index_by_file: dict[UUID, int] = {}
    seen_chunk_ids: set[str] = set()
    for index, seed in enumerate(seeds):
        if seed.regulatory_chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(seed.regulatory_chunk_id)
        positions_by_file.setdefault(seed.user_file_id, []).append(seed.position)
        first_index_by_file.setdefault(seed.user_file_id, index)

    eligible_files = sorted(
        (
            user_file_id
            for user_file_id, positions in positions_by_file.items()
            if len(positions) >= minimum_seed_count
        ),
        key=lambda user_file_id: (
            -len(positions_by_file[user_file_id]),
            first_index_by_file[user_file_id],
            str(user_file_id),
        ),
    )
    return [
        (user_file_id, tuple(sorted(set(positions_by_file[user_file_id]))))
        for user_file_id in eligible_files
    ]


def select_dominant_regulatory_navigation_seed_file(
    seeds: Sequence[RegulatoryNavigationSeed],
    *,
    minimum_seed_count: int = 2,
) -> tuple[UUID, tuple[int, ...]] | None:
    """Choose one eligible source while preserving search-result dominance."""
    ranked_files = _rank_regulatory_navigation_seed_files(
        seeds,
        minimum_seed_count=minimum_seed_count,
    )
    return ranked_files[0] if ranked_files else None


def select_bounded_referenced_provisions(
    candidates: Iterable[RegulatoryChunkSiblingCandidate],
    seed_chunk_ids: Sequence[str],
    references: Sequence[RegulatoryProvisionReference],
    *,
    as_of_date: datetime.date | None,
    query: str = "",
    max_provisions: int = DEFAULT_REFERENCE_MAX_PROVISIONS,
    max_chunks_per_provision: int = DEFAULT_REFERENCE_MAX_CHUNKS_PER_PROVISION,
    max_chars_per_provision: int = DEFAULT_REFERENCE_MAX_CHARS_PER_PROVISION,
) -> list[RegulatoryChunkProjection]:
    """Resolve explicit one-hop article references within one visible source.

    A bare article number is followed only when it identifies exactly one
    structural provision anchor in the dominant source. Multiple paragraphs
    under that anchor are expected; multiple enclosing scopes or overlapping
    versions are treated as ambiguous and are skipped.
    """

    if max_provisions <= 0:
        raise ValueError("max_provisions must be positive")
    if max_chunks_per_provision <= 0:
        raise ValueError("max_chunks_per_provision must be positive")
    if max_chars_per_provision <= 0:
        raise ValueError("max_chars_per_provision must be positive")
    if not seed_chunk_ids or not references:
        return []

    projected_by_file, projected_by_id = _project_candidates(
        candidates,
        as_of_date=as_of_date,
    )
    unique_seed_ids = list(dict.fromkeys(seed_chunk_ids))
    visible_seeds = [
        seed
        for seed_id in unique_seed_ids
        if (seed := projected_by_id.get(seed_id)) is not None
        and _is_valid_on(seed, as_of_date)
    ]
    dominant = select_dominant_regulatory_navigation_seed_file(
        [
            RegulatoryNavigationSeed(
                regulatory_chunk_id=seed.regulatory_chunk_id,
                user_file_id=seed.user_file_id,
                position=seed.position,
            )
            for seed in visible_seeds
        ],
        minimum_seed_count=1,
    )
    if dominant is None:
        return []
    user_file_id, _ = dominant
    file_rows = projected_by_file[user_file_id]

    represented_anchor_keys: set[tuple[str, ...]] = set()
    for seed in visible_seeds:
        if seed.user_file_id != user_file_id:
            continue
        seed_anchor = _last_explicit_article_anchor(seed.heading_path)
        if seed_anchor is not None:
            represented_anchor_keys.add(seed_anchor.key)
    selected: list[RegulatoryChunkProjection] = []
    selected_ids: set[str] = set()
    followed_references: set[RegulatoryProvisionReference] = set()
    followed_count = 0
    query_terms = _lexical_terms(query)

    for reference in references:
        if followed_count >= max_provisions:
            break
        if reference in followed_references:
            continue
        followed_references.add(reference)
        article_no = _normalize_article_no(reference.article_no)
        if article_no is None:
            continue

        matching_indices_by_anchor: dict[tuple[str, ...], list[int]] = {}
        for index, row in enumerate(file_rows):
            if not _is_valid_on(row, as_of_date):
                continue
            anchor = _last_explicit_article_anchor(row.heading_path)
            if (
                anchor is None
                or anchor.article_no != article_no
                or anchor.qualifier != reference.qualifier
            ):
                continue
            matching_indices_by_anchor.setdefault(anchor.key, []).append(index)

        if len(matching_indices_by_anchor) != 1:
            continue
        anchor_key, matching_indices = next(iter(matching_indices_by_anchor.items()))
        if anchor_key in represented_anchor_keys:
            continue

        # Amendment versions intentionally share a logical position. Seeing
        # more than one visible row at that position means the snapshot is
        # contradictory, so this automatic navigation lane must not choose.
        visible_position_counts: dict[int, int] = {}
        for index in matching_indices:
            position = file_rows[index].position
            visible_position_counts[position] = (
                visible_position_counts.get(position, 0) + 1
            )
        if any(count > 1 for count in visible_position_counts.values()):
            continue

        seed_index = min(matching_indices)
        span = _provision_span_for_seed(
            file_rows,
            seed_index,
            as_of_date=as_of_date,
        )
        if not set(matching_indices).issubset(span) or (
            _has_overlapping_visible_positions(
                file_rows,
                span,
                as_of_date=as_of_date,
            )
        ):
            continue
        group = _ProvisionSelectionGroup(
            user_file_id=user_file_id,
            projection_indices=span,
            seed_ids=[file_rows[seed_index].regulatory_chunk_id],
        )
        provision_rows = _select_group_within_budget(
            file_rows,
            group,
            query_terms=query_terms,
            as_of_date=as_of_date,
            max_chunks=max_chunks_per_provision,
            max_chars=max_chars_per_provision,
        )
        if not provision_rows:
            continue
        followed_count += 1
        represented_anchor_keys.add(anchor_key)
        for row in provision_rows:
            if row.regulatory_chunk_id in selected_ids:
                continue
            selected_ids.add(row.regulatory_chunk_id)
            selected.append(row)
    return selected


def is_regulatory_navigation_candidate_visible(
    candidate: RegulatoryProvisionHeadingCandidate,
    *,
    as_of_date: datetime.date | None,
) -> bool:
    """Apply the same current/historical visibility contract as retrieval."""
    if as_of_date is None:
        return candidate.status == RegulatoryChunkStatus.ACTIVE.value
    if (
        candidate.validity_start_date is not None
        and candidate.validity_start_date > as_of_date
    ):
        return False
    if (
        candidate.validity_end_date is not None
        and candidate.validity_end_date <= as_of_date
    ):
        return False
    return True


def get_regulatory_provision_heading_source(
    db_session: Session,
    seed_chunk_ids: Sequence[str],
    *,
    as_of_date: datetime.date | None,
) -> RegulatoryProvisionHeadingSource | None:
    """Load structural headings for one dominant, date-visible source.

    Only ids, positions, paths, validity metadata, and the document title are
    loaded. Legal text is deliberately excluded from this navigation query.
    """
    unique_seed_ids = list(dict.fromkeys(seed_chunk_ids))
    if not unique_seed_ids:
        return None

    visibility_conditions = []
    if as_of_date is None:
        visibility_conditions.append(
            RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value
        )
    else:
        visibility_conditions.extend(
            [
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of_date,
                ),
            ]
        )

    seed_records = list(
        db_session.execute(
            select(
                RegulatoryChunk.id,
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
            ).where(RegulatoryChunk.id.in_(unique_seed_ids), *visibility_conditions)
        ).all()
    )
    seed_by_id = {
        record.id: RegulatoryNavigationSeed(
            regulatory_chunk_id=record.id,
            user_file_id=record.user_file_id,
            position=record.position,
        )
        for record in seed_records
    }
    ordered_seeds = [
        seed_by_id[seed_id] for seed_id in unique_seed_ids if seed_id in seed_by_id
    ]
    ranked_sources = _rank_regulatory_navigation_seed_files(ordered_seeds)
    if not ranked_sources:
        return None
    eligible_file_ids = [user_file_id for user_file_id, _ in ranked_sources]

    heading_records = list(
        db_session.execute(
            select(
                RegulatoryChunk.id,
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
                RegulatoryChunk.heading_path,
                RegulatoryChunk.chunk_metadata,
                RegulatoryChunk.status,
                RegulatoryChunk.validity_start_date,
                RegulatoryChunk.validity_end_date,
            )
            .where(
                RegulatoryChunk.user_file_id.in_(eligible_file_ids),
                *visibility_conditions,
            )
            .order_by(
                RegulatoryChunk.user_file_id,
                RegulatoryChunk.position,
                RegulatoryChunk.id,
            )
        ).all()
    )
    candidates_by_file: dict[UUID, list[RegulatoryProvisionHeadingCandidate]] = {}
    for record in heading_records:
        candidate = RegulatoryProvisionHeadingCandidate(
            regulatory_chunk_id=record.id,
            user_file_id=record.user_file_id,
            position=record.position,
            heading_path=tuple(record.heading_path),
            status=record.status,
            validity_start_date=record.validity_start_date,
            validity_end_date=record.validity_end_date,
            article_title=(
                str(record.chunk_metadata["article_title"])
                if record.chunk_metadata.get("article_title") is not None
                else None
            ),
        )
        # Recheck the pure visibility rule so mock-backed callers and future
        # query refactors cannot accidentally broaden the outline.
        if is_regulatory_navigation_candidate_visible(candidate, as_of_date=as_of_date):
            candidates_by_file.setdefault(candidate.user_file_id, []).append(candidate)

    selected_source: (
        tuple[
            UUID,
            tuple[int, ...],
            tuple[RegulatoryProvisionHeadingCandidate, ...],
        ]
        | None
    ) = None
    for user_file_id, seed_positions in ranked_sources:
        candidates = tuple(candidates_by_file.get(user_file_id, ()))
        if any(
            parse_regulatory_article_heading(heading.strip()) is not None
            for candidate in candidates
            for heading in candidate.heading_path
        ):
            selected_source = (user_file_id, seed_positions, candidates)
            break
    if selected_source is None:
        return None
    user_file_id, seed_positions, candidates = selected_source

    document_title = db_session.scalar(
        select(UserFile.name).where(UserFile.id == user_file_id)
    )
    if document_title is None:
        return None
    return RegulatoryProvisionHeadingSource(
        user_file_id=user_file_id,
        document_title=document_title,
        seed_positions=seed_positions,
        candidates=candidates,
    )


def _regulatory_sibling_candidate(
    row: RegulatoryChunk,
) -> RegulatoryChunkSiblingCandidate:
    return RegulatoryChunkSiblingCandidate(
        regulatory_chunk_id=row.id,
        user_file_id=row.user_file_id,
        position=row.position,
        text=row.text,
        heading_path=tuple(row.heading_path),
        article_no=(
            str(row.chunk_metadata["article_no"])
            if row.chunk_metadata.get("article_no") is not None
            else None
        ),
        article_title=(
            str(row.chunk_metadata["article_title"])
            if row.chunk_metadata.get("article_title") is not None
            else None
        ),
        chunk_type=row.chunk_type,
        paragraph_no=(
            str(row.chunk_metadata["paragraph_no"])
            if row.chunk_metadata.get("paragraph_no") is not None
            else None
        ),
        clause_label=(
            str(row.chunk_metadata["clause_label"])
            if row.chunk_metadata.get("clause_label") is not None
            else None
        ),
        validity_start_date=row.validity_start_date,
        validity_end_date=row.validity_end_date,
        status=row.status,
    )


def get_bounded_same_provision_siblings(
    db_session: Session,
    seed_chunk_ids: Sequence[str],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_chunks_per_provision: int = DEFAULT_PROVISION_MAX_CHUNKS,
    max_chars_per_provision: int = DEFAULT_PROVISION_MAX_CHARS,
) -> list[RegulatoryChunkProjection]:
    """Load seed files and return immutable, bounded provision projections."""
    unique_seed_ids = list(dict.fromkeys(seed_chunk_ids))
    if not unique_seed_ids:
        return []

    seed_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.id.in_(unique_seed_ids))
        ).all()
    )
    user_file_ids = {row.user_file_id for row in seed_rows}
    if not user_file_ids:
        return []

    all_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_(user_file_ids)
            )
        ).all()
    )
    candidates = [_regulatory_sibling_candidate(row) for row in all_rows]
    return select_bounded_same_provision_siblings(
        candidates,
        unique_seed_ids,
        query=query,
        as_of_date=as_of_date,
        max_chunks_per_provision=max_chunks_per_provision,
        max_chars_per_provision=max_chars_per_provision,
    )


def get_bounded_adjacent_provisions(
    db_session: Session,
    seed_chunk_ids: Sequence[str],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_provisions: int = DEFAULT_ADJACENT_MAX_PROVISIONS,
    max_chunks_per_provision: int = DEFAULT_ADJACENT_MAX_CHUNKS_PER_PROVISION,
    max_total_chars: int = DEFAULT_ADJACENT_MAX_TOTAL_CHARS,
) -> list[RegulatoryChunkProjection]:
    """Load seed files and return a bounded adjacent-provision context window."""

    unique_seed_ids = list(dict.fromkeys(seed_chunk_ids))
    if not unique_seed_ids:
        return []
    seed_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.id.in_(unique_seed_ids))
        ).all()
    )
    user_file_ids = {row.user_file_id for row in seed_rows}
    if not user_file_ids:
        return []
    all_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_(user_file_ids)
            )
        ).all()
    )
    return select_bounded_adjacent_provisions(
        [_regulatory_sibling_candidate(row) for row in all_rows],
        unique_seed_ids,
        query=query,
        as_of_date=as_of_date,
        max_provisions=max_provisions,
        max_chunks_per_provision=max_chunks_per_provision,
        max_total_chars=max_total_chars,
    )


def get_visible_regulatory_chunk_ids(
    db_session: Session,
    chunk_ids: Sequence[str],
    *,
    as_of_date: datetime.date | None,
) -> set[str]:
    """Return authoritative chunk ids visible in the requested legal snapshot."""

    unique_chunk_ids = list(dict.fromkeys(chunk_ids))
    if not unique_chunk_ids:
        return set()

    visibility_conditions = []
    if as_of_date is None:
        visibility_conditions.append(
            RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value
        )
    else:
        visibility_conditions.extend(
            [
                or_(
                    RegulatoryChunk.validity_start_date.is_(None),
                    RegulatoryChunk.validity_start_date <= as_of_date,
                ),
                or_(
                    RegulatoryChunk.validity_end_date.is_(None),
                    RegulatoryChunk.validity_end_date > as_of_date,
                ),
            ]
        )
    return set(
        db_session.scalars(
            select(RegulatoryChunk.id).where(
                RegulatoryChunk.id.in_(unique_chunk_ids),
                *visibility_conditions,
            )
        ).all()
    )


def get_bounded_referenced_provisions(
    db_session: Session,
    seed_chunk_ids: Sequence[str],
    references: Sequence[RegulatoryProvisionReference],
    *,
    as_of_date: datetime.date | None,
    query: str = "",
    max_provisions: int = DEFAULT_REFERENCE_MAX_PROVISIONS,
    max_chunks_per_provision: int = DEFAULT_REFERENCE_MAX_CHUNKS_PER_PROVISION,
    max_chars_per_provision: int = DEFAULT_REFERENCE_MAX_CHARS_PER_PROVISION,
) -> list[RegulatoryChunkProjection]:
    """Load and resolve explicit references inside the selected source only."""

    unique_seed_ids = list(dict.fromkeys(seed_chunk_ids))
    if not unique_seed_ids or not references:
        return []
    seed_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(RegulatoryChunk.id.in_(unique_seed_ids))
        ).all()
    )
    user_file_ids = {row.user_file_id for row in seed_rows}
    if not user_file_ids:
        return []
    all_rows = list(
        db_session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id.in_(user_file_ids)
            )
        ).all()
    )
    return select_bounded_referenced_provisions(
        [_regulatory_sibling_candidate(row) for row in all_rows],
        unique_seed_ids,
        references,
        as_of_date=as_of_date,
        query=query,
        max_provisions=max_provisions,
        max_chunks_per_provision=max_chunks_per_provision,
        max_chars_per_provision=max_chars_per_provision,
    )


def make_regulatory_chunk_id(user_file_id: UUID, position: int, text: str) -> str:
    """Deterministic chunk id: re-processing an unchanged file yields the same
    ids, so re-indexing is idempotent."""
    digest = hashlib.sha256(
        f"{user_file_id}:{position}:{text}".encode("utf-8")
    ).hexdigest()
    return f"rc_{digest[:40]}"


def common_reindex_validity_window(
    chunks: Iterable[RegulatoryChunkValidityState],
) -> RegulatoryFileValidityWindow | None:
    """Return the one safe file window that may survive re-chunking.

    Re-chunking can change ids, positions, and row counts, so per-provision
    windows cannot safely be matched onto the new rows. A window is portable
    only when every existing row is an unversioned, active indexed row and all
    rows carry the same boundaries. Any amendment/supersession history or
    mixed window disables inheritance for the whole file.
    """

    chunk_list = list(chunks)
    if not chunk_list:
        return None
    if any(
        chunk.source != RegulatoryChunkSource.INDEXED.value
        or chunk.status != RegulatoryChunkStatus.ACTIVE.value
        or chunk.supersedes_chunk_id is not None
        or chunk.superseded_by_chunk_id is not None
        for chunk in chunk_list
    ):
        return None

    windows = {
        (chunk.validity_start_date, chunk.validity_end_date) for chunk in chunk_list
    }
    if len(windows) != 1:
        return None
    start, end = next(iter(windows))
    if start is not None and end is not None and start >= end:
        return None
    return RegulatoryFileValidityWindow(start=start, end=end)


def replace_indexed_chunks_for_file(
    db_session: Session,
    user_file_id: UUID,
    chunker_chunks: list[ChunkerChunk],
) -> list[RegulatoryChunk]:
    """Replace all pipeline-produced chunks for a file with a fresh chunking.

    Amendment-produced chunks are preserved: they were approved by an admin
    and are not derivable from the file content. A single file-level validity
    window survives re-chunking only when every existing row is unversioned
    and carries that same window. Does not commit; caller owns the transaction
    and the Elasticsearch re-projection.
    """
    # Read only scalar columns here. Loading the old ORM objects into the
    # identity map before the bulk delete can conflict with adding replacement
    # rows that intentionally reuse the same deterministic primary keys.
    existing_states = [
        RegulatoryChunkValidityState(
            source=row.source,
            status=row.status,
            validity_start_date=row.validity_start_date,
            validity_end_date=row.validity_end_date,
            supersedes_chunk_id=row.supersedes_chunk_id,
            superseded_by_chunk_id=row.superseded_by_chunk_id,
        )
        for row in db_session.execute(
            select(
                RegulatoryChunk.source,
                RegulatoryChunk.status,
                RegulatoryChunk.validity_start_date,
                RegulatoryChunk.validity_end_date,
                RegulatoryChunk.supersedes_chunk_id,
                RegulatoryChunk.superseded_by_chunk_id,
            )
            .where(RegulatoryChunk.user_file_id == user_file_id)
            .with_for_update()
        ).all()
    ]
    inherited_window = common_reindex_validity_window(existing_states)

    db_session.execute(
        delete(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == user_file_id,
            RegulatoryChunk.source == RegulatoryChunkSource.INDEXED.value,
        )
    )

    chunk_by_order = {chunk.metadata.chunk_order: chunk for chunk in chunker_chunks}
    if len(chunk_by_order) != len(chunker_chunks):
        raise ValueError("regulatory chunks contain duplicate positions")
    chunk_id_by_order = {
        order: make_regulatory_chunk_id(user_file_id, order, chunk.text)
        for order, chunk in chunk_by_order.items()
    }

    rows: list[RegulatoryChunk] = []
    for chunk in chunker_chunks:
        meta = chunk.metadata
        stored_metadata = meta.to_storage_dict()
        source_regulatory_chunk_ids: list[str] = []
        if meta.chunk_variant == HIERARCHICAL_AGGREGATE_CHUNK_VARIANT:
            for source_order in meta.source_chunk_orders:
                source_chunk = chunk_by_order.get(source_order)
                if source_chunk is None:
                    raise ValueError("aggregate references an unknown source chunk")
                if source_chunk.metadata.chunk_variant != ATOMIC_CHUNK_VARIANT:
                    raise ValueError("aggregate source chunk must be atomic")
                source_regulatory_chunk_ids.append(chunk_id_by_order[source_order])
        stored_metadata["source_regulatory_chunk_ids"] = source_regulatory_chunk_ids
        row = RegulatoryChunk(
            id=chunk_id_by_order[meta.chunk_order],
            user_file_id=user_file_id,
            text=chunk.text,
            position=meta.chunk_order,
            chunk_type=meta.chunk_type,
            heading_path=list(meta.heading_path),
            chunk_metadata=stored_metadata,
            status=RegulatoryChunkStatus.ACTIVE.value,
            source=RegulatoryChunkSource.INDEXED.value,
            validity_start_date=(
                inherited_window.start if inherited_window is not None else None
            ),
            validity_end_date=(
                inherited_window.end if inherited_window is not None else None
            ),
        )
        db_session.add(row)
        rows.append(row)
    return rows


def get_chunks_for_file(
    db_session: Session,
    user_file_id: UUID,
    *,
    include_superseded: bool = True,
) -> list[RegulatoryChunk]:
    stmt = (
        select(RegulatoryChunk)
        .where(RegulatoryChunk.user_file_id == user_file_id)
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
    )
    if not include_superseded:
        stmt = stmt.where(RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value)
    return list(db_session.scalars(stmt).all())


def get_next_chunk_position(db_session: Session, user_file_id: UUID) -> int:
    """Return the next position without materializing a file's chunks."""

    max_position = db_session.scalar(
        select(func.max(RegulatoryChunk.position)).where(
            RegulatoryChunk.user_file_id == user_file_id
        )
    )
    return 0 if max_position is None else max_position + 1


def has_regulatory_chunks_for_file(db_session: Session, user_file_id: UUID) -> bool:
    """Return whether PostgreSQL owns the canonical chunks for this user file."""

    return bool(
        db_session.scalar(
            select(exists().where(RegulatoryChunk.user_file_id == user_file_id))
        )
    )


def get_chunk_by_id(db_session: Session, chunk_id: str) -> RegulatoryChunk | None:
    return db_session.get(RegulatoryChunk, chunk_id)


def get_active_chunks_by_ids(
    db_session: Session, chunk_ids: Sequence[str]
) -> dict[str, RegulatoryChunk]:
    """Load canonical active source chunks for exact search projection ids."""

    unique_chunk_ids = list(dict.fromkeys(chunk_ids))
    if not unique_chunk_ids:
        return {}
    rows = db_session.scalars(
        select(RegulatoryChunk).where(
            RegulatoryChunk.id.in_(unique_chunk_ids),
            RegulatoryChunk.status == RegulatoryChunkStatus.ACTIVE.value,
            RegulatoryChunk.chunk_type.is_distinct_from(
                HIERARCHICAL_AGGREGATE_CHUNK_VARIANT
            ),
        )
    ).all()
    return {row.id: row for row in rows}


def is_hierarchical_aggregate_chunk(chunk: RegulatoryChunk) -> bool:
    return (
        chunk.chunk_type == HIERARCHICAL_AGGREGATE_CHUNK_VARIANT
        or chunk.chunk_metadata.get("chunk_variant")
        == HIERARCHICAL_AGGREGATE_CHUNK_VARIANT
    )


def delete_hierarchical_aggregates_referencing_chunk(
    db_session: Session,
    *,
    user_file_id: UUID,
    source_chunk_id: str,
) -> int:
    """Remove stale derived rows before an atomic source row is mutated."""

    candidates = list(
        db_session.scalars(
            select(RegulatoryChunk)
            .where(
                RegulatoryChunk.user_file_id == user_file_id,
                RegulatoryChunk.chunk_type == HIERARCHICAL_AGGREGATE_CHUNK_VARIANT,
            )
            .with_for_update()
        ).all()
    )
    affected = [
        candidate
        for candidate in candidates
        if source_chunk_id
        in candidate.chunk_metadata.get("source_regulatory_chunk_ids", [])
    ]
    for candidate in affected:
        db_session.delete(candidate)
    return len(affected)


def update_chunk(
    chunk: RegulatoryChunk,
    *,
    text: str | None = None,
    heading_path: list[str] | None = None,
    chunk_metadata: dict[str, Any] | None = None,
    validity_start_date: datetime.date | None | str = "unset",
    validity_end_date: datetime.date | None | str = "unset",
) -> RegulatoryChunk:
    """Apply a partial edit to a chunk row.

    The `"unset"` sentinel distinguishes "leave unchanged" from an explicit
    None (= clear the date). Does not commit; caller owns the transaction and
    the Elasticsearch re-projection.
    """
    if text is not None:
        chunk.text = text
    if heading_path is not None:
        chunk.heading_path = heading_path
    if chunk_metadata is not None:
        chunk.chunk_metadata = chunk_metadata
    if validity_start_date != "unset":
        chunk.validity_start_date = cast(datetime.date | None, validity_start_date)
    if validity_end_date != "unset":
        chunk.validity_end_date = cast(datetime.date | None, validity_end_date)
    return chunk


def apply_file_validity_window(
    chunks: Iterable[RegulatoryChunk],
    *,
    validity_start_date: ValidityDateUpdate = "unset",
    validity_end_date: ValidityDateUpdate = "unset",
) -> RegulatoryFileValidityUpdateResult:
    """Apply explicit source-file dates without rewriting amendment history.

    A file-level window describes an uploaded source snapshot. Amendment rows
    and indexed rows that already participate in a supersession chain have
    provision-specific windows, so they must continue to be edited through the
    amendment/chunk workflow instead of inheriting this bulk value.

    Validation is completed for every eligible row before any row is changed,
    keeping the in-memory update atomic when a proposed half-open window is
    invalid.
    """

    chunk_list = list(chunks)
    eligible = [
        chunk
        for chunk in chunk_list
        if chunk.source == RegulatoryChunkSource.INDEXED.value
        and chunk.status == RegulatoryChunkStatus.ACTIVE.value
        and chunk.supersedes_chunk_id is None
        and chunk.superseded_by_chunk_id is None
    ]

    def uniform_window(
        windows: set[tuple[datetime.date | None, datetime.date | None]],
    ) -> RegulatoryFileValidityWindow | None:
        if len(windows) != 1:
            return None
        start, end = next(iter(windows))
        if start is not None and end is not None and start >= end:
            return None
        return RegulatoryFileValidityWindow(start=start, end=end)

    previous_window = uniform_window(
        {(chunk.validity_start_date, chunk.validity_end_date) for chunk in eligible}
    )

    proposed_windows: list[
        tuple[RegulatoryChunk, datetime.date | None, datetime.date | None]
    ] = []
    for chunk in eligible:
        start = (
            chunk.validity_start_date
            if validity_start_date == "unset"
            else validity_start_date
        )
        end = (
            chunk.validity_end_date
            if validity_end_date == "unset"
            else validity_end_date
        )
        if start is not None and end is not None and start >= end:
            raise ValueError(
                "Validity start date must be earlier than validity end date."
            )
        proposed_windows.append((chunk, start, end))

    for chunk, start, end in proposed_windows:
        chunk.validity_start_date = start
        chunk.validity_end_date = end

    updated_window = uniform_window(
        {(start, end) for _, start, end in proposed_windows}
    )

    return RegulatoryFileValidityUpdateResult(
        updated_chunk_count=len(eligible),
        skipped_versioned_chunk_count=len(chunk_list) - len(eligible),
        previous_window=previous_window,
        updated_window=updated_window,
    )


def update_file_validity_window(
    db_session: Session,
    user_file_id: UUID,
    *,
    validity_start_date: ValidityDateUpdate = "unset",
    validity_end_date: ValidityDateUpdate = "unset",
) -> RegulatoryFileValidityUpdateResult:
    """Lock a file's chunks and apply explicit source-snapshot dates."""

    chunks = list(
        db_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == user_file_id)
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
            .with_for_update()
        ).all()
    )
    return apply_file_validity_window(
        chunks,
        validity_start_date=validity_start_date,
        validity_end_date=validity_end_date,
    )


def get_chunk_counts_for_files(
    db_session: Session, user_file_ids: list[UUID]
) -> dict[UUID, int]:
    from sqlalchemy import func as sa_func

    if not user_file_ids:
        return {}
    rows = db_session.execute(
        select(RegulatoryChunk.user_file_id, sa_func.count())
        .where(RegulatoryChunk.user_file_id.in_(user_file_ids))
        .group_by(RegulatoryChunk.user_file_id)
    ).all()
    return {row[0]: row[1] for row in rows}
