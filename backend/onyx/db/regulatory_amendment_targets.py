"""Read-only source identity and bounded canonical context for amendments."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import bindparam, select, text
from sqlalchemy.orm import Session

from onyx.db.models import RegulatoryChunk, UserFile
from onyx.db.regulatory_chunks import RegulatoryChunkStructuralMatch
from onyx.regulatory.source_identity import source_identity_distinguishing_tokens

_MAX_STRUCTURAL_SOURCE_FILES = 256
_MAX_STRUCTURAL_SOURCE_TITLES = 2048
_ARTICLE_OPENING_PATTERN = (
    r"^(?:(?:gecici|mukerrer|ek)[[:space:]]+)?madde[[:space:]:.]*[0-9]+[a-z]?"
)
_PROVISION_OPENING_PATTERN = (
    _ARTICLE_OPENING_PATTERN + r"|^(ek|ekler|annex|appendix)[[:space:]]*$"
    r"|^(ek|annex|appendix)([[:space:]:.-]+[0-9a-z]+|[0-9]+)"
    r"([[:space:][:punct:]]|$)|^\(?([0-9]{1,3}[.)]|[a-z][)])"
)
_INSTRUMENT_TITLE_END_PATTERN = (
    r"(^|[[:space:]])(kanun(u)?|teblig(i)?|yonetmelik|yonetmeligi|"
    r"genelge(si)?|yonerge(si)?|karar(i)?|sozlesme(si)?|andlasma(si)?|"
    r"protokol(u)?)([[:space:]]*\([^\n)]{1,80}\))?[[:space:]]*$"
)
_INSTRUMENT_TITLE_END_RE = re.compile(
    r"\b(?P<kind>kanun(?:u)?|teblig(?:i)?|yonetmelik|yonetmeligi|"
    r"genelge(?:si)?|yonerge(?:si)?|karar(?:i)?|sozlesme(?:si)?|"
    r"andlasma(?:si)?|protokol(?:u)?)"
    r"(?:\s*\([^\n)]{1,80}\))?\s*$"
)


_STRUCTURAL_SOURCE_TITLES_SQL = text(
    """
    WITH scoped_chunks AS MATERIALIZED (
        SELECT user_file_id, heading_path
        FROM regulatory_chunk
        WHERE user_file_id IN :user_file_ids
          AND status = 'active'
          AND superseded_by_chunk_id IS NULL
          AND chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
          AND chunk_metadata->>'bound_to_regulatory_chunk_id' IS NULL
          AND regexp_replace(
              translate(lower(normalize(heading_path::text, NFKD)), 'ı', 'i'),
              '[̀-ͯ]', '', 'g'
          ) LIKE ALL(CAST(:title_patterns AS text[]))
    ), candidate_files AS (
        SELECT DISTINCT chunk.user_file_id
        FROM scoped_chunks AS chunk
        CROSS JOIN LATERAL (
            SELECT label, regexp_replace(
                translate(lower(normalize(label, NFKD)), 'ı', 'i'),
                '[̀-ͯ]', '', 'g'
            ) AS folded_label
            FROM jsonb_array_elements_text(chunk.heading_path) AS element(label)
        ) AS heading
        WHERE char_length(heading.label) BETWEEN 8 AND 400
          AND heading.folded_label ~ :title_end_pattern
          AND heading.folded_label LIKE ALL(CAST(:title_patterns AS text[]))
        LIMIT :file_limit
    )
    SELECT chunk.user_file_id, heading.label AS title,
           count(DISTINCT regexp_replace(
               substring(structure.article FROM :article_pattern),
               '[[:space:]:.]+', ' ', 'g'
           )) AS article_count,
           (SELECT count(*) FROM candidate_files) AS candidate_file_count
    FROM regulatory_chunk AS chunk
    JOIN candidate_files USING (user_file_id)
    CROSS JOIN LATERAL (
        SELECT min(part.ordinal) FILTER (
                   WHERE part.label ~ :provision_pattern
               ) AS boundary,
               (array_agg(part.label ORDER BY part.ordinal)
                   FILTER (WHERE part.label ~ :article_pattern))[1]
                   AS article
        FROM (
            SELECT ordinal, regexp_replace(
                translate(lower(normalize(label, NFKD)), 'ı', 'i'),
                '[̀-ͯ]', '', 'g'
            ) AS label
            FROM jsonb_array_elements_text(chunk.heading_path)
                WITH ORDINALITY AS element(label, ordinal)
        ) AS part
    ) AS structure
    CROSS JOIN LATERAL jsonb_array_elements_text(chunk.heading_path)
        WITH ORDINALITY AS heading(label, ordinal)
    WHERE chunk.status = 'active'
      AND chunk.superseded_by_chunk_id IS NULL
      AND chunk.chunk_type IS DISTINCT FROM 'hierarchical_aggregate'
      AND chunk.chunk_metadata->>'bound_to_regulatory_chunk_id' IS NULL
      AND heading.ordinal < structure.boundary
      AND structure.article IS NOT NULL
      AND char_length(heading.label) BETWEEN 8 AND 400
      AND regexp_replace(
          translate(lower(normalize(heading.label, NFKD)), 'ı', 'i'),
          '[̀-ͯ]', '', 'g'
      ) ~ :title_end_pattern
    GROUP BY chunk.user_file_id, heading.label
    LIMIT :title_limit
    """
).bindparams(bindparam("user_file_ids", expanding=True))


def _structural_title_identity(value: str) -> tuple[str, tuple[str, ...]] | None:
    folded = "".join(
        character
        for character in unicodedata.normalize(
            "NFKD", value.casefold().replace("ı", "i")
        )
        if not unicodedata.combining(character)
    )
    match = _INSTRUMENT_TITLE_END_RE.search(folded)
    tokens = source_identity_distinguishing_tokens(value)
    if match is None or not tokens:
        return None
    kind = match.group("kind")
    if kind == "yonetmeligi":
        return "yonetmelik", tokens
    for canonical in (
        "kanun",
        "teblig",
        "yonetmelik",
        "genelge",
        "yonerge",
        "karar",
        "sozlesme",
        "andlasma",
        "protokol",
    ):
        if kind.startswith(canonical):
            return canonical, tokens
    return None


@dataclass(frozen=True)
class AmendmentSourceIdentity:
    user_file_id: UUID
    name: str
    root_heading: str


def load_amendment_source_identities(
    db_session: Session,
    user_file_ids: Sequence[UUID],
) -> list[AmendmentSourceIdentity]:
    if not user_file_ids:
        return []
    roots = (
        select(RegulatoryChunk.user_file_id, RegulatoryChunk.heading_path)
        .where(
            RegulatoryChunk.user_file_id.in_(user_file_ids),
            RegulatoryChunk.status == "active",
        )
        .distinct(RegulatoryChunk.user_file_id)
        .order_by(
            RegulatoryChunk.user_file_id, RegulatoryChunk.position, RegulatoryChunk.id
        )
        .subquery()
    )
    rows = db_session.execute(
        select(UserFile.id, UserFile.name, roots.c.heading_path)
        .outerjoin(roots, roots.c.user_file_id == UserFile.id)
        .where(UserFile.id.in_(user_file_ids))
    ).all()
    return [
        AmendmentSourceIdentity(row[0], row[1], row[2][0] if row[2] else "")
        for row in rows
    ]


def load_amendment_structural_source_titles(
    db_session: Session,
    user_file_ids: Sequence[UUID],
    target_source: str,
) -> dict[UUID, str]:
    """Recover an own title above multiple articles after extraction reordering.

    A body citation or a heading inside a provision/annex is not source identity.
    Every qualifying title in a candidate file participates in the ambiguity check.
    """
    target_identity = _structural_title_identity(target_source)
    if not user_file_ids or target_identity is None:
        return {}
    rows = db_session.execute(
        _STRUCTURAL_SOURCE_TITLES_SQL,
        {
            "user_file_ids": user_file_ids,
            "title_patterns": [f"%{token}%" for token in target_identity[1]],
            "title_end_pattern": _INSTRUMENT_TITLE_END_PATTERN,
            "article_pattern": _ARTICLE_OPENING_PATTERN,
            "provision_pattern": _PROVISION_OPENING_PATTERN,
            "file_limit": _MAX_STRUCTURAL_SOURCE_FILES + 1,
            "title_limit": _MAX_STRUCTURAL_SOURCE_TITLES + 1,
        },
    ).all()
    if len(rows) > _MAX_STRUCTURAL_SOURCE_TITLES or any(
        row.candidate_file_count > _MAX_STRUCTURAL_SOURCE_FILES for row in rows
    ):
        return {}
    titles_by_file: dict[UUID, dict[tuple[str, tuple[str, ...]], tuple[str, int]]] = {}
    for row in rows:
        identity = _structural_title_identity(row.title)
        if identity is None:
            continue
        file_titles = titles_by_file.setdefault(row.user_file_id, {})
        existing = file_titles.get(identity)
        if existing is None or row.article_count > existing[1]:
            file_titles[identity] = (row.title, row.article_count)
    return {
        file_id: file_titles[target_identity][0]
        for file_id, file_titles in titles_by_file.items()
        if len(file_titles) == 1
        and target_identity in file_titles
        and file_titles[target_identity][1] >= 2
    }


def load_amendment_source_chunks(
    db_session: Session,
    user_file_id: UUID,
    *,
    limit: int = 4096,
) -> list[RegulatoryChunkStructuralMatch]:
    """Return a complete bounded file; an oversized file cannot imply a boundary."""
    rows = db_session.execute(
        select(RegulatoryChunk, UserFile.name)
        .join(UserFile, UserFile.id == RegulatoryChunk.user_file_id)
        .where(
            RegulatoryChunk.user_file_id == user_file_id,
            RegulatoryChunk.status == "active",
            RegulatoryChunk.superseded_by_chunk_id.is_(None),
            RegulatoryChunk.chunk_type.is_distinct_from("hierarchical_aggregate"),
            RegulatoryChunk.chunk_metadata["bound_to_regulatory_chunk_id"].astext.is_(
                None
            ),
        )
        .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        .limit(limit + 1)
    ).all()
    if len(rows) > limit:
        return []
    return [
        RegulatoryChunkStructuralMatch(chunk=row[0], source_name=row[1]) for row in rows
    ]
