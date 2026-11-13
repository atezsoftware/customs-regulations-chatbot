"""Read-only source identity and bounded canonical context for amendments."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import and_, bindparam, func, or_, select, text, true
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from onyx.db.enums import UserFileStatus
from onyx.db.models import (
    DocumentSet__UserFile,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatoryDocumentIntakeReviewDecision,
    RegulatoryFilePublication,
    UserFile,
)
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
_APPENDIX_ROOT_RE = re.compile(
    r"^(?:ek(?:ler)?|annex(?:es)?|appendi(?:x|ces))"
    r"(?:\s*[-–—:.]?\s*(?:\d+[a-z]?|[ivxlcdm]+|[a-z]))?"
    r"(?:\s*[:–—-]\s*.+)?$",
    re.IGNORECASE,
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
    is_attachment: bool = False


def published_amendment_source_conditions() -> tuple[ColumnElement[bool], ...]:
    """A pending preparation or unfinished writer cannot establish target identity."""
    member = RegulatoryDocumentIntakeFile
    has_intake = (
        select(member.id)
        .where(member.user_file_id == UserFile.id)
        .correlate(UserFile)
        .exists()
    )
    unfinished_intake = (
        select(member.id)
        .join(RegulatoryDocumentIntake, RegulatoryDocumentIntake.id == member.intake_id)
        .where(
            member.user_file_id == UserFile.id,
            or_(
                member.status != "completed",
                RegulatoryDocumentIntake.status != "completed",
            ),
        )
        .correlate(UserFile)
        .exists()
    )
    disposition = (
        select(RegulatoryDocumentIntakeReviewDecision.disposition)
        .join(member, member.id == RegulatoryDocumentIntakeReviewDecision.document_id)
        .where(member.user_file_id == UserFile.id)
        .order_by(RegulatoryDocumentIntakeReviewDecision.sequence.desc())
        .limit(1)
        .correlate(UserFile)
        .scalar_subquery()
    )
    return (
        UserFile.status == UserFileStatus.COMPLETED,
        ~unfinished_intake,
        or_(disposition.is_(None), disposition != "source_evidence"),
        or_(
            and_(
                RegulatoryFilePublication.epoch > 0,
                RegulatoryFilePublication.gate_closed.is_(False),
                RegulatoryFilePublication.writer_manifest.is_(None),
                RegulatoryFilePublication.writer_manifest_sha256.is_(None),
            ),
            and_(RegulatoryFilePublication.user_file_id.is_(None), ~has_intake),
        ),
    )


def _own_communique_header_title(
    root: str, opening: str, document_number: str | None
) -> str:
    """Recover identifiers adjacent to an own official opening, never a citation."""
    if (
        len(root) > 400
        or not document_number
        or len(document_number) > 40
        or _structural_title_identity(root) is None
    ):
        return root
    lines = [line.strip() for line in opening.splitlines() if line.strip()]
    if (
        len(lines) < 7
        or re.fullmatch(r"\d{1,2}\s+\w+\s+(?:19|20)\d{2}\s+\w+", lines[0]) is None
        or lines[1] not in {"Resmî Gazete", "Resmi Gazete"}
        or re.fullmatch(r"Sayı\s*:\s*\d+", lines[2]) is None
        or lines[3] != "TEBLİĞ"
        or not lines[4].endswith("Bakanlığından:")
        or not root.endswith("TEBLİĞ")
    ):
        return root
    body = "\n".join(lines[5:])
    title = re.match(r"\s+".join(re.escape(word) for word in root.split()), body)
    if title is None:
        return root
    suffix = re.match(
        r"\s*(?P<eu>\((?:19|20)\d{2}/[1-9]\d{0,4}/(?:AB|AT|AET)\))"
        r"\s+(?P<domestic>\([A-ZÇĞİÖŞÜ]{2,12}-(?:19|20)\d{2}/[1-9]\d{0,4}\))"
        r"(?=[ \t]*(?:\n|$))",
        body[title.end() :],
    )
    if suffix is None or document_number != suffix["domestic"][1:-1]:
        return root
    tail = body[title.end() + suffix.end() :].strip()
    # The next article caption may share the opening row; a continued act title may not.
    if tail and (
        len(tail) > 120
        or re.fullmatch(r"[^\W\d_]+(?:[ ,/-]+[^\W\d_]+)*", tail) is None
        or re.search(
            r"tebliğ|yönetmelik|kanun|değişiklik|yapılmasına", tail, re.IGNORECASE
        )
    ):
        return root
    return f"{root} {suffix['eu']} {suffix['domestic']}"


def load_amendment_source_identities(
    db_session: Session,
    user_file_ids: Sequence[UUID],
    *,
    document_set_id: int | None = None,
) -> list[AmendmentSourceIdentity]:
    if not user_file_ids:
        return []
    roots = (
        select(
            RegulatoryChunk.user_file_id,
            RegulatoryChunk.id.label("chunk_id"),
            RegulatoryChunk.heading_path,
        )
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
        select(
            UserFile.id,
            UserFile.name,
            RegulatoryDocumentIntakeFile.role.label("intake_role"),
            roots.c.heading_path,
            RegulatoryChunk.position,
            RegulatoryChunk.chunk_type,
            RegulatoryChunk.superseded_by_chunk_id,
            RegulatoryChunk.chunk_metadata["bound_to_regulatory_chunk_id"].astext.label(
                "bound_to"
            ),
            RegulatoryChunk.chunk_metadata["document_number"].astext.label(
                "document_number"
            ),
            func.left(RegulatoryChunk.text, 4096).label("opening"),
        )
        .select_from(UserFile)
        .outerjoin(
            RegulatoryFilePublication,
            RegulatoryFilePublication.user_file_id == UserFile.id,
        )
        .outerjoin(roots, roots.c.user_file_id == UserFile.id)
        .outerjoin(
            RegulatoryDocumentIntakeFile,
            RegulatoryDocumentIntakeFile.user_file_id == UserFile.id,
        )
        .outerjoin(RegulatoryChunk, RegulatoryChunk.id == roots.c.chunk_id)
        .where(UserFile.id.in_(user_file_ids), *published_amendment_source_conditions())
        .where(
            select(DocumentSet__UserFile.user_file_id)
            .where(
                DocumentSet__UserFile.user_file_id == UserFile.id,
                DocumentSet__UserFile.document_set_id == document_set_id,
            )
            .correlate(UserFile)
            .exists()
            if document_set_id is not None
            else true()
        )
    ).all()
    result = []
    for row in rows:
        root = row.heading_path[0] if row.heading_path else ""
        if (
            row.position == 0
            and row.chunk_type != "hierarchical_aggregate"
            and row.superseded_by_chunk_id is None
            and row.bound_to is None
        ):
            root = _own_communique_header_title(root, row.opening, row.document_number)
        result.append(
            AmendmentSourceIdentity(
                row.id,
                row.name,
                root,
                is_attachment=row.intake_role == "annex"
                or _APPENDIX_ROOT_RE.fullmatch(root.strip()) is not None,
            )
        )
    return result


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
