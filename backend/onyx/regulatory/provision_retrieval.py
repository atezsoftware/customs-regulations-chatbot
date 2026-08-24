"""Bounded same-provision expansion for selected regulatory search hits."""

import datetime
import logging
import re
import unicodedata
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypedDict
from uuid import UUID

from sqlalchemy.orm import Session

from onyx.context.search.models import InferenceChunk, InferenceSection
from onyx.context.search.utils import inference_section_from_single_chunk
from onyx.db.regulatory_chunks import (
    DEFAULT_NAVIGATION_MAX_HEADINGS,
    RegulatoryChunkProjection,
    RegulatoryProvisionHeadingCandidate,
    RegulatoryProvisionHeadingSource,
    get_bounded_adjacent_provisions,
    get_bounded_referenced_provisions,
    get_bounded_same_provision_siblings,
    get_bounded_source_lexical_matches,
    get_regulatory_provision_heading_source,
    is_regulatory_navigation_candidate_visible,
)
from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_regulatory_provision_references,
    normalize_regulatory_heading_path,
    parse_regulatory_article_heading,
)

_SIBLING_BLURB_CHARS = 800
_ARTICLE_PREFIX_RE = re.compile(
    r"^(?:(?:gecici|geçici|mukerrer|mükerrer)\s+)?madde"
    r"\s*[:.]?\s*\d+[a-z]?\b[\s:.-]*",
    flags=re.IGNORECASE,
)
_BARE_DESCENDANT_RE = re.compile(
    r"^(?:\(?\d+[a-z]?\)?|[a-zçğıöşü])(?:[.):/-])?$",
    flags=re.IGNORECASE,
)
_MAX_TOPIC_HINT_CHARS = 120
_REFERENCED_PROVISION_MAX_TOTAL_CHARS = 6_000
_LEXICAL_TERM_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_NAVIGATION_GENERIC_TERMS = {
    "article",
    "bolum",
    "chapter",
    "ek",
    "ilave",
    "madde",
    "section",
}
_NAVIGATION_LEAD_MAX_SECTIONS = 2
_SOURCE_LEXICAL_MAX_MATCHES = 2
_SOURCE_LEXICAL_MAX_SECTIONS = 4
logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RegulatoryProvisionNavigationEntry:
    """One structural heading lead; it contains no provision text."""

    article_key: str
    heading_label: str
    regulatory_chunk_id: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class RegulatoryProvisionNavigation:
    """A compact outline for one source selected by the search ranking."""

    document_title: str
    entries: tuple[RegulatoryProvisionNavigationEntry, ...]
    user_file_id: str | None = field(default=None, repr=False, compare=False)


class RegulatoryProvisionNavigationPayloadEntry(TypedDict):
    article_key: str
    heading_label: str


class RegulatoryProvisionNavigationPayload(TypedDict):
    type: Literal["regulatory_provision_heading_navigation"]
    document_title: str
    usage_note: str
    headings: list[RegulatoryProvisionNavigationPayloadEntry]


@dataclass(frozen=True, slots=True)
class RegulatoryRerankPacket:
    """One reranker document backed by independently citable atomic chunks."""

    candidate: InferenceChunk
    members: tuple[InferenceChunk, ...]


@dataclass(frozen=True, slots=True)
class _RawArticleHeading:
    article_base_key: str
    scope_labels: tuple[str, ...]
    scope_keys: tuple[str, ...]
    heading_label: str
    position: int
    label_priority: int
    regulatory_chunk_id: str | None


@dataclass(frozen=True, slots=True)
class _ArticleHeadingOption:
    article_key: str
    article_base_key: str
    heading_label: str
    position: int
    label_priority: int
    regulatory_chunk_id: str | None


def _fold_heading(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_marks.replace("ı", "i").split())


def _focused_query_overlap_score(query_terms: set[str], text: str) -> int:
    """Prefer siblings that cover the model's focused search anchors.

    Exact numeric and alphanumeric identifiers carry more information than an
    ordinary word in legal retrieval (dates, rates, codes, and provision
    labels). This remains document-agnostic and only orders rows that the
    bounded same-provision selector already admitted.
    """

    text_terms = set(_LEXICAL_TERM_RE.findall(_fold_heading(text)))
    score = 0
    for query_term in query_terms:
        if query_term in text_terms:
            score += 4 if query_term.isdigit() else 2
            continue
        if len(query_term) < 4 or query_term.isdigit():
            continue
        if any(
            len(text_term) >= 4 and text_term[:4] == query_term[:4]
            for text_term in text_terms
        ):
            score += 1
    return score


def _navigation_query_terms(value: str) -> set[str]:
    return {
        term
        for term in _LEXICAL_TERM_RE.findall(_fold_heading(value))
        if len(term) >= 3
        and not term.isdigit()
        and term not in _NAVIGATION_GENERIC_TERMS
    }


def _heading_key_component(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", _fold_heading(value)).strip("-")


def _raw_article_heading(
    candidate: RegulatoryProvisionHeadingCandidate,
) -> _RawArticleHeading | None:
    article_headings = []
    for index, heading in enumerate(candidate.heading_path):
        stripped_heading = heading.strip()
        parsed_heading = parse_regulatory_article_heading(stripped_heading)
        if stripped_heading and parsed_heading is not None:
            article_headings.append((index, stripped_heading, parsed_heading))
    if not article_headings:
        return None

    article_index, original_heading_label, parsed = article_headings[-1]
    scope_labels = tuple(
        heading.strip()
        for heading in candidate.heading_path[:article_index]
        if heading.strip() and parse_regulatory_article_heading(heading) is None
    )
    qualifier_prefix = f"{parsed.qualifier}-" if parsed.qualifier else ""
    article_base_key = (
        f"{qualifier_prefix}madde:{_heading_key_component(parsed.article_no)}"
    )
    qualifier_label = {
        "gecici": "GEÇİCİ ",
        "mukerrer": "MÜKERRER ",
    }.get(parsed.qualifier or "", "")
    canonical_article_label = f"{qualifier_label}MADDE {parsed.article_no}"

    inline_title = ""
    if not parsed.is_reverse:
        prefix_match = _ARTICLE_PREFIX_RE.match(original_heading_label)
        if prefix_match is not None:
            inline_title = original_heading_label[prefix_match.end() :].strip(" -–—:.")
    metadata_title = (candidate.article_title or "").strip()
    title = metadata_title or inline_title

    topic_hint = ""
    if not title:
        for descendant in candidate.heading_path[article_index + 1 :]:
            descendant = " ".join(descendant.split())
            if (
                not descendant
                or parse_regulatory_article_heading(descendant) is not None
                or _BARE_DESCENDANT_RE.fullmatch(_fold_heading(descendant)) is not None
            ):
                continue
            topic_hint = descendant[:_MAX_TOPIC_HINT_CHARS].rstrip()
            break

    if title:
        heading_label = (
            f"{canonical_article_label} — {title[:_MAX_TOPIC_HINT_CHARS].rstrip()}"
        )
        label_priority = 2
    elif topic_hint:
        heading_label = f"{canonical_article_label} — {topic_hint}"
        label_priority = 1
    else:
        heading_label = canonical_article_label
        label_priority = 0
    return _RawArticleHeading(
        article_base_key=article_base_key,
        scope_labels=scope_labels,
        scope_keys=tuple(_fold_heading(label) for label in scope_labels),
        heading_label=heading_label,
        position=candidate.position,
        label_priority=label_priority,
        regulatory_chunk_id=candidate.regulatory_chunk_id,
    )


def _strip_common_document_root(
    headings: Sequence[_RawArticleHeading],
) -> list[_RawArticleHeading]:
    """Drop at most one shared path node while retaining annex/part scopes."""
    common_root = (
        headings[0].scope_keys[0] if headings and headings[0].scope_keys else None
    )
    if common_root is None or any(
        not heading.scope_keys or heading.scope_keys[0] != common_root
        for heading in headings
    ):
        return list(headings)
    return [
        _RawArticleHeading(
            article_base_key=heading.article_base_key,
            scope_labels=heading.scope_labels[1:],
            scope_keys=heading.scope_keys[1:],
            heading_label=heading.heading_label,
            position=heading.position,
            label_priority=heading.label_priority,
            regulatory_chunk_id=heading.regulatory_chunk_id,
        )
        for heading in headings
    ]


def select_regulatory_provision_navigation(
    source: RegulatoryProvisionHeadingSource,
    *,
    as_of_date: datetime.date | None,
    max_headings: int = DEFAULT_NAVIGATION_MAX_HEADINGS,
    focused_query: str = "",
    query_references: Sequence[RegulatoryProvisionReference] = (),
    result_references: Sequence[RegulatoryProvisionReference] = (),
) -> RegulatoryProvisionNavigation | None:
    """Derive a bounded outline led by explicit and lexical query anchors."""
    if max_headings <= 0:
        raise ValueError("max_headings must be positive")
    if not source.seed_positions:
        return None
    effective_limit = min(max_headings, DEFAULT_NAVIGATION_MAX_HEADINGS)

    raw_headings: list[_RawArticleHeading] = []
    for candidate in source.candidates:
        if candidate.user_file_id != source.user_file_id:
            continue
        if not is_regulatory_navigation_candidate_visible(
            candidate,
            as_of_date=as_of_date,
        ):
            continue
        raw_heading = _raw_article_heading(candidate)
        if raw_heading is not None:
            raw_headings.append(raw_heading)
    if not raw_headings:
        return None
    raw_headings = _strip_common_document_root(raw_headings)

    seed_positions = source.seed_positions
    options_by_key: dict[str, _ArticleHeadingOption] = {}
    for heading in raw_headings:
        scope_key = "/".join(
            component
            for label in heading.scope_keys
            if (component := _heading_key_component(label))
        )
        article_key = (
            f"{scope_key}::{heading.article_base_key}"
            if scope_key
            else heading.article_base_key
        )
        heading_label = " > ".join([*heading.scope_labels, heading.heading_label])
        current_option = _ArticleHeadingOption(
            article_key=article_key,
            article_base_key=heading.article_base_key,
            heading_label=heading_label,
            position=heading.position,
            label_priority=heading.label_priority,
            regulatory_chunk_id=heading.regulatory_chunk_id,
        )
        previous = options_by_key.get(article_key)
        if previous is None:
            options_by_key[article_key] = current_option
            continue

        preferred_label = max(
            [previous, current_option],
            key=lambda option: (
                option.label_priority,
                len(option.heading_label),
                -option.position,
            ),
        )
        anchor = min(
            (previous, current_option),
            key=lambda option: (
                min(abs(option.position - seed) for seed in seed_positions),
                option.position,
            ),
        )
        options_by_key[article_key] = _ArticleHeadingOption(
            article_key=article_key,
            article_base_key=heading.article_base_key,
            heading_label=preferred_label.heading_label,
            position=anchor.position,
            label_priority=preferred_label.label_priority,
            regulatory_chunk_id=anchor.regulatory_chunk_id,
        )

    query_base_key_order = {
        (f"{reference.qualifier}-" if reference.qualifier else "")
        + f"madde:{_heading_key_component(reference.article_no)}": index
        for index, reference in enumerate(dict.fromkeys(query_references))
    }
    result_base_key_order = {
        (f"{reference.qualifier}-" if reference.qualifier else "")
        + f"madde:{_heading_key_component(reference.article_no)}": index
        for index, reference in enumerate(dict.fromkeys(result_references))
        if (
            (f"{reference.qualifier}-" if reference.qualifier else "")
            + f"madde:{_heading_key_component(reference.article_no)}"
        )
        not in query_base_key_order
    }
    query_terms = _navigation_query_terms(focused_query)
    ranked = sorted(
        options_by_key.values(),
        key=lambda option: (
            0 if option.article_base_key in query_base_key_order else 1,
            query_base_key_order.get(option.article_base_key, 0),
            -_focused_query_overlap_score(query_terms, option.heading_label),
            0 if option.article_base_key in result_base_key_order else 1,
            result_base_key_order.get(option.article_base_key, 0),
            min(abs(option.position - position) for position in seed_positions),
            option.position,
            option.article_key,
        ),
    )
    selected = ranked[:effective_limit]
    return RegulatoryProvisionNavigation(
        document_title=source.document_title,
        entries=tuple(
            RegulatoryProvisionNavigationEntry(
                article_key=option.article_key,
                heading_label=option.heading_label,
                regulatory_chunk_id=option.regulatory_chunk_id,
            )
            for option in selected
        ),
        user_file_id=str(source.user_file_id),
    )


def _explicit_references_from_sections(
    query: str,
    sections: Sequence[InferenceSection],
) -> tuple[RegulatoryProvisionReference, ...]:
    """Read literal references from the query and clean retrieved chunk text."""

    ordered_references: list[RegulatoryProvisionReference] = []
    seen: set[RegulatoryProvisionReference] = set()
    texts = [query] if query else []
    texts.extend(section.center_chunk.content for section in sections)
    for text in texts:
        for reference in extract_regulatory_provision_references(text):
            if reference in seen:
                continue
            seen.add(reference)
            ordered_references.append(reference)
    return tuple(ordered_references)


def build_regulatory_provision_navigation(
    db_session: Session,
    sections: Sequence[InferenceSection],
    *,
    query: str = "",
    as_of_date: datetime.date | None,
    max_headings: int = DEFAULT_NAVIGATION_MAX_HEADINGS,
) -> RegulatoryProvisionNavigation | None:
    """Build an outline only when selected regulatory hits establish a source."""
    seed_chunk_ids = [
        chunk_id
        for section in sections
        if (chunk_id := section.center_chunk.regulatory_chunk_id) is not None
    ]
    if not seed_chunk_ids:
        return None
    source = get_regulatory_provision_heading_source(
        db_session,
        seed_chunk_ids,
        as_of_date=as_of_date,
    )
    if source is None:
        return None
    query_references = _explicit_references_from_sections(query, ())
    result_references = _explicit_references_from_sections("", sections)
    return select_regulatory_provision_navigation(
        source,
        as_of_date=as_of_date,
        max_headings=max_headings,
        focused_query=query,
        query_references=query_references,
        result_references=result_references,
    )


def regulatory_provision_navigation_payload(
    navigation: RegulatoryProvisionNavigation,
) -> RegulatoryProvisionNavigationPayload:
    """Return a JSON-safe payload whose epistemic status is explicit."""
    return RegulatoryProvisionNavigationPayload(
        type="regulatory_provision_heading_navigation",
        document_title=navigation.document_title,
        usage_note=(
            "Headings and title/topic hints are navigation leads only; they are "
            "not legal evidence, and an omitted heading is not evidence that a "
            "provision is absent. A lead that could materially change a requested "
            "conclusion remains unresolved until its operative text is retrieved "
            "or the resulting source gap is expressly qualified."
        ),
        headings=[
            RegulatoryProvisionNavigationPayloadEntry(
                article_key=entry.article_key,
                heading_label=entry.heading_label,
            )
            for entry in navigation.entries
        ],
    )


def expand_selected_regulatory_navigation_leads(
    db_session: Session,
    sections: list[InferenceSection],
    *,
    navigation: RegulatoryProvisionNavigation | None,
    query: str,
    as_of_date: datetime.date | None,
    max_total_sections: int,
) -> list[InferenceSection]:
    """Replace weak tail hits with at most two strongly matched provision leads."""

    if navigation is None or not sections or max_total_sections <= 0:
        return sections[:max_total_sections]

    query_terms = _navigation_query_terms(query)
    explicit_references = set(_explicit_references_from_sections(query, ()))
    represented_article_keys: set[str] = set()
    for section in sections:
        for heading in reversed(section.center_chunk.heading_path or []):
            parsed_heading = parse_regulatory_article_heading(heading)
            if parsed_heading is None:
                continue
            represented_article_keys.add(
                (f"{parsed_heading.qualifier}-" if parsed_heading.qualifier else "")
                + f"madde:{_heading_key_component(parsed_heading.article_no)}"
            )
            break

    eligible_entries: list[
        tuple[int, int, int, int, RegulatoryProvisionNavigationEntry]
    ] = []
    for entry_index, entry in enumerate(navigation.entries):
        chunk_id = entry.regulatory_chunk_id
        if chunk_id is None:
            continue
        parsed_key = entry.article_key.rsplit("::", 1)[-1]
        explicit_match = any(
            parsed_key
            == (
                (f"{reference.qualifier}-" if reference.qualifier else "")
                + f"madde:{_heading_key_component(reference.article_no)}"
            )
            for reference in explicit_references
        )
        overlap = _focused_query_overlap_score(query_terms, entry.heading_label)
        represented_match = parsed_key in represented_article_keys
        if explicit_match or represented_match or overlap > 0:
            eligible_entries.append(
                (
                    0 if explicit_match else 1,
                    0 if represented_match else 1,
                    -overlap,
                    entry_index,
                    entry,
                )
            )
    if not eligible_entries:
        return sections[:max_total_sections]
    prioritized_entries = [
        item[-1] for item in sorted(eligible_entries)[:_NAVIGATION_LEAD_MAX_SECTIONS]
    ]

    template = next(
        (
            section.center_chunk
            for section in sections
            if navigation.user_file_id is not None
            and section.center_chunk.document_id == navigation.user_file_id
        ),
        None,
    )
    if template is None:
        return sections[:max_total_sections]

    existing_identities = {
        _chunk_identity(section.center_chunk) for section in sections
    }
    lead_sections: list[InferenceSection] = []
    for entry in prioritized_entries:
        if entry.regulatory_chunk_id is None:
            continue
        projections = get_bounded_same_provision_siblings(
            db_session,
            [entry.regulatory_chunk_id],
            query=query,
            as_of_date=as_of_date,
            max_chunks_per_provision=2,
            max_chars_per_provision=4_000,
        )
        for projection in projections:
            identity = (str(projection.user_file_id), projection.regulatory_chunk_id)
            if identity in existing_identities:
                continue
            lead_chunk = _chunk_from_projection(
                projection,
                template,
                relevance_explanation="Strong lexical regulatory heading lead",
            )
            lead_sections.append(inference_section_from_single_chunk(lead_chunk))
            existing_identities.add(identity)
            if len(lead_sections) >= _NAVIGATION_LEAD_MAX_SECTIONS:
                break
        if len(lead_sections) >= _NAVIGATION_LEAD_MAX_SECTIONS:
            break
    if not lead_sections:
        return sections[:max_total_sections]

    retained = sections[: max(0, max_total_sections - len(lead_sections))]
    retained_identities = {
        _chunk_identity(section.center_chunk) for section in retained
    }
    return retained + [
        section
        for section in lead_sections
        if _chunk_identity(section.center_chunk) not in retained_identities
    ]


def expand_selected_regulatory_source_lexical_matches(
    db_session: Session,
    sections: list[InferenceSection],
    *,
    navigation: RegulatoryProvisionNavigation | None,
    query: str,
    as_of_date: datetime.date | None,
    max_total_sections: int,
) -> list[InferenceSection]:
    """Replace weak tail hits with bounded matches from one known source."""

    if (
        navigation is None
        or navigation.user_file_id is None
        or not sections
        or max_total_sections <= 0
    ):
        return sections[:max_total_sections]
    source_template = next(
        (
            section.center_chunk
            for section in sections
            if section.center_chunk.document_id == navigation.user_file_id
        ),
        None,
    )
    if source_template is None:
        return sections[:max_total_sections]

    existing_chunk_ids = [
        chunk_id
        for section in sections
        if (chunk_id := section.center_chunk.regulatory_chunk_id) is not None
    ]
    source_matches = get_bounded_source_lexical_matches(
        db_session,
        user_file_id=UUID(navigation.user_file_id),
        query=query,
        as_of_date=as_of_date,
        excluded_chunk_ids=existing_chunk_ids,
        max_matches=_SOURCE_LEXICAL_MAX_MATCHES,
    )
    if not source_matches:
        return sections[:max_total_sections]

    sibling_projections = get_bounded_same_provision_siblings(
        db_session,
        [match.regulatory_chunk_id for match in source_matches],
        query=query,
        as_of_date=as_of_date,
        max_chunks_per_provision=2,
        max_chars_per_provision=4_000,
    )
    ordered_projections = [*source_matches, *sibling_projections]
    fallback_sections: list[InferenceSection] = []
    seen_identities = {_chunk_identity(section.center_chunk) for section in sections}
    for projection in ordered_projections:
        identity = (str(projection.user_file_id), projection.regulatory_chunk_id)
        if identity in seen_identities:
            continue
        chunk = _chunk_from_projection(
            projection,
            source_template,
            relevance_explanation="Focused lexical match within selected source",
        )
        fallback_sections.append(inference_section_from_single_chunk(chunk))
        seen_identities.add(identity)
        if len(fallback_sections) >= min(
            _SOURCE_LEXICAL_MAX_SECTIONS,
            max_total_sections,
        ):
            break
    if not fallback_sections:
        return sections[:max_total_sections]

    retained = sections[: max(0, max_total_sections - len(fallback_sections))]
    return [*retained, *fallback_sections]


def _chunk_identity(chunk: InferenceChunk) -> tuple[str, str | int]:
    if chunk.regulatory_chunk_id is not None:
        return (chunk.document_id, chunk.regulatory_chunk_id)
    return (chunk.document_id, chunk.chunk_id)


def _projection_provision_identity(
    projection: RegulatoryChunkProjection,
) -> tuple[str, ...]:
    last_article_index: int | None = None
    for index, heading in enumerate(projection.heading_path):
        if parse_regulatory_article_heading(heading) is not None:
            last_article_index = index
    structural_path = (
        projection.heading_path[: last_article_index + 1]
        if last_article_index is not None
        else projection.heading_path[:-1]
    )
    return (
        str(projection.user_file_id),
        *(_fold_heading(heading) for heading in structural_path),
    )


def _packet_document(members: Sequence[InferenceChunk]) -> str:
    passages: list[str] = []
    for member in members:
        heading = (
            " > ".join(member.heading_path)
            if isinstance(member.heading_path, list)
            else member.semantic_identifier
        )
        passages.append(f"[{heading}]\n{member.content}" if heading else member.content)
    return "\n\n".join(passages)


def _singleton_rerank_packets(
    chunks: Sequence[InferenceChunk],
) -> list[RegulatoryRerankPacket]:
    return [
        RegulatoryRerankPacket(candidate=chunk, members=(chunk,)) for chunk in chunks
    ]


def build_regulatory_rerank_packets(
    db_session: Session,
    chunks: Sequence[InferenceChunk],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_chunks_per_provision: int = 12,
    max_chars_per_provision: int = 6_000,
) -> list[RegulatoryRerankPacket]:
    """Build bounded structural reranker documents in one metadata lookup.

    The synthetic candidate is used only for scoring. Every packet retains the
    exact atomic chunks that downstream citation and authorization code consumes.
    Any metadata failure fails closed to the original singleton candidates.
    """

    deduplicated: list[InferenceChunk] = []
    seen_chunk_identities: set[tuple[str, str | int]] = set()
    for chunk in chunks:
        identity = _chunk_identity(chunk)
        if identity not in seen_chunk_identities:
            seen_chunk_identities.add(identity)
            deduplicated.append(chunk)
    seed_ids = [
        chunk_id
        for chunk in deduplicated
        if (chunk_id := chunk.regulatory_chunk_id) is not None
    ]
    if not seed_ids:
        return _singleton_rerank_packets(deduplicated)

    try:
        projections = get_bounded_same_provision_siblings(
            db_session,
            seed_ids,
            query=query,
            as_of_date=as_of_date,
            max_chunks_per_provision=max_chunks_per_provision,
            max_chars_per_provision=max_chars_per_provision,
        )
    except Exception:
        logger.exception(
            "Regulatory rerank packet metadata lookup failed; using singletons"
        )
        return _singleton_rerank_packets(deduplicated)

    template_by_document = {
        chunk.document_id: chunk
        for chunk in deduplicated
        if chunk.regulatory_chunk_id is not None
    }
    members_by_family: dict[tuple[str, ...], list[InferenceChunk]] = {}
    member_ids_by_family: dict[tuple[str, ...], set[str]] = {}
    family_by_seed_id: dict[str, tuple[str, ...]] = {}
    for projection in projections:
        template = template_by_document.get(str(projection.user_file_id))
        if template is None:
            continue
        family = _projection_provision_identity(projection)
        family_member_ids = member_ids_by_family.setdefault(family, set())
        if projection.regulatory_chunk_id in family_member_ids:
            if projection.regulatory_chunk_id in seed_ids:
                family_by_seed_id[projection.regulatory_chunk_id] = family
            continue
        family_member_ids.add(projection.regulatory_chunk_id)
        member = _chunk_from_projection(
            projection,
            template,
            relevance_explanation="Atomic member of structural rerank packet",
        )
        members_by_family.setdefault(family, []).append(member)
        if projection.regulatory_chunk_id in seed_ids:
            family_by_seed_id[projection.regulatory_chunk_id] = family

    packets: list[RegulatoryRerankPacket] = []
    emitted_families: set[tuple[str, ...]] = set()
    for seed in deduplicated:
        seed_id = seed.regulatory_chunk_id
        family = family_by_seed_id.get(seed_id) if seed_id is not None else None
        if family is None:
            packets.append(RegulatoryRerankPacket(candidate=seed, members=(seed,)))
            continue
        if family in emitted_families:
            continue
        emitted_families.add(family)
        all_members = sorted(
            members_by_family[family],
            key=lambda member: (member.chunk_id, member.regulatory_chunk_id or ""),
        )
        prioritized_members = [
            member for member in all_members if member.regulatory_chunk_id in seed_ids
        ]
        prioritized_members.extend(
            member
            for member in all_members
            if member.regulatory_chunk_id not in seed_ids
        )
        bounded_members: list[InferenceChunk] = []
        rendered_chars = 0
        for member in prioritized_members:
            if len(bounded_members) >= max_chunks_per_provision:
                break
            rendered_member = _packet_document((member,))
            separator_chars = 2 if bounded_members else 0
            if (
                bounded_members
                and rendered_chars + separator_chars + len(rendered_member)
                > max_chars_per_provision
            ):
                continue
            bounded_members.append(member)
            rendered_chars += separator_chars + len(rendered_member)
        members = tuple(
            sorted(
                bounded_members,
                key=lambda member: (member.chunk_id, member.regulatory_chunk_id or ""),
            )
        )
        if len(members) == 1:
            packets.append(
                RegulatoryRerankPacket(candidate=members[0], members=members)
            )
            continue
        first = members[0]
        candidate_document = _packet_document(members)[:max_chars_per_provision]
        candidate = first.model_copy(
            deep=True,
            update={
                "content": candidate_document,
                "blurb": candidate_document[:_SIBLING_BLURB_CHARS],
                "match_highlights": [],
                "relevance_explanation": "Structural provision packet for reranking",
            },
        )
        packets.append(RegulatoryRerankPacket(candidate=candidate, members=members))
    return packets


def expand_ranked_regulatory_rerank_packets(
    ranked_candidates: Sequence[InferenceChunk],
    packets: Sequence[RegulatoryRerankPacket],
    packet_scores: dict[tuple[str, int], float],
) -> tuple[list[InferenceChunk], dict[tuple[str, int], float]]:
    """Project packet ordering/scores back onto exact, deduplicated atomics."""

    packet_by_candidate = {
        _chunk_identity(packet.candidate): packet for packet in packets
    }
    expanded: list[InferenceChunk] = []
    expanded_scores: dict[tuple[str, int], float] = {}
    seen: set[tuple[str, str | int]] = set()
    for candidate in ranked_candidates:
        packet = packet_by_candidate.get(_chunk_identity(candidate))
        if packet is None:
            packet = RegulatoryRerankPacket(candidate=candidate, members=(candidate,))
        packet_score = packet_scores.get((candidate.document_id, candidate.chunk_id))
        for member in packet.members:
            identity = _chunk_identity(member)
            if identity in seen:
                continue
            seen.add(identity)
            expanded.append(member)
            if packet_score is not None:
                expanded_scores[(member.document_id, member.chunk_id)] = packet_score
    return expanded, expanded_scores


def _structural_provision_family_identity(
    chunk: InferenceChunk,
) -> tuple[str, ...] | None:
    """Identify one local provision family without using its mutable chunk ID."""

    if chunk.regulatory_chunk_id is None or not isinstance(chunk.heading_path, list):
        return None
    last_article_index: int | None = None
    for heading_index, heading in enumerate(chunk.heading_path):
        if isinstance(heading, str) and parse_regulatory_article_heading(heading):
            last_article_index = heading_index
    if last_article_index is None:
        return None
    return (
        chunk.document_id,
        *(
            _fold_heading(heading)
            for heading in chunk.heading_path[: last_article_index + 1]
            if isinstance(heading, str)
        ),
    )


def _document_semantic_identifier(chunk: InferenceChunk) -> str:
    if chunk.title:
        return chunk.title
    return chunk.semantic_identifier.split(" — ", 1)[0]


def _chunk_from_projection(
    projection: RegulatoryChunkProjection,
    template: InferenceChunk,
    *,
    relevance_explanation: str = "Same bounded regulatory provision",
) -> InferenceChunk:
    heading_path = normalize_regulatory_heading_path(
        projection.heading_path,
        article_no=projection.article_no,
        chunk_type=projection.chunk_type,
        paragraph_no=projection.paragraph_no,
        clause_label=projection.clause_label,
    )
    document_identifier = _document_semantic_identifier(template)
    semantic_identifier = document_identifier
    if heading_path:
        semantic_identifier = f"{document_identifier} — {' > '.join(heading_path)}"

    return template.model_copy(
        deep=True,
        update={
            "chunk_id": projection.projection_index,
            "content": projection.text,
            "blurb": projection.text[:_SIBLING_BLURB_CHARS],
            "semantic_identifier": semantic_identifier,
            "score": template.score,
            "is_relevant": True,
            "relevance_explanation": relevance_explanation,
            "match_highlights": [],
            "regulatory_chunk_id": projection.regulatory_chunk_id,
            "heading_path": heading_path,
            "validity_start_date": projection.validity_start_date,
            "validity_end_date": projection.validity_end_date,
        },
    )


def expand_selected_regulatory_references(
    db_session: Session,
    sections: list[InferenceSection],
    *,
    reference_sections: Sequence[InferenceSection],
    query: str,
    as_of_date: datetime.date | None,
    max_total_sections: int,
) -> list[InferenceSection]:
    """Expose bounded, exact chunks named by one-hop result references.

    References are extracted only from the original query and clean ranked hit
    content. Newly fetched provision text is never traversed recursively.
    """

    if not sections or max_total_sections <= 0:
        return sections[:max_total_sections]

    expanded: list[InferenceSection] = []
    seen_identities: set[tuple[str, str | int]] = set()
    for section in sections:
        identity = _chunk_identity(section.center_chunk)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        expanded.append(section)
    if len(expanded) >= max_total_sections:
        return expanded[:max_total_sections]

    references = _explicit_references_from_sections(query, reference_sections)
    seed_chunk_ids = [
        chunk_id
        for section in reference_sections
        if (chunk_id := section.center_chunk.regulatory_chunk_id) is not None
    ]
    if not references or not seed_chunk_ids:
        return expanded

    projections = get_bounded_referenced_provisions(
        db_session,
        seed_chunk_ids,
        references,
        as_of_date=as_of_date,
        query=query,
    )
    templates_by_document = {
        section.center_chunk.document_id: section.center_chunk
        for section in reference_sections
        if section.center_chunk.regulatory_chunk_id is not None
    }
    query_terms = set(_LEXICAL_TERM_RE.findall(_fold_heading(query)))
    prioritized_projections = sorted(
        projections,
        key=lambda item: (
            item.expansion_priority,
            -_focused_query_overlap_score(query_terms, item.text),
            item.projection_index,
            item.regulatory_chunk_id,
        ),
    )
    admitted: list[tuple[RegulatoryChunkProjection, InferenceChunk]] = []
    admitted_identities: set[tuple[str, str]] = set()
    referenced_chars = 0
    available_slots = max_total_sections - len(expanded)
    for projection in prioritized_projections:
        if len(admitted) >= available_slots:
            break
        projection_identity = (
            str(projection.user_file_id),
            projection.regulatory_chunk_id,
        )
        if (
            projection_identity in seen_identities
            or projection_identity in admitted_identities
        ):
            continue
        if (
            referenced_chars + len(projection.text)
            > _REFERENCED_PROVISION_MAX_TOTAL_CHARS
        ):
            continue
        template = templates_by_document.get(str(projection.user_file_id))
        if template is None:
            continue
        admitted.append((projection, template))
        admitted_identities.add(projection_identity)
        referenced_chars += len(projection.text)

    for projection, template in sorted(
        admitted,
        key=lambda item: (
            item[0].projection_index,
            item[0].regulatory_chunk_id,
        ),
    ):
        referenced_chunk = _chunk_from_projection(
            projection,
            template,
            relevance_explanation="Explicit one-hop regulatory reference",
        )
        seen_identities.add(_chunk_identity(referenced_chunk))
        expanded.append(inference_section_from_single_chunk(referenced_chunk))
    return expanded


def expand_selected_regulatory_adjacent_provisions(
    db_session: Session,
    sections: list[InferenceSection],
    *,
    seed_sections: Sequence[InferenceSection],
    query: str,
    as_of_date: datetime.date | None,
    max_total_sections: int,
) -> list[InferenceSection]:
    """Use spare slots for immediate provisions without evicting exact evidence."""

    if not sections or max_total_sections <= 0:
        return sections[:max_total_sections]
    seed_chunk_ids = [
        chunk_id
        for section in seed_sections
        if (chunk_id := section.center_chunk.regulatory_chunk_id) is not None
    ]
    if not seed_chunk_ids:
        return sections[:max_total_sections]
    projections = get_bounded_adjacent_provisions(
        db_session,
        seed_chunk_ids,
        query=query,
        as_of_date=as_of_date,
    )
    if not projections:
        return sections[:max_total_sections]

    templates_by_document = {
        section.center_chunk.document_id: section.center_chunk
        for section in seed_sections
        if section.center_chunk.regulatory_chunk_id is not None
    }
    existing_identities = {
        _chunk_identity(section.center_chunk) for section in sections
    }
    available_slots = max(0, max_total_sections - len(sections))
    if available_slots == 0:
        return sections[:max_total_sections]
    adjacent_sections: list[InferenceSection] = []
    for projection in projections:
        identity = (str(projection.user_file_id), projection.regulatory_chunk_id)
        if identity in existing_identities:
            continue
        template = templates_by_document.get(str(projection.user_file_id))
        if template is None:
            continue
        adjacent = _chunk_from_projection(
            projection,
            template,
            relevance_explanation="Immediate same-scope regulatory provision",
        )
        adjacent_sections.append(inference_section_from_single_chunk(adjacent))
        existing_identities.add(identity)
        if len(adjacent_sections) >= min(2, available_slots):
            break
    if not adjacent_sections:
        return sections[:max_total_sections]

    retained = sections[:max_total_sections]
    retained_identities = {
        _chunk_identity(section.center_chunk) for section in retained
    }
    return (
        retained
        + [
            section
            for section in adjacent_sections
            if _chunk_identity(section.center_chunk) not in retained_identities
        ][:available_slots]
    )


def expand_selected_regulatory_sections(
    db_session: Session,
    sections: list[InferenceSection],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_total_sections: int,
    structural_seed_sections: Sequence[InferenceSection] | None = None,
) -> list[InferenceSection]:
    """Add independently citable siblings without exceeding the chat budget.

    The LLM relevance selector still chooses the controlling seed chunks. This
    deterministic step only fills in local paragraphs, exact numbered families,
    and bounded adjacent structural context; it neither opens a whole document
    nor performs another model call.
    """
    if not sections or max_total_sections <= 0:
        return sections[:max_total_sections]

    deduplicated_sections: list[InferenceSection] = []
    seen_identities: set[tuple[str, str | int]] = set()
    regulatory_seeds: list[InferenceChunk] = []
    for section in sections:
        identity = _chunk_identity(section.center_chunk)
        if identity in seen_identities:
            continue
        seen_identities.add(identity)
        deduplicated_sections.append(section)
        if section.center_chunk.regulatory_chunk_id is not None:
            regulatory_seeds.append(section.center_chunk)

    if not regulatory_seeds:
        return deduplicated_sections[:max_total_sections]

    expansion_seed_by_id = {
        seed.regulatory_chunk_id: seed
        for seed in regulatory_seeds
        if seed.regulatory_chunk_id is not None
    }
    represented_provision_families = {
        family_identity
        for seed in regulatory_seeds
        if (family_identity := _structural_provision_family_identity(seed)) is not None
    }
    for structural_seed_section in structural_seed_sections or ():
        structural_seed = structural_seed_section.center_chunk
        if structural_seed.regulatory_chunk_id is None:
            continue
        family_identity = _structural_provision_family_identity(structural_seed)
        if (
            family_identity is not None
            and family_identity in represented_provision_families
        ):
            continue
        expansion_seed_by_id.setdefault(
            structural_seed.regulatory_chunk_id,
            structural_seed,
        )
        if family_identity is not None:
            represented_provision_families.add(family_identity)
    expansion_seed_chunk_ids = list(expansion_seed_by_id)
    projections = get_bounded_same_provision_siblings(
        db_session,
        expansion_seed_chunk_ids,
        query=query,
        as_of_date=as_of_date,
    )
    projection_by_id = {
        projection.regulatory_chunk_id: projection for projection in projections
    }
    repaired_sections: list[InferenceSection] = []
    for section in deduplicated_sections:
        template = section.center_chunk
        seed_projection = (
            projection_by_id.get(template.regulatory_chunk_id)
            if template.regulatory_chunk_id is not None
            else None
        )
        if seed_projection is None:
            repaired_sections.append(section)
            continue
        repaired_seed = _chunk_from_projection(
            seed_projection,
            template,
            relevance_explanation=(
                template.relevance_explanation or "Selected regulatory search result"
            ),
        ).model_copy(
            update={
                "is_relevant": template.is_relevant,
                "match_highlights": template.match_highlights,
            }
        )
        repaired_sections.append(inference_section_from_single_chunk(repaired_seed))
    deduplicated_sections = repaired_sections
    regulatory_seeds = [
        section.center_chunk
        for section in deduplicated_sections
        if section.center_chunk.regulatory_chunk_id is not None
    ]
    selected_identities = {
        _chunk_identity(section.center_chunk) for section in deduplicated_sections
    }
    available_sibling_count = sum(
        1
        for projection in projections
        if (str(projection.user_file_id), projection.regulatory_chunk_id)
        not in selected_identities
    )
    reserved_sibling_slots = min(
        5,
        available_sibling_count,
        max_total_sections // 2,
    )
    retained_seed_count = max_total_sections - reserved_sibling_slots
    original_sections = deduplicated_sections
    deduplicated_sections = deduplicated_sections[:retained_seed_count]
    seed_by_id = {
        seed.regulatory_chunk_id: seed
        for seed in regulatory_seeds
        if seed.regulatory_chunk_id is not None
    }
    seed_by_id.update(
        {
            seed_id: seed
            for seed_id, seed in expansion_seed_by_id.items()
            if seed_id not in seed_by_id
        }
    )
    primary_seed_ids = {
        seed.regulatory_chunk_id
        for seed in regulatory_seeds
        if seed.regulatory_chunk_id is not None
    }
    sibling_queues: dict[str, deque[RegulatoryChunkProjection]] = {
        seed_id: deque() for seed_id in seed_by_id
    }
    query_terms = set(_LEXICAL_TERM_RE.findall(_fold_heading(query)))

    for projection in sorted(
        projections,
        key=lambda item: (
            item.expansion_priority,
            -_focused_query_overlap_score(query_terms, item.text),
            item.projection_index,
            item.regulatory_chunk_id,
        ),
    ):
        projection_identity = (
            str(projection.user_file_id),
            projection.regulatory_chunk_id,
        )
        if projection_identity in seen_identities:
            continue
        compatible_seeds = [
            seed
            for seed in seed_by_id.values()
            if seed.regulatory_chunk_id is not None
            and seed.document_id == str(projection.user_file_id)
        ]
        if not compatible_seeds:
            continue
        closest_seed = min(
            compatible_seeds,
            key=lambda seed: (
                abs(seed.chunk_id - projection.projection_index),
                seed.chunk_id,
                seed.regulatory_chunk_id or "",
            ),
        )
        assert closest_seed.regulatory_chunk_id is not None
        sibling_queues[closest_seed.regulatory_chunk_id].append(projection)

    expanded = list(deduplicated_sections)
    seed_rank_by_id = {
        seed.regulatory_chunk_id: seed_rank
        for seed_rank, seed in enumerate(seed_by_id.values())
        if seed.regulatory_chunk_id is not None
    }
    while len(expanded) < max_total_sections:
        priority_seed_ids = [
            seed_id
            for seed_id, queue in sibling_queues.items()
            if queue and queue[0].expansion_priority < 0
        ]
        if priority_seed_ids:
            priority_seed_id = min(
                priority_seed_ids,
                key=lambda seed_id: (
                    sibling_queues[seed_id][0].expansion_priority,
                    -_focused_query_overlap_score(
                        query_terms, sibling_queues[seed_id][0].text
                    ),
                    seed_rank_by_id[seed_id],
                    sibling_queues[seed_id][0].projection_index,
                    sibling_queues[seed_id][0].regulatory_chunk_id,
                ),
            )
            priority_projection = sibling_queues[priority_seed_id].popleft()
            priority_sibling = _chunk_from_projection(
                priority_projection,
                seed_by_id[priority_seed_id],
            )
            priority_identity = _chunk_identity(priority_sibling)
            if priority_identity not in seen_identities:
                seen_identities.add(priority_identity)
                expanded.append(inference_section_from_single_chunk(priority_sibling))
            continue

        added_in_round = False
        active_seed_ids = [
            seed_id for seed_id, queue in sibling_queues.items() if queue
        ]
        primary_active_seed_ids = [
            seed_id for seed_id in active_seed_ids if seed_id in primary_seed_ids
        ]
        if primary_active_seed_ids:
            # Navigation-only seeds are discovery leads. They must not displace
            # same-provision text belonging to the stronger selected hit.
            active_seed_ids = primary_active_seed_ids
        active_seed_ids.sort(
            key=lambda seed_id: (
                sibling_queues[seed_id][0].expansion_priority,
                -_focused_query_overlap_score(
                    query_terms, sibling_queues[seed_id][0].text
                ),
                seed_rank_by_id[seed_id],
                sibling_queues[seed_id][0].projection_index,
                sibling_queues[seed_id][0].regulatory_chunk_id,
            )
        )
        for seed_id in active_seed_ids:
            projection = sibling_queues[seed_id].popleft()
            seed = seed_by_id[seed_id]
            sibling = _chunk_from_projection(projection, seed)
            identity = _chunk_identity(sibling)
            if identity in seen_identities:
                continue
            seen_identities.add(identity)
            expanded.append(inference_section_from_single_chunk(sibling))
            added_in_round = True
            if len(expanded) >= max_total_sections:
                break
        if not added_in_round:
            break

    # If a provision has fewer usable siblings than the reserved allowance,
    # restore the original ranked seeds rather than returning a short result.
    expanded_identities = {
        _chunk_identity(section.center_chunk) for section in expanded
    }
    for section in original_sections:
        if len(expanded) >= max_total_sections:
            break
        identity = _chunk_identity(section.center_chunk)
        if identity in expanded_identities:
            continue
        expanded.append(section)
        expanded_identities.add(identity)

    return expanded
