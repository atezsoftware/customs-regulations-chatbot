"""Bounded same-provision expansion for selected regulatory search hits."""

import datetime
import re
import unicodedata
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict

from sqlalchemy.orm import Session

from onyx.context.search.models import InferenceChunk, InferenceSection
from onyx.context.search.utils import inference_section_from_single_chunk
from onyx.db.regulatory_chunks import (
    DEFAULT_NAVIGATION_MAX_HEADINGS,
    RegulatoryChunkProjection,
    RegulatoryProvisionHeadingCandidate,
    RegulatoryProvisionHeadingSource,
    get_bounded_referenced_provisions,
    get_bounded_same_provision_siblings,
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


@dataclass(frozen=True, slots=True)
class RegulatoryProvisionNavigationEntry:
    """One structural heading lead; it contains no provision text."""

    article_key: str
    heading_label: str


@dataclass(frozen=True, slots=True)
class RegulatoryProvisionNavigation:
    """A compact outline for one source selected by the search ranking."""

    document_title: str
    entries: tuple[RegulatoryProvisionNavigationEntry, ...]


class RegulatoryProvisionNavigationPayloadEntry(TypedDict):
    article_key: str
    heading_label: str


class RegulatoryProvisionNavigationPayload(TypedDict):
    type: Literal["regulatory_provision_heading_navigation"]
    document_title: str
    usage_note: str
    headings: list[RegulatoryProvisionNavigationPayloadEntry]


@dataclass(frozen=True, slots=True)
class _RawArticleHeading:
    article_base_key: str
    scope_labels: tuple[str, ...]
    scope_keys: tuple[str, ...]
    heading_label: str
    position: int
    label_priority: int


@dataclass(frozen=True, slots=True)
class _ArticleHeadingOption:
    article_key: str
    article_base_key: str
    heading_label: str
    position: int
    label_priority: int


def _fold_heading(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_marks = "".join(
        character for character in decomposed if unicodedata.category(character) != "Mn"
    )
    return " ".join(without_marks.split())


def _focused_query_overlap_score(query_terms: set[str], text: str) -> int:
    """Prefer siblings that cover the model's focused search anchors.

    Exact numeric and alphanumeric identifiers carry more information than an
    ordinary word in legal retrieval (dates, rates, codes, and provision
    labels). This remains document-agnostic and only orders rows that the
    bounded same-provision selector already admitted.
    """

    text_terms = set(_LEXICAL_TERM_RE.findall(_fold_heading(text)))
    return sum(
        4 if any(character.isdigit() for character in term) else 1
        for term in query_terms.intersection(text_terms)
    )


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
        previous = options_by_key.get(article_key)
        if previous is None:
            options_by_key[article_key] = _ArticleHeadingOption(
                article_key=article_key,
                article_base_key=heading.article_base_key,
                heading_label=heading_label,
                position=heading.position,
                label_priority=heading.label_priority,
            )
            continue

        preferred_label = max(
            [
                previous,
                _ArticleHeadingOption(
                    article_key=article_key,
                    article_base_key=heading.article_base_key,
                    heading_label=heading_label,
                    position=heading.position,
                    label_priority=heading.label_priority,
                ),
            ],
            key=lambda option: (
                option.label_priority,
                len(option.heading_label),
                -option.position,
            ),
        )
        options_by_key[article_key] = _ArticleHeadingOption(
            article_key=article_key,
            article_base_key=heading.article_base_key,
            heading_label=preferred_label.heading_label,
            position=min(
                (previous.position, heading.position),
                key=lambda position: (
                    min(abs(position - seed) for seed in seed_positions),
                    position,
                ),
            ),
            label_priority=preferred_label.label_priority,
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
    query_terms = set(_LEXICAL_TERM_RE.findall(_fold_heading(focused_query)))
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
            )
            for option in selected
        ),
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


def _chunk_identity(chunk: InferenceChunk) -> tuple[str, str | int]:
    if chunk.regulatory_chunk_id is not None:
        return (chunk.document_id, chunk.regulatory_chunk_id)
    return (chunk.document_id, chunk.chunk_id)


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


def expand_selected_regulatory_sections(
    db_session: Session,
    sections: list[InferenceSection],
    *,
    query: str,
    as_of_date: datetime.date | None,
    max_total_sections: int,
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

    projections = get_bounded_same_provision_siblings(
        db_session,
        [
            seed.regulatory_chunk_id
            for seed in regulatory_seeds
            if seed.regulatory_chunk_id
        ],
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
    if len(deduplicated_sections) >= max_total_sections:
        return deduplicated_sections[:max_total_sections]
    seed_by_id = {
        seed.regulatory_chunk_id: seed
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
            for seed in regulatory_seeds
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
        for seed_rank, seed in enumerate(regulatory_seeds)
        if seed.regulatory_chunk_id is not None
    }
    while len(expanded) < max_total_sections:
        added_in_round = False
        active_seed_ids = [
            seed_id for seed_id, queue in sibling_queues.items() if queue
        ]
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

    return expanded
