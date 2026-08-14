"""Canonicalize structural paths projected from regulatory source rows."""

import re
import unicodedata
from collections.abc import Sequence
from typing import NamedTuple

_FORWARD_ARTICLE_HEADING_RE = re.compile(
    r"^(?:(?P<qualifier>gecici|geçici|mukerrer|mükerrer)\s+)?"
    r"(?:madde|article|art\.?)"
    r"(?:\s+|\s*[:.]\s*)"
    r"(?P<number>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_REVERSE_ARTICLE_HEADING_RE = re.compile(
    r"^(?P<number>\d+[a-z]?)\s+maddes[iı]\s*[:.]?\s*$",
    flags=re.IGNORECASE,
)
_QUERY_FORWARD_ARTICLE_RE = re.compile(
    r"(?<![a-z0-9])"
    r"(?:(?P<qualifier>gecici|mukerrer)\s+)?"
    r"(?:madde|md|article|art)\.?\s*:?[ \t]*"
    r"(?P<number>\d+[a-z]?)(?![a-z0-9]|\.\d)",
    flags=re.IGNORECASE,
)
_INFLECTED_ARTICLE_WORD = (
    r"(?:madde(?:de|den|nin|ye|yi)?|maddes(?:i|ı)(?:nde|nden|nin|ne|ni)?)"
)
_QUERY_REVERSE_ARTICLE_RE = re.compile(
    r"(?<![a-z0-9])(?P<number>\d+[a-z]?)(?![a-z0-9]|\.\d)"
    r"(?:"
    rf"\.\s*{_INFLECTED_ARTICLE_WORD}"
    rf"|\s*['’]?\s*(?:inci|nci|uncu|ıncı)\s+{_INFLECTED_ARTICLE_WORD}"
    r"|\s+maddes(?:i|ı)(?:nde|nden|nin|ne|ni)?"
    r")\b",
    flags=re.IGNORECASE,
)
_QUERY_ANNEX_SCOPE_RE = re.compile(
    r"(?<![a-z0-9])(?:ek|annex)\s*[-.:]?\s*"
    r"(?P<identifier>\d+[a-z]?|[ivxlcdm]+)\b",
    flags=re.IGNORECASE,
)
_QUERY_SERIES_SCOPE_RE = re.compile(
    r"(?<![a-z0-9])(?:seri|series)\s*"
    r"(?:(?:no|numara|number)\.?\s*[:.]?\s*)?"
    r"(?P<identifier>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_QUERY_SOURCE_HINT_TERM_RE = re.compile(r"[^\W_]+", flags=re.UNICODE)
_MAX_QUERY_SOURCE_HINT_TERMS = 5
_SOURCE_DESIGNATOR_STEMS = (
    "agreement",
    "anlasma",
    "anlasmas",
    "convention",
    "decision",
    "decree",
    "directive",
    "genelge",
    "kanun",
    "karar",
    "protokol",
    "protocol",
    "regulation",
    "rules",
    "sozlesme",
    "teblig",
    "treaty",
    "tuzuk",
    "yonerge",
    "yonetmelik",
)
_SOURCE_DESIGNATOR_EXACT = frozenset({"act", "code", "law"})
_ANONYMOUS_SOURCE_TERMS = frozenset({"a", "an", "bu", "isbu", "the", "this"})
_QUERY_PROVISION_LABEL_TERMS = frozenset({"art", "article", "madde", "md"})
_QUERY_PROVISION_NUMBER_RE = re.compile(r"^\d+[a-z]?$", flags=re.IGNORECASE)
_STRUCTURAL_UNIT_HEADING_RE = re.compile(
    r"^(?:\(\d{1,3}\)|\d{1,3}[.)]|\([a-zçğıöşü]\)|[a-zçğıöşü]\)|"
    r"\([ivxlcdm]+\))\s+(?P<body>\S.*)",
    flags=re.IGNORECASE,
)
_NUMBERED_UNIT_HEADING_RE = re.compile(
    r"^(?:\((?P<parenthesized>\d{1,3})\)|(?P<plain>\d{1,3})[.)])\s+\S"
)
_CLAUSE_UNIT_HEADING_RE = re.compile(
    r"^(?:\((?P<parenthesized>[a-zçğıöşü]|[ivxlcdm]+)\)|"
    r"(?P<plain>[a-zçğıöşü])\))\s+\S",
    flags=re.IGNORECASE,
)


class RegulatoryArticleHeading(NamedTuple):
    """Canonical identity parsed from a structural article heading."""

    article_no: str
    qualifier: str | None
    is_reverse: bool


class RegulatoryProvisionReference(NamedTuple):
    """One explicit structural provision identity extracted from a query."""

    article_no: str
    qualifier: str | None


def _fold(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.casefold())
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _canonical_article_number(value: str) -> str:
    folded = " ".join(_fold(value).split()).upper()
    folded = folded.removeprefix("GEÇICI ").removeprefix("GECICI ")
    folded = folded.removeprefix("MÜKERRER ").removeprefix("MUKERRER ")
    return folded


def extract_single_regulatory_provision_reference(
    query: str,
) -> RegulatoryProvisionReference | None:
    """Extract one unambiguous article reference without interpreting its topic.

    Dotted identifiers are deliberately excluded: a query such as ``8.2.2.8``
    names a nested provision, not article 8. When a query names multiple distinct
    articles, retrieval remains unconstrained so the search model can split or
    reformulate the request itself.
    """

    references = set(extract_regulatory_provision_references(query))

    if len(references) != 1:
        return None
    return next(iter(references))


def extract_regulatory_provision_references(
    text: str,
) -> tuple[RegulatoryProvisionReference, ...]:
    """Extract explicit article references in textual order.

    This is deliberately structural: it recognizes only an article designator
    paired with an article number. It does not infer a provision from topic
    words, and duplicate references are returned only once.
    """

    folded_text = _fold(text)
    occurrences: list[tuple[int, RegulatoryProvisionReference]] = []
    for match in _QUERY_FORWARD_ARTICLE_RE.finditer(folded_text):
        occurrences.append(
            (
                match.start(),
                RegulatoryProvisionReference(
                    article_no=_canonical_article_number(match.group("number")),
                    qualifier=match.group("qualifier"),
                ),
            )
        )
    for match in _QUERY_REVERSE_ARTICLE_RE.finditer(folded_text):
        occurrences.append(
            (
                match.start(),
                RegulatoryProvisionReference(
                    article_no=_canonical_article_number(match.group("number")),
                    qualifier=None,
                ),
            )
        )

    references: list[RegulatoryProvisionReference] = []
    seen: set[RegulatoryProvisionReference] = set()
    for _, reference in sorted(occurrences, key=lambda occurrence: occurrence[0]):
        if reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    return tuple(references)


def _matching_provision_starts(
    folded_query: str,
    reference: RegulatoryProvisionReference,
) -> list[int]:
    matching_starts: list[int] = []
    for pattern in (_QUERY_FORWARD_ARTICLE_RE, _QUERY_REVERSE_ARTICLE_RE):
        for match in pattern.finditer(folded_query):
            qualifier = match.groupdict().get("qualifier")
            candidate = RegulatoryProvisionReference(
                article_no=_canonical_article_number(match.group("number")),
                qualifier=qualifier,
            )
            if candidate == reference:
                matching_starts.append(match.start())
    return matching_starts


def extract_regulatory_provision_source_hint(query: str) -> str | None:
    """Return a bounded source-name lead written before one explicit article.

    Search-tool queries are intentionally focused, so the terms immediately
    preceding ``Madde N``/``Article N`` are the least assumptive source signal.
    The hint is a ranking boost only; it never filters out another source.
    """

    reference = extract_single_regulatory_provision_reference(query)
    if reference is None:
        return None

    folded_query = _fold(query)
    matching_starts = _matching_provision_starts(folded_query, reference)
    if not matching_starts:
        return None

    prefix = folded_query[: min(matching_starts)]
    terms = _QUERY_SOURCE_HINT_TERM_RE.findall(prefix)
    if not terms:
        return None
    return " ".join(terms[-_MAX_QUERY_SOURCE_HINT_TERMS:])


def regulatory_query_scope_heading_phrases(query: str) -> tuple[str, ...]:
    """Return explicit annex/series leads written before one provision.

    Scope is deliberately a ranking signal rather than a filter: older indexed
    instruments do not always retain every enclosing annex label in each chunk.
    """

    reference = extract_single_regulatory_provision_reference(query)
    if reference is None:
        return ()

    folded_query = _fold(query)
    matching_starts = _matching_provision_starts(folded_query, reference)
    if not matching_starts:
        return ()
    prefix = folded_query[: min(matching_starts)]

    phrases: list[str] = []
    for match in _QUERY_ANNEX_SCOPE_RE.finditer(prefix):
        identifier = match.group("identifier").upper()
        phrases.extend((f"EK {identifier}", f"ANNEX {identifier}"))
    for match in _QUERY_SERIES_SCOPE_RE.finditer(prefix):
        identifier = match.group("identifier").upper()
        phrases.extend(
            (
                f"SERİ NO {identifier}",
                f"SERI NO {identifier}",
                f"SERIES NO {identifier}",
            )
        )
    return tuple(dict.fromkeys(phrases))


def _is_source_designator(term: str) -> bool:
    return term in _SOURCE_DESIGNATOR_EXACT or any(
        term.startswith(stem) for stem in _SOURCE_DESIGNATOR_STEMS
    )


def extract_regulatory_instrument_source_hint(query: str) -> str | None:
    """Extract one conservative named-instrument lead from a focused query.

    Detection is folded, while the returned phrase preserves the original
    Unicode spelling for the configured analyzer. The hint is a positive boost,
    never a source filter.
    """

    original_terms = _QUERY_SOURCE_HINT_TERM_RE.findall(query)
    folded_terms = [_fold(term) for term in original_terms]
    candidates: dict[str, str] = {}
    for designator_index, term in enumerate(folded_terms):
        if not _is_source_designator(term):
            continue

        start_index = max(0, designator_index - _MAX_QUERY_SOURCE_HINT_TERMS + 1)
        for index in range(start_index, designator_index):
            if folded_terms[index] not in _QUERY_PROVISION_LABEL_TERMS:
                continue
            start_index = index + 1
            if start_index < designator_index and _QUERY_PROVISION_NUMBER_RE.fullmatch(
                folded_terms[start_index]
            ):
                start_index += 1

        # In a sentence such as ``... gönderimi Basel Sözleşmesi bakımından``,
        # a fixed-width prefix also captures unrelated issue words. Prefer the
        # contiguous title-cased name immediately before the legal-form word.
        # Lower-case/model-normalized queries retain the bounded fallback above.
        title_start_index = designator_index
        while title_start_index > start_index:
            preceding_term = original_terms[title_start_index - 1]
            if not preceding_term[:1].isupper():
                break
            title_start_index -= 1
        if title_start_index < designator_index:
            start_index = title_start_index

        prefix_terms = folded_terms[start_index:designator_index]
        if not prefix_terms or all(
            term in _ANONYMOUS_SOURCE_TERMS for term in prefix_terms
        ):
            continue
        candidate = " ".join(original_terms[start_index : designator_index + 1])
        candidates[_fold(candidate)] = candidate

    if len(candidates) != 1:
        return None
    return next(iter(candidates.values()))


def extract_regulatory_source_hint(query: str) -> str | None:
    """Prefer a named instrument, with the explicit-provision fallback."""

    return extract_regulatory_instrument_source_hint(
        query
    ) or extract_regulatory_provision_source_hint(query)


def extract_regulatory_distinctive_source_hint(query: str) -> str | None:
    """Return source-name terms without a generic legal-form designator.

    A translated query may preserve a proper name while translating words such
    as ``Convention`` or ``Sözleşmesi``. This hint remains a ranking signal and
    never excludes another instrument.
    """

    instrument_hint = extract_regulatory_instrument_source_hint(query)
    if instrument_hint is None:
        provision_hint = extract_regulatory_provision_source_hint(query)
        if provision_hint is None:
            return None
        distinctive_terms = [
            term
            for term in _QUERY_SOURCE_HINT_TERM_RE.findall(provision_hint)
            if term not in _ANONYMOUS_SOURCE_TERMS and not _is_source_designator(term)
        ]
        return " ".join(distinctive_terms) or None
    terms = _QUERY_SOURCE_HINT_TERM_RE.findall(instrument_hint)
    if len(terms) < 2:
        return instrument_hint
    return " ".join(terms[:-1])


def regulatory_provision_heading_phrases(
    reference: RegulatoryProvisionReference,
) -> tuple[str, ...]:
    """Return exact analyzed-heading alternatives for one provision identity."""

    qualifier = {
        "gecici": "GEÇİCİ ",
        "mukerrer": "MÜKERRER ",
    }.get(reference.qualifier or "", "")
    article_no = reference.article_no
    if qualifier:
        return (
            f"{qualifier}MADDE {article_no}",
            f"{qualifier}MADDE{article_no}",
        )
    return (
        f"MADDE {article_no}",
        f"MADDE{article_no}",
        f"{article_no} MADDESİ",
        f"{article_no} MADDESI",
        f"ARTICLE {article_no}",
        f"ARTICLE{article_no}",
        f"ART {article_no}",
    )


def regulatory_heading_path_matches_reference(
    heading_path: Sequence[str],
    reference: RegulatoryProvisionReference,
) -> bool:
    """Check the normalized structural anchor, ignoring descendant prose.

    Regulatory projection keeps the current article as the first article-like
    node in a normalized path. A later descendant can begin with a cross-reference
    and must not be allowed to relabel the chunk.
    """

    for heading in heading_path:
        parsed = parse_regulatory_article_heading(heading)
        if parsed is None:
            continue
        return (
            parsed.article_no == reference.article_no
            and parsed.qualifier == reference.qualifier
        )
    return False


def parse_regulatory_article_heading(
    heading: str,
) -> RegulatoryArticleHeading | None:
    """Parse forward headings and exact reverse-form article headings.

    Forward headings may retain their descriptive suffix. The reverse form is
    deliberately full-line-only so prose such as ``4A maddesi uyarınca`` is
    never promoted to a structural article boundary.
    """

    stripped = heading.strip()
    forward_match = _FORWARD_ARTICLE_HEADING_RE.match(stripped)
    if forward_match is not None:
        qualifier = forward_match.group("qualifier")
        return RegulatoryArticleHeading(
            article_no=_canonical_article_number(forward_match.group("number")),
            qualifier=_fold(qualifier) if qualifier is not None else None,
            is_reverse=False,
        )

    reverse_match = _REVERSE_ARTICLE_HEADING_RE.fullmatch(_fold(stripped))
    if reverse_match is None:
        return None
    return RegulatoryArticleHeading(
        article_no=_canonical_article_number(reverse_match.group("number")),
        qualifier=None,
        is_reverse=True,
    )


def _canonical_article_heading_from_metadata(article_no: str) -> str | None:
    target = _canonical_article_number(article_no)
    if re.fullmatch(r"\d+[A-Z]?", target) is None:
        return None
    folded_article_no = " ".join(_fold(article_no).split())
    qualifier = ""
    if folded_article_no.startswith("gecici "):
        qualifier = "GEÇİCİ "
    elif folded_article_no.startswith("mukerrer "):
        qualifier = "MÜKERRER "
    return f"{qualifier}MADDE {target}"


def _terminal_heading_matches_unit_metadata(
    heading: str,
    *,
    chunk_type: str | None,
    paragraph_no: str | None,
    clause_label: str | None,
) -> bool:
    if chunk_type in {"clause", "subclause"} and clause_label is not None:
        match = _CLAUSE_UNIT_HEADING_RE.match(heading)
        if match is None:
            return False
        marker = match.group("parenthesized") or match.group("plain") or ""
        return _fold(marker) == _fold(clause_label)
    if chunk_type == "paragraph" and paragraph_no is not None:
        match = _NUMBERED_UNIT_HEADING_RE.match(heading)
        if match is None:
            return False
        marker = match.group("parenthesized") or match.group("plain") or ""
        return marker == paragraph_no
    return (
        chunk_type in {"clause", "numbered_section", "paragraph", "subclause"}
        and _STRUCTURAL_UNIT_HEADING_RE.match(heading) is not None
    )


def _is_probable_nested_unit_parent(heading: str) -> bool:
    if _STRUCTURAL_UNIT_HEADING_RE.match(heading) is None:
        return False
    letters = "".join(character for character in heading if character.isalpha())
    return not (
        letters
        and letters == letters.upper()
        and letters != letters.lower()
        and not heading.rstrip().endswith(":")
    )


def _is_legacy_nested_unit_before_article(heading: str) -> bool:
    """Recognize a legacy descendant leaked across a later article boundary."""

    match = _STRUCTURAL_UNIT_HEADING_RE.match(heading)
    if match is None or not _is_probable_nested_unit_parent(heading):
        return False
    body = match.group("body").rstrip()
    if body.endswith(("...", "…")):
        return True
    # Legacy labels use a 90-character body limit. A complete sentence exactly
    # on that boundary was retained without the otherwise diagnostic ellipsis.
    return len(body) == 90 and body.endswith(".")


def normalize_regulatory_heading_path(
    heading_path: Sequence[str],
    *,
    article_no: str | None,
    chunk_type: str | None = None,
    paragraph_no: str | None = None,
    clause_label: str | None = None,
) -> list[str]:
    """Repair legacy article lineage without inventing substantive headings.

    A structural section before the first article remains intact. Metadata
    identifies a forward article boundary. An exact reverse-form article at
    the end of a legacy path is structural and takes precedence over stale
    parent-article metadata. When an older row has authoritative article
    metadata but no article node at all, a canonical label is inserted before
    its trailing structural unit.
    """

    path = [heading.strip() for heading in heading_path if heading.strip()]
    if not path:
        article_heading = (
            _canonical_article_heading_from_metadata(article_no)
            if article_no is not None
            else None
        )
        return [article_heading] if article_heading is not None else []

    parsed_headings = [parse_regulatory_article_heading(heading) for heading in path]
    article_indices = [
        index
        for index, parsed_heading in enumerate(parsed_headings)
        if parsed_heading is not None
    ]
    if not article_indices:
        if article_no is None:
            return path
        article_heading = _canonical_article_heading_from_metadata(article_no)
        if article_heading is None:
            return path
        insertion_index = len(path)
        if _terminal_heading_matches_unit_metadata(
            path[-1],
            chunk_type=chunk_type,
            paragraph_no=paragraph_no,
            clause_label=clause_label,
        ):
            insertion_index -= 1
            parent_budget = {"clause": 1, "subclause": 2}.get(chunk_type or "", 0)
            while (
                insertion_index > 0
                and parent_budget > 0
                and _is_probable_nested_unit_parent(path[insertion_index - 1])
            ):
                insertion_index -= 1
                parent_budget -= 1
        return path[:insertion_index] + [article_heading] + path[insertion_index:]

    first_article_index = article_indices[0]
    stable_scope_end_index = first_article_index
    while stable_scope_end_index > 0 and _is_legacy_nested_unit_before_article(
        path[stable_scope_end_index - 1]
    ):
        stable_scope_end_index -= 1
    last_article_index = article_indices[-1]
    last_article = parsed_headings[last_article_index]
    if last_article is not None and last_article.is_reverse:
        current_article_index = last_article_index
    else:
        if article_no is None:
            return path
        target = _canonical_article_number(article_no)
        matching_indices: list[int] = []
        for index in article_indices:
            parsed_heading = parsed_headings[index]
            if parsed_heading is not None and parsed_heading.article_no == target:
                matching_indices.append(index)
        if not matching_indices:
            return path
        current_article_index = matching_indices[-1]
    if (
        current_article_index <= first_article_index
        and stable_scope_end_index == first_article_index
    ):
        return path

    return path[:stable_scope_end_index] + path[current_article_index:]
