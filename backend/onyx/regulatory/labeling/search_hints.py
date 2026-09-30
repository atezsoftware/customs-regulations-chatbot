"""Small, source-neutral taxonomy hints attached to individual planned queries."""

import re
import unicodedata
from collections.abc import Sequence
from typing import TYPE_CHECKING

from onyx.regulatory.labeling.search_models import LabelSearchHint, LabelSearchSnapshot

if TYPE_CHECKING:
    from onyx.regulatory.coverage_plan import RegulatoryCoverageItem

MAX_CATALOG_CARDS = 32


def _words(value: str) -> set[str]:
    value = "".join(
        c
        for c in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(c)
    ).replace("ı", "i")
    return {word[:6] for word in re.findall(r"\w+", value) if len(word) >= 4}


def planning_label_catalog(
    snapshot: LabelSearchSnapshot, query: str
) -> list[dict[str, str]]:
    words = _words(query)
    scored = [
        (len(words & _words(f"{label.name} {label.description}")), label)
        for label in snapshot.taxonomy.labels
    ]
    scored.sort(key=lambda pair: (-pair[0], pair[1].id))
    return [
        {"id": label.id, "name": label.name, "description": label.description[:280]}
        for score, label in scored[:MAX_CATALOG_CARDS]
        if score > 0
    ]


def hint_for_query(
    items: Sequence["RegulatoryCoverageItem"], query: str
) -> dict[str, list[str]]:
    identity = " ".join(query.casefold().split())
    labels: list[str] = []
    for item in items:
        if identity not in {
            " ".join(q.casefold().split()) for q in item.retrieval_queries
        }:
            continue
        for hint in item.label_hints:
            if " ".join(hint.query.casefold().split()) == identity:
                labels.extend(hint.label_ids)
    return {"label_ids": list(dict.fromkeys(labels))[:12]}


def explicit_subject_hint(snapshot: LabelSearchSnapshot, query: str) -> LabelSearchHint:
    """Recognize explicit taxonomy names/acronyms when a planner omits its hint."""
    from onyx.regulatory.labeling.search_ranking import resolve_label_facets

    def tokens(value: str) -> tuple[str, ...]:
        folded = "".join(
            c
            for c in unicodedata.normalize("NFKD", value.casefold())
            if not unicodedata.combining(c)
        ).replace("ı", "i")
        return tuple(re.findall(r"\w+", folded))

    query_tokens = tokens(query)
    padded_query = " " + " ".join(query_tokens) + " "
    facets = resolve_label_facets(snapshot.taxonomy)
    matched: list[str] = []
    for label in snapshot.taxonomy.labels:
        if facets.get(label.id) != "subject":
            continue
        name = tokens(label.name)
        name_match = bool(name) and " " + " ".join(name) + " " in padded_query
        acronyms = re.findall(r"(?<![.\w])[A-ZÇĞİÖŞÜ]{3,6}(?![.\w])", label.description)
        acronym_match = any(tokens(acronym)[0] in query_tokens for acronym in acronyms)
        if name_match or acronym_match:
            matched.append(label.id)
    return LabelSearchHint(label_ids=tuple(matched[:12]))
