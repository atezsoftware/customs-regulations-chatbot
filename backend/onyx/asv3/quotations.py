"""Identify quoted wording that needs explicit source-specific verification."""

import re

from pydantic import JsonValue

from onyx.asv3.authority import folded
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem

_QUOTES = re.compile(r"“([^”\n]{1,600})”|\"([^\"\n]{1,600})\"")


def normalized(text: str) -> str:
    return " ".join(folded(text).split())


def canonical_heading_labels(item: EvidenceItem) -> set[str]:
    """Literal labels are structural context, not witnesses to operative rules."""
    canonical = item.metadata.get("canonical_metadata")
    headings = item.metadata.get("heading_path")
    if (
        item.metadata.get("external")
        or item.metadata.get("derived")
        or not item.chunk_id
        or not isinstance(canonical, dict)
        or canonical.get("document_id") != item.source_id
        or canonical.get("regulatory_chunk_id") != item.chunk_id
        or not isinstance(headings, list)
        or canonical.get("heading_path") != headings
    ):
        return set()
    return {
        normalized(label)
        for label in headings
        if isinstance(label, str)
        and label.strip()
        and not label.rstrip().endswith(("...", "…"))
    }


def unmatched_quoted_terms(
    answer: str, scenario: str, ledger: EvidenceLedger
) -> list[dict[str, JsonValue]]:
    """Literal matches and scenario quotes need no additional model work."""
    terms: list[dict[str, JsonValue]] = []
    seen: set[tuple[str, tuple[int, ...]]] = set()
    facts = normalized(scenario)
    for passage in re.split(r"\n\s*\n", answer):
        citations = extract_citation_numbers(passage)
        if not citations:
            continue
        originals = [
            normalized(item.text)
            for n in citations
            if (item := ledger.get(n)) is not None
        ]
        labels = {
            label
            for n in citations
            if (item := ledger.get(n)) is not None
            for label in canonical_heading_labels(item)
        }
        for match in _QUOTES.finditer(passage):
            term = match[1] or match[2]
            key = normalized(term), citations
            if (
                key in seen
                or key[0] in facts
                or any(key[0] in text for text in originals)
                or key[0] in labels
            ):
                continue
            seen.add(key)
            terms.append(
                {
                    "term_id": f"qt{len(terms)}",
                    "term": term,
                    "passage": passage,
                    "evidence_numbers": list(citations),
                }
            )
    return terms
