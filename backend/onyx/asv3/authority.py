"""Track explicit governing references separately from original legal evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.regulatory.heading_path import extract_regulatory_provision_references


def folded(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(char)
    ).replace("ı", "i")


_STATUTE = re.compile(
    r"(?<!\d)(?P<number>\d{2,7})\s+sayili\s+"
    r"[^\n.;:]{0,100}?\bkanun[a-z]*\b"
    r"|\b(?:law|act|statute)\s+(?:no\.?\s*)?(?P<english_number>\d{2,7})\b"
)
_COMPACT_ARTICLE = re.compile(r"\(?\b(\d{1,4})(?:/\d+(?:-[a-z])?)\)?\s*madd")


@dataclass(frozen=True)
class StatuteReference:
    number: str
    article: str | None
    reference_text: str


def statute_references(text: str) -> tuple[StatuteReference, ...]:
    normalized = folded(text.replace("**", "").replace("__", ""))
    found: dict[tuple[str, str | None], StatuteReference] = {}
    for match in _STATUTE.finditer(normalized):
        tail = normalized[match.end() : match.end() + 180]
        # Do not attach a later sentence's article to this instrument.
        tail = re.split(r"[;\n]|\.(?=\s+[A-ZÇĞİÖŞÜa-zçğıöşü])", tail, 1)[0]
        explicit = extract_regulatory_provision_references(tail)
        compact = _COMPACT_ARTICLE.search(tail)
        article = (
            compact[1] if compact else explicit[0].article_no if explicit else None
        )
        number = match["number"] or match["english_number"]
        found.setdefault(
            (number, article),
            StatuteReference(
                number, article, normalized[match.start() : match.end() + 100][:240]
            ),
        )
    return tuple(found.values())


def authority_obligations(
    answer: str, ledger: EvidenceLedger
) -> list[dict[str, JsonValue]]:
    """A named statute's discussion in another instrument is not the statute itself."""
    cited = set(extract_citation_numbers(answer))
    originals = ledger.authority_metadata()
    obligations: list[dict[str, JsonValue]] = []
    for reference in statute_references(answer):
        matching: list[int] = []
        for row in originals:
            number = row.get("citation")
            if not isinstance(number, int) or not row.get("citable"):
                continue
            metadata = row
            kind = folded(str(metadata.get("document_type", "")))
            if kind not in {"kanun", "law", "act", "statute"}:
                continue
            headings = metadata.get("heading_path")
            root = headings[0] if isinstance(headings, list) and headings else ""
            identity = str(metadata.get("title", "")) + " " + str(root)
            if not re.search(rf"(?<!\d){re.escape(reference.number)}(?!\d)", identity):
                continue
            if (
                reference.article is not None
                and str(metadata.get("article_no")) != reference.article
            ):
                continue
            matching.append(number)
        obligations.append(
            {
                "instrument_number": reference.number,
                "article": reference.article,
                "reference_text": reference.reference_text,
                "matching_original_evidence": matching,
                "cited_original_evidence": sorted(cited.intersection(matching)),
                "status": "original_cited"
                if cited.intersection(matching)
                else "unresolved_original",
            }
        )
    return obligations


def unresolved_authority_gap(
    answer: str, ledger: EvidenceLedger
) -> dict[str, JsonValue] | None:
    missing = [
        row
        for row in authority_obligations(answer, ledger)
        if row["status"] != "original_cited"
    ]
    if not missing:
        return None
    return {
        "gaps": [
            "A statute explicitly used as the governing basis has no cited original operative evidence."
        ],
        "missing": missing,
        "instruction": (
            "Follow the referenced instrument and provision in the permitted corpus. Reopen a matching recorded original if available; otherwise resolve the source and read the relevant provision. "
            "A lower instrument's reference cannot close this need. Choose the method and useful parallel work yourself; if the original cannot be obtained, disclose that exact gap instead of asserting its rule."
        ),
    }
