"""Track explicit governing references separately from original legal evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.regulatory.heading_path import (
    extract_regulatory_provision_reference_occurrences,
)


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
_COMPACT_ARTICLE = re.compile(
    r"\(?\b(?P<article>\d{1,4})\s*/\s*(?P<paragraph>\d+)"
    r"(?:\s*[-(]\s*(?P<clause>[a-zçğıöşü])\)?)?\)?\s*madd"
)
_SHORTHAND_ARTICLE = re.compile(
    r"\b(?:(?P<qualifier>geçici|gecici|mükerrer|mukerrer|ek)\s+)?m\.?\s*(?P<article>\d{1,4})(?!\d)"
    r"(?:\s*/\s*(?P<paragraph>\d+)(?:\s*[-(]\s*(?P<clause>[a-zçğıöşü])\)?)?)?"
)
_INSTRUMENT_DESIGNATOR = re.compile(
    r"\b(?:kanun[a-z]*|yonetmeli[kg][a-z]*|teblig[a-z]*|genelge[a-z]*|karar[a-z]*"
    r"|laws?|acts?|statutes?|regulations?|directives?|decrees?|decisions?|circulars?)\b"
)


def _reference_tail(text: str) -> str:
    for boundary in re.finditer(r"[;\n]|\.(?=\s+\S)", text):
        if boundary[0] == ".":
            prefix = folded(text[: boundary.start()])
            suffix = folded(text[boundary.end() :]).lstrip()
            if re.search(r"\b(?:m|md|art|no)$", prefix) or (
                re.search(r"\d$", prefix) and re.match(r"madd[a-z]*\b", suffix)
            ):
                continue
        text = text[: boundary.start()]
        break
    instrument = _INSTRUMENT_DESIGNATOR.search(folded(text))
    return text[: instrument.start()] if instrument is not None else text


@dataclass(frozen=True)
class StatuteReference:
    number: str
    article: str | None
    reference_text: str
    paragraph: str | None = None
    clause: str | None = None
    qualifier: str | None = None


def statute_references(text: str) -> tuple[StatuteReference, ...]:
    # Keep Turkish clause letters distinct while normalizing instrument wording.
    normalized = "".join(
        char if char in "çğıöşü" else folded(char)
        for char in text.replace("**", "").replace("__", "").casefold()
    ).replace("ı", "i")
    identity_text = folded(normalized)
    found: dict[
        tuple[str, str | None, str | None, str | None, str | None], StatuteReference
    ] = {}
    for match in _STATUTE.finditer(identity_text):
        tail = _reference_tail(normalized[match.end() : match.end() + 180])
        candidates: list[tuple[int, str, str | None, str | None, str | None]] = [
            (offset, reference.article_no, None, None, reference.qualifier)
            for offset, reference in extract_regulatory_provision_reference_occurrences(
                tail
            )
        ]
        for pattern in (_COMPACT_ARTICLE, _SHORTHAND_ARTICLE):
            for subunit in pattern.finditer(tail):
                qualifier = subunit.groupdict().get("qualifier")
                candidates.append(
                    (
                        subunit.start(),
                        subunit["article"],
                        subunit["paragraph"],
                        subunit["clause"],
                        folded(qualifier) if qualifier else None,
                    )
                )
        article, paragraph, clause, qualifier = (
            min(candidates, key=lambda candidate: candidate[0])[1:]
            if candidates
            else (None, None, None, None)
        )
        number = match["number"] or match["english_number"]
        found.setdefault(
            (number, article, paragraph, clause, qualifier),
            StatuteReference(
                number,
                article,
                (normalized[match.start() : match.end()] + tail)[:240],
                paragraph,
                clause,
                qualifier,
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
            if reference.qualifier != metadata.get("article_qualifier"):
                continue
            if (
                reference.paragraph is not None
                and str(metadata.get("paragraph_no")) != reference.paragraph
            ):
                continue
            if (
                reference.clause is not None
                and str(metadata.get("clause_label")) != reference.clause
            ):
                continue
            matching.append(number)
        obligations.append(
            {
                "instrument_number": reference.number,
                "article": reference.article,
                "paragraph": reference.paragraph,
                "clause": reference.clause,
                "qualifier": reference.qualifier,
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
