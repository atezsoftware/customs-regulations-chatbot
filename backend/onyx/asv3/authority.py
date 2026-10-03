"""Track explicit governing references separately from original legal evidence."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import model_evidence_metadata
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
    r"(?:\s*[-(]\s*\(?(?P<clause>[a-zçğıöşü])\)?)?\)?\s*madd"
)
_SHORTHAND_ARTICLE = re.compile(
    r"\b(?:(?P<qualifier>geçici|gecici|mükerrer|mukerrer|ek)\s+)?m\.?\s*(?P<article>\d{1,4})(?!\d)"
    r"(?:\s*/\s*(?P<paragraph>\d+)(?:\s*[-(]\s*(?P<clause>[a-zçğıöşü])\)?)?)?"
)
_INSTRUMENT_DESIGNATOR = re.compile(
    r"\b(?:kanun[a-z]*|yonetmeli[kg][a-z]*|teblig[a-z]*|genelge[a-z]*|karar[a-z]*"
    r"|sozlesme[a-z]*|anlasma[a-z]*|konvansiyon[a-z]*"
    r"|laws?|acts?|statutes?|regulations?|directives?|decrees?|decisions?|circulars?"
    r"|conventions?|treaties|treaty|agreements?)\b"
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
    clause_shorthand: bool = False


def statute_references(text: str) -> tuple[StatuteReference, ...]:
    # Keep Turkish clause letters distinct while normalizing instrument wording.
    original = text.replace("**", "").replace("__", "")
    parts: list[str] = []
    offsets: list[int] = []
    for index, char in enumerate(original):
        part = (
            char.casefold() if char.casefold() in "çğıöşü" else folded(char)
        ).replace("ı", "i")
        parts.append(part)
        offsets.extend([index] * len(part))
    offsets.append(len(original))
    normalized = "".join(parts)
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
        original_reference = original[
            offsets[match.start()] : offsets[match.end() + len(tail)]
        ]
        slash_letter = re.search(
            r"\b\d+\s*/\s*([a-zçğıöşüA-ZÇĞİÖŞÜ])\b", original_reference
        )
        found.setdefault(
            (number, article, paragraph, clause, qualifier),
            StatuteReference(
                number,
                article,
                original_reference[:240],
                paragraph,
                clause,
                qualifier,
                slash_letter is not None and slash_letter[1].islower(),
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
        instrument_originals: list[dict[str, JsonValue]] = []
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
            instrument_originals.append(row)
        # Slash letters can denote an inserted article or a clause shorthand.
        # An actual inserted article takes precedence; never collapse its identity.
        slash = re.fullmatch(r"(\d+)/([A-ZÇĞİÖŞÜ])", reference.article or "")
        if (
            reference.article is not None
            and reference.clause is None
            and reference.clause_shorthand
            and slash is not None
            and not any(
                row.get("article_no") == reference.article
                and row.get("article_qualifier") == reference.qualifier
                for row in instrument_originals
            )
        ):
            clause = slash[2].lower().replace("i̇", "i")
            if any(
                row.get("article_no") == slash[1]
                and row.get("article_qualifier") == reference.qualifier
                and row.get("clause_label") == clause
                for row in instrument_originals
            ):
                reference = replace(reference, article=slash[1], clause=clause)
        matching: list[int] = []
        for metadata in instrument_originals:
            number = metadata["citation"]
            assert isinstance(number, int)
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


_LAW_KINDS = {"kanun", "law", "act", "statute"}
_UNCITED_ORIGINAL_GAP = re.compile(
    r"(?P<subject>[^.!?;,]+?)\s+(?:ozgun (?:metni|metin|metnin|hukum|hukmu)|"
    r"original (?:text|provision))\s+(?:henuz\s+)?"
    r"(?:ulasilamadi|elde edilemedi|dogrulanamadi|incelenemedi|incelenmedi|"
    r"could not be (?:obtained|verified|examined)|has not been (?:obtained|verified|examined)|"
    r"(?:is|remains) unavailable)"
    r"\s*[.!?]?\s*$"
)
_LITERAL_QUOTE = re.compile(r'"([^"\n]+)"|“([^”\n]+)”')
_GAP_LOCATOR_WORDS = {
    "nun",
    "un",
    "nin",
    "in",
    "gecici",
    "mukerrer",
    "ek",
    "m",
    "md",
    "inci",
    "nci",
    "uncu",
    "ncu",
    "maddesi",
    "maddesinin",
    "maddenin",
    "article",
    "art",
    "section",
    "of",
    "the",
}


def _formal_law_name(value: str) -> str | None:
    value = re.sub(r"\s+", " ", folded(value)).strip()
    if re.search(r"[/\\_]|\.(?:md|docx?|pdf|txt)$", value):
        return None
    value = re.sub(r"^\d{2,7}\s+sayili\s+", "", value)
    match = re.fullmatch(r"(.+?)\s+kanun(?:u|un|unun)?", value)
    if match:
        return f"{match[1]} kanun"
    return value if re.fullmatch(r".+\s+(?:law|act|statute)", value) else None


def _reference_name(reference: StatuteReference) -> str | None:
    match = _STATUTE.search(folded(reference.reference_text))
    return _formal_law_name(match[0]) if match else None


def _native_original_rows(ledger: EvidenceLedger) -> list[dict[str, JsonValue]]:
    rows: list[dict[str, JsonValue]] = []
    for row in ledger.authority_metadata():
        number = row["citation"]
        if not isinstance(number, int):
            continue
        item = ledger.get(number)
        if item is None:
            continue
        metadata = model_evidence_metadata(item.metadata)
        doc = item.search_doc
        if (
            doc is None
            or not item.chunk_id
            or doc.document_id != item.source_id
            or doc.metadata.get("regulatory_chunk_id") != item.chunk_id
            or metadata.get("derived")
            or metadata.get("external")
        ):
            continue
        headings = row.get("heading_path")
        root = str(headings[0]) if isinstance(headings, list) and headings else ""
        title = str(row.get("title") or "")
        identities = [root, title]
        kind = folded(str(row.get("document_type") or ""))
        # Missing kind can be recovered from an exact official statute heading,
        # never from a filename or a cross-reference in the passage's body.
        if kind not in _LAW_KINDS and (
            kind or not any(_formal_law_name(value) for value in identities)
        ):
            continue
        numbers = {
            ref.number for value in identities for ref in statute_references(value)
        }
        if kind in _LAW_KINDS:
            numbers.update(
                value.strip() for value in identities if value.strip().isdigit()
            )
            if _formal_law_name(root):
                basename = title.replace("\\", "/").rsplit("/", 1)[-1]
                prefix = re.match(r"^(\d{2,7})[_-](?=[a-z])", folded(basename))
                if prefix:
                    numbers.add(prefix[1])
        rows.append(
            {
                **row,
                "instrument_numbers": sorted(numbers),
                "formal_names": sorted(
                    {name for value in identities if (name := _formal_law_name(value))}
                ),
            }
        )
    return rows


def _named_native_references(
    text: str, aliases: dict[str, set[str]]
) -> list[tuple[StatuteReference, str | None]]:
    references = [(ref, _reference_name(ref)) for ref in statute_references(text)]
    normalized = folded(text.replace("**", "").replace("__", ""))
    for name, numbers in aliases.items():
        if len(numbers) > 1:
            continue
        # The canonical stem tolerates Turkish case endings but not an acronym.
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?:u[a-z]*)?(?!\w)")
        for match in pattern.finditer(normalized):
            tail = _reference_tail(normalized[match.end() : match.end() + 180])
            locators = extract_regulatory_provision_reference_occurrences(tail)
            article = locators[0][1].article_no if locators else None
            qualifier = locators[0][1].qualifier if locators else None
            subunits = [
                part
                for pattern in (_COMPACT_ARTICLE, _SHORTHAND_ARTICLE)
                if (part := pattern.search(tail)) is not None
            ]
            if subunits:
                first = min(subunits, key=lambda part: part.start())
                article = first["article"]
                qualifier = first.groupdict().get("qualifier") or qualifier
            references.append(
                (
                    StatuteReference(
                        next(iter(numbers), ""),
                        article,
                        (match[0] + tail)[:240],
                        qualifier=qualifier,
                    ),
                    name,
                )
            )
    return list(dict.fromkeys(references))


def _without_verified_quotes(text: str, cited: set[int], ledger: EvidenceLedger) -> str:
    originals = [
        " ".join(item.text.split())
        for number in cited
        if (item := ledger.get(number)) is not None
    ]

    def replace_quote(match: re.Match[str]) -> str:
        quote = " ".join((match[1] or match[2]).split())
        return " " if any(quote in original for original in originals) else match[0]

    return _LITERAL_QUOTE.sub(replace_quote, text)


def _precise_original_gap(text: str, aliases: dict[str, set[str]]) -> bool:
    notice = folded(text.replace("**", "").replace("__", ""))
    notice = re.sub(r"(?<=\d)\.(?=\s*madd)|\b(?:m|md|no)\.", "", notice)
    statement = _UNCITED_ORIGINAL_GAP.fullmatch(notice)
    if statement is None:
        return False
    subject = statement["subject"]
    match = _STATUTE.match(subject)
    if match is None:
        match = next(
            (
                found
                for name in aliases
                if (found := re.match(rf"{re.escape(name)}(?:u[a-z]*)?(?!\w)", subject))
            ),
            None,
        )
    return match is not None and all(
        word in _GAP_LOCATOR_WORDS or len(word) == 1
        for word in re.findall(r"[a-z]+", subject[match.end() :])
    )


def native_named_authority_gap(
    answer: str, ledger: EvidenceLedger
) -> dict[str, JsonValue] | None:
    """Check local named-statute identity, not legal entailment or unnamed omissions."""
    rows = _native_original_rows(ledger)
    aliases: dict[str, set[str]] = {}
    for row in rows:
        for name in (
            row["formal_names"] if isinstance(row["formal_names"], list) else []
        ):
            if isinstance(name, str):
                known = row["instrument_numbers"]
                if isinstance(known, list):
                    aliases.setdefault(name, set()).update(
                        number for number in known if isinstance(number, str)
                    )
    # Lower references supply recognition aliases only, never matching originals.
    for number in ledger.citation_mapping():
        item = ledger.get(number)
        if item is None:
            continue
        for reference in statute_references(item.text):
            if name := _reference_name(reference):
                aliases.setdefault(name, set()).add(reference.number)
    missing: list[dict[str, JsonValue]] = []
    for unit in assertion_inventory(answer):
        if unit["presentation_only"]:
            continue
        cited = set(unit["evidence_numbers"])
        if not cited and _precise_original_gap(unit["text"], aliases):
            continue
        attributed = _without_verified_quotes(unit["text"], cited, ledger)
        for reference, name in _named_native_references(attributed, aliases):
            matching: list[int] = []
            for row in rows:
                numbers, names = row["instrument_numbers"], row["formal_names"]
                if not isinstance(numbers, list) or not isinstance(names, list):
                    continue
                if reference.number and numbers and reference.number not in numbers:
                    continue
                if not (
                    reference.number in numbers
                    or (name is not None and name in names and not numbers)
                ):
                    continue
                if reference.article is not None and (
                    str(row.get("article_no")) != reference.article
                    or row.get("article_qualifier") != reference.qualifier
                ):
                    continue
                citation = row["citation"]
                if isinstance(citation, int):
                    matching.append(citation)
            if not cited.intersection(matching):
                missing.append(
                    {
                        "unit_id": unit["unit_id"],
                        "instrument_number": reference.number or None,
                        "reference_text": reference.reference_text,
                        "article": reference.article,
                        "inline_evidence": sorted(cited),
                        "matching_original_evidence": matching,
                    }
                )
    if not missing:
        return None
    return {
        "named_authority_gaps": missing,
        "instruction": (
            "Each explicitly named governing statute needs its own matching canonical original "
            "and adjacent citation in this answer unit. Another instrument's quotation or a "
            "citation elsewhere cannot substitute. Reuse a supplied matching original or "
            "resolve/read its relevant provision with the useful method you choose. "
            "Preserve supported details; if its original cannot be obtained, disclose only "
            "that precise source gap without asserting the statute's result."
        ),
    }
