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


_ENGLISH_STATUTE = r"\b(?:law|act|statute)\s+(?:no\.?\s*)?(?P<english_number>\d{2,7})\b"
_STATUTE = re.compile(
    r"(?<!\d)(?P<number>\d{2,7})\s+sayili\s+"
    r"[^\n.;:]{0,100}?\bkanun[a-z]*\b"
    r"|" + _ENGLISH_STATUTE
)
_STRICT_STATUTE = re.compile(
    r"(?<![\d/])(?P<number>\d{2,7})\s+sayili\s+"
    # A decision number cannot consume a subsequent numbered statute identity.
    r"(?:(?!\b\d{2,7}\s+sayili\b)[^\n.;:]){0,100}?\bkanun[a-z]*\b"
    r"|" + _ENGLISH_STATUTE
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
_OTHER_INSTRUMENT_END = (
    r"(?:yonetmeli[kg](?:i|in|inin)?|teblig(?:i|in|inin)?|genelge(?:si|nin)?"
    r"|karar(?:i|in|inin)?|sozlesme(?:si|nin)?|anlasma(?:si|nin)?|konvansiyon(?:u|un|unun)?"
    r"|regulation|directive|decree|decision|circular|convention|treaty|agreement)"
)
_NUMBERED_OTHER_INSTRUMENT = re.compile(
    r"(?<![\d/])\d{2,7}\s+sayili\s+"
    r"(?:(?!\b\d{2,7}\s+sayili\b)[^\n.;:]){0,100}?\b"
    + _OTHER_INSTRUMENT_END
    + r"\b|\b(?:regulation|directive|decree|decision|circular|convention|treaty|agreement)"
    r"\s+(?:no\.?\s*)?\d{2,7}\b"
)
_BOUND_NUMBER = (
    r"(?P<article>\d{1,4}[a-zçğıöşü]?)"
    r"(?:(?:\s*/\s*(?P<paragraph>\d+)"
    r"(?:\s*[-(]\s*\(?(?P<clause>[a-zçğıöşü])\)?)?)"
    r"|(?:\s*/\s*(?P<inserted>[a-zçğıöşü])))?"
    r"(?![a-z0-9/]|\.\d)"
)
_BOUND_QUALIFIER = r"(?:(?P<qualifier>gecici|geçici|mukerrer|mükerrer|ek)\s+)?"
_BOUND_FORWARD = re.compile(
    _BOUND_QUALIFIER
    + r"(?:madde|article|section|art|md|m)\b\.?\s*:?\s*"
    + _BOUND_NUMBER
)
_BOUND_REVERSE = re.compile(
    _BOUND_QUALIFIER
    + _BOUND_NUMBER
    + r"\s*(?:\.\s*|['’]?\s*(?:inci|nci|uncu|ıncı)\s+|\s+)"
    + r"madd(?:e(?:de|den|nin|ye|yi)?|es[iı](?:nde|nden|nin|ne|ni)?|eleri(?:nin|ne)?)\b"
)
_BOUND_COMPACT = re.compile(
    r"\(?" + _BOUND_QUALIFIER + _BOUND_NUMBER + r"\)?\s*madd[a-z]*\b"
)
_BOUND_LIST_NUMBER = re.compile(_BOUND_QUALIFIER + _BOUND_NUMBER)
_BOUND_PREFIX = re.compile(
    r"\s*(?:['’]\s*(?:nun|nin|un|in|nın|inin|unun)\b\s*)?"
    r"(?:[,:(|]\s*)?"
)
_BOUND_SEPARATOR = re.compile(r"\s*(?:,|\bve\b|\bile\b|\band\b)\s*")
_ENUMERATION_END = re.compile(r"\s*(?:maddeler[a-z]*|articles?|sections?)\b")


def _explicit_bare_enumeration(text: str, start: int) -> bool:
    match = _BOUND_LIST_NUMBER.match(text, start)
    while match is not None:
        if _ENUMERATION_END.match(text, match.end()):
            return True
        separator = _BOUND_SEPARATOR.match(text, match.end())
        if separator is None:
            return False
        match = _BOUND_LIST_NUMBER.match(text, separator.end())
    return False


@dataclass(frozen=True)
class _BoundLocator:
    start: int
    end: int
    article: str
    paragraph: str | None
    clause: str | None
    qualifier: str | None


def _bound_locator(
    text: str, start: int, *, bare: bool = False
) -> _BoundLocator | None:
    patterns = (
        (_BOUND_FORWARD, _BOUND_REVERSE, _BOUND_COMPACT, _BOUND_LIST_NUMBER)
        if bare
        else (_BOUND_FORWARD, _BOUND_REVERSE, _BOUND_COMPACT)
    )
    matches = [
        match
        for pattern in patterns
        if (match := pattern.match(text, start))
        and (
            pattern is not _BOUND_LIST_NUMBER or _explicit_bare_enumeration(text, start)
        )
    ]
    if not matches:
        return None
    match = max(matches, key=lambda value: value.end())
    if bare and re.match(r"\s*(?:sayili|tarihli|yil[a-z]*)\b", text[match.end() :]):
        return None
    article = match["article"].upper()
    if inserted := match["inserted"]:
        article += "/" + inserted.upper()
    qualifier = match["qualifier"]
    return _BoundLocator(
        start,
        match.end(),
        article,
        match["paragraph"],
        match["clause"],
        folded(qualifier) if qualifier else None,
    )


def _explicit_reverse_instrument(text: str, start: int) -> bool:
    tail = folded(text[start:])
    if link := re.match(r"\s+of\s+(?:the\s+)?", tail):
        target = tail[link.end() :]
        if _STRICT_STATUTE.match(target) or _NUMBERED_OTHER_INSTRUMENT.match(target):
            return True
        formal = re.match(
            r"(.+?\b(?:kanun[a-z]*|law|act|statute|" + _OTHER_INSTRUMENT_END + r")\b)",
            target,
        )
        return formal is not None and _explicit_instrument_name(formal[1])
    if parenthesized := re.match(r"\s*\(([^)\n]+)\)", tail):
        identity = parenthesized[1].strip()
        return (
            _STRICT_STATUTE.fullmatch(identity) is not None
            or _NUMBERED_OTHER_INSTRUMENT.fullmatch(identity) is not None
            or _explicit_instrument_name(identity)
        )
    return False


def _explicit_instrument_name(value: str) -> bool:
    if _formal_law_name(value) is not None:
        return True
    if re.search(r"[/\\_]|\.(?:md|docx?|pdf|txt)$", value):
        return False
    return re.fullmatch(r".+\s+" + _OTHER_INSTRUMENT_END, value) is not None


def _bound_locators(text: str, start: int, end: int) -> list[_BoundLocator]:
    """Bind locators through explicit syntax, never through a later narrative clause."""
    found: list[_BoundLocator] = []
    tail = text[end:]
    prefix = _BOUND_PREFIX.match(tail)
    assert prefix is not None
    cursor = prefix.end()
    locator = _bound_locator(tail, cursor)
    while locator is not None:
        if _explicit_reverse_instrument(tail, locator.end):
            break
        found.append(replace(locator, start=end + locator.start, end=end + locator.end))
        separator = _BOUND_SEPARATOR.match(tail, locator.end)
        if separator is None:
            break
        if _STRICT_STATUTE.match(folded(tail), separator.end()):
            break
        locator = _bound_locator(tail, separator.end(), bare=True)
    # Reverse links require an explicit possessive/prepositional relationship.
    before = text[:start]
    boundary = max(before.rfind("\n"), before.rfind(";"), before.rfind("|")) + 1
    for pattern in (_BOUND_FORWARD, _BOUND_REVERSE, _BOUND_COMPACT):
        for match in pattern.finditer(before, boundary):
            connector = before[match.end() :]
            if not re.fullmatch(r"\s+(?:of\s+(?:the\s+)?)|\s*\(\s*", connector):
                continue
            if "(" in connector and not re.match(r"\s*\)", text[end:]):
                continue
            locator = _bound_locator(before, match.start())
            if locator is not None:
                found.append(locator)
    return list(dict.fromkeys(found))


def _reference_bounds(
    text: str, start: int, end: int, locator: _BoundLocator | None
) -> tuple[int, int]:
    if locator is not None:
        start, end = min(start, locator.start), max(end, locator.end)
        if closing := re.match(r"\s*\)", text[end:]):
            end += closing.end()
    return start, end


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


def _lowercase_slash_clause(text: str, article: str | None) -> bool:
    slash = re.fullmatch(r"(\d+)/([A-ZÇĞİÖŞÜ])", article or "")
    if slash is None:
        return False
    return any(
        match[1].islower()
        and match[1].casefold().replace("ı", "i")
        == slash[2].casefold().replace("i̇", "i").replace("ı", "i")
        for match in re.finditer(
            rf"\b{re.escape(slash[1])}\s*/\s*([a-zçğıöşüA-ZÇĞİÖŞÜ])\b", text
        )
    )


def statute_references(
    text: str,
    *,
    strict_reference_boundaries: bool = False,
    syntactic_reference_binding: bool = False,
) -> tuple[StatuteReference, ...]:
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
    found: dict[tuple[str | bool | None, ...], StatuteReference] = {}
    pattern = _STRICT_STATUTE if strict_reference_boundaries else _STATUTE
    for match in pattern.finditer(identity_text):
        if syntactic_reference_binding:
            bound = _bound_locators(normalized, match.start(), match.end())
            references: list[_BoundLocator | None] = []
            references.extend(bound)
            if not references:
                references.append(None)
            for locator in references:
                number = match["number"] or match["english_number"]
                reference_start, reference_end = _reference_bounds(
                    normalized, match.start(), match.end(), locator
                )
                original_reference = original[
                    offsets[reference_start] : offsets[reference_end]
                ]
                article = locator.article if locator else None
                paragraph = locator.paragraph if locator else None
                clause = locator.clause if locator else None
                qualifier = locator.qualifier if locator else None
                clause_shorthand = _lowercase_slash_clause(
                    original[offsets[locator.start] : offsets[locator.end]]
                    if locator
                    else "",
                    article,
                )
                found.setdefault(
                    (number, article, paragraph, clause, qualifier, clause_shorthand),
                    StatuteReference(
                        number,
                        article,
                        original_reference[:240],
                        paragraph,
                        clause,
                        qualifier,
                        clause_shorthand,
                    ),
                )
            continue
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
    text: str,
    aliases: dict[str, set[str]],
    *,
    strict_reference_boundaries: bool = False,
    syntactic_reference_binding: bool = False,
) -> list[tuple[StatuteReference, str | None]]:
    references = [
        (ref, _reference_name(ref))
        for ref in statute_references(
            text,
            strict_reference_boundaries=strict_reference_boundaries,
            syntactic_reference_binding=syntactic_reference_binding,
        )
    ]
    raw_text = text.replace("**", "").replace("__", "")
    normalized = folded(raw_text)
    offsets: list[int] = []
    if syntactic_reference_binding:
        parts: list[str] = []
        for index, char in enumerate(raw_text):
            part = (
                char.casefold() if char.casefold() in "çğıöşü" else folded(char)
            ).replace("ı", "i")
            parts.append(part)
            offsets.extend([index] * len(part))
        offsets.append(len(raw_text))
        normalized = "".join(parts)
    for name, numbers in aliases.items():
        if len(numbers) > 1:
            continue
        # The canonical stem tolerates Turkish case endings but not an acronym.
        pattern = re.compile(rf"(?<!\w){re.escape(name)}(?:u[a-z]*)?(?!\w)")
        for match in pattern.finditer(
            folded(normalized) if syntactic_reference_binding else normalized
        ):
            if syntactic_reference_binding:
                bound = _bound_locators(normalized, match.start(), match.end())
                for locator in list(bound) or [None]:
                    reference_start, reference_end = _reference_bounds(
                        normalized, match.start(), match.end(), locator
                    )
                    raw_reference = raw_text[
                        offsets[reference_start] : offsets[reference_end]
                    ]
                    references.append(
                        (
                            StatuteReference(
                                next(iter(numbers), ""),
                                locator.article if locator else None,
                                raw_reference[:240],
                                locator.paragraph if locator else None,
                                locator.clause if locator else None,
                                locator.qualifier if locator else None,
                                _lowercase_slash_clause(
                                    raw_text[
                                        offsets[locator.start] : offsets[locator.end]
                                    ]
                                    if locator
                                    else "",
                                    locator.article if locator else None,
                                ),
                            ),
                            name,
                        )
                    )
                continue
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


def _canonical_law_aliases(
    rows: list[dict[str, JsonValue]],
) -> dict[str, set[str]]:
    aliases: dict[str, set[str]] = {}
    for row in rows:
        names, numbers = row["formal_names"], row["instrument_numbers"]
        if isinstance(names, list) and isinstance(numbers, list):
            for name in names:
                if isinstance(name, str):
                    aliases.setdefault(name, set()).update(
                        number for number in numbers if isinstance(number, str)
                    )
    return aliases


def _authority_aliases(
    ledger: EvidenceLedger,
    rows: list[dict[str, JsonValue]],
    *,
    strict_reference_boundaries: bool,
    syntactic_reference_binding: bool,
) -> dict[str, set[str]]:
    aliases = _canonical_law_aliases(rows)
    for number in ledger.citation_mapping():
        item = ledger.get(number)
        if item is None:
            continue
        for reference in statute_references(
            item.text,
            strict_reference_boundaries=strict_reference_boundaries,
            syntactic_reference_binding=syntactic_reference_binding,
        ):
            if name := _reference_name(reference):
                aliases.setdefault(name, set()).add(reference.number)
    return aliases


def _canonical_source(item: object) -> bool:
    from onyx.asv3.models import EvidenceItem

    if not isinstance(item, EvidenceItem):
        return False
    metadata = model_evidence_metadata(item.metadata)
    return bool(
        item.search_doc is not None
        and item.chunk_id
        and item.search_doc.document_id == item.source_id
        and item.search_doc.metadata.get("regulatory_chunk_id") == item.chunk_id
        and not metadata.get("derived")
        and not metadata.get("external")
    )


def _normalized_native_reference(
    reference: StatuteReference,
    name: str | None,
    rows: list[dict[str, JsonValue]],
) -> StatuteReference:
    """Disambiguate lowercase clause shorthand using canonical own-instrument locators."""
    slash = re.fullmatch(r"(\d+)/([A-ZÇĞİÖŞÜ])", reference.article or "")
    if not reference.clause_shorthand or reference.clause is not None or slash is None:
        return reference
    own_rows = []
    for row in rows:
        numbers, names = row["instrument_numbers"], row["formal_names"]
        if not isinstance(numbers, list) or not isinstance(names, list):
            continue
        if reference.number and numbers and reference.number not in numbers:
            continue
        if (
            reference.number in numbers
            or (name is not None and name in names and not numbers)
        ) and row.get("article_qualifier") == reference.qualifier:
            own_rows.append(row)
    if any(row.get("article_no") == reference.article for row in own_rows):
        return reference
    clause = slash[2].lower().replace("i̇", "i")
    if any(
        row.get("article_no") == slash[1]
        and _canonical_clause_label(row.get("clause_label")) == clause
        for row in own_rows
    ):
        return replace(reference, article=slash[1], clause=clause)
    return reference


def _canonical_clause_label(value: JsonValue) -> str | None:
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\(?([a-zçğıöşü])\)?", value.strip().lower())
    return match[1] if match else None


def _matching_native_rows(
    reference: StatuteReference,
    name: str | None,
    rows: list[dict[str, JsonValue]],
    *,
    clause_specific: bool = False,
) -> list[int]:
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
        if (
            clause_specific
            and reference.clause_shorthand
            and reference.clause is not None
            and _canonical_clause_label(row.get("clause_label")) != reference.clause
        ):
            continue
        if type(citation := row["citation"]) is int:
            matching.append(citation)
    return matching


def explicit_reference_leads(
    ledger: EvidenceLedger,
    complete_originals: list[dict[str, JsonValue]],
    *,
    syntactic_reference_binding: bool = False,
) -> list[dict[str, JsonValue]]:
    """Project literal source references as navigation, never mandatory legal needs."""
    rows = _native_original_rows(ledger)
    aliases = _canonical_law_aliases(rows)
    complete: set[int] = set()
    for record in complete_originals:
        number = record.get("citation")
        if type(number) is not int:
            continue
        item = ledger.get(number)
        if (
            item is not None
            and _canonical_source(item)
            and record.get("source_id") == item.source_id
            and record.get("chunk_id") == item.chunk_id
            and record.get("text_hash") == item.text_hash
            and record.get("text") == item.text
            and record.get("start_char", 0) == 0
            and record.get("end_char", len(item.text)) == len(item.text)
        ):
            complete.add(number)
    leads: dict[tuple[str, str | None, str, str | None], dict[str, JsonValue]] = {}
    for number in sorted(complete):
        item = ledger.get(number)
        assert item is not None
        for reference, name in _named_native_references(
            item.text,
            aliases,
            strict_reference_boundaries=True,
            syntactic_reference_binding=syntactic_reference_binding,
        ):
            if reference.article is None:
                continue
            if syntactic_reference_binding:
                reference = _normalized_native_reference(reference, name, rows)
            assert reference.article is not None
            matching = _matching_native_rows(
                reference, name, rows, clause_specific=syntactic_reference_binding
            )
            if complete.intersection(matching):
                continue
            key = (
                reference.number,
                name if not reference.number else None,
                reference.article,
                reference.qualifier,
            )
            lead = leads.setdefault(
                key,
                {
                    "instrument_number": reference.number or None,
                    "formal_name": name,
                    "article": reference.article,
                    "qualifier": reference.qualifier,
                    "origin_citations": [],
                    "matching_original_evidence": matching,
                },
            )
            origins = lead["origin_citations"]
            assert isinstance(origins, list)
            if number not in origins:
                origins.append(number)
    return list(leads.values())


_DEFINED_ABBREVIATION = re.compile(r"\(\s*([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ0-9]{1,11})\s*\)")


def _defined_statute_abbreviations(
    answer: str,
    aliases: dict[str, set[str]],
    *,
    syntactic_reference_binding: bool = False,
) -> dict[str, tuple[str, str | None] | None]:
    defined: dict[str, set[tuple[str, str | None]]] = {}
    for match in _DEFINED_ABBREVIATION.finditer(answer):
        prefix = answer[max(0, match.start() - 180) : match.start()]
        tail = folded(prefix.replace("**", "").replace("__", "")).strip()
        identities = {
            (reference.number, name)
            for reference, name in _named_native_references(
                prefix,
                aliases,
                strict_reference_boundaries=True,
                syntactic_reference_binding=syntactic_reference_binding,
            )
            if (
                (numbered := _STRICT_STATUTE.search(folded(reference.reference_text)))
                is not None
                and tail.endswith(numbered[0])
            )
            or (
                name is not None
                and re.search(rf"(?<!\w){re.escape(name)}(?:u[a-z]*)?$", tail)
            )
        }
        if identities:
            defined.setdefault(folded(match[1]), set()).update(identities)
    return {
        label: next(iter(identities)) if len(identities) == 1 else None
        for label, identities in defined.items()
    }


def _abbreviated_statute_references(
    text: str,
    defined: dict[str, tuple[str, str | None] | None],
    *,
    syntactic_reference_binding: bool = False,
) -> list[tuple[StatuteReference, str | None]]:
    references: list[tuple[StatuteReference, str | None]] = []
    for label, identity in defined.items():
        if identity is None:
            continue
        remainder = re.sub(r"\[\d+\]", "", folded(text))
        remainder = re.sub(rf"(?<!\w){re.escape(label)}(?!\w)", "", remainder)
        if all(
            word in _GAP_LOCATOR_WORDS or len(word) == 1
            for word in re.findall(r"[a-z]+", remainder)
        ):
            continue
        number, name = identity
        for reference, _ in _named_native_references(
            text,
            {label: {number}},
            strict_reference_boundaries=True,
            syntactic_reference_binding=syntactic_reference_binding,
        ):
            if reference.article is not None:
                references.append((reference, name))
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


_GAP_RESEARCH_CONTEXT = (
    r"(?:(?:bu|mevcut|sinirli)\s+)*(?:(?:usul|hukuki)\s+)?"
    r"(?:incelemede|incelemesinde|arastirmada|arastirmasinda)"
    r"|in (?:this|the present|this limited) (?:review|research|procedural review)"
)
_GAP_ORIGINAL_STATEMENT = re.compile(
    r"(?P<subject>.+?)\s+(?:ozgun (?:metni|metnin|metinleri|metin|hukmu|hukumleri|hukum)|"
    r"original (?:texts?|provisions?))\s+(?:henuz\s+)?"
    r"(?:(?:" + _GAP_RESEARCH_CONTEXT + r")\s+)?"
    r"(?:ulasilamadi|elde edilemedi|dogrulanamadi|incelenemedi|incelenmedi|"
    r"degerlendirilmedi|ele alinmadi|incelenmemistir|degerlendirilmemistir|"
    r"ele alinmamistir|could not be (?:obtained|verified|examined)|"
    r"(?:has|have) not been (?:obtained|verified|examined|reviewed)|"
    r"(?:is|are|remains|remain) unavailable)"
)
_GAP_SCOPE_STATEMENT = re.compile(
    r"(?:(?:bu yanit|bu inceleme),?\s+)?(?P<subject>.+?)\s+"
    r"(?:sonuc(?:u|unu|lari|larini)(?: hakkinda)?|hakkinda)(?: da)?\s+"
    r"(?:bir )?(?:belirleme yapmiyor|belirleme yapilmiyor|belirlemiyor|"
    r"sonuc bildirmemektedir|sonuc bildirmiyor)"
    r"|this (?:answer|review) (?:does not determine|makes no determination about)\s+"
    r"(?P<english_subject>.+)"
)
# Scope subjects are nominal topics, not arbitrary prose or inferred legal claims.
_GAP_SCOPE_NOMINAL = re.compile(
    r"(?:sonuc|etki|ceza|faiz|vergi|oran|odeme|beyan|izin|istisna|itiraz|hak|usul|"
    r"belge|teminat|sure|sorumluluk|yukumluluk|yetki|iade|kiymet|gecikme|"
    r"yapilmasi|yapilmamasi|verilmesi|verilmemesi|alinmasi|alinmamasi|"
    r"tamamlanmasi|tamamlanmamasi)"
    r"(?:nin|nun|in|un|i|u|si|su|sinin|sunun|sinde|sinda|na|ne|ya|ye|"
    r"lar|ler|lari|leri|larin|lerin|larinin|lerinin)?"
)
_GAP_SCOPE_WORDS = _GAP_LOCATOR_WORDS | {
    "bir",
    "bu",
    "herhangi",
    "eksik",
    "gec",
    "tamamlayici",
    "hukuki",
    "kapsaminda",
    "geri",
    "verme",
    "kaldirma",
    "ve",
    "veya",
    "ile",
    "yahut",
    "and",
    "or",
    "a",
    "an",
    "payment",
    "interest",
    "penalty",
    "liability",
    "refund",
    "permission",
    "procedure",
    "objection",
    "appeal",
    "right",
    "scope",
    "effect",
    "applicability",
}
_GAP_SUBJECT_SUFFIXES = _GAP_LOCATOR_WORDS | {
    "hukumleri",
    "hukumlerinin",
    "maddeleri",
    "maddelerinin",
    "ve",
    "ile",
    "and",
    "s",
}


def _gap_reference_identity(
    reference: StatuteReference, name: str | None
) -> tuple[str, str | None, str | None, str | None]:
    return (
        reference.number,
        None if reference.number else name,
        reference.article,
        reference.qualifier,
    )


_GAP_NUMBER_ONLY_IDENTITY = re.compile(
    r"\d{2,7}\s+sayili\s+kanun(?:u|un|unun)?"
    r"|(?:law|act|statute)\s+(?:no\s*)?\d{2,7}"
)


def _gap_masking_aliases(
    rows: list[dict[str, JsonValue]],
    declared: dict[str, tuple[str, str | None] | None],
) -> dict[str, set[str]]:
    """Only own canonical identities authorize consuming formal title words."""
    aliases = _canonical_law_aliases(rows)
    for label, identity in declared.items():
        if identity is None:
            continue
        number, name = identity
        if (
            name is not None
            and name in aliases
            and (aliases[name] == {number} or (not number and not aliases[name]))
        ):
            aliases[label] = {number} if number else set()
    return aliases


def _gap_identity_remainder(subject: str, masking_aliases: dict[str, set[str]]) -> str:
    spans: list[tuple[int, int]] = []
    rejected: list[tuple[int, int]] = []
    for match in _STRICT_STATUTE.finditer(subject):
        number = match["number"] or match["english_number"]
        name = _reference_name(StatuteReference(number, None, match[0]))
        if _GAP_NUMBER_ONLY_IDENTITY.fullmatch(match[0]) or (
            name is not None and masking_aliases.get(name) == {number}
        ):
            spans.append(match.span())
        else:
            rejected.append(match.span())
    alias_matches = sorted(
        (
            (_formal_law_name(name) is None, match)
            for name, numbers in masking_aliases.items()
            if len(numbers) <= 1
            for match in re.finditer(
                rf"(?<!\w){re.escape(name)}(?:u|un|unun)?(?!\w)", subject
            )
        ),
        key=lambda item: (item[0], item[1].start(), -item[1].end()),
    )
    for is_label, match in alias_matches:
        if any(
            left <= match.start() and match.end() <= right for left, right in rejected
        ):
            continue
        if is_label:
            prefix = list(subject[: match.start()])
            for start, end in spans:
                for left, right in [(start, end)] + [
                    (locator.start, locator.end)
                    for locator in _bound_locators(subject, start, end)
                ]:
                    if right <= match.start():
                        prefix[left:right] = " " * (right - left)
            remainder = "".join(prefix)
            if (
                not _bound_locators(subject, match.start(), match.end())
                or re.search(r"\d|[^a-z\s,'’()/:-]", remainder)
                or any(
                    word not in _GAP_SUBJECT_SUFFIXES
                    for word in re.findall(r"[a-z]+", remainder)
                )
            ):
                continue
        spans.append(match.span())
    retained: list[tuple[int, int]] = []
    for start, end in sorted(set(spans), key=lambda span: (span[0], -span[1])):
        if any(left <= start and end <= right for left, right in retained):
            continue
        retained.append((start, end))
    masked = list(subject)
    for start, end in retained:
        for left, right in [(start, end)] + [
            (locator.start, locator.end)
            for locator in _bound_locators(subject, start, end)
        ]:
            masked[left:right] = " " * (right - left)
    return "".join(masked)


def _gap_subject_references(
    subject: str,
    aliases: dict[str, set[str]],
    masking_aliases: dict[str, set[str]],
) -> list[tuple[StatuteReference, str | None]] | None:
    """Consume only explicit identities, their bound locators and possessive links."""
    references = _gap_references(subject, aliases)
    if not references:
        return None
    remainder = _gap_identity_remainder(subject, masking_aliases)
    if re.search(r"\d|[^a-z\s,'’()/:-]", remainder) or any(
        word not in _GAP_SUBJECT_SUFFIXES for word in re.findall(r"[a-z]+", remainder)
    ):
        return None
    return references


def _gap_scope_references(
    subject: str,
    aliases: dict[str, set[str]],
    masking_aliases: dict[str, set[str]],
) -> list[tuple[StatuteReference, str | None]] | None:
    remainder = _gap_identity_remainder(subject, masking_aliases)
    if re.search(r"\d|[^a-z\s,'’()/:-]", remainder) or any(
        word not in _GAP_SCOPE_WORDS and _GAP_SCOPE_NOMINAL.fullmatch(word) is None
        for word in re.findall(r"[a-z]+", remainder)
    ):
        return None
    return _gap_references(subject, aliases)


def _gap_references(
    text: str, aliases: dict[str, set[str]]
) -> list[tuple[StatuteReference, str | None]]:
    references = _named_native_references(
        text,
        aliases,
        strict_reference_boundaries=True,
        syntactic_reference_binding=True,
    )
    unique: dict[
        tuple[str, str | None, str | None, str | None],
        tuple[StatuteReference, str | None],
    ] = {}
    for reference, name in references:
        unique.setdefault(
            _gap_reference_identity(reference, name),
            (reference, name),
        )
    return list(unique.values())


def _original_gap_references(
    text: str,
    aliases: dict[str, set[str]],
    *,
    masking_aliases: dict[str, set[str]],
) -> list[tuple[StatuteReference, str | None]] | None:
    """Recognize a fully negative disclosure, never a mixed legal answer block."""
    notice = folded(text.replace("**", "").replace("__", ""))
    notice = re.sub(r"(?<=\d)\.(?=\s*madd)", "", notice)
    notice = re.sub(r"\b(m|md|art|no)\.", r"\1 ", notice)
    if re.search(r"\[\d+\]", notice):
        return None
    clauses = [part.strip() for part in re.split(r"[.!?;]", notice) if part.strip()]
    disclosed: list[tuple[StatuteReference, str | None]] = []
    pending: list[tuple[StatuteReference, str | None]] | None = None
    for clause in clauses:
        if scope := _GAP_SCOPE_STATEMENT.fullmatch(clause):
            if pending is not None:
                return None
            pending = _gap_scope_references(
                scope["subject"] or scope["english_subject"], aliases, masking_aliases
            )
            if pending is None:
                return None
            continue
        clause = re.sub(r"^(?:" + _GAP_RESEARCH_CONTEXT + r")\s+", "", clause)
        original = _GAP_ORIGINAL_STATEMENT.fullmatch(clause)
        if original is None:
            return None
        subject = original["subject"]
        singular = re.fullmatch(
            r"bu (?:hukmun|maddenin)|this provision(?:'s)?", subject
        )
        plural = re.fullmatch(r"bu hukumlerin|these provisions'?", subject)
        if singular or plural:
            if pending is None or (
                len(pending) != 1 or pending[0][0].article is None
                if singular
                else len(pending) < 2
            ):
                return None
            references = pending
        else:
            references = _gap_subject_references(subject, aliases, masking_aliases)
            if references is None:
                return None
            if pending and not {
                _gap_reference_identity(ref, name) for ref, name in pending
            }.issubset(_gap_reference_identity(ref, name) for ref, name in references):
                return None
        disclosed.extend(references)
        pending = None
    return list(dict.fromkeys(disclosed)) if disclosed and pending is None else None


def native_named_authority_gap(
    answer: str,
    ledger: EvidenceLedger,
    *,
    strict_reference_boundaries: bool = False,
    resolve_defined_abbreviations: bool = False,
    syntactic_reference_binding: bool = False,
) -> dict[str, JsonValue] | None:
    """Check local named-statute identity, not legal entailment or unnamed omissions."""
    rows = _native_original_rows(ledger)
    # Lower references supply recognition aliases only, never matching originals.
    aliases = _authority_aliases(
        ledger,
        rows,
        strict_reference_boundaries=strict_reference_boundaries,
        syntactic_reference_binding=syntactic_reference_binding,
    )
    units = assertion_inventory(answer)
    defined = (
        _defined_statute_abbreviations(
            "\n".join(
                _without_verified_quotes(
                    unit["text"], set(unit["evidence_numbers"]), ledger
                )
                for unit in units
            ),
            aliases,
            syntactic_reference_binding=syntactic_reference_binding,
        )
        if resolve_defined_abbreviations
        else {}
    )
    gap_aliases = {
        **aliases,
        **{
            label: {identity[0]}
            for label, identity in defined.items()
            if identity is not None
        },
    }
    masking_aliases = (
        _gap_masking_aliases(rows, defined) if syntactic_reference_binding else {}
    )
    missing: list[dict[str, JsonValue]] = []
    for unit in units:
        if unit["presentation_only"]:
            continue
        cited = set(unit["evidence_numbers"])
        if not cited and (
            _original_gap_references(
                unit["text"], gap_aliases, masking_aliases=masking_aliases
            )
            is not None
            if syntactic_reference_binding
            else _precise_original_gap(unit["text"], gap_aliases)
        ):
            continue
        attributed = _without_verified_quotes(unit["text"], cited, ledger)
        references = _named_native_references(
            attributed,
            aliases,
            strict_reference_boundaries=strict_reference_boundaries,
            syntactic_reference_binding=syntactic_reference_binding,
        )
        if resolve_defined_abbreviations:
            references.extend(
                _abbreviated_statute_references(
                    attributed,
                    defined,
                    syntactic_reference_binding=syntactic_reference_binding,
                )
            )
        seen: set[tuple[str | None, ...]] = set()
        for reference, name in references:
            if syntactic_reference_binding:
                reference = _normalized_native_reference(reference, name, rows)
            identity = (
                reference.number,
                name if not reference.number else None,
                reference.article,
                reference.qualifier,
                reference.clause
                if syntactic_reference_binding and reference.clause_shorthand
                else None,
            )
            if identity in seen:
                continue
            seen.add(identity)
            matching = _matching_native_rows(
                reference, name, rows, clause_specific=syntactic_reference_binding
            )
            if not cited.intersection(matching):
                missing.append(
                    {
                        "unit_id": unit["unit_id"],
                        "instrument_number": reference.number or None,
                        "reference_text": reference.reference_text,
                        "article": reference.article,
                        **(
                            {"clause": reference.clause}
                            if syntactic_reference_binding
                            and reference.clause_shorthand
                            and reference.clause
                            else {}
                        ),
                        "inline_evidence": sorted(cited),
                        "matching_original_evidence": matching,
                    }
                )
    if not missing:
        return None
    return {
        "named_authority_gaps": missing,
        "citation_format": "[n]",
        "instruction": (
            "Each explicitly named governing statute needs its own matching canonical original "
            "and adjacent recorded global [n] citation in this answer unit. Parentheses (n) "
            "are not citation markers; preserve actual legal article/paragraph numbering. "
            "inline_evidence lists only citation markers parsed in the current unit. "
            "matching_original_evidence lists identity/navigation candidates, not approval "
            "of the claim or confirmation of full delivery; choose the actual supporting "
            "original. Another instrument's quotation or a "
            "citation elsewhere cannot substitute. Reuse a supplied matching original or "
            "resolve/read its relevant provision with the useful method you choose. "
            "Preserve supported details; if its original cannot be obtained, disclose only "
            "that precise source gap without asserting the statute's result."
        ),
    }
