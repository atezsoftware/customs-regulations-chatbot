"""Deterministic structural anchors for amendment target retrieval."""

import re
import unicodedata
from dataclasses import dataclass

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.heading_path import (
    extract_regulatory_provision_references,
)

_CLAUSE_REFERENCE_RE = re.compile(
    r"\((?P<label>[a-zçğıöşü])\)\s*bend",
    flags=re.IGNORECASE,
)
# A Turkish amendment instruction always opens with its own article number
# ("MADDE 10-"), then names the target ("11 inci maddesinin"). Extracting a
# reference from the whole sentence therefore sees two numbers and can neither
# choose nor safely refuse; the opening designator is dropped first so the
# remaining references describe the amended source only.
_INSTRUCTION_HEADER_RE = re.compile(
    r"^\s*(?:(?:geçici|gecici|mükerrer|mukerrer)\s+)?madde\s*\d+[a-zçğıöşü]*\s*"
    r"[-–—.)]\s*",
    flags=re.IGNORECASE,
)
_PARAGRAPH_ORDINALS = {
    "birinci": "1",
    "ikinci": "2",
    "üçüncü": "3",
    "ucuncu": "3",
    "dördüncü": "4",
    "dorduncu": "4",
    "beşinci": "5",
    "besinci": "5",
    "altıncı": "6",
    "altinci": "6",
    "yedinci": "7",
    "sekizinci": "8",
    "dokuzuncu": "9",
    "onuncu": "10",
}
_PARAGRAPH_ORDINAL_ALTERNATION: str = "|".join(
    sorted(_PARAGRAPH_ORDINALS, key=lambda name: (-len(name), name))
)
_PARAGRAPH_REFERENCE_RE = re.compile(
    rf"(?:(?P<ordinal>{_PARAGRAPH_ORDINAL_ALTERNATION})"
    r"|(?P<number>\d{1,3})\s*(?:inci|ıncı|incı|uncu|üncü|uncü|nci|ncı|ncu|ncü)?)"
    r"\s+f[ıi]kra",
    flags=re.IGNORECASE,
)
_APPENDIX_REFERENCE_RE = re.compile(
    r"(?<![\w])ek\s*[-–—:.]?\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
)
_QUALIFIED_ARTICLE_RE = re.compile(
    r"(?<!\w)(?P<kind>ek|geçici|gecici|mükerrer|mukerrer)\s+"
    r"(?:madde\s+(?P<forward>\d+[a-z]?)\b|"
    r"(?P<reverse>\d+[a-z]?)\s*(?:[.'’]?\s*"
    r"(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü))?\s+madd\w*)",
    re.IGNORECASE,
)
_EDIT_VERB_RE = re.compile(
    r"\b(?:değiştirilmiş|degistirilmis|eklenmiş|eklenmis|"
    r"ilave\s+edilmiş|ilave\s+edilmis|kaldırılmış|kaldirilmis|"
    r"çıkarılmış|cikarilmis)(?:tir|tır|tur|tür|ti)?\b",
    re.IGNORECASE,
)
_QUOTED_TEXT_RE = re.compile(r'"[^"]*"|“[^”]*”', re.DOTALL)
_QUOTED_APPENDIX_TARGET_RE = re.compile(
    r'[“"]((?:ek|annex|appendix)[\s:–—-]*(?:\d+[a-z]?|[ivxlcdm]+|[a-z])'
    r'(?:[/.-][0-9a-z]+)*)[”"](?=\s+(?:başlıklı|numaralı|sayılı|ek\w*))',
    re.IGNORECASE,
)
_ATTACHED_REPLACEMENT_RE = re.compile(
    r"ekteki\s+şekilde\s+değiştirilmiştir\s*[.!:]?",
    flags=re.IGNORECASE,
)
_SOURCE_GENERIC_TOKENS = frozenset(
    {
        "genel",
        "gumruk",
        "karar",
        "karari",
        "kanun",
        "kanunu",
        "no",
        "sayili",
        "seri",
        "tebligi",
        "teblig",
        "urun",
        "guvenligi",
        "denetimi",
        "ve",
        "yonetmeligi",
        "yonetmelik",
    }
)


@dataclass(frozen=True, slots=True)
class AmendmentStructuralTarget:
    article_no: str | None = None
    clause_label: str | None = None
    appendix_label: str | None = None
    paragraph_no: str | None = None


def normalize_appendix_label(value: str) -> str:
    match = _APPENDIX_REFERENCE_RE.search(value)
    if match is None:
        return "".join(
            character for character in value.casefold() if character.isalnum()
        )
    return f"ek{match.group('label').casefold()}"


def _source_identity_tokens(value: str) -> set[str]:
    decoded = re.sub(r"_?x[12]", " ", value, flags=re.IGNORECASE)
    folded = unicodedata.normalize(
        "NFKD",
        decoded.casefold().translate(
            str.maketrans({"ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u"})
        ),
    )
    ascii_value = "".join(
        character for character in folded if not unicodedata.combining(character)
    )
    return {
        str(int(token)) if token.isdigit() else token
        for token in re.findall(r"[a-z0-9]+", ascii_value)
    }


def source_identity_matches(target_source: str | None, source_name: str) -> bool:
    """Require explicit instrument-specific title tokens when they are available."""

    if not target_source:
        return True
    law_number = named_law_number(target_source)
    source_number = named_law_number(source_name)
    if law_number is not None:
        return law_number == source_number
    distinguishing_tokens = set(source_identity_distinguishing_tokens(target_source))
    if not distinguishing_tokens:
        return True
    return distinguishing_tokens <= _source_identity_tokens(source_name)


def named_law_number(source_name: str) -> str | None:
    """Identify a law title, not a law cited inside another document's name."""
    folded = unicodedata.normalize("NFKD", source_name.casefold().replace("ı", "i"))
    title = "".join(
        character for character in folded if not unicodedata.combining(character)
    ).replace("_", " ")
    if not re.search(r"\bkanun", title):
        return None
    match = re.search(r"\b(\d{3,5})\s+sayili\b", title)
    if match is not None:
        own_title = re.sub(
            r"\.(?:md|txt|pdf|docx|html?)$", "", title[match.end() :]
        ).strip()
        # A title may itself amend other laws. Its final instrument type, not
        # the word 'amendment', distinguishes it from a regulation citing a law.
        if not re.search(r"\bkanun(?:u)?(?:\s*\(.*\))?\s*$", own_title):
            return None
        if re.match(r"kanun(?:da|unda|a|una)\b", own_title):
            return None
    if match is None:
        match = re.search(r"\bkanun\s+(?:no|numarasi)\s*[:.]?\s*(\d{3,5})\b", title)
    return str(int(match.group(1))) if match else None


def source_identity_distinguishing_tokens(
    target_source: str | None,
) -> tuple[str, ...]:
    if not target_source:
        return ()
    return tuple(
        sorted(_source_identity_tokens(target_source) - _SOURCE_GENERIC_TOKENS)
    )


def amended_body(instruction_text: str) -> str:
    """Drop the instruction's own ``MADDE N-`` designator from its text."""

    return _INSTRUCTION_HEADER_RE.sub("", instruction_text, count=1)


def unquoted_text(text: str) -> str:
    """Remove quoted content while retaining surrounding article boundaries."""
    return _QUOTED_TEXT_RE.sub(lambda match: "\n" * match.group().count("\n"), text)


def amendment_operation_text(instruction_text: str) -> str:
    """Read the edit command without references inside its supplied new body."""
    body = amended_body(instruction_text)
    body = _QUOTED_APPENDIX_TARGET_RE.sub(r"\1", body)
    # Strip quoted phrases before locating the verb so a quoted edit is not
    # mistaken for the command being performed.
    unquoted = unquoted_text(body)
    command = _EDIT_VERB_RE.search(unquoted)
    return unquoted[: command.end()] if command is not None else unquoted


def article_identity(reference: str) -> str | None:
    """Preserve the normal/additional/temporary/repeated article namespace."""
    references = extract_regulatory_provision_references(reference)
    if not references:
        return None
    number = references[0].article_no
    qualified = _QUALIFIED_ARTICLE_RE.search(reference)
    if (
        qualified
        and (qualified.group("forward") or qualified.group("reverse")).upper() == number
    ):
        kind = qualified.group("kind").casefold().replace("\u0307", "")
        prefix = (
            "EK"
            if kind == "ek"
            else "GEÇİCİ"
            if kind in {"geçici", "gecici"}
            else "MÜKERRER"
        )
        return f"{prefix} {number}"
    return number


def _target_paragraph_no(instruction_body: str) -> str | None:
    """Return the single amended paragraph number, if the text names one."""

    numbers = {
        _PARAGRAPH_ORDINALS[
            match.group("ordinal").casefold().replace("\u0307", "").replace("ı", "i")
        ]
        if match.group("ordinal")
        else match.group("number").lstrip("0")
        for match in _PARAGRAPH_REFERENCE_RE.finditer(instruction_body)
    }
    numbers.discard("")
    if len(numbers) != 1:
        return None
    return next(iter(numbers))


def appendix_reference_text(text: str) -> str:
    """Remove additional-article references before detecting document annexes."""
    return _QUALIFIED_ARTICLE_RE.sub("", text)


def parse_amendment_structural_target(
    instruction: AmendmentInstruction,
) -> AmendmentStructuralTarget | None:
    """Resolve the amended provision, ignoring the instruction's own number.

    The first reference in the amended body is used when several remain: Turkish
    amendment drafting cites the target (or the neighbour a new unit follows)
    before quoting any replacement text, so a later number belongs to the new
    body rather than to the provision being changed.
    """

    instruction_body = amendment_operation_text(instruction.instruction_text)
    article_no = article_identity(instruction_body) or article_identity(
        instruction.article_reference or ""
    )
    if article_no is None and re.search(
        r"aşağıdaki\s+(?:(?:yeni|geçici|ek|mükerrer)\s+)*madde.*eklenmiş",
        instruction_body,
        re.IGNORECASE,
    ):
        heading = re.search(
            r'(?:^|\n)\s*["“]*(?:(?:GEÇİCİ|EK|MÜKERRER)\s+)?MADDE\s+\d+[a-z]?',
            amended_body(instruction.instruction_text),
            re.IGNORECASE,
        )
        if heading:
            article_no = article_identity(heading.group())
    clause_match = _CLAUSE_REFERENCE_RE.search(instruction_body)
    appendix_match = _APPENDIX_REFERENCE_RE.search(
        appendix_reference_text(instruction_body)
    )
    target = AmendmentStructuralTarget(
        article_no=article_no,
        clause_label=(clause_match.group("label").casefold() if clause_match else None),
        appendix_label=(
            f"EK-{appendix_match.group('label').upper()}" if appendix_match else None
        ),
        paragraph_no=_target_paragraph_no(instruction_body),
    )
    if target.article_no is None and target.appendix_label is None:
        return None
    return target


def canonical_structural_query_anchor(
    target: AmendmentStructuralTarget | None,
) -> str | None:
    """Render the target as the forward ``madde N`` form retrieval indexes.

    ``extract_legal_exact_fields`` — which feeds the exact provision boost — only
    recognizes the forward designator. An instruction spells its target as
    ``11 inci maddesinin``, so passing the raw sentence through boosts the
    instruction's own article instead. Rendering the resolved target keeps that
    boost pointed at the amended provision.
    """

    if target is None:
        return None
    if target.article_no is not None:
        anchor = f"madde {target.article_no}"
        if target.paragraph_no is not None:
            anchor = f"{anchor} ({target.paragraph_no}) fıkra"
        if target.clause_label is not None:
            anchor = f"{anchor} ({target.clause_label}) bent"
        return anchor
    if target.appendix_label is not None:
        return target.appendix_label
    return None


def _candidate_matches_appendix(candidate: CandidateChunk, appendix_label: str) -> bool:
    candidate_label = candidate.metadata.get("appendix_label")
    return isinstance(candidate_label, str) and (
        normalize_appendix_label(candidate_label)
        == normalize_appendix_label(appendix_label)
    )


def deterministic_structural_candidate(
    instruction: AmendmentInstruction, candidates: list[CandidateChunk]
) -> CandidateChunk | None:
    """Resolve a uniquely named canonical target without asking the matcher.

    Search wording is weak evidence for amendments because it usually describes
    the new text. Exact source, article and subunit metadata is stronger. Annex
    edits use any exact annex member only as a representative; drafting later
    loads and replaces the complete canonical annex scope atomically.
    """

    target = parse_amendment_structural_target(instruction)
    if target is None:
        return None
    exact = [
        candidate
        for candidate in candidates
        if bool(getattr(candidate, "structured_match", False))
        and source_identity_matches(
            instruction.target_source, str(getattr(candidate, "source_name", ""))
        )
    ]
    if target.appendix_label is not None:
        appendix = [
            candidate
            for candidate in exact
            if _candidate_matches_appendix(candidate, target.appendix_label)
        ]
        return appendix[0] if appendix else None

    def matches_named_unit(candidate: CandidateChunk) -> bool:
        metadata = candidate.metadata
        if (
            str(metadata.get("article_no") or "").casefold()
            != str(target.article_no or "").casefold()
        ):
            return False
        if (
            target.paragraph_no is not None
            and str(metadata.get("paragraph_no") or "") != target.paragraph_no
        ):
            return False
        if (
            target.clause_label is not None
            and str(metadata.get("clause_label") or "").casefold()
            != target.clause_label.casefold()
        ):
            return False
        return True

    named = [candidate for candidate in exact if matches_named_unit(candidate)]
    return named[0] if len(named) == 1 else None


def _has_inline_appendix_replacement_body(
    instruction_text: str, appendix_label: str
) -> bool:
    replacement_match = _ATTACHED_REPLACEMENT_RE.search(instruction_text)
    if replacement_match is None:
        return True
    remainder = instruction_text[replacement_match.end() :].strip()
    if not remainder:
        return False
    first_line = next(
        (line.strip() for line in remainder.splitlines() if line.strip()), ""
    )
    if normalize_appendix_label(first_line) != normalize_appendix_label(appendix_label):
        return False
    return len(re.findall(r"[^\W_]+", remainder, flags=re.UNICODE)) >= 4


def has_explicit_legacy_annex_edit(instruction_text: str) -> bool:
    """Recognize self-contained edits in checkpoints predating semantic routing.

    Unknown new-version wording stays in document comparison rather than being
    treated as permission to synthesize a replacement from the old annex.
    """
    if _ATTACHED_REPLACEMENT_RE.search(instruction_text):
        return False
    if re.search(r"yürürlükten\s+kaldırılmıştır", instruction_text, re.IGNORECASE):
        return True
    if re.search(
        r"(?:ibare\w*|gtip\w*|gtıp\w*).*değiştirilmiştir",
        instruction_text,
        re.IGNORECASE | re.DOTALL,
    ):
        return len(re.findall(r"[“\"][^”\"]+[”\"]", instruction_text)) >= 2
    supplied = re.search(
        r"aşağıdaki.*?(?:eklenmiştir|ilave\s+edilmiştir)[.!:]?\s*(.+)",
        instruction_text,
        re.IGNORECASE | re.DOTALL,
    )
    return supplied is not None and bool(re.search(r"\w", supplied.group(1)))


def appendix_replacement_attention_message(
    instruction: AmendmentInstruction,
    candidates: list[CandidateChunk],
) -> str | None:
    """Explain a safe refusal when an appendix exists but its new body does not."""

    target = parse_amendment_structural_target(instruction)
    if target is None or target.appendix_label is None:
        return None
    matching_candidates = [
        candidate
        for candidate in candidates
        if _candidate_matches_appendix(candidate, target.appendix_label)
    ]
    if not matching_candidates:
        return None
    chunk_word = "chunk" if len(matching_candidates) == 1 else "chunks"
    if _has_inline_appendix_replacement_body(
        instruction.instruction_text, target.appendix_label
    ):
        # The number of physical search chunks says nothing about the legal
        # scope of a row/cell edit. Multi-chunk drafting resolves and reviews
        # the complete scope atomically downstream.
        return None
    return (
        f"{instruction.instruction_text}\n\n"
        f"Target found: {target.appendix_label} "
        f"({len(matching_candidates)} {chunk_word}). No replacement appendix "
        "content was included after ‘ekteki şekilde değiştirilmiştir’, so no "
        "partial proposal was generated."
    )
