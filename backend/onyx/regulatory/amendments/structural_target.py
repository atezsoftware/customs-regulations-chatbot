"""Deterministic structural anchors for amendment target retrieval."""

import re
import unicodedata
from dataclasses import dataclass

from onyx.regulatory.amendments.models import AmendmentInstruction
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.heading_path import (
    extract_single_regulatory_provision_reference,
)

_CLAUSE_REFERENCE_RE = re.compile(
    r"\((?P<label>[a-zçğıöşü])\)\s*bend",
    flags=re.IGNORECASE,
)
_APPENDIX_REFERENCE_RE = re.compile(
    r"(?<![\w])ek\s*[-–—:.]?\s*(?P<label>\d+[a-z]?)\b",
    flags=re.IGNORECASE,
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
        "yonetmeligi",
        "yonetmelik",
    }
)


@dataclass(frozen=True, slots=True)
class AmendmentStructuralTarget:
    article_no: str | None = None
    clause_label: str | None = None
    appendix_label: str | None = None


def normalize_appendix_label(value: str) -> str:
    match = _APPENDIX_REFERENCE_RE.search(value)
    if match is None:
        return "".join(
            character for character in value.casefold() if character.isalnum()
        )
    return f"ek{match.group('label').casefold()}"


def _source_identity_tokens(value: str) -> set[str]:
    decoded = re.sub(r"_?x[12]", " ", value, flags=re.IGNORECASE)
    folded = unicodedata.normalize("NFKD", decoded.casefold())
    ascii_value = "".join(
        character for character in folded if not unicodedata.combining(character)
    )
    return set(re.findall(r"[a-z0-9]+", ascii_value))


def source_identity_matches(target_source: str | None, source_name: str) -> bool:
    """Require explicit instrument-specific title tokens when they are available."""

    if not target_source:
        return True
    distinguishing_tokens = set(source_identity_distinguishing_tokens(target_source))
    if not distinguishing_tokens:
        return True
    return distinguishing_tokens <= _source_identity_tokens(source_name)


def source_identity_distinguishing_tokens(
    target_source: str | None,
) -> tuple[str, ...]:
    if not target_source:
        return ()
    return tuple(
        sorted(_source_identity_tokens(target_source) - _SOURCE_GENERIC_TOKENS)
    )


def parse_amendment_structural_target(
    instruction: AmendmentInstruction,
) -> AmendmentStructuralTarget | None:
    combined_reference = "\n".join(
        value
        for value in (instruction.article_reference, instruction.instruction_text)
        if value
    )
    article_reference = extract_single_regulatory_provision_reference(
        combined_reference
    )
    clause_match = _CLAUSE_REFERENCE_RE.search(instruction.instruction_text)
    appendix_match = _APPENDIX_REFERENCE_RE.search(instruction.instruction_text)
    target = AmendmentStructuralTarget(
        article_no=(article_reference.article_no if article_reference else None),
        clause_label=(clause_match.group("label").casefold() if clause_match else None),
        appendix_label=(
            f"EK-{appendix_match.group('label').upper()}" if appendix_match else None
        ),
    )
    if target.article_no is None and target.appendix_label is None:
        return None
    return target


def _candidate_matches_appendix(candidate: CandidateChunk, appendix_label: str) -> bool:
    candidate_label = candidate.metadata.get("appendix_label")
    return isinstance(candidate_label, str) and (
        normalize_appendix_label(candidate_label)
        == normalize_appendix_label(appendix_label)
    )


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
        if len(matching_candidates) == 1:
            return None
        return (
            f"{instruction.instruction_text}\n\n"
            f"Target found: {target.appendix_label} "
            f"({len(matching_candidates)} {chunk_word}). This appendix spans "
            "multiple canonical chunks, so an atomic multi-chunk replacement is "
            "required; no partial proposal was generated."
        )
    return (
        f"{instruction.instruction_text}\n\n"
        f"Target found: {target.appendix_label} "
        f"({len(matching_candidates)} {chunk_word}). No replacement appendix "
        "content was included after ‘ekteki şekilde değiştirilmiştir’, so no "
        "partial proposal was generated."
    )
