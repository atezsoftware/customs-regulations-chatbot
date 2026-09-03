"""Deterministic guards for amendment drafts produced by an LLM."""

import re
import unicodedata
from collections.abc import Sequence

from onyx.regulatory.amendments.models import AmendmentInstruction

_FULL_REPLACEMENT_RE = re.compile(
    r"aşağıdaki\s+şekilde\s+değiştirilmiştir\s*[.:]?",
    flags=re.IGNORECASE,
)
_OMISSION_RE = re.compile(r"\.{3,}|…")
_MARKDOWN_RE = re.compile(r"(?:\*\*|__|`|<[^>]+>)")


class DraftIntegrityError(ValueError):
    """The generated proposal cannot faithfully represent its instruction."""


def explicit_replacement_body(instruction_text: str) -> str | None:
    """Return the quoted body of an explicit full-provision replacement."""

    replacement = _FULL_REPLACEMENT_RE.search(instruction_text)
    if replacement is None:
        return None
    remainder = instruction_text[replacement.end() :].strip()
    for opening, closing in (("“", "”"), ('"', '"')):
        start = remainder.find(opening)
        end = remainder.rfind(closing)
        if start >= 0 and end > start:
            return remainder[start + len(opening) : end].strip()
    return None


def _comparison_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _MARKDOWN_RE.sub("", normalized)
    return " ".join(normalized.split()).casefold()


def validate_explicit_replacements(
    instructions: Sequence[AmendmentInstruction], draft_text: str
) -> None:
    """Require every authoritative replacement body to appear in the draft."""

    validate_explicit_replacement_texts(
        [instruction.instruction_text for instruction in instructions],
        draft_text,
    )


def validate_explicit_replacement_texts(
    instruction_texts: Sequence[str], draft_text: str
) -> None:
    """Validate stored instruction text again at the approval boundary."""

    normalized_draft = _comparison_text(draft_text)
    for instruction_text in instruction_texts:
        body = explicit_replacement_body(instruction_text)
        if body is None:
            continue
        if _comparison_text(body) not in normalized_draft:
            raise DraftIntegrityError(
                "The generated draft does not contain the explicit replacement "
                "body from the amendment instruction."
            )


def reject_unsupported_descendant_replacement(
    instructions: Sequence[AmendmentInstruction], *, has_active_descendants: bool
) -> None:
    """Refuse a one-row replacement when the target owns canonical children."""

    if not has_active_descendants:
        return
    reject_unsupported_descendant_replacement_texts(
        [instruction.instruction_text for instruction in instructions],
        has_active_descendants=has_active_descendants,
    )


def reject_unsupported_descendant_replacement_texts(
    instruction_texts: Sequence[str], *, has_active_descendants: bool
) -> None:
    """Apply the descendant guard to instructions persisted in the database."""

    if not has_active_descendants:
        return
    bodies = [
        body
        for instruction_text in instruction_texts
        if (body := explicit_replacement_body(instruction_text)) is not None
    ]
    if not bodies:
        return
    detail = (
        " and contains omission markers"
        if any(_OMISSION_RE.search(body) for body in bodies)
        else ""
    )
    raise DraftIntegrityError(
        "The replacement targets a provision with active descendant chunks"
        f"{detail}; an atomic multi-chunk replacement is required."
    )


def reconcile_existing_heading_path(
    heading_path: Sequence[str],
    *,
    amended_text: str,
    chunk_type: str | None,
    article_no: str | None = None,
    article_title: str | None = None,
    paragraph_no: str | None = None,
    clause_label: str | None = None,
    subclause_label: str | None = None,
) -> list[str]:
    """Update a mutable terminal unit label while preserving stable ancestors."""

    path = list(heading_path)
    plain_text = _MARKDOWN_RE.sub("", unicodedata.normalize("NFKC", amended_text))
    if chunk_type == "article" and article_no:
        qualifier_match = re.fullmatch(
            r"(?P<qualifier>GEÇİCİ|GECICI|MÜKERRER|MUKERRER)\s+(?P<number>.+)",
            article_no.strip(),
            flags=re.IGNORECASE,
        )
        if qualifier_match:
            canonical_qualifier = (
                "GEÇİCİ"
                if qualifier_match.group("qualifier").upper().startswith(("GEÇ", "GEC"))
                else "MÜKERRER"
            )
            marker = f"{canonical_qualifier} MADDE {qualifier_match.group('number')}"
        else:
            marker = f"MADDE {article_no.strip()}"
        terminal = f"{marker} - {article_title}" if article_title else marker
        if path:
            path[-1] = terminal
        else:
            path.append(terminal)
        return path

    marker: str | None = None
    if chunk_type == "paragraph" and paragraph_no:
        marker = rf"(?:\({re.escape(paragraph_no)}\)|{re.escape(paragraph_no)}[.)])"
    elif chunk_type in {"clause", "subclause"} and clause_label:
        marker = rf"(?:\({re.escape(clause_label)}\)|{re.escape(clause_label)}\))"
    if chunk_type == "subclause" and subclause_label:
        marker = rf"\({re.escape(subclause_label)}\)"
    if marker is None:
        return path

    unit = re.search(rf"(?<!\w)(?P<marker>{marker})\s+(?P<body>\S.*)", plain_text)
    if unit is None:
        return path
    body = re.split(r"[.;]\s+", unit.group("body"), maxsplit=1)[0].strip()
    body = " ".join(body.split()).strip(" -–—:.;")
    if len(body) > 90:
        body = f"{body[:87].rstrip()}..."
    terminal = f"{unit.group('marker')} {body}".strip()
    if path:
        path[-1] = terminal
    else:
        path.append(terminal)
    return path
