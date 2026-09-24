"""Evidence boundaries for a complete, ordered source fragment sequence."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from onyx.regulatory.provision_identity import article_identity

ARTICLE_OPENING = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*{1,2})?\s*"
    r"(?:(?:ek|geçici|gecici|mükerrer|mukerrer)\s+)?madde\s+"
    r"\d+[a-z]?(?=\s*[-–—:.])",
    re.IGNORECASE | re.MULTILINE,
)
_QUOTED = re.compile(r'"[^"]*"|“[^”]*”', re.DOTALL)
_ANNEX_OPENING = re.compile(
    r"^[ \t]*(?:#{1,6}\s*)?(?:\*{1,2})?\s*"
    r"(?:ekler\b|ek\s*[-–—:/]?\s*(?:\d+[a-z]?|[ivxlcdm]+)\b)",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class SourceFragment:
    source_id: str
    text: str


def article_scope_indices(
    rows: Sequence[SourceFragment],
    article_no: str,
    *,
    max_chunks: int = 64,
    stop_at_annex: bool = False,
) -> list[int]:
    found: list[int] = []
    anchor: SourceFragment | None = None
    occurrences = 0
    for index, row in enumerate(rows):
        if re.search(
            r"kanuna\s+[iİı]şlenemeyen\s+hükümler", row.text[:200], re.IGNORECASE
        ):
            break
        if anchor is not None and row.source_id != anchor.source_id:
            anchor = None
        unquoted = _QUOTED.sub(lambda match: "\n" * match.group().count("\n"), row.text)
        annex = _ANNEX_OPENING.search(unquoted) if stop_at_annex else None
        if annex is not None:
            if not unquoted[: annex.start()].strip():
                break
            if anchor is not None or any(
                article_identity(item.group()) == article_no
                for item in ARTICLE_OPENING.finditer(unquoted[: annex.start()])
            ):
                return []
        headings = list(ARTICLE_OPENING.finditer(unquoted))
        if len(headings) > 1 or (headings and headings[0].start() > 0):
            if anchor is not None or any(
                article_identity(item.group()) == article_no for item in headings
            ):
                return []
        heading = ARTICLE_OPENING.match(row.text)
        if heading:
            anchor = row if article_identity(heading.group()) == article_no else None
            if anchor is not None:
                occurrences += 1
        if anchor is not None:
            found.append(index)
    return found if occurrences == 1 and len(found) <= max_chunks else []
