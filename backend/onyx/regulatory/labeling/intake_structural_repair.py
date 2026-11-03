"""Narrow legacy structural sources eligible for model reconsideration."""

import re
from collections.abc import Mapping
from typing import Any

from onyx.regulatory.document_intake.structural_readers import (
    is_ordinal_chapter_heading,
)

_FOLIO = re.compile(r"Yönetmelik Adı – [1-9][0-9]{0,3}")
_OPERATIVE_PREFIX = re.compile(
    r"^(?:(?:GEÇİCİ\s+|EK\s+)?MADDE\b|EK\s*[-–]?\s*\d|"
    r"\(\d+\)|\d+[.)]|[a-zçğıöşü]\))",
    re.I,
)
_LOCATORS = (
    "article_no",
    "paragraph_no",
    "clause_label",
    "subclause_label",
    "table_index",
    "table_row_index",
)


def legacy_structural_repair_candidate(
    *, chunk_id: str, text: str, source_snapshot: Mapping[str, Any]
) -> bool:
    """Eligibility only: the model must still return a strict non-abstained result.

    Callers must independently bind this entire snapshot to the current owned raw
    source. Inherited annex scope is retained even for an isolated page folio.
    """
    metadata = source_snapshot.get("chunk_metadata")
    position = source_snapshot.get("position")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("intake_chunking_version") not in ("2", "3")
        or metadata.get("chunk_variant") != "atomic"
        or source_snapshot.get("chunk_type") != "free_text"
        or type(position) is not int
        or position < 0
        or not text
        or text != text.strip()
        or any(character in text for character in "\n\r|")
        or any(metadata.get(field) is not None for field in _LOCATORS)
        or _OPERATIVE_PREFIX.match(text)
    ):
        return False
    if _FOLIO.fullmatch(text):
        return True
    headings = source_snapshot.get("heading_path")
    audit = metadata.get("intake_legal_metadata")
    if (
        not isinstance(headings, list)
        or len(headings) < 2
        or not all(isinstance(heading, str) for heading in headings)
        or not is_ordinal_chapter_heading(headings[-1])
        or metadata.get("heading_path") != headings
        or metadata.get("appendix_label") is not None
        or metadata.get("section_title") != text
        or not isinstance(audit, Mapping)
        or audit.get("input_stale") is not False
    ):
        return False
    fields = audit.get("fields")
    field = fields.get("section_title") if isinstance(fields, Mapping) else None
    return (
        isinstance(field, Mapping)
        and field.get("stale") is False
        and all(
            field.get(key) == value
            for key, value in {
                "source": "llm",
                "basis": "explicit",
                "evidence_scope": "body",
                "evidence_quote": text,
                "source_chunk_id": chunk_id,
            }.items()
        )
    )
