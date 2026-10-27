"""Choose new-document display names from reviewed, source-backed metadata."""

from collections.abc import Mapping, Sequence
from typing import Any

from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY, _fold

UNKNOWN_DOCUMENT_TITLE = "Başlık belirlenemedi"


def _normalized(value: str) -> str:
    return _fold(" ".join(value.split()))


def intake_document_name(
    metadata: Sequence[Mapping[str, Any]], *, source_text: str
) -> str | None:
    """Manual titles win; conflicting inferred titles never pick an arbitrary file name."""
    manual_fields: list[list[str]] = []
    inferred_fields: list[list[str]] = []
    manually_reviewed = False
    for field in ("document_title", "annex_title"):
        manual: list[str] = []
        inferred: list[str] = []
        for row in metadata:
            audit = row.get(METADATA_AUDIT_KEY)
            if not isinstance(audit, dict):
                continue
            fields = audit.get("fields")
            entry = fields.get(field) if isinstance(fields, dict) else None
            if not isinstance(entry, dict):
                continue
            source = entry.get("source")
            value = row.get(field)
            if source == "manual":
                manually_reviewed = True
                destination = manual
            elif (
                source == "llm"
                and not entry.get("stale")
                and isinstance(value, str)
                and _normalized(value) in _normalized(source_text)
            ):
                destination = inferred
            else:
                continue
            if isinstance(value, str) and value.strip():
                destination.append(" ".join(value.split()))
        manual_fields.append(manual)
        inferred_fields.append(inferred)
    for candidates in manual_fields if manually_reviewed else inferred_fields:
        if not candidates:
            continue
        # A shorter exact heading may accompany its complete multi-line title.
        longest = max(candidates, key=lambda value: len(value))
        if all(_normalized(value) in _normalized(longest) for value in candidates):
            return longest
        return None
    return None
