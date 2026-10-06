"""Recognize an explicit single-instrument provision acquisition target."""

import re

from pydantic import JsonValue

from onyx.regulatory.heading_path import (
    extract_regulatory_provision_reference_occurrences,
)

_INSTRUMENT = re.compile(
    r"\b(?:kanun(?:u|un)?|yönetmeli(?:k|ği)|tebli(?:ğ|ği)|genelge(?:si)?|"
    r"law|act|statute|regulation|directive|circular)\b",
    re.IGNORECASE,
)


def focused_source_target(args: dict[str, JsonValue]) -> tuple[str, str] | None:
    target = args.get("evidence_target")
    if args.get("discover_related_sources") is True or not isinstance(target, str):
        return None
    references = extract_regulatory_provision_reference_occurrences(target)
    instruments = list(_INSTRUMENT.finditer(target))
    if len(references) != 1 or len(instruments) != 1:
        return None
    start, reference = references[0]
    source_name = target[:start].strip(" \t\n,:;()'’\"")
    instrument = instruments[0]
    if instrument.end() > start or source_name != target[: instrument.end()].strip():
        return None
    if not re.search(r"\w+\s+", source_name):
        return None
    if re.match(
        r"^(?:bu|şu|o|ilgili|söz\s*konusu|this|the|applicable|relevant)\s+",
        source_name,
        re.IGNORECASE,
    ):
        return None
    article = " ".join(
        value for value in (reference.qualifier, reference.article_no) if value
    )
    return source_name, article
