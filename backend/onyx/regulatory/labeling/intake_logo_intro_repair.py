"""Prove one legacy logo-rule introduction for strict model reconsideration."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.intake_dependent_clause_repair import (
    DependentClauseRepairSource,
)

_INTRO = "1. Logo Tasarımı\nBu kapsamda;"
_SECTION = "1. Logo Tasarımı"
_TITLE = "Ortak Şarj Cihazı Logosu"
_NEXT_TITLE = "Ölçümler ve Hesaplamalar"
_PARENT = ["Ek I", "EK iii", _SECTION]
_SNAPSHOT = (
    "position",
    "heading_path",
    "chunk_type",
    "chunk_metadata",
    "validity_start_date",
    "validity_end_date",
)
_NO_LOCATORS = (
    "article_no",
    "article_title",
    "subclause_label",
    "table_index",
    "table_row_index",
)
_CHILD_PREFIXES = (
    "a) Logo, etikete iliştirildiğinde ",
    "b) Logonun referans renkleri ",
    "c) Logoda kullanılacak yazı tipi ",
    'ç) "XX", ',
    "d) Logo koyu renkli bir zemin üzerinde kullanılacaksa, ",
    "e) Ürünün etiketi, muhafazası, ambalajı veya kullanım kılavuzu ",
)


def legacy_logo_intro_shape(text: str) -> bool:
    """Routing only; the complete current owned list must also be proved."""
    return text == _INTRO


def _metadata(source: DependentClauseRepairSource) -> Mapping[str, Any]:
    value = source.snapshot.get("chunk_metadata")
    return value if isinstance(value, Mapping) else {}


def _fields(source: DependentClauseRepairSource) -> Mapping[str, Any]:
    audit = _metadata(source).get("intake_legal_metadata")
    fields = audit.get("fields") if isinstance(audit, Mapping) else None
    return fields if isinstance(fields, Mapping) else {}


def _fact(
    source: DependentClauseRepairSource,
    field: str,
    quote: str,
    scope: str,
    witness: DependentClauseRepairSource,
) -> bool:
    fact = _fields(source).get(field)
    return (
        isinstance(fact, Mapping)
        and all(
            fact.get(key) == value
            for key, value in {
                "source": "llm",
                "basis": "explicit",
                "stale": False,
                "evidence_quote": quote,
                "evidence_scope": scope,
                "source_chunk_id": witness.chunk_id,
                "reference_quote": None,
                "reference_chunk_id": None,
            }.items()
        )
        and isinstance(fact.get("request_hash"), str)
        and re.fullmatch(r"[a-f0-9]{64}", fact["request_hash"]) is not None
        and (
            quote in witness.text
            if scope == "body"
            else quote in witness.snapshot.get("heading_path", [])
        )
    )


def _absent(source: DependentClauseRepairSource, *fields: str) -> bool:
    return all(
        _metadata(source).get(field) is None and field not in _fields(source)
        for field in fields
    )


def _base(source: DependentClauseRepairSource, version: str) -> bool:
    meta = _metadata(source)
    audit = meta.get("intake_legal_metadata")
    return (
        meta.get("chunk_variant") == "atomic"
        and meta.get("intake_chunking_version") == version
        and meta.get("heading_path") == source.snapshot.get("heading_path")
        and isinstance(audit, Mapping)
        and audit.get("input_stale") is False
        and _absent(source, *_NO_LOCATORS)
    )


def _title(source: DependentClauseRepairSource, title: str, appendix: str) -> bool:
    return (
        source.text == title
        and source.snapshot.get("chunk_type") == "free_text"
        and source.snapshot.get("heading_path") == ["Ek I", appendix]
        and _metadata(source).get("annex_title") == title
        and _metadata(source).get("appendix_label") == appendix
        and _absent(source, "paragraph_no", "clause_label", "section_title")
        and _fact(source, "annex_title", title, "body", source)
        and _fact(source, "appendix_label", appendix, "heading", source)
    )


def _child(
    source: DependentClauseRepairSource,
    label: str,
    prefix: str,
    target: DependentClauseRepairSource,
    title: DependentClauseRepairSource,
) -> bool:
    meta = _metadata(source)
    headings = source.snapshot.get("heading_path")
    fact = _fields(source).get("clause_label")
    if (
        source.snapshot.get("chunk_type") != "clause"
        or meta.get("clause_label") != label
        or meta.get("paragraph_no") != "1"
        or meta.get("appendix_label") != "EK iii"
        or not isinstance(headings, list)
        or len(headings) != 4
        or headings[:-1] != _PARENT
        or not isinstance(headings[-1], str)
        or not headings[-1].startswith(f"{label}) ")
        or not source.text.startswith(
            headings[-1].removesuffix("...").removesuffix("…")
        )
        or not source.text.startswith(prefix)
        or not 1 <= len(source.text) <= 1500
        or re.search(r"[\n\r|]", source.text)
        or not isinstance(fact, Mapping)
        or not isinstance(fact.get("evidence_quote"), str)
    ):
        return False
    scope = "heading" if label in ("d", "e") else "body"
    quote = fact["evidence_quote"]
    if not quote.startswith(f"{label}) ") or not _fact(
        source, "clause_label", quote, scope, source
    ):
        return False
    if label == "a":
        return (
            _absent(source, "annex_title", "section_title")
            and _fact(source, "paragraph_no", _SECTION, "heading", source)
            and _fact(source, "appendix_label", "EK iii", "heading", source)
        )
    return (
        meta.get("section_title") == _SECTION
        and meta.get("annex_title") == _TITLE
        and _fact(source, "paragraph_no", _SECTION, "body", target)
        and _fact(source, "section_title", _SECTION, "body", target)
        and _fact(source, "annex_title", _TITLE, "body", title)
        and _fact(source, "appendix_label", "EK iii", "heading", title)
    )


def legacy_logo_intro_repair_candidate(
    *, chunk_id: str, sources: Sequence[DependentClauseRepairSource]
) -> bool:
    """Admit one unchanged-context reconsideration, never a source-only result."""
    by_id = {source.chunk_id: source for source in sources}
    target = by_id.get(chunk_id)
    if target is None or len(sources) != 27 or len(by_id) != len(sources):
        return False
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not all(isinstance(member, str) for member in members)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
    ):
        return False
    positions: dict[int, DependentClauseRepairSource] = {}
    for source in sources:
        position = source.snapshot.get("position")
        if (
            type(position) is not int
            or position < 0
            or position in positions
            or not source.active
            or source.file_id != target.file_id
            or _metadata(source).get("chunk_variant") != "atomic"
            or _metadata(source).get("intake_chunking_version") not in ("2", "3")
            or source.text != source.current_text
            or context_hash(source.text) != source.canonical_text_sha256
            or any(
                source.snapshot.get(k) != source.current_snapshot.get(k)
                for k in _SNAPSHOT
            )
        ):
            return False
        positions[position] = source
    meta = _metadata(target)
    version = meta.get("intake_chunking_version")
    if (
        not legacy_logo_intro_shape(target.text)
        or version not in ("2", "3")
        or target.snapshot.get("chunk_type") != "paragraph"
        or target.snapshot.get("heading_path") != _PARENT
        or meta.get("paragraph_no") != "1"
        or meta.get("section_title") != _SECTION
        or meta.get("appendix_label") != "EK iii"
        or not _absent(target, "clause_label", "annex_title")
        or not _fact(target, "paragraph_no", _SECTION, "body", target)
        or not _fact(target, "section_title", _SECTION, "body", target)
        or not _fact(target, "appendix_label", "EK iii", "heading", target)
    ):
        return False
    position = target.snapshot["position"]
    required = [positions.get(position + offset) for offset in range(-1, 8)]
    if any(source is None for source in required):
        return False
    witnesses = [source for source in required if source is not None]
    title, _intro, *tail = witnesses
    return (
        all(_base(source, version) for source in witnesses)
        and _title(title, _TITLE, "EK iii")
        and _title(tail[-1], _NEXT_TITLE, "EK iv")
        and all(
            _child(source, label, prefix, target, title)
            for source, label, prefix in zip(
                tail[:-1], "abcçde", _CHILD_PREFIXES, strict=True
            )
        )
    )
