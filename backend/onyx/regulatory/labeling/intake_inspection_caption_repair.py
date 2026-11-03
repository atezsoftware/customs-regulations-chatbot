"""Prove an unchanged annex criterion caption for one strict reconsideration."""

import re
from collections.abc import Mapping, Sequence
from typing import Any

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.intake_dependent_clause_repair import (
    DependentClauseRepairSource,
)

_CAPTION = "B.MUAYENE ÜNİTESİ KRİTERLERİ"
_NEXT = "C.ALTYAPI VE PERSONEL KRİTERLERİ"
_HEADINGS = ["EK–1", "EK 4"]
_SNAPSHOT_FIELDS = (
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
    "paragraph_no",
    "clause_label",
    "subclause_label",
    "table_index",
    "table_row_index",
)


def legacy_inspection_caption_shape(text: str) -> bool:
    """Routing only; the full current criterion table must also be proved."""
    return text == _CAPTION


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


def _base(source: DependentClauseRepairSource) -> bool:
    metadata = _metadata(source)
    audit = metadata.get("intake_legal_metadata")
    return (
        metadata.get("chunk_variant") == "atomic"
        and metadata.get("intake_chunking_version") == "3"
        and metadata.get("heading_path") == source.snapshot.get("heading_path")
        and isinstance(audit, Mapping)
        and audit.get("input_stale") is False
    )


def _caption(source: DependentClauseRepairSource, title: str) -> bool:
    return (
        source.text == title
        and source.snapshot.get("chunk_type") == "free_text"
        and source.snapshot.get("heading_path") == _HEADINGS
        and _metadata(source).get("section_title") == title
        and _metadata(source).get("appendix_label") == "EK 4"
        and all(
            _metadata(source).get(field) is None and field not in _fields(source)
            for field in _NO_LOCATORS
        )
        and _fact(source, "section_title", title, "body", source)
        and _fact(source, "appendix_label", "EK 4", "heading", source)
    )


def legacy_inspection_caption_repair_candidate(
    *, chunk_id: str, sources: Sequence[DependentClauseRepairSource]
) -> bool:
    """Admit one bounded model attempt; continued abstention stays failed."""
    by_id = {source.chunk_id: source for source in sources}
    target = by_id.get(chunk_id)
    if target is None or len(sources) != 26 or len(by_id) != len(sources):
        return False
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not all(isinstance(member, str) for member in members)
        or len(set(members)) != len(members)
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
            or source.text != source.current_text
            or context_hash(source.text) != source.canonical_text_sha256
            or any(
                source.snapshot.get(key) != source.current_snapshot.get(key)
                for key in _SNAPSHOT_FIELDS
            )
        ):
            return False
        positions[position] = source
    if not _base(target) or not _caption(target, _CAPTION):
        return False
    position = target.snapshot["position"]
    witnesses = [positions.get(position + offset) for offset in range(-1, 9)]
    if any(source is None or not _base(source) for source in witnesses):
        return False
    proved = [source for source in witnesses if source is not None]
    previous, _target, *following = proved
    tables, boundary = following[:-1], following[-1]
    if (
        previous.snapshot.get("chunk_type") != "table"
        or previous.snapshot.get("heading_path") != _HEADINGS
        or _metadata(previous).get("table_index") != 3
        or _metadata(previous).get("appendix_label") != "EK 4"
        or not previous.text.startswith(
            "| 1 | ADR /Taşıt Uygunluk Belgesi İncelemeleri |"
        )
        or not _fact(previous, "table_index", previous.text, "body", previous)
        or not _fact(previous, "appendix_label", "EK 4", "heading", previous)
        or not _caption(boundary, _NEXT)
        or len(tables) != 7
    ):
        return False
    numbered_rows: list[int] = []
    for index, table in enumerate(tables):
        if (
            table.snapshot.get("chunk_type") != "table"
            or table.snapshot.get("heading_path") != _HEADINGS
            or _metadata(table).get("table_index") != (4 if index < 4 else 5)
            or _metadata(table).get("appendix_label") != "EK 4"
            or not _fact(table, "appendix_label", "EK 4", "heading", table)
        ):
            return False
        if index:
            if _metadata(table).get("section_title") != _CAPTION or not _fact(
                table, "section_title", _CAPTION, "body", target
            ):
                return False
        elif (
            "section_title" in _metadata(table)
            or "section_title" in _fields(table)
            or not table.text.startswith("| SIRA | KRİTER | KAPSAM |\n")
            or "| 1 | İşletme aşağıda sıralanan belgeleri olmak zorundadır."
            not in table.text
            or not _fact(table, "table_index", table.text, "body", table)
        ):
            return False
        numbered_rows.extend(
            int(match.group(1))
            for match in re.finditer(r"(?m)^\|[ \t]*(\d+)[ \t]*\|", table.text)
        )
    return numbered_rows == list(range(1, 21))
