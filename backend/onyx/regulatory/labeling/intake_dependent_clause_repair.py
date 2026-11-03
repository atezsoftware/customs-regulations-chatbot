"""Positive legacy handling-list evidence for one strict model reconsideration."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash

_HANDLING_TARGET = re.compile(r"\(a\) [^\n\r|]{1,450} (?:ya da|veya)")
_REQUIREMENT_TARGET = re.compile(r"\([a-d]\) [^\n\r|]{1,450}\.")
_SECTION = re.compile(r"^(\d+(?:\.\d+){1,4})\. ([^\n]+)$", re.M)
_BOUNDARY = re.compile(r"^(?:MADDE\b|GEÇİCİ MADDE\b|EK(?:\s|-)\w|\w+ BÖLÜM\b)", re.M)
_HANDLING_SECTION = (
    "Makine veya bir ilgili ürünün taşınmasını kolaylaştıracak şekilde tasarımı"
)
_HANDLING_INTRO = (
    "Makine veya bir ilgili ürünün veya bu ürünün aksam parçalarından birinin "
    "elle taşınacağı durumlarda, aşağıdakilerden biri söz konusu olur;"
)
_FOLLOWING = "(b) güvenli bir şekilde kaldırılıp taşınabilmesi için donatılır."
_REQUIREMENT_LISTS = (
    (
        "Zincirler, halatlar ve kayışlar",
        "Yukarıda belirtilen belge en azından aşağıda belirtilen bilgileri gösterir:",
        "Kaldırma aksesuarları",
    ),
    (
        "Kaldırma aksesuarları",
        "Her kaldırma aksesuarı veya ticari olarak bölünmez her kaldırma aksesuarı "
        "grubuna en azından aşağıdaki hususları belirleyen talimatlar eşlik eder:",
        "Kaldırma makineleri veya ilgili ürünler",
    ),
)
_SNAPSHOT_FIELDS = (
    "position",
    "heading_path",
    "chunk_type",
    "chunk_metadata",
    "validity_start_date",
    "validity_end_date",
)
_LOCATORS = (
    "article_no",
    "paragraph_no",
    "subclause_label",
    "table_index",
    "table_row_index",
)


@dataclass(frozen=True)
class DependentClauseRepairSource:
    chunk_id: str
    file_id: str
    text: str
    canonical_text_sha256: str
    snapshot: dict[str, Any]
    current_text: str
    current_snapshot: dict[str, Any]
    active: bool


def legacy_dependent_clause_shape(text: str) -> bool:
    """A routing hint; full current source/list evidence is required before use."""
    return (
        _HANDLING_TARGET.fullmatch(text) is not None
        or _REQUIREMENT_TARGET.fullmatch(text) is not None
    )


def _metadata(source: DependentClauseRepairSource) -> Mapping[str, Any]:
    value = source.snapshot.get("chunk_metadata")
    return value if isinstance(value, Mapping) else {}


def _heading_prefix(text: str) -> str:
    return text[:-3] if text.endswith("...") else text.removesuffix("…")


def _clause(
    source: DependentClauseRepairSource,
    label: str,
    parent: list[str],
    *,
    closing: bool = False,
    requirement: bool = False,
) -> bool:
    metadata = _metadata(source)
    headings = source.snapshot.get("heading_path")
    audit = metadata.get("intake_legal_metadata")
    fields = audit.get("fields") if isinstance(audit, Mapping) else None
    fact = fields.get("clause_label") if isinstance(fields, Mapping) else None
    return (
        source.snapshot.get("chunk_type") == "clause"
        and metadata.get("intake_chunking_version") in ("2", "3")
        and metadata.get("chunk_variant") == "atomic"
        and metadata.get("clause_label") == label
        and all(metadata.get(field) is None for field in _LOCATORS)
        and isinstance(headings, list)
        and len(headings) == len(parent) + 1
        and headings[:-1] == parent
        and all(isinstance(value, str) for value in headings)
        and metadata.get("heading_path") == headings
        and headings[-1].startswith(f"({label}) ")
        and source.text.startswith(_heading_prefix(headings[-1]))
        and metadata.get("appendix_label") == parent[-2]
        and isinstance(fields, Mapping)
        and (
            metadata.get("section_title") == parent[-1]
            if not (closing or requirement)
            else metadata.get("section_title") is None and "section_title" not in fields
        )
        and isinstance(audit, Mapping)
        and audit.get("input_stale") is False
        and isinstance(fact, Mapping)
        and all(
            fact.get(key) == value
            for key, value in {
                "source": "llm" if closing or requirement else "structural",
                "basis": "explicit" if closing or requirement else "chunker",
                "stale": False,
                "source_chunk_id": source.chunk_id,
            }.items()
        )
        and (
            isinstance(fact.get("evidence_quote"), str)
            and fact["evidence_quote"].startswith(_heading_prefix(headings[-1]))
            and source.text.startswith(fact["evidence_quote"])
            if requirement
            else fact.get("evidence_quote") == (_FOLLOWING if closing else None)
        )
        and (
            not (closing or requirement)
            or fact.get("evidence_scope") == "body"
            and fact.get("reference_chunk_id") is None
            and fact.get("reference_quote") is None
            and isinstance(fact.get("request_hash"), str)
            and re.fullmatch(r"[a-f0-9]{64}", fact["request_hash"]) is not None
        )
    )


def _requirement_list(
    target: DependentClauseRepairSource,
    positions: Mapping[int, DependentClauseRepairSource],
) -> bool:
    headings = target.snapshot["heading_path"]
    if len(headings) != 4 or not all(isinstance(value, str) for value in headings):
        return False
    parent = headings[:-1]
    if re.fullmatch(r"EK\s*-?\s*[1-9]\d{0,2}", parent[-2]) is None:
        return False
    found = 0
    for position in range(target.snapshot["position"] - 4, target.snapshot["position"]):
        intro = positions.get(position)
        siblings = [positions.get(position + offset) for offset in range(1, 5)]
        if intro is None or any(source is None for source in siblings):
            continue
        body_sections = list(_SECTION.finditer(intro.text))
        if (
            not body_sections
            or intro.snapshot["heading_path"][:-1] != parent
            or _metadata(intro).get("heading_path") != intro.snapshot["heading_path"]
            or _metadata(intro).get("appendix_label") != parent[-2]
            or any(_metadata(intro).get(field) is not None for field in _LOCATORS)
        ):
            continue
        section = body_sections[-1]
        if _BOUNDARY.search(intro.text[section.start() :]):
            continue
        for title, instruction, next_title in _REQUIREMENT_LISTS:
            if section[2] != title or not " ".join(intro.text.split()).endswith(
                instruction
            ):
                continue
            complete = [source for source in siblings if source is not None]
            if not all(
                _clause(source, label, parent, requirement=True)
                and _BOUNDARY.search(source.text) is None
                for label, source in zip("abcd", complete, strict=True)
            ):
                continue
            closing = _SECTION.search(complete[-1].text)
            if (
                any(_SECTION.search(source.text) for source in complete[:-1])
                or closing is None
                or closing[2] != next_title
            ):
                continue
            start, end = section[1].split("."), closing[1].split(".")
            if start[:-1] == end[:-1] and int(end[-1]) == int(start[-1]) + 1:
                found += 1
    return found == 1


def legacy_dependent_clause_repair_candidate(
    *, chunk_id: str, sources: Sequence[DependentClauseRepairSource]
) -> bool:
    """Prove a dependent alternatives/requirement list, never assign its labels.

    The preceding chunk contains an earlier list item followed by a new intro.
    Only that intro's tail governs this target; the next subsection closes it.
    Callers retain owned-original, full-context and strict model-result guards.
    """
    by_id = {source.chunk_id: source for source in sources}
    target = by_id.get(chunk_id)
    if (
        target is None
        or len(sources) != len(by_id)
        or not 1 <= len(sources) <= 40
        or not legacy_dependent_clause_shape(target.text)
    ):
        return False
    members = target.snapshot.get("context_member_ids")
    if (
        not isinstance(members, list)
        or not all(isinstance(value, str) for value in members)
        or len(members) != len(set(members))
        or set(members) != set(by_id)
    ):
        return False
    positions = {}
    for source in sources:
        position = source.snapshot.get("position")
        if (
            type(position) is not int
            or position in positions
            or not source.active
            or source.file_id != target.file_id
            or not isinstance(source.snapshot.get("heading_path"), list)
            or not isinstance(source.snapshot.get("chunk_metadata"), Mapping)
            or source.text != source.current_text
            or context_hash(source.current_text) != source.canonical_text_sha256
            or any(
                source.snapshot.get(field) != source.current_snapshot.get(field)
                for field in _SNAPSHOT_FIELDS
            )
        ):
            return False
        positions[position] = source
    if _REQUIREMENT_TARGET.fullmatch(target.text):
        return _requirement_list(target, positions)
    position = target.snapshot["position"]
    previous, following = positions.get(position - 1), positions.get(position + 1)
    headings = target.snapshot.get("heading_path")
    if (
        previous is None
        or following is None
        or not isinstance(headings, list)
        or len(headings) != 4
        or not all(isinstance(value, str) for value in headings)
    ):
        return False
    parent = headings[:-1]
    previous_label = _metadata(previous).get("clause_label")
    if (
        not re.fullmatch(r"EK\s*-?\s*[1-9]\d{0,2}", parent[-2])
        or not re.match(r"[1-9]\d{0,2}\. ", parent[-1])
        or not isinstance(previous_label, str)
        or not re.fullmatch(r"[a-zçğıöşü]", previous_label)
        or not _clause(target, "a", parent)
        or not _clause(previous, previous_label, parent)
        or not _clause(following, "b", parent, closing=True)
        or not following.text.startswith(_FOLLOWING)
    ):
        return False
    previous_lines = previous.text.splitlines()
    if (
        len(previous_lines) < 2
        or not previous_lines[0].endswith(".")
        or " ".join(" ".join(previous_lines[1:]).split()) != _HANDLING_INTRO
    ):
        return False
    sections = [
        (source.snapshot["position"], match.start(), match)
        for source in sources
        if source.snapshot["position"] <= position - 1
        and source.snapshot.get("heading_path", [])[: len(parent)] == parent
        for match in _SECTION.finditer(source.text)
    ]
    if not sections:
        return False
    _position, _offset, section = max(sections, key=lambda row: (row[0], row[1]))
    closing = _SECTION.search(following.text)
    if section[2] != _HANDLING_SECTION or closing is None:
        return False
    for offset in range(_position, position + 2):
        member = positions.get(offset)
        if (
            member is None
            or member.snapshot.get("heading_path", [])[:-1] != parent
            or _metadata(member).get("heading_path")
            != member.snapshot.get("heading_path")
            or any(_metadata(member).get(field) is not None for field in _LOCATORS)
            or _BOUNDARY.search(member.text)
        ):
            return False
    start, end = section[1].split("."), closing[1].split(".")
    return (
        start[0] == parent[-1].split(".", 1)[0]
        and start[:-1] == end[:-1]
        and int(end[-1]) == int(start[-1]) + 1
    )
