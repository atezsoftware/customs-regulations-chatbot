"""Recognize preliminary judicial sections without interpreting a legal holding."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from onyx.asv3.models import EvidenceItem, model_evidence_metadata

JudicialSectionRole = Literal["preliminary", "argument_only", "unknown"]
JudicialTextSection = Literal[
    "preliminary", "argument_only", "reasoning", "disposition", "unknown"
]


def _normalized_heading(text: str) -> str:
    plain = re.sub(r"[*_#]", "", text).strip().casefold().replace("ı", "i")
    plain = "".join(
        char
        for char in unicodedata.normalize("NFKD", plain)
        if not unicodedata.combining(char)
    )
    return re.sub(r"^(?:[ivxlcdm]+|\d+|[a-z])[.)-]\s*", "", plain)


def _heading_role(
    text: str, *, metadata_heading: bool = False
) -> JudicialTextSection | None:
    heading = _normalized_heading(text)
    administrative = r"^(?:karar tarihi|esas sayisi|karar sayisi|applicant|case information)\s*(?::|$)"
    if re.match(administrative, heading):
        return None if metadata_heading else "preliminary"
    if re.match(
        r"^(?:itirazin konusu|basvurunun konusu|basvuru konusu|itiraz yoluna basvuran|"
        r"basvuran|dava konusu|olay|iptali istenen (?:kanun (?:hukmu|hukumleri)|kurallar?)|"
        r"subject of (?:the )?application|requested relief|procedural history|"
        r"challenged (?:provisions?|rules?))\s*(?::|$)",
        heading,
    ):
        return "preliminary"
    if re.match(r"^[\w\s]+mahkemesi\s+baskanligindan\s*:", heading):
        return "preliminary"
    if re.match(
        r"^(?:taraflarin iddialari|basvurucunun iddialari|davacinin iddialari|"
        r"davalinin savunmasi|itirazin gerekcesi|grounds of (?:the )?application|"
        r"(?:applicant|claimant|defendant)(?:'s)? (?:arguments|submissions))\s*(?::|$)",
        heading,
    ):
        return "argument_only"
    if re.match(
        r"^(?:esas(?:in|inin) incelenmesi|merits|reasons)\s*(?::|$)",
        heading,
    ):
        return "reasoning"
    if re.match(r"^(?:hukum|sonuc|disposition|holding)\s*(?::|$)", heading):
        return "disposition"
    if metadata_heading:
        return None
    if re.match(r"^\s*(?:#{1,6}\s|[IVXLCDM]+[.)-]\s+)", text) or re.match(
        r"^\s*\*\*[^*\n]+\*\*\s*$", text
    ):
        return "unknown"
    return None


def _canonical_position(item: EvidenceItem) -> int | None:
    position = item.metadata.get("position")
    doc = item.search_doc
    metadata = model_evidence_metadata(item.metadata)
    if (
        not isinstance(position, int)
        or isinstance(position, bool)
        or position < 0
        or not item.chunk_id
        or doc is None
        or doc.document_id != item.source_id
        or doc.metadata.get("regulatory_chunk_id") != item.chunk_id
        or any(
            metadata.get(key)
            for key in ("derived", "external", "untrusted", "truncated")
        )
    ):
        return None
    return position


def _section_transition(
    text: str,
    role: JudicialTextSection,
    parent: JudicialTextSection,
    *,
    metadata_heading: bool = False,
) -> tuple[JudicialTextSection, JudicialTextSection]:
    found = _heading_role(text, metadata_heading=metadata_heading)
    plain = re.sub(r"[*_#]", "", text).strip()
    # Lettered subheadings return to their major section after party submissions.
    if re.match(r"^[A-HJ-UWYZ][.)-]\s+", plain):
        return (found if found not in {None, "unknown"} else parent), parent
    return (found, found) if found is not None else (role, parent)


def _preceding_sections(
    item: EvidenceItem, source_context: Sequence[EvidenceItem]
) -> tuple[JudicialTextSection, JudicialTextSection]:
    position = _canonical_position(item)
    if position is None:
        return "unknown", "unknown"
    by_position: dict[int, list[EvidenceItem]] = {}
    for original in source_context:
        candidate_position = _canonical_position(original)
        if (
            original.source_id == item.source_id
            and candidate_position is not None
            and candidate_position < position
            and all(
                original.metadata.get(key) == item.metadata.get(key)
                for key in ("read_as_of_date", "validity_start", "validity_end")
            )
        ):
            by_position.setdefault(candidate_position, []).append(original)
    preceding: list[EvidenceItem] = []
    for previous in range(position - 1, -1, -1):
        candidates = by_position.get(previous, [])
        if len({candidate.identity for candidate in candidates}) != 1:
            break
        preceding.append(candidates[0])
    role: JudicialTextSection = "unknown"
    parent: JudicialTextSection = "unknown"
    for original in reversed(preceding):
        for line in original.text.splitlines():
            role, parent = _section_transition(line, role, parent)
    return role, parent


def judicial_witness_section(
    item: EvidenceItem,
    start_char: int,
    end_char: int,
    *,
    source_context: Sequence[EvidenceItem] = (),
) -> JudicialTextSection:
    """Identify section text, never the meaning or legal sufficiency of a holding."""
    role, parent = _preceding_sections(item, source_context)
    headings = item.metadata.get("heading_path")
    if isinstance(headings, list):
        for heading in headings:
            if not isinstance(heading, str):
                continue
            role, parent = _section_transition(
                heading, role, parent, metadata_heading=True
            )
    covered: set[JudicialTextSection] = set()
    position = 0
    for line in item.text.splitlines(keepends=True):
        found = _heading_role(line)
        role, parent = _section_transition(line, role, parent)
        line_end = position + len(line)
        if (
            position < end_char
            and start_char < line_end
            and line.strip()
            and found not in {"unknown", "reasoning", "disposition"}
        ):
            covered.add(role)
        position = line_end
        if position >= end_char:
            break
    if len(covered) == 1:
        return next(iter(covered))
    return "unknown"


def nonoperative_judicial_witness_role(
    item: EvidenceItem,
    start_char: int,
    end_char: int,
    *,
    source_context: Sequence[EvidenceItem] = (),
) -> JudicialSectionRole:
    role = judicial_witness_section(
        item, start_char, end_char, source_context=source_context
    )
    if role == "preliminary":
        return "preliminary"
    if role == "argument_only":
        return "argument_only"
    return "unknown"


def judicial_disposition_missing(originals: Sequence[EvidenceItem]) -> bool:
    """Recognized reasoning cannot stand in for the connected disposition body."""
    sections = {
        judicial_witness_section(item, 0, len(item.text), source_context=originals)
        for item in originals
        if _canonical_position(item) is not None
    }
    return "reasoning" in sections and "disposition" not in sections
