"""Recognize preliminary judicial sections without interpreting a legal holding."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from typing import Literal

from onyx.asv3.models import EvidenceItem, model_evidence_metadata

JudicialSectionRole = Literal["preliminary", "argument_only", "unknown"]


def _normalized_heading(text: str) -> str:
    plain = re.sub(r"[*_#]", "", text).strip().casefold().replace("ı", "i")
    plain = "".join(
        char
        for char in unicodedata.normalize("NFKD", plain)
        if not unicodedata.combining(char)
    )
    return re.sub(r"^(?:[ivxlcdm]+|\d+)[.)-]\s*", "", plain)


def _heading_role(
    text: str, *, metadata_heading: bool = False
) -> JudicialSectionRole | None:
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
        r"^(?:hukum|sonuc|esas(?:in|inin) incelenmesi|merits|reasons|disposition|holding)\s*(?::|$)",
        heading,
    ):
        return "unknown"
    if re.match(r"^\s*(?:#{1,6}\s|(?:[IVXLCDM]+|\d+)[.)-]\s+)", text) or re.match(
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


def _preceding_role(
    item: EvidenceItem, source_context: Sequence[EvidenceItem]
) -> JudicialSectionRole:
    position = _canonical_position(item)
    if position is None:
        return "unknown"
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
    role: JudicialSectionRole = "unknown"
    for original in reversed(preceding):
        for line in original.text.splitlines():
            found = _heading_role(line)
            if found is not None:
                role = found
    return role


def nonoperative_judicial_witness_role(
    item: EvidenceItem,
    start_char: int,
    end_char: int,
    *,
    source_context: Sequence[EvidenceItem] = (),
) -> JudicialSectionRole:
    """Return a negative structural signal; unknown never approves legal support."""
    role = _preceding_role(item, source_context)
    headings = item.metadata.get("heading_path")
    if isinstance(headings, list):
        for heading in headings:
            if not isinstance(heading, str):
                continue
            found = _heading_role(heading, metadata_heading=True)
            if found is not None:
                role = found
    covered: set[JudicialSectionRole] = set()
    position = 0
    for line in item.text.splitlines(keepends=True):
        found = _heading_role(line)
        if found is not None:
            role = found
        line_end = position + len(line)
        if (
            position < end_char
            and start_char < line_end
            and line.strip()
            and found != "unknown"
        ):
            covered.add(role)
        position = line_end
        if position >= end_char:
            break
    if covered == {"preliminary"}:
        return "preliminary"
    if covered == {"argument_only"}:
        return "argument_only"
    return "unknown"
