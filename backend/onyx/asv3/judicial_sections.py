"""Recognize preliminary judicial sections without interpreting a legal holding."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from onyx.asv3.models import EvidenceItem

JudicialSectionRole = Literal["preliminary", "argument_only", "unknown"]


def _normalized_heading(text: str) -> str:
    plain = re.sub(r"[*_#]", "", text).strip().casefold().replace("ı", "i")
    plain = "".join(
        char
        for char in unicodedata.normalize("NFKD", plain)
        if not unicodedata.combining(char)
    )
    return re.sub(r"^(?:[ivxlcdm]+|\d+)[.)-]\s*", "", plain)


def _heading_role(text: str) -> JudicialSectionRole | None:
    heading = _normalized_heading(text)
    if re.match(
        r"^(?:itirazin konusu|basvurunun konusu|basvuru konusu|itiraz yoluna basvuran|"
        r"basvuran|karar tarihi|esas sayisi|karar sayisi|dava konusu|subject of (?:the )?application|"
        r"requested relief|procedural history|applicant|case information)\s*(?::|$)",
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
    if re.match(r"^\s*(?:#{1,6}\s|(?:[IVXLCDM]+|\d+)[.)-]\s+)", text) or re.match(
        r"^\s*\*\*[^*\n]+\*\*\s*$", text
    ):
        return "unknown"
    return None


def nonoperative_judicial_witness_role(
    item: EvidenceItem, start_char: int, end_char: int
) -> JudicialSectionRole:
    """Return a negative structural signal; unknown never approves legal support."""
    role: JudicialSectionRole = "unknown"
    headings = item.metadata.get("heading_path")
    if isinstance(headings, list):
        for heading in headings:
            if not isinstance(heading, str):
                continue
            found = _heading_role(heading)
            if found is not None:
                role = found
    covered: set[JudicialSectionRole] = set()
    position = 0
    for line in item.text.splitlines(keepends=True):
        found = _heading_role(line)
        if found is not None:
            role = found
        line_end = position + len(line)
        if position < end_char and start_char < line_end and line.strip():
            covered.add(role)
        position = line_end
        if position >= end_char:
            break
    if covered == {"preliminary"}:
        return "preliminary"
    if covered == {"argument_only"}:
        return "argument_only"
    return "unknown"
