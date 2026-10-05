"""Uncitable source-title relationships anchored to a read governing original."""

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from onyx.asv3.models import EvidenceItem, model_evidence_metadata
from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_regulatory_provision_reference_occurrences,
    parse_regulatory_article_heading,
)
from onyx.regulatory.source_identity import named_law_number

RelatedSourceRole = Literal[
    "judicial_candidate",
    "referral_candidate",
    "executive_candidate",
    "amendment_candidate",
]


@dataclass(frozen=True)
class ProvisionNavigationAnchor:
    source_id: str
    instrument_name: str
    instrument_number: str | None
    article_no: str
    qualifier: str | None


def _fold(value: str) -> str:
    normalized = unicodedata.normalize(
        "NFKD", value.casefold().translate(str.maketrans("ışçğöü", "iscgou"))
    )
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _formal_title(value: str) -> str | None:
    title = " ".join(value.split())
    if re.search(r"[/\\_]|\.(?:md|docx?|pdf|txt)$", title, re.IGNORECASE):
        return None
    title = re.sub(r"^\d{2,7}\s+say[iı]l[iı]\s+", "", title, flags=re.IGNORECASE)
    if re.fullmatch(r".+\s+(?:kanunu?|law|act|statute)", _fold(title)):
        return title
    return None


def derive_provision_navigation_anchor(
    source_id: str,
    evidence: Sequence[EvidenceItem],
    article_no: str | None = None,
    qualifier: str | None = None,
) -> ProvisionNavigationAnchor | None:
    """Use own canonical headings/title, never a law quoted in another source's text."""
    anchors: set[ProvisionNavigationAnchor] = set()
    for item in evidence:
        doc = item.search_doc
        metadata = model_evidence_metadata(item.metadata)
        if (
            item.source_id != source_id
            or not item.chunk_id
            or doc is None
            or doc.document_id != source_id
            or doc.metadata.get("regulatory_chunk_id") != item.chunk_id
            or metadata.get("derived")
            or metadata.get("external")
            or metadata.get("untrusted")
        ):
            continue
        kind = _fold(str(metadata.get("document_type") or ""))
        if kind and kind not in {"kanun", "law", "act", "statute"}:
            continue
        headings = metadata.get("heading_path")
        if not isinstance(headings, list) or not headings:
            continue
        own = None
        for heading in headings:
            if isinstance(heading, str):
                own = parse_regulatory_article_heading(heading)
                if own is not None:
                    break
        if own is None or (
            article_no is not None
            and (own.article_no, own.qualifier) != (article_no, qualifier)
        ):
            continue
        root, title = str(headings[0]), str(metadata.get("title") or "")
        root_name, title_name = _formal_title(root), _formal_title(title)
        if root_name and title_name and _fold(root_name) != _fold(title_name):
            continue
        official_name = root_name or title_name
        if official_name is None:
            continue
        root_number, title_number = named_law_number(root), named_law_number(title)
        if root_number and title_number and root_number != title_number:
            continue
        number = root_number or title_number
        anchors.add(
            ProvisionNavigationAnchor(
                source_id, official_name, number, own.article_no, own.qualifier
            )
        )
    # Conflicting instrument identities stay unresolved rather than broadening a query.
    return next(iter(anchors)) if len(anchors) == 1 else None


def match_related_source_name(
    anchor: ProvisionNavigationAnchor, source_name: str
) -> RelatedSourceRole | None:
    """A title explicitly targeting this instrument/provision is only a reading lead."""
    basename = source_name.replace("\\", "/").rsplit("/", 1)[-1]
    title = _fold(re.sub(r"\.(?:md|docx?|pdf|txt|html?)$", "", basename, flags=re.I))
    title = re.sub(r"[_\-]+", " ", title)
    name = re.escape(_fold(anchor.instrument_name))
    name = re.sub(r"kanunu?$", r"kanun(?:u|un|unun|a|una|da|unda|dan|undan)?", name)
    explicit_numbers = {
        str(int(match[1]))
        for match in re.finditer(
            rf"(?<!\d)(\d{{2,7}})\s+sayili\s+{name}(?![a-z0-9])", title
        )
    }
    if anchor.instrument_number and explicit_numbers - {anchor.instrument_number}:
        return None
    instrument_matches = list(re.finditer(rf"(?<![a-z0-9]){name}(?![a-z0-9])", title))
    if anchor.instrument_number is not None:
        instrument_matches.extend(
            re.finditer(
                rf"(?<!\d){re.escape(anchor.instrument_number)}\s+sayili\s+kanun[a-z]*\b",
                title,
            )
        )
    if not instrument_matches:
        return None
    # Keep offsets aligned while accepting adjectival title wording such as maddesindeki.
    reference_title = re.sub(
        r"\b(madde(?:si)?(?:nde|de))ki\b", lambda match: match[1] + "  ", title
    )
    references = extract_regulatory_provision_reference_occurrences(reference_title)
    target = RegulatoryProvisionReference(anchor.article_no, anchor.qualifier)
    intervening_instrument = (
        r"\b(?:kanun[a-z]*|yonetmelik[a-z]*|teblig[a-z]*|law|act|statute|regulation)\b"
    )
    if not any(
        reference == target
        and match.end() <= start
        and re.search(intervening_instrument, title[match.end() : start]) is None
        for start, reference in references
        for match in instrument_matches
    ):
        return None
    tokens = set(re.findall(r"[a-z]+", title))
    if any(
        token.startswith(("basvuru", "itiraz", "dilekce", "referral"))
        for token in tokens
    ):
        return "referral_candidate"
    if any(
        token.startswith(
            ("mahkeme", "danistay", "yargitay", "court", "tribunal", "judgment")
        )
        for token in tokens
    ):
        return "judicial_candidate"
    if tokens & {"cumhurbaskani", "bakanlar", "presidential", "cabinet"} and any(
        token.startswith(("karar", "decision", "decree")) for token in tokens
    ):
        return "executive_candidate"
    if any(
        token.startswith(("degisiklik", "degistir", "iptal", "amendment", "annulment"))
        for token in tokens
    ):
        return "amendment_candidate"
    return None
