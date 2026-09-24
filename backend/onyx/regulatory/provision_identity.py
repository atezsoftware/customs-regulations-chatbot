"""Legal article namespaces shared by amendment and chat lookup."""

import re

from onyx.regulatory.heading_path import extract_regulatory_provision_references

QUALIFIED_ARTICLE_RE = re.compile(
    r"(?<!\w)(?P<kind>ek|geçici|gecici|mükerrer|mukerrer)\s+"
    r"(?:madde\s+(?P<forward>\d+[a-z]?)\b|"
    r"(?P<reverse>\d+[a-z]?)\s*(?:[.'’]?\s*"
    r"(?:inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü))?\s+madd\w*)",
    re.IGNORECASE,
)


def article_identity(reference: str) -> str | None:
    """Preserve the normal/additional/temporary/repeated article namespace."""
    references = extract_regulatory_provision_references(reference)
    if not references:
        return None
    number = references[0].article_no
    qualified = QUALIFIED_ARTICLE_RE.search(reference)
    if (
        qualified
        and (qualified.group("forward") or qualified.group("reverse")).upper() == number
    ):
        kind = qualified.group("kind").casefold().replace("\u0307", "")
        prefix = (
            "EK"
            if kind == "ek"
            else "GEÇİCİ"
            if kind in {"geçici", "gecici"}
            else "MÜKERRER"
        )
        return f"{prefix} {number}"
    return number
