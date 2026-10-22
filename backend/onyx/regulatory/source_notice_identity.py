"""Conservative typed identities for content-free regulatory source notices."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class NoticeIdentity:
    instrument_type: str
    number: str
    year: int | None = None
    instrument_date: date | None = None

    @property
    def key(self) -> str:
        parts = [self.instrument_type]
        if self.instrument_date is not None:
            parts.append(self.instrument_date.isoformat())
        elif self.year is not None:
            parts.append(str(self.year))
        parts.append(str(int(self.number)))
        return ":".join(parts)

    @property
    def aliases(self) -> tuple[str, ...]:
        number = str(int(self.number))
        if self.instrument_type == "law":
            return (self.key, f"{number} sayılı Kanun")
        if self.instrument_type == "administrative_letter":
            assert self.instrument_date is not None
            return (
                self.key,
                f"{self.instrument_date:%d.%m.%Y} tarihli {number} sayılı yazı",
            )
        label = {"circular": "Genelge", "decision": "Karar", "communique": "Tebliğ"}[
            self.instrument_type
        ]
        return (self.key, f"{self.year}/{number} sayılı {label}")


def normalize_notice_alias(value: str) -> str:
    """Fold filename punctuation/accents only; never infer missing identities."""
    value = value.replace("ı", "i").replace("İ", "I")
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    ).casefold()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def notice_title_alias(value: str) -> str | None:
    """Address an exact catalogue title without treating its repeal note as its name."""
    marker = r"\(\s*m[üu]lga\s*\)"
    value = re.sub(rf"^\s*(?:{marker}\s*)+", "", value, flags=re.IGNORECASE)
    title = re.split(marker, value, maxsplit=1, flags=re.IGNORECASE)[0]
    normalized = normalize_notice_alias(title)
    if not normalized:
        return None
    return "title:" + hashlib.sha256(normalized.encode()).hexdigest()


_TYPE_WORDS = {
    "genelge": "circular",
    "karar": "decision",
    "karari": "decision",
    "bkk": "decision",
    "ck": "decision",
    "teblig": "communique",
    "tebligi": "communique",
}
_YEAR_NUMBER = r"(?P<year>(?:18|19|20)\d{2})\s*[/–—-]\s*(?P<number>\d{1,7})"
_TYPE = r"(?P<kind>genelge(?:si|nin|sinin|ye|yi|de|den)?|karar(?:i|in|inin|ina|a|da|inda|indan)?|bkk|ck|teblig(?:i|in|inin|ine|e|de|inde|inden)?)"
_REFERENCE_PATTERNS = (
    re.compile(
        rf"(?<!\d){_YEAR_NUMBER}\s*(?:sayili\s+(?:[a-z]{{2,}}\s+){{0,6}}?)?{_TYPE}\b"
    ),
    re.compile(rf"\b{_TYPE}\s*(?:no\s*[:.]?\s*)?\(?\s*{_YEAR_NUMBER}(?!\d)"),
)
_LAW_PATTERNS = (
    re.compile(
        r"(?<!\d)(?P<number>\d{1,7})\s+sayili\s+(?:[a-z]+\s+){0,12}?kanun(?:u|un|unun|a|una|unda|undan|daki|undaki)?\b"
    ),
    re.compile(r"\bkanun\s*\(\s*(?P<number>\d{1,7})\s*(?:-|–|/|\))"),
)
_LETTER = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-](?P<year>(?:19|20)\d{2})"
    r"\s+tarihli\s+(?:ve\s+)?(?P<number>\d{1,15})\s+sayili\s+(?:tasarruflu\s+)?yazi\b"
)


def notice_identity_queries(text: str) -> list[str]:
    """Extract explicit first-hop references, never bare numbers or guessed titles."""
    # Keep structural date/year delimiters, while applying the same Turkish fold.
    folded = text.replace("ı", "i").replace("İ", "I")
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", folded)
        if not unicodedata.combining(character)
    ).casefold()
    result: dict[str, None] = {}
    for canonical in re.findall(
        r"(?<![a-z0-9_])(?:law:\d+|(?:circular|decision|communique):(?:18|19|20)\d{2}:\d+|administrative_letter:\d{4}-\d{2}-\d{2}:\d+|(?:catalog|title):[a-f0-9]{64})(?![a-z0-9_])",
        folded,
    ):
        result[canonical] = None
    folded = folded.replace("_", " ")
    for pattern in _REFERENCE_PATTERNS:
        for match in pattern.finditer(folded):
            raw_kind = match["kind"]
            if raw_kind.startswith("genelge"):
                kind = "circular"
            elif raw_kind.startswith("karar"):
                kind = "decision"
            elif raw_kind.startswith("teblig"):
                kind = "communique"
            else:
                kind = _TYPE_WORDS[raw_kind]
            identity = NoticeIdentity(kind, match["number"], int(match["year"]))
            result[identity.key] = None
    for pattern in _LAW_PATTERNS:
        for match in pattern.finditer(folded):
            if re.search(r"\b(?:genelge|teblig|karar)", match[0]):
                continue
            result[NoticeIdentity("law", match["number"]).key] = None
    for match in _LETTER.finditer(folded):
        try:
            issued = date(int(match["year"]), int(match["month"]), int(match["day"]))
        except ValueError:
            continue
        result[
            NoticeIdentity(
                "administrative_letter", match["number"], instrument_date=issued
            ).key
        ] = None
    return list(result)
