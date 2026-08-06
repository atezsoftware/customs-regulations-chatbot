import re
import unicodedata
from datetime import date

from pydantic import BaseModel, Field


class LegalExactFields(BaseModel):
    """Normalized legal identifiers shared by indexing and query construction."""

    model_config = {"frozen": True}

    provision_identifiers: list[str] = Field(default_factory=list)
    decision_numbers: list[str] = Field(default_factory=list)
    legal_dates: list[str] = Field(default_factory=list)


_TURKISH_CASE_TRANSLATION = str.maketrans({"I": "ı", "İ": "i"})
_PROVISION_PATTERN = re.compile(
    r"(?<!\w)(?:(geçici|ek|mükerrer)\s+)?(?:madde|md\.?)[ \t]*"
    r"([0-9]+(?:[./-][0-9a-zçğıöşü]+|[a-zçğıöşü])?)",
    flags=re.IGNORECASE,
)
_DECISION_NUMBER_PATTERN = re.compile(r"(?<!\d)([12]\d{3}/\d{1,9})(?!\d)")
_DAY_FIRST_DATE_PATTERN = re.compile(r"(?<!\d)(\d{1,2})[./](\d{1,2})[./](\d{4})(?!\d)")
_ISO_DATE_PATTERN = re.compile(r"(?<!\d)(\d{4})-(\d{1,2})-(\d{1,2})(?!\d)")


def _turkish_lower(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    return normalized.translate(_TURKISH_CASE_TRANSLATION).lower()


def _append_unique(values: list[str], seen: set[str], value: str) -> None:
    if value not in seen:
        seen.add(value)
        values.append(value)


def _valid_iso_date(year: int, month: int, day: int) -> str | None:
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def extract_legal_exact_fields(*texts: str | None) -> LegalExactFields:
    """Extract stable Turkish provision, decision-number, and date values.

    Multiple inputs let indexing include chunk text, headings, and serialized
    metadata without giving this parser knowledge of connector-specific models.
    Values are de-duplicated in first-seen order so indexed payloads are stable.
    """

    searchable_text = "\n".join(text for text in texts if text)

    provision_identifiers: list[str] = []
    seen_provisions: set[str] = set()
    for match in _PROVISION_PATTERN.finditer(searchable_text):
        qualifier = match.group(1)
        identifier = _turkish_lower(match.group(2))
        normalized = (
            f"{_turkish_lower(qualifier)} madde {identifier}"
            if qualifier
            else f"madde {identifier}"
        )
        _append_unique(provision_identifiers, seen_provisions, normalized)

    decision_numbers: list[str] = []
    seen_decisions: set[str] = set()
    for match in _DECISION_NUMBER_PATTERN.finditer(searchable_text):
        _append_unique(decision_numbers, seen_decisions, match.group(1))

    legal_dates: list[str] = []
    seen_dates: set[str] = set()
    for match in _DAY_FIRST_DATE_PATTERN.finditer(searchable_text):
        normalized_date = _valid_iso_date(
            int(match.group(3)), int(match.group(2)), int(match.group(1))
        )
        if normalized_date is not None:
            _append_unique(legal_dates, seen_dates, normalized_date)
    for match in _ISO_DATE_PATTERN.finditer(searchable_text):
        normalized_date = _valid_iso_date(
            int(match.group(1)), int(match.group(2)), int(match.group(3))
        )
        if normalized_date is not None:
            _append_unique(legal_dates, seen_dates, normalized_date)

    return LegalExactFields(
        provision_identifiers=provision_identifiers,
        decision_numbers=decision_numbers,
        legal_dates=legal_dates,
    )
