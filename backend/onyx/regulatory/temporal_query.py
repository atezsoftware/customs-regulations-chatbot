import re
import unicodedata
from datetime import date
from typing import Final

_TURKISH_MONTHS: Final[dict[str, int]] = {
    "ocak": 1,
    "subat": 2,
    "mart": 3,
    "nisan": 4,
    "mayis": 5,
    "haziran": 6,
    "temmuz": 7,
    "agustos": 8,
    "eylul": 9,
    "ekim": 10,
    "kasim": 11,
    "aralik": 12,
}

_WRITTEN_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})\s+"
    r"(?P<month>ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|"
    r"eylul|ekim|kasim|aralik)\s+"
    r"(?P<year>\d{4})(?!\d)"
)
_DAY_FIRST_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-]"
    r"(?P<year>\d{4})(?!\d)"
)
_ISO_DATE_RE: Final[re.Pattern[str]] = re.compile(
    r"(?<!\d)(?P<year>\d{4})-(?P<month>\d{1,2})-(?P<day>\d{1,2})(?!\d)"
)


def _fold_turkish_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    ).replace("ı", "i")


def _validated_date(*, year: str, month: int | str, day: str) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def extract_regulatory_as_of_date(message: str) -> date | None:
    """Return one unambiguous full calendar date from a regulatory query."""
    folded_message = _fold_turkish_text(message)
    dates: set[date] = set()

    for match in _WRITTEN_DATE_RE.finditer(folded_message):
        parsed = _validated_date(
            year=match.group("year"),
            month=_TURKISH_MONTHS[match.group("month")],
            day=match.group("day"),
        )
        if parsed is not None:
            dates.add(parsed)

    for pattern in (_DAY_FIRST_DATE_RE, _ISO_DATE_RE):
        for match in pattern.finditer(folded_message):
            parsed = _validated_date(
                year=match.group("year"),
                month=match.group("month"),
                day=match.group("day"),
            )
            if parsed is not None:
                dates.add(parsed)

    return next(iter(dates)) if len(dates) == 1 else None
