"""Conservative Turkish paragraph references shared by retrieval and updates."""

import re
import unicodedata

_UNITS = ("", "bir", "iki", "uc", "dort", "bes", "alti", "yedi", "sekiz", "dokuz")
_TENS = (
    "",
    "on",
    "yirmi",
    "otuz",
    "kirk",
    "elli",
    "altmis",
    "yetmis",
    "seksen",
    "doksan",
)
_ORDINALS = dict(
    zip(
        (*_UNITS[1:], *_TENS[1:], "yuz"),
        (
            "birinci",
            "ikinci",
            "ucuncu",
            "dorduncu",
            "besinci",
            "altinci",
            "yedinci",
            "sekizinci",
            "dokuzuncu",
            "onuncu",
            "yirminci",
            "otuzuncu",
            "kirkinci",
            "ellinci",
            "altmisinci",
            "yetmisinci",
            "sekseninci",
            "doksaninci",
            "yuzuncu",
        ),
        strict=True,
    )
)


def _ordinal_words(number: int) -> str:
    hundreds, rest = divmod(number, 100)
    tens, units = divmod(rest, 10)
    words = [*([_UNITS[hundreds]] if hundreds > 1 else []), "yuz"] if hundreds else []
    if tens:
        words.append(_TENS[tens])
    if units:
        words.append(_UNITS[units])
    words[-1] = _ORDINALS[words[-1]]
    return " ".join(words)


_WORD_NUMBERS = {_ordinal_words(number): str(number) for number in range(1, 1000)}
_NUMBER_WORDS = "|".join(
    sorted(
        {*_ORDINALS, *_ORDINALS.values(), "bin", "milyon", "ve", "veya", "ile"},
        key=lambda word: (-len(word), word),
    )
)
_REFERENCE = re.compile(
    rf"(?<!\w)(?:(?P<words>(?:(?:{_NUMBER_WORDS})\s+)*(?:{_NUMBER_WORDS}))"
    r"|(?P<number>\d{1,3})(?:\s*[.'’]?\s*(?:inci|uncu|nci|ncu))?)\s+fikra"
)
_JOINED_REFERENCE_PREFIX = re.compile(
    rf"(?<!\w)(?:{_NUMBER_WORDS}|\d+(?:\s*[.'’]?\s*(?:inci|uncu|nci|ncu))?)"
    r"\s*(?:,|ve|veya|ile|ila|[-–—])\s*$"
)


def extract_single_paragraph_reference(text: str) -> str | None:
    folded = unicodedata.normalize("NFKD", text.casefold().replace("ı", "i"))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    numbers: set[str] = set()
    for match in _REFERENCE.finditer(folded):
        if _JOINED_REFERENCE_PREFIX.search(folded[: match.start()]):
            return None
        number = (
            _WORD_NUMBERS.get(" ".join(match.group("words").split()))
            if match.group("words")
            else str(int(match.group("number")))
        )
        if number is None or number == "0":
            return None
        numbers.add(number)
    return next(iter(numbers)) if len(numbers) == 1 else None
