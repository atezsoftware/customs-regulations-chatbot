"""Source identity checks shared by chat and amendment retrieval."""

import re
import unicodedata

_SOURCE_GENERIC_TOKENS = frozenset(
    {
        "genel",
        "gumruk",
        "karar",
        "karari",
        "kanun",
        "kanunu",
        "no",
        "sayili",
        "seri",
        "tebligi",
        "teblig",
        "urun",
        "guvenligi",
        "denetimi",
        "ve",
        "yonetmeligi",
        "yonetmelik",
    }
)


def _source_identity_tokens(value: str) -> set[str]:
    decoded = re.sub(r"_?x[12]", " ", value, flags=re.IGNORECASE)
    folded = unicodedata.normalize(
        "NFKD",
        decoded.casefold().translate(
            str.maketrans({"ı": "i", "ş": "s", "ç": "c", "ğ": "g", "ö": "o", "ü": "u"})
        ),
    )
    ascii_value = "".join(
        character for character in folded if not unicodedata.combining(character)
    )
    return {
        str(int(token)) if token.isdigit() else token
        for token in re.findall(r"[a-z0-9]+", ascii_value)
    }


def source_identity_matches(target_source: str | None, source_name: str) -> bool:
    """Require explicit instrument-specific title tokens when they are available."""

    if not target_source:
        return True
    law_number = named_law_number(target_source)
    source_number = named_law_number(source_name)
    if law_number is not None:
        return law_number == source_number
    distinguishing_tokens = set(source_identity_distinguishing_tokens(target_source))
    if not distinguishing_tokens:
        return True
    return distinguishing_tokens <= _source_identity_tokens(source_name)


def _series_identities(value: str) -> set[tuple[int, int]]:
    return {
        (int(match[0]), int(match[1]))
        for match in re.findall(
            r"(?<!\d)((?:19|20)\d{2})\s*[/_-]\s*(\d{1,5})(?!\d|[/_-]\d)",
            value,
        )
    }


def source_file_identity_matches(target: str, name: str, root: str) -> bool:
    """Verify a file's own title and reject conflicting instrument identifiers."""
    number = named_law_number(target)
    filename_number, root_number = named_law_number(name), named_law_number(root)
    if filename_number and root_number and filename_number != root_number:
        return False
    if number is not None:
        return (filename_number or root_number) == number
    series = _series_identities(target)
    name_series, root_series = _series_identities(name), _series_identities(root)
    if name_series and root_series and name_series != root_series:
        return False
    if series and any(
        identities and identities != series for identities in (name_series, root_series)
    ):
        return False
    return source_identity_matches(target, name) or source_identity_matches(
        target, root
    )


def named_law_number(source_name: str) -> str | None:
    """Identify a law title, not a law cited inside another document's name."""
    folded = unicodedata.normalize("NFKD", source_name.casefold().replace("ı", "i"))
    title = "".join(
        character for character in folded if not unicodedata.combining(character)
    ).replace("_", " ")
    if not re.search(r"\bkanun", title):
        return None
    match = re.search(r"\b(\d{3,5})\s+sayili\b", title)
    if match is not None:
        own_title = re.sub(
            r"\.(?:md|txt|pdf|docx|html?)$", "", title[match.end() :]
        ).strip()
        # A title may itself amend other laws. Its final instrument type, not
        # the word 'amendment', distinguishes it from a regulation citing a law.
        if not re.search(r"\bkanun(?:u)?(?:\s*\(.*\))?\s*$", own_title):
            return None
        if re.match(r"kanun(?:da|unda|a|una)\b", own_title):
            return None
    if match is None:
        match = re.search(r"\bkanun\s+(?:no|numarasi)\s*[:.]?\s*(\d{3,5})\b", title)
    return str(int(match.group(1))) if match else None


def source_identity_distinguishing_tokens(
    target_source: str | None,
) -> tuple[str, ...]:
    if not target_source:
        return ()
    return tuple(
        sorted(_source_identity_tokens(target_source) - _SOURCE_GENERIC_TOKENS)
    )
