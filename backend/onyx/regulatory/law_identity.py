"""Exact own-law identity from a canonical title and its opening fields."""

import re
import unicodedata


def _fold(value: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    ).replace("ı", "i")


def canonical_law_number(value: str) -> str | None:
    if not re.fullmatch(r"\d{2,7}", value) or int(value) == 0:
        return None
    return str(int(value))


def _own_title(value: str) -> str | None:
    title = " ".join(_fold(value).split())
    title = re.sub(r"(?:\s*\[\d+\])+$", "", title).strip()
    if re.search(r"[/\\_]|\.(?:md|txt|pdf)$", title):
        return None
    numbered = re.match(r"^(\d{2,7})\s+sayili\s+", title)
    if numbered is not None and canonical_law_number(numbered[1]) is None:
        return None
    title = re.sub(r"^\d{2,7}\s+sayili\s+", "", title)
    match = re.fullmatch(r"(.+?)\s+kanunu?", title)
    return f"{match[1]} kanun" if match else None


def own_law_filename_number(name: str, root: str) -> str | None:
    """Bind an exact numbered filename or presentation name to its own law title."""
    title = _own_title(root)
    if title is None:
        return None
    display = re.fullmatch(r"(\d{2,7})\s+sayili\s+(.+)", " ".join(_fold(name).split()))
    if display is not None and _own_title(display[2]) == title:
        number = canonical_law_number(display[1])
        root_number = own_law_title_number(root)
        return (
            number
            if number is not None and (root_number is None or root_number == number)
            else None
        )
    basename = _fold(name.replace("\\", "/").rsplit("/", 1)[-1])
    match = re.fullmatch(
        r"(\d{2,7})[ _-]+(?:sayili[ _-]+)?(.+?)\.(?:md|txt|pdf)", basename
    )
    if match is None:
        return None
    number = canonical_law_number(match[1])
    filename_title = _own_title(match[2].replace("_", " "))
    numeric_law_filename = re.fullmatch(
        r"\d{2,7}[ _-]+sayili[ _-]+kanunu?\.(?:md|txt|pdf)", basename
    )
    if number is None or filename_title != title and not numeric_law_filename:
        return None
    numbered_root = re.match(r"^(\d{2,7})\s+sayili\b", _fold(root))
    if numbered_root and canonical_law_number(numbered_root[1]) != number:
        return None
    return number


def own_law_title_number(value: str) -> str | None:
    match = re.match(r"^(\d{2,7})\s+sayili\b", " ".join(_fold(value).split()))
    return canonical_law_number(match[1]) if match else None


def compatible_own_law_title_fragment(root: str, full_title: str) -> bool:
    """Only an exact suffix caption can use a separately proven full title."""
    fragment, title = _own_title(root), _own_title(full_title)
    return (
        fragment is not None
        and title is not None
        and len(fragment.split()) >= 2
        and (fragment == title or title.endswith(" " + fragment))
        and (
            own_law_title_number(root) is None
            or own_law_title_number(full_title) in {None, own_law_title_number(root)}
        )
    )


def own_law_opening_numbers(text: str, root: str) -> frozenset[str]:
    """Read a complete singular own-number field beneath its contiguous own title.

    Ambiguous recognized mastheads are invalid, rather than an absent witness
    that could be overridden by a numbered canonical root.
    """
    title = _own_title(root)
    if title is None:
        return frozenset()
    lines = text.lstrip("\ufeff \t\r\n").splitlines()
    if lines and re.fullmatch(r"```[a-zA-Z0-9_+-]*\s*", lines[0]):
        lines = lines[1:]
    lines = [line.strip(" #*\t") for line in lines]
    if not lines or re.match(r"(?:madde|amac)\b", _fold(lines[0])):
        return frozenset()
    header: list[str] = []
    for line in lines:
        if re.match(r"(?:madde|amac)\b", _fold(line)):
            break
        header.append(line)
    fields = [
        i
        for i, line in enumerate(header)
        if re.match(r"kanun\s+(?:numarasi|no)\b", _fold(line))
    ]
    if not fields:
        return frozenset()
    if len(fields) != 1 or len(header) > 12 or len("\n".join(header)) > 1600:
        raise ValueError(
            "The own law-number masthead is ambiguous or exceeds its complete-read bound"
        )
    index = fields[0]
    opening_title = " ".join(header[:index])
    if _own_title(opening_title) != title:
        raise ValueError(
            "The own law-number field does not immediately follow its own title"
        )
    field = re.fullmatch(
        r"kanun\s+(?:numarasi|no)\s*[:.]?\s*(\d{2,7})"
        r"(?:\s+kabul\s+tarihi\s*:\s*\d{1,2}/\d{1,2}/\d{4})?\s*",
        _fold(header[index]),
    )
    number = canonical_law_number(field[1]) if field else None
    if number is None or any(
        candidate is not None and candidate != number
        for candidate in (
            own_law_title_number(opening_title),
            own_law_title_number(root),
        )
    ):
        raise ValueError("The own law number is malformed or contradicts its title")
    return frozenset({number})
