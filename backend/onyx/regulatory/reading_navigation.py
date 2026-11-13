"""Conservative removal of navigation lists from retained reading copies."""

import re
import unicodedata
from typing import NamedTuple

_ARTICLE = re.compile(r"^madde\s*(\d+(?:\s*[a-z](?=\s|$))?)\b[\s.:–—-]*(.*)$")
_BARE_PART = re.compile(
    r"[ivxlcdm\d]+\.?\s*kisim(?:\s*[-–]\s*[ivxlcdm\d]+\.?\s*bolum)?$"
)
_STRUCTURE = re.compile(r"^(kisim|bolum|baslik)\b\s*([ivxlcdm]+|\d+)?")
_BODY_PART = re.compile(r"^(?:birinci|ikinci|[ivxlcdm]+\.?)\s+kisim$")
_ORDINAL_STRUCTURE = re.compile(
    r"^(?:birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|"
    r"dokuzuncu|onuncu|onbirinci|onikinci|onucuncu)\s+(?:kisim|bolum)$"
)
_ANNEX_ENTRY = re.compile(r"^ek\s+[a-z\d]+(?:\s+[a-z]|\s*\([a-z\d]+\))?\s*:\s*\S")


class _Navigation(NamedTuple):
    end: int
    preserved: tuple[str, ...] = ()


def _plain(line: str) -> str:
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    line = re.sub(r"^\s*(?:#{1,8}\s*|[-+]\s+)", "", line)
    line = line.replace("*", "").replace("_", "").strip()
    folded = unicodedata.normalize("NFKD", line.casefold().replace("ı", "i"))
    return "".join(c for c in folded if not unicodedata.combining(c)).strip()


def _key(line: str) -> str:
    return re.sub(r"\s+", " ", _plain(line))


def _article(line: str) -> tuple[str, str] | None:
    match = _ARTICLE.fullmatch(line)
    return (match[1].replace(" ", ""), match[2].strip()) if match else None


def _title_tokens(title: str) -> tuple[str, ...]:
    return tuple(
        word[:8] for word in re.findall(r"[a-z0-9]+", title.replace("eur.l", "eur.1"))
    )


def _headed_navigation(
    lines: list[str], keys: list[str], start: int
) -> _Navigation | None:
    articles: dict[str, str] = {}
    structures: set[tuple[str, str]] = set()
    continued: dict[str, str] = {}
    last_number: str | None = None
    pending_structure: int | None = None
    last_annex = False
    preserved: list[str] = []
    end = start + 1
    for i in range(start + 1, len(lines)):
        key = keys[i]
        if key.startswith("|") and key.endswith("|"):
            cells = [cell.strip() for cell in key.strip("|").split("|")]
            if len(cells) != 2:
                return None
            key = " ".join(cells)
            if re.fullmatch(r"[-: ]+", key):
                key = ""
        if not key:
            end = i + 1
            continue
        article = _article(key)
        structure = _STRUCTURE.match(key)
        if article:
            number, title = article
            if number in articles:
                if pending_structure is not None:
                    end = pending_structure
                break
            if not title or len(title) > 240:
                return None
            # Contents titles cannot be an operative sentence or paragraph.
            if title.endswith((".", ";", ":")):
                return None
            articles[number] = title
            last_number = number
            last_annex = False
            pending_structure = None
        elif structure:
            identity = (structure[1], structure[2])
            if identity in structures:
                break
            if articles and pending_structure is None:
                pending_structure = i
            structures.add(identity)
            last_number = None
            last_annex = False
        elif key.rstrip(":") in {"ekler", "eklerin listesi"} or _ANNEX_ENTRY.match(key):
            last_number = None
            last_annex = bool(_ANNEX_ENTRY.match(key))
        elif key == "ortak deklarasyon":
            following = next((key for key in keys[i + 1 : i + 5] if key), "")
            following_structure = _STRUCTURE.match(following)
            if (
                following_structure is None
                or (following_structure[1], following_structure[2]) not in structures
            ):
                break
            last_number = None
            last_annex = False
        elif (
            last_annex
            and len(key) < 240
            and not key.endswith((".", ";", ":"))
            and _ANNEX_ENTRY.match(
                next((key for key in keys[i + 1 : i + 5] if key), "")
            )
        ):
            pass
        elif re.match(r"^\d+\s", key) and next(
            (key for key in keys[i + 1 : i + 5] if key), ""
        ).rstrip(":") in {"ekler", "eklerin listesi"}:
            preserved.append(lines[i])
            for j in range(i + 1, min(i + 5, len(lines))):
                if keys[j]:
                    break
                preserved.append(lines[j])
            last_number = None
            last_annex = False
        elif (
            last_number is not None
            and not key.endswith((".", ";", ":"))
            and len(articles[last_number] + key) < 240
            and (
                next_article := _article(
                    next((key for key in keys[i + 1 : i + 5] if key), "")
                )
            )
            and next_article[0] not in articles
        ):
            continued.setdefault(last_number, articles[last_number])
            articles[last_number] += " " + key
        else:
            break
        end = i + 1

    if pending_structure is not None:
        end = min(end, pending_structure)
    if len(articles) < 3:
        return None

    # Unknown text followed by more list entries is an uncertain boundary.
    for key in keys[end:]:
        structure = _STRUCTURE.match(key)
        if key == "icindekiler" or (
            structure and (structure[1], structure[2]) in structures
        ):
            break
        article = _article(key)
        if article:
            if article[0] in articles:
                break
            return None
        if _ANNEX_ENTRY.match(key) or key.rstrip(":") in {"ekler", "eklerin listesi"}:
            return None

    # Corroborate against actual article headings, before another contents list.
    def matches_title(number: str, title: str) -> bool:
        prefix = continued.get(number, "")
        return title == articles[number] or (
            len(prefix) >= 40
            and title.startswith(prefix)
            and _title_tokens(title) == _title_tokens(articles[number])
        )

    corroborated: set[str] = set()
    for i in range(end, len(lines)):
        if keys[i] == "icindekiler":
            break
        article = _article(keys[i])
        if article is None or article[0] not in articles:
            continue
        number, title = article
        if matches_title(number, title):
            corroborated.add(number)
            continue
        for key in keys[i + 1 : i + 7]:
            if not key:
                continue
            if _article(key) or _STRUCTURE.match(key) or key == "icindekiler":
                break
            title = (title + " " + key).strip()
            if matches_title(number, title):
                corroborated.add(number)
                break
            if len(title) >= len(articles[number]):
                break
    return (
        _Navigation(end, tuple(preserved))
        if len(corroborated) >= 3 and continued.keys() <= corroborated
        else None
    )


def _dotted_navigation(keys: list[str], start: int) -> _Navigation | None:
    prior = re.sub(r"[^a-z0-9]", "", " ".join(keys[:start]))
    corroborated: set[str] = set()
    leaders = 0
    end = start
    uncertain: int | None = None
    for i in range(start + 1, len(keys)):
        key = keys[i]
        if not key:
            continue
        if "madde no" in key and "sayfa no" in key:
            continue
        if _article(key) or key == "icindekiler":
            break
        match = re.fullmatch(r"(.+?)(?:\s*\.{2,}\s*|\s*)(\d[\d ]*)", key)
        title = match[1].strip() if match else key
        title = re.sub(r"^(?:\d+|[a-z])[)\-.]\s*", "", title)
        compact = re.sub(r"[^a-z0-9]", "", title)
        known = len(compact) >= 3 and compact in prior
        if match:
            if not known:
                uncertain = i
                break
            leaders += 1
            corroborated.add(compact)
            end = i + 1
        elif len(key) > 240 or not (known or _ORDINAL_STRUCTURE.fullmatch(key)):
            uncertain = i
            break
    # Do not hide just a prefix of a contents list with unknown formatting.
    if uncertain is not None:
        for key in keys[uncertain + 1 :]:
            if _article(key) or key == "icindekiler":
                break
            match = re.fullmatch(r"(.+?)\s*\.{2,}\s*\d[\d ]*", key)
            if match and re.sub(r"[^a-z0-9]", "", match[1]) in prior:
                return None
    return _Navigation(end) if leaders >= 3 and len(corroborated) >= 3 else None


def _bare_end(lines: list[str], keys: list[str], start: int) -> int | None:
    entries: list[str] = []
    end = start
    for i in range(start, len(keys)):
        if not keys[i]:
            if lines[i].strip():
                break
            end = i + 1
            continue
        if not _BARE_PART.fullmatch(keys[i]):
            break
        entries.append(keys[i])
        end = i + 1
    if (
        len(entries) < 3
        or len(set(entries)) != len(entries)
        or sum("bolum" in entry for entry in entries) < 2
    ):
        return None
    following = [key for key in keys[end : end + 24] if key]
    body_part_index = next(
        (i for i, key in enumerate(following) if _BODY_PART.fullmatch(key)), None
    )
    if body_part_index is None or any(
        "kisim" in key or "bolum" in key for key in following[:body_part_index]
    ):
        return None
    article_index = next(
        (i for i, key in enumerate(following) if _article(key) is not None), None
    )
    if article_index is None:
        return None
    article = _article(following[article_index])
    assert article is not None
    number, text = article
    if not text and article_index + 1 < len(following):
        text = following[article_index + 1]
    if (
        not any(_BODY_PART.fullmatch(key) for key in following)
        or number != "1"
        or len(text) < 40
        or _article(text) is not None
    ):
        return None
    return end


def remove_reading_navigation(markdown: str) -> str:
    """Remove corroborated lists while preserving all other source substrings."""
    lines = markdown.splitlines(keepends=True)
    keys = [_key(line) for line in lines]
    result: list[str] = []
    i = 0
    seen_article = False
    checked_unlabelled = False
    fence: str | None = None
    while i < len(lines):
        stripped = lines[i].lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if fence is None:
                fence = marker[1]
            elif stripped.rstrip().startswith(fence):
                fence = None
        if fence is None and marker is None:
            navigation = None
            if keys[i] == "icindekiler":
                navigation = _headed_navigation(lines, keys, i)
                if navigation is None:
                    navigation = _dotted_navigation(keys, i)
            elif i < 160 and not seen_article and not checked_unlabelled:
                if _BARE_PART.fullmatch(keys[i]):
                    checked_unlabelled = True
                    end = _bare_end(lines, keys, i)
                    if end is not None:
                        navigation = _Navigation(end)
                elif _STRUCTURE.match(keys[i]) or _article(keys[i]):
                    checked_unlabelled = True
                    navigation = _headed_navigation(lines, keys, i - 1)
            if navigation is not None:
                result.extend(navigation.preserved)
                i = navigation.end
                continue
            seen_article = seen_article or _article(keys[i]) is not None
        result.append(lines[i])
        i += 1
    return "".join(result)
