"""Conservative instrument identity and topic scopes without model calls."""

import re
import unicodedata
from collections.abc import Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from onyx.regulatory.heading_path import extract_regulatory_instrument_source_hint


class CatalogSourceTitle(BaseModel):
    file_id: UUID
    name: str
    root_heading: str = ""


class CatalogSearchScope(BaseModel):
    source_hint: str | None = None
    file_ids: list[UUID] = Field(default_factory=list)
    heading_prefixes: list[list[str]] = Field(default_factory=list)
    # Complete distinct canonical paths, not inferred article ranges.
    heading_paths: list[list[str]] = Field(default_factory=list)
    heading_paths_by_file: dict[str, list[list[str]]] = Field(default_factory=dict)


def catalog_fold(value: str) -> str:
    value = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    return "".join(c for c in value if not unicodedata.combining(c))


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", catalog_fold(value))


def _instrument_tokens(value: str) -> list[str]:
    tokens = _tokens(value)
    result: list[str] = []
    for token in tokens:
        if token.startswith("kanun"):
            token = "kanun"
        elif token.startswith(("yonetmelik", "yonetmelig")):
            token = "yonetmelik"
        elif token.startswith("teblig"):
            token = "teblig"
        elif token.startswith("genelge"):
            token = "genelge"
        elif token.startswith("karar"):
            token = "karar"
        result.append(token)
    return result


def _own_title(value: str) -> list[str]:
    title = value.replace("\\", "/").rsplit("/", 1)[-1]
    title = re.sub(r"\.(?:md|pdf|docx?|txt)$", "", title, flags=re.I)
    title = re.sub(r"^\s*\d{3,6}[_\s]+(?:say[iı]l[iı][_\s]+)?", "", title, flags=re.I)
    return _instrument_tokens(title)


def extract_catalog_source_hint(query: str) -> str | None:
    # Notification is a topic, whereas a tebliğ is an instrument type.
    routing_query = re.sub(r"\btebli[gğ]at\w*", "bildirim", query, flags=re.I)
    return extract_regulatory_instrument_source_hint(routing_query)


def resolve_catalog_source_titles(
    query: str, titles: Sequence[CatalogSourceTitle]
) -> list[CatalogSourceTitle]:
    hint = extract_catalog_source_hint(query)
    if not hint:
        return []
    target = _instrument_tokens(hint)
    number = next((t for t in target if t.isdigit() and len(t) >= 3), None)
    query_tokens = _instrument_tokens(query)
    series = re.search(r"(?<!\d)(\d{4})[/.-](\d{1,6})(?!\d)", query)
    form = target[-1] if target else ""
    matched: list[CatalogSourceTitle] = []
    for title in titles:
        own_titles = [_own_title(title.name), _own_title(title.root_heading)]
        if series:
            identity = (int(series.group(1)), int(series.group(2)))
            filename = title.name.replace("\\", "/").rsplit("/", 1)[-1]
            own_series = re.search(
                r"^(?:[a-z]+[_ -])?(\d{4})[/_.-](\d{1,6})(?!\d)", catalog_fold(filename)
            )
            found = bool(
                own_series
                and (int(own_series.group(1)), int(own_series.group(2))) == identity
                and any(tokens[-1:] == [form] for tokens in own_titles)
            )
        elif number:
            # A numeric source must own that number; citations in other titles do not qualify.
            found = any(
                number in _tokens(v)
                and _instrument_tokens(
                    re.sub(r"\.(?:md|pdf|docx?|txt)$", "", v, flags=re.I)
                )[-1:]
                == ["kanun"]
                for v in (title.name.rsplit("/", 1)[-1], title.root_heading)
            )
        else:
            found = any(
                len(tokens) >= 2
                and any(
                    query_tokens[start : start + len(tokens)] == tokens
                    for start in range(len(query_tokens) - len(tokens) + 1)
                )
                for tokens in own_titles
            )
        if found:
            matched.append(title)
    return matched


_TOPIC_STOP_WORDS = frozenset(
    {
        "ile",
        "ve",
        "veya",
        "ilgili",
        "iliskin",
        "hakkinda",
        "hukum",
        "hukumler",
        "hukumleri",
        "hukumlerini",
        "maddeler",
        "maddeleri",
        "bolum",
        "bolumu",
        "bul",
        "getir",
        "goster",
        "nelerdir",
        "nedir",
        "neler",
        "hangileri",
        "olan",
        "bir",
        "bu",
        "su",
        "da",
        "de",
        "yer",
        "alan",
        "bana",
        "tum",
        "butun",
        "acikla",
        "soruyorum",
    }
)


def _topic_stem(token: str) -> str:
    return (
        re.sub(
            r"(?:lar|ler)(?:i|in|ini|inin|a|e|da|de|dan|den)?$|(?:inin|inin|ini|sini|si|yi|ya|ye|i)$",
            "",
            token,
        )
        if len(token) > 4
        else token
    )


def select_catalog_heading_prefixes(
    query: str,
    source_hint: str,
    paths: Sequence[Sequence[str]],
    *,
    source_title: str | None = None,
) -> list[tuple[str, ...]]:
    # The instrument lead is a source constraint, never a topic match.
    folded_query = catalog_fold(query)
    if source_title:
        source_words = re.findall(r"[^\W_]+", source_hint)
        title_length = len(_own_title(source_title))
        if title_length:
            source_hint = " ".join(source_words[-title_length:])
    folded_hint = catalog_fold(source_hint)
    remainder = folded_query.replace(folded_hint, "", 1)
    topics = {
        _topic_stem(t)
        for t in _tokens(remainder)
        if t not in _TOPIC_STOP_WORDS and len(t) > 2 and not t.isdigit()
    }
    if not topics:
        return []
    matches: set[tuple[str, ...]] = set()
    for path in paths:
        for depth in range(2, len(path) + 1):
            label = path[depth - 1]
            # Article-specific questions retain semantic retrieval, not section-wide narrowing.
            if (
                re.search(r"\b(?:madde|article|ek|gecici)\s*\d", catalog_fold(label))
                or re.match(
                    r"^(?:\d+\s*[.)]|[a-zçğıöşü]\)|\([^)]+\))", label, flags=re.I
                )
                or label.endswith(("...", "…"))
            ):
                break
            ordinal = r"^(?:birinci|ikinci|ucuncu|dorduncu|besinci|altinci|yedinci|sekizinci|dokuzuncu|onuncu|onbirinci|onikinci|onucuncu)$"
            words = {
                _topic_stem(t)
                for t in _tokens(label)
                if t not in {"bolum", "kisim", "kitap", "ve", "ile"}
                and not re.fullmatch(ordinal, t)
            }
            # A broad topic must not be narrowed to one more-specific subtopic.
            if topics == words:
                matches.add(tuple(path[:depth]))
                break
    return sorted(
        prefix
        for prefix in matches
        if not any(
            prefix[: len(parent)] == parent
            for parent in matches
            if len(parent) < len(prefix)
        )
    )
