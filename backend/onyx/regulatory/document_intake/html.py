"""Recover legal paragraph boundaries from verified HTML layout containers."""

import re

from bs4 import BeautifulSoup, Tag

from onyx.file_processing.html_utils import parse_html_page_basic

_ARTICLE_START = re.compile(
    r"^(?:(?:GEÇİCİ|GECICI|MÜKERRER|MUKERRER|EK)\s+)?MADDE\s+"
    r"\d+(?:/?[A-Z])?\s*[-–—:]",
    re.I,
)
_MASTHEAD_CELL = re.compile(
    r"(?:\d{1,2}\s+\w+\s+\d{4}(?:\s+\w+)?|resm[îi]\s+gazete|"
    r"say[ıi]\s*:\s*\d+|yönetmelik|tebliğ|kanun|karar)",
    re.I,
)


def _legal_layout_table(table: Tag) -> bool:
    cells = [
        cell
        for cell in table.find_all(("td", "th"))
        if cell.find_parent("table") is table
    ]
    body = [
        cell
        for cell in cells
        if sum(
            bool(_ARTICLE_START.match(paragraph.get_text(" ", strip=True)))
            for paragraph in cell.find_all("p")
        )
        >= 2
    ]
    return len(body) == 1 and all(
        cell is body[0]
        or not (text := cell.get_text(" ", strip=True))
        or _MASTHEAD_CELL.fullmatch(text) is not None
        for cell in cells
    )


def canonical_legal_html(root: Tag | BeautifulSoup) -> str:
    """Restore only layout whitespace; real grids and existing link text stay exact."""
    baseline = parse_html_page_basic(str(root))
    layouts = [table for table in root.find_all("table") if _legal_layout_table(table)]
    if not layouts:
        return baseline
    # The legacy formatter stops rendering links after entering its first table.
    # Preserve that rendering contract while fixing only paragraph boundaries.
    in_table = False
    for element in list(root.descendants):
        if isinstance(element, Tag):
            if element.name == "table":
                in_table = True
            elif in_table and element.name == "a":
                element.name = "span"
    for table in reversed(layouts):
        table.unwrap()
    factory = BeautifulSoup("", "html.parser")
    for table in list(root.find_all("table")):
        if table.find_parent("table") is not None:
            continue
        replacement = factory.new_tag("div")
        for line in parse_html_page_basic(str(table)).splitlines():
            row = factory.new_tag("div")
            row.string = line
            replacement.append(row)
        table.replace_with(replacement)
    normalized = parse_html_page_basic(str(root))
    if "".join(normalized.split()) != "".join(baseline.split()):
        raise ValueError("HTML layout reconstruction changed the source wording")
    return normalized
