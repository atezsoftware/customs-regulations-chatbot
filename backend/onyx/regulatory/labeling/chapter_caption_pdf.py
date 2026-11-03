"""Bounded glyph advances for the simple embedded native print font profile."""

import re
from typing import Any, cast


def caption_font_metrics(font: Any) -> tuple[dict[str, float], float, float]:
    if (
        font is None
        or font.get("/Subtype") != "/Type0"
        or font.get("/Encoding") != "/Identity-H"
    ):
        raise ValueError("The native caption font has no supported glyph metrics")
    descendants = font.get("/DescendantFonts")
    if not isinstance(descendants, list) or len(descendants) != 1:
        raise ValueError("The native caption descendant font is ambiguous")
    descendant = descendants[0].get_object()
    descriptor = descendant.get("/FontDescriptor").get_object()
    if (
        descendant.get("/Subtype") != "/CIDFontType2"
        or descendant.get("/CIDToGIDMap") != "/Identity"
        or not descriptor.get("/FontFile2")
    ):
        raise ValueError("The native caption font is not an embedded Identity font")
    cmap = (
        font.get("/ToUnicode").get_object().get_data().decode("ascii", errors="strict")
    )
    if len(cmap) > 256 * 1024:
        raise ValueError("The native caption font mapping is too broad")
    glyphs: dict[str, int] = {}

    def put(cid: int, codepoint: int) -> None:
        char = chr(codepoint)
        if char in glyphs and glyphs[char] != cid:
            raise ValueError("The native caption character has ambiguous glyphs")
        glyphs[char] = cid
        if len(glyphs) > 4096:
            raise ValueError("The native caption glyph map is too broad")

    for block in re.finditer(r"(\d+) beginbfchar\s*(.*?)\s*endbfchar", cmap, re.S):
        pairs = re.findall(r"<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>", block[2])
        if len(pairs) != int(block[1]):
            raise ValueError("Unsupported native caption Unicode mapping")
        for cid, codepoint in pairs:
            put(int(cid, 16), int(codepoint, 16))
    for block in re.finditer(r"(\d+) beginbfrange\s*(.*?)\s*endbfrange", cmap, re.S):
        ranges = re.findall(
            r"<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>\s*<([0-9A-Fa-f]{4})>", block[2]
        )
        if len(ranges) != int(block[1]):
            raise ValueError("Unsupported native caption Unicode range")
        for first, last, codepoint in ranges:
            start, end, unicode_start = (
                int(first, 16),
                int(last, 16),
                int(codepoint, 16),
            )
            if not 0 <= end - start <= 4096:
                raise ValueError("The native caption glyph range is too broad")
            for offset, cid in enumerate(range(start, end + 1)):
                put(cid, unicode_start + offset)
    widths: dict[int, float] = {}
    values = descendant.get("/W", [])
    if len(values) > 8192:
        raise ValueError("The native caption width table is too broad")
    i = 0
    while i < len(values):
        first, data = int(values[i]), values[i + 1]
        i += 2
        if isinstance(data, list):
            widths.update(
                {first + n: float(cast(Any, width)) for n, width in enumerate(data)}
            )
        else:
            last, width = int(data), float(values[i])
            i += 1
            if not 0 <= last - first <= 4096:
                raise ValueError("The native caption font widths are too broad")
            widths.update({cid: width for cid in range(first, last + 1)})
    default = float(descendant.get("/DW", 1000))
    metrics = {char: widths.get(cid, default) for char, cid in glyphs.items()}
    return (
        metrics,
        abs(float(descriptor.get("/Descent", 0))),
        float(descriptor.get("/Ascent", 1000)),
    )
