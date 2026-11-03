"""Fail-closed visibility checks for the closed native Word lead-in profile."""

import re
from collections.abc import Sequence

from bs4 import BeautifulSoup, NavigableString, Tag
from soupsieve import compile as compile_selector
from soupsieve import match as selector_matches
from soupsieve.util import SelectorSyntaxError

from onyx.regulatory.labeling.self_repeal import (
    NativeSourceStyles,
    require_native_visibility,
)

_PROFILE_CSS_PROPERTIES = frozenset(
    {
        "border",
        "border-bottom",
        "border-collapse",
        "border-left",
        "border-right",
        "border-top",
        "color",
        "display",
        "visibility",
        "font-family",
        "font-size",
        "font-style",
        "font-weight",
        "height",
        "layout-grid-mode",
        "letter-spacing",
        "line-height",
        "margin",
        "margin-bottom",
        "margin-left",
        "margin-right",
        "margin-top",
        "mso-style-link",
        "mso-style-name",
        "padding",
        "page",
        "page-break-after",
        "page-break-before",
        "panose-1",
        "size",
        "text-align",
        "text-autospace",
        "text-decoration",
        "text-indent",
        "vertical-align",
        "width",
        "word-wrap",
    }
)
_PROFILE_COLORS = frozenset(
    {
        "black",
        "blue",
        "red",
        "teal",
        "#5a5a5a",
        "#954f72",
        "windowtext",
        "inherit",
        "initial",
    }
)
_PROFILE_TAGS = frozenset(
    {
        "html",
        "body",
        "div",
        "p",
        "b",
        "strong",
        "span",
        "i",
        "em",
        "a",
        "br",
        "sup",
        "sub",
        "o:p",
    }
)
_PROFILE_HTML_COLORS = frozenset(
    {"#0563c1", "#954f72", "#5a5a5a", "black", "blue", "red", "teal"}
)


def lead_in_prefix_text_is_visible(
    node: NavigableString, styles: NativeSourceStyles
) -> bool:
    """A text node depends on its ancestors, never on unrelated sibling content."""
    parents = [parent for parent in node.parents if isinstance(parent, Tag)]
    return any(parent.name == "body" for parent in parents) and all(
        parent.name not in {"head", "style", "script", "noscript", "template"}
        and not parent.has_attr("hidden")
        and styles.computed(parent)["display"] != "none"
        and styles.computed(parent)["visibility"] == "visible"
        for parent in parents
    )


def article_lead_in_quotes_closed(text: str) -> bool:
    straight = curly = False
    for index, char in enumerate(text):
        if char not in "'‘’":
            continue
        # Turkish proper-name apostrophes are not quotation boundaries.
        if (
            char != "‘"
            and 0 < index < len(text) - 1
            and text[index - 1].isalnum()
            and text[index + 1].isalnum()
        ):
            continue
        if char == "'":
            straight = not straight
        elif char == "‘":
            if curly:
                return False
            curly = True
        elif not curly:
            return False
        else:
            curly = False
    return not straight and not curly


def _profile_declarations(text: str) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for declaration in re.sub(r"/\*.*?\*/", "", text, flags=re.S).split(";"):
        key, separator, value = declaration.partition(":")
        if not separator:
            continue
        key, value = (
            key.strip().casefold(),
            value.strip().casefold().removesuffix("!important").strip(),
        )
        if key not in _PROFILE_CSS_PROPERTIES or (
            key == "color" and value not in _PROFILE_COLORS
        ):
            raise ValueError("The structural source has unmodeled visibility CSS")
        values.append((key, value))
    return values


def _profile_layout_is_visible(
    tag: Tag, declarations: Sequence[tuple[str, str]]
) -> bool:
    for key, value in declarations:
        if key == "display" and value not in {"block", "inline", "inline-block"}:
            return False
        if key == "visibility" and value != "visible":
            return False
        if key == "font-weight" and value not in {"normal", "bold", "400", "700"}:
            return False
        if key in {"width", "height"}:
            return False
        if key.startswith("margin") or key in {
            "padding",
            "text-indent",
            "letter-spacing",
            "font-size",
            "line-height",
        }:
            if key == "line-height" and value == "normal":
                continue
            for token in value.split():
                number = re.fullmatch(
                    r"(-?(?:\d+(?:\.\d+)?|\.\d+))(pt|px|cm|%)?", token
                )
                if number is None:
                    return False
                amount, unit = float(number[1]), number[2]
                if unit == "%":
                    if key != "line-height" or not 50 <= amount <= 200:
                        return False
                    continue
                points = (
                    amount * {"pt": 1.0, "px": 0.75, "cm": 72 / 2.54, None: 1.0}[unit]
                )
                if key == "font-size" and not 6 <= points <= 48:
                    return False
                if key == "line-height" and not 6 <= points <= 100:
                    return False
                if key == "letter-spacing" and not -1 <= points <= 5:
                    return False
                if (key.startswith("margin") or key in {"padding", "text-indent"}) and (
                    not 0 <= points <= 80 or (points != 0 and tag.name != "p")
                ):
                    return False
        if key == "text-decoration" and value not in {"none", "underline"}:
            return False
        if (
            key.startswith("border")
            and key != "border-collapse"
            and value not in {"none", "0", "0pt", "0cm", "0px"}
        ):
            return False
        if key == "vertical-align" and value not in {
            "baseline",
            "top",
            "middle",
            "bottom",
            "super",
            "sub",
            "text-top",
            "text-bottom",
        }:
            return False
        if key == "font-family" and any(
            family.strip().strip("\"'")
            not in {
                "times new roman",
                "times",
                "calibri",
                "arial",
                "arial unicode ms",
                "new york",
                "serif",
                "sans-serif",
            }
            for family in value.split(",")
        ):
            return False
    return True


def require_article_lead_in_visibility(
    soup: BeautifulSoup, paragraphs: Sequence[Tag]
) -> None:
    """Fail closed outside the retained Word profile instead of guessing layout."""
    if soup.find(
        [
            "script",
            "iframe",
            "object",
            "embed",
            "template",
            "svg",
            "canvas",
            "video",
            "audio",
            "base",
        ]
    ) or any(
        any(str(key).casefold().startswith("on") for key in tag.attrs)
        for tag in soup.find_all(True)
    ):
        raise ValueError("The structural source contains active content")
    if any(
        "stylesheet"
        in {str(token).casefold() for token in link.get_attribute_list("rel")}
        for link in soup.find_all("link")
    ) or any(
        str(meta.get("http-equiv", "")).casefold() == "refresh"
        for meta in soup.find_all("meta")
    ):
        raise ValueError("The structural source has external styles or navigation")
    require_native_visibility(soup)
    NativeSourceStyles(soup)
    tags: dict[int, Tag] = {}
    for paragraph in paragraphs:
        for tag in (paragraph, *paragraph.parents, *paragraph.find_all(True)):
            if isinstance(tag, BeautifulSoup):
                continue
            if isinstance(tag, Tag):
                if (
                    tag.name not in _PROFILE_TAGS
                    or any(
                        tag.has_attr(key)
                        for key in (
                            "popover",
                            "bgcolor",
                            "background",
                            "color",
                            "inert",
                            "leftmargin",
                            "rightmargin",
                            "topmargin",
                            "bottommargin",
                            "marginwidth",
                            "marginheight",
                        )
                    )
                    or str(tag.get("aria-hidden", "")).casefold() == "true"
                    or str(tag.get("dir", "ltr")).casefold() != "ltr"
                    or any(
                        tag.has_attr(key)
                        and str(tag.get(key)).strip().casefold()
                        not in _PROFILE_HTML_COLORS
                        for key in ("text", "link", "vlink", "alink")
                    )
                ):
                    raise ValueError(
                        "The structural source has hidden or unsupported native ancestry"
                    )
                tags[id(tag)] = tag
    for style in soup.find_all("style"):
        css = (
            re.sub(r"/\*.*?\*/", "", style.get_text(), flags=re.S)
            .replace("<!--", "")
            .replace("-->", "")
        )
        for selectors, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
            declarations = _profile_declarations(body)
            if selectors.lstrip().startswith("@"):
                continue
            try:
                compile_selector(selectors)
            except SelectorSyntaxError:
                # NativeSourceStyles has already rejected all but invalid simple Word rules.
                continue
            if any(
                selector_matches(selectors, tag)
                and not _profile_layout_is_visible(tag, declarations)
                for tag in tags.values()
            ):
                raise ValueError("The structural source has unsupported native layout")
    for tag in soup.find_all(style=True):
        declarations = _profile_declarations(str(tag.get("style", "")))
        if id(tag) in tags and not _profile_layout_is_visible(tag, declarations):
            raise ValueError("The structural source has unsupported inline layout")
