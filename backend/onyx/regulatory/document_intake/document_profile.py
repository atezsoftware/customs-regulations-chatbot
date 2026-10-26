"""Ground document identity in its own header while retaining source coordinates."""

import re
from dataclasses import replace
from pathlib import Path

from onyx.regulatory.chunker import (
    _DOCUMENT_TITLE_TYPE_PATTERNS,
    DocumentMetadata,
    SourceBlock,
    _article_unit,
    _classify_block,
    _clean_inline_markdown,
    _fold_text,
    _infer_document_metadata,
)

_MONTH = (
    r"(?:ocak|subat|mart|nisan|mayis|haziran|temmuz|agustos|eylul|ekim|kasim|aralik)"
)
_DATE = rf"\d{{1,2}}\s+{_MONTH}\s+\d{{4}}"
_URL = r"(?:https?://)?(?:www\.)?[a-z0-9.-]+\.[a-z]{2,}/\S*"
_PRINT_HEADER = re.compile(
    rf"\d{{1,2}}[./]\d{{1,2}}[./]\d{{4}}\s+\d{{1,2}}:\d{{2}}\s+{_URL}", re.I
)
_PRINT_FOOTER = re.compile(rf"{_URL}\s+\d+/\d+", re.I)
_PAGE_MASTHEAD = re.compile(
    rf"(?:sayfa\s*:\s*\d+\s+resmi\s+gazete\s+{_DATE}\s*[–—-]\s*sayi\s*:\s*\d+|"
    rf"{_DATE}\s*[–—-]\s*sayi\s*:\s*\d+\s+resmi\s+gazete\s+sayfa\s*:\s*\d+)"
)
_PUBLICATION_LINE = re.compile(
    rf"(?:{_DATE}(?:\s+[a-z]+)?|resmi\s+gazete|sayi\s*:\s*\d+)"
)
_LABELED_NUMBER = re.compile(
    r"^(?:kanun|tebli[gğ]|karar|genelge|sirk[uü]ler|belge)\s+"
    r"(?:no\.?|numaras[ıi]|say[ıi]s[ıi])\s*[:：]?\s*"
    r"(?P<number>[A-ZÇĞİÖŞÜ]{1,10}[-:]\s*\d{4}/\d{1,6}|\d{1,6}(?:[-/]\d{1,6}){0,2})\s*$",
    re.I,
)
_OWN_HEADER_CODE = re.compile(r"\(([A-ZÇĞİÖŞÜ]{2,10}[-:]\s*\d{4}/\d{1,6})\)")
_PARENTHETICAL_CODES = re.compile(r"(?:\([^()\n]+\)\s*)+")
_TITLE_NUMBER = re.compile(r"^(\d{1,6})\s+say[ıi]l[ıi]\s+", re.I)
_BARE_TYPES = frozenset(
    {"kanun", "teblig", "yonetmelik", "karar", "genelge", "yonerge"}
)


def _print_chrome(block: SourceBlock) -> bool:
    if block.kind == "table":
        return False
    text = block.text.strip()
    return bool(
        _PRINT_HEADER.fullmatch(text)
        or _PRINT_FOOTER.fullmatch(text)
        or _PAGE_MASTHEAD.fullmatch(_fold_text(text))
    )


def intake_document_profile(
    *, blocks: list[SourceBlock], metadata: DocumentMetadata
) -> tuple[DocumentMetadata, frozenset[int]]:
    """Inspect a bounded header view; original text and offsets never change."""
    ignored = frozenset(block.block_id for block in blocks if _print_chrome(block))
    header: list[str] = []
    for block in blocks:
        if block.block_id in ignored:
            continue
        clean = _clean_inline_markdown(block.text)
        if (
            _classify_block(block)["kind"]
            in {
                "article",
                "appendix",
                "supplement",
                "section",
                "numbered_section",
                "dotted_section",
            }
            or _article_unit(clean, block.text) is not None
            or block.kind == "table"
        ):
            break
        header.append(block.text)
        if len(header) >= 60:
            break
    clean_header = [_clean_inline_markdown(line) for line in header]
    title = None
    title_end = None
    for index, line in enumerate(clean_header):
        folded = _fold_text(line)
        if folded in _BARE_TYPES or not any(
            pattern.search(folded) for _kind, pattern in _DOCUMENT_TITLE_TYPE_PATTERNS
        ):
            continue
        start = index
        while start > 0:
            previous = clean_header[start - 1]
            if (
                not previous
                or previous.upper() != previous
                or _fold_text(previous) in _BARE_TYPES
                or not re.search(r"[A-ZÇĞİÖŞÜ]", previous)
                or previous.endswith(":")
            ):
                break
            start -= 1
        title = " ".join(clean_header[start : index + 1])
        title_end = index
        break
    number = None
    for line in clean_header:
        labeled = _LABELED_NUMBER.fullmatch(line)
        if labeled:
            number = labeled["number"]
            break
    if number is None and title_end is not None:
        candidates = _OWN_HEADER_CODE.findall(clean_header[title_end])
        for line in clean_header[title_end + 1 : title_end + 4]:
            if _PARENTHETICAL_CODES.fullmatch(line) is None:
                break
            candidates.extend(_OWN_HEADER_CODE.findall(line))
        if len(set(candidates)) == 1:
            number = candidates[0]
    if number is None and title is not None:
        title_number = _TITLE_NUMBER.match(title)
        if title_number:
            number = title_number[1]
    unsafe_root = bool(ignored) or bool(
        blocks
        and _PUBLICATION_LINE.fullmatch(
            _fold_text(_clean_inline_markdown(blocks[0].text))
        )
    )
    view = "\n".join(header)
    inspected = _infer_document_metadata(
        path=Path("<memory>"),
        root_path=None,
        content=view,
        parser=metadata.parser,
        source_path_override=metadata.source_path,
    )
    return replace(
        metadata,
        title=title if title is not None else None if unsafe_root else metadata.title,
        document_number=number,
        document_date=inspected.document_date,
        file_date=None,
        document_type=inspected.document_type
        if unsafe_root or title is not None
        else metadata.document_type,
    ), ignored
