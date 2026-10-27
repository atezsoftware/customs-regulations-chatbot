"""Recognize a plain article title only beside an actual article definition."""

import re
from dataclasses import replace

from onyx.regulatory.chunker import (
    SourceBlock,
    _article_unit,
    _classify_block,
    _clean_inline_markdown,
    _is_probable_standalone_heading,
)
from onyx.regulatory.document_intake.text import _mask_code

_OPERATIVE_END = re.compile(
    r"\b(?:uygulanır|yapılır|düzenlenir|bulunur|saklanır|sunulur|verilir|sağlanır|"
    r"edilir|eder|girer|yürütür|zorundadır|gerekir|ödenir|alınır|olmalıdır|olur|sayılır|kapsar)$",
    re.I,
)


def promote_plain_article_titles(
    blocks: list[SourceBlock], *, source_text: str | None = None
) -> list[SourceBlock]:
    masked = _mask_code(source_text)[0] if source_text is not None else None
    return [
        replace(block, kind="heading")
        if (
            index + 1 < len(blocks)
            and block.kind == "paragraph"
            and _classify_block(block)["kind"] == "paragraph"
            and _classify_block(blocks[index + 1])["kind"] == "article"
            and len(block.text) <= 90
            and _OPERATIVE_END.search(block.text) is None
            and (masked is None or masked[block.start_char : block.end_char].strip())
            and _article_unit(_clean_inline_markdown(block.text), block.text) is None
            and _is_probable_standalone_heading(block.text)
        )
        else block
        for index, block in enumerate(blocks)
    ]
