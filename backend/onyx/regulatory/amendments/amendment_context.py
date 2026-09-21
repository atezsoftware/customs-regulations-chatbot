"""The context one amendment's instructions share with one another.

An amendment is segmented into instructions that are then matched and drafted
one at a time, so each instruction reaches the model stripped of everything the
rest of the document said. That loses real meaning: the article naming the
source ("Aynı Tebliğ'in ..."), the article stating when the whole amendment
takes effect, and the definitions an instruction leans on all live in *other*
articles. This module carries that document-level context alongside each
instruction, without letting it become something the instruction applies.
"""

import re
from dataclasses import dataclass

from onyx.regulatory.amendments.structural_target import unquoted_text

# An amendment's own commencement article speaks about the amendment itself
# ("Bu Tebliğ ... yürürlüğe girer"), which is what separates it from a quoted
# replacement body that happens to describe some other text's entry into force.
_SELF_REFERENCE_RE = re.compile(
    r"\b(?:bu|işbu|isbu)\s+"
    r"(?:tebliğ|teblig|yönetmelik|yonetmelik|karar|kanun|genelge|yönerge|yonerge)",
    re.IGNORECASE,
)
_COMMENCEMENT_RE = re.compile(
    r"y[üu]r[üu]rl[üu][ğg]e\s+gir",
    re.IGNORECASE,
)
_ARTICLE_BOUNDARY_RE = re.compile(
    r"(?=(?:^|\n)\s*(?:GEÇİCİ\s+)?MADDE\s*\d+)",
    re.IGNORECASE,
)

# A commencement article is a sentence or two. Anything longer than this is a
# block the split failed to separate, and only its opening is worth quoting.
_MAX_COMMENCEMENT_CHARS = 800
# The background exists to be read in full for every instruction; a bound keeps
# an unusually long amendment (or a mis-pasted whole regulation) from crowding
# the candidates and the instruction itself out of the prompt.
_MAX_BACKGROUND_CHARS = 24000


def _bounded(text: str, limit: int) -> str:
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}\n[truncated]"


def commencement_provisions(raw_text: str) -> list[str]:
    """Articles of this amendment that state when it takes effect."""

    blocks = [
        block
        for block in _ARTICLE_BOUNDARY_RE.split(unquoted_text(raw_text))
        if block.strip()
    ]
    if not blocks:
        blocks = [raw_text]
    found: list[str] = []
    for block in blocks:
        if not _COMMENCEMENT_RE.search(block):
            continue
        if not _SELF_REFERENCE_RE.search(block):
            continue
        statement = _bounded(block, _MAX_COMMENCEMENT_CHARS)
        if statement not in found:
            found.append(statement)
    return found


@dataclass(frozen=True)
class AmendmentContext:
    """What every instruction of one amendment may read but must not apply."""

    background: str
    commencement: tuple[str, ...]

    def prompt_section(self) -> str:
        parts = [
            "Full text of the amendment these instructions were taken from. "
            "It is background for interpreting the instruction you were given "
            "— resolving what a source, a defined term or a cross-reference "
            "means, and what date applies. Apply ONLY the instruction(s) "
            "listed above; never apply a change that appears here but is not "
            "in them.\n\n"
            f"{self.background}"
        ]
        if self.commencement:
            parts.append(
                "This amendment states its own entry into force in the "
                "following article(s). It governs every instruction that does "
                "not state a date of its own:\n\n" + "\n\n".join(self.commencement)
            )
        return "\n\n".join(parts)


def build_amendment_context(raw_text: str) -> AmendmentContext | None:
    background = _bounded(raw_text, _MAX_BACKGROUND_CHARS)
    if not background:
        return None
    return AmendmentContext(
        background=background,
        commencement=tuple(commencement_provisions(raw_text)),
    )
