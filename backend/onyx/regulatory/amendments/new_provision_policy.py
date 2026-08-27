"""Deterministic safety policy for brand-new top-level provisions."""

import re

_SUBORDINATE_ADDITION_RE = re.compile(
    r"maddes(?:ine|inin|inde|inden).*?"
    r"(?:fıkra|fikra|bent|cümle|cumle|ibare|paragraf).*?"
    r"(?:eklenmiştir|eklenmistir|ilave edilmiştir|ilave edilmistir)",
    re.IGNORECASE | re.DOTALL,
)
_TOP_LEVEL_ADDITION_PATTERNS = (
    re.compile(
        r"(?:geçici\s+|gecici\s+)?madde\s+\d+[a-zçğıöşü]*\b.{0,160}?"
        r"(?:eklenmiştir|eklenmistir|ilave edilmiştir|ilave edilmistir)",
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        r"aşağıdaki\s+(?:yeni\s+)?(?:geçici\s+|gecici\s+)?madde\b.{0,200}?"
        r"(?:eklenmiştir|eklenmistir|ilave edilmiştir|ilave edilmistir)",
        re.IGNORECASE | re.DOTALL,
    ),
)


def explicitly_adds_top_level_provision(instruction_text: str) -> bool:
    """Return true only when the text itself adds a top-level article.

    Adding a paragraph, clause, sentence, or phrase to an existing article is
    an update to that article and must never enter the new-article path.
    """

    normalized = " ".join(instruction_text.split())
    if _SUBORDINATE_ADDITION_RE.search(normalized):
        return False
    return any(pattern.search(normalized) for pattern in _TOP_LEVEL_ADDITION_PATTERNS)
