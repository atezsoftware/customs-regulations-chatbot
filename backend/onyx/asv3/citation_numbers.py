"""Read the numeric citation formats supported by the chat renderer."""

import re

_CITATION_PATTERN = re.compile(
    r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])"
)


def extract_citation_numbers(text: str) -> tuple[int, ...]:
    """Preserve grouped source IDs throughout delivery, checking and publication."""
    return tuple(
        dict.fromkeys(
            int(number)
            for match in _CITATION_PATTERN.finditer(text)
            for number in re.findall(r"\d+", match.group())
        )
    )
