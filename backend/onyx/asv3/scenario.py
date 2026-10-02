"""Preserve explicit user questions without another planning model call."""

import re

_NUMBERED_QUESTION = re.compile(r"(?m)^[ \t]*(\d{1,3})[.)][ \t]+(?=\S)")


def initial_questions(request: str) -> list[str]:
    """Split one consecutive numbered list; keep ambiguous requests intact."""
    matches = list(_NUMBERED_QUESTION.finditer(request))
    if len(matches) < 2 or [int(match[1]) for match in matches] != list(
        range(1, len(matches) + 1)
    ):
        return [request]
    boundaries = [match.start() for match in matches[1:]] + [len(request)]
    return [
        request[match.end() : end].strip()
        for match, end in zip(matches, boundaries, strict=True)
    ]
