"""Preserve explicit user questions without another planning model call."""

import re
from typing import TypedDict

_NUMBERED_QUESTION = re.compile(r"(?m)^[ \t]*(\d{1,3})[.)][ \t]+(?=\S)")


class RequestedDetermination(TypedDict):
    determination_id: str
    question_id: str
    question: str
    start_char: int
    end_char: int


def question_determinations(questions: list[str]) -> list[RequestedDetermination]:
    """Retain literal top-level interrogative clauses without naming expected laws."""
    result: list[RequestedDetermination] = []
    for question_index, text in enumerate(questions):
        boundaries: list[int] = []
        closing_quotes: list[str] = []
        brackets: list[str] = []
        interrogative = False
        for index, character in enumerate(text):
            if closing_quotes:
                if character == closing_quotes[-1]:
                    closing_quotes.pop()
                continue
            if character in {'"', "“", "«"}:
                closing_quotes.append({'"': '"', "“": "”", "«": "»"}[character])
            elif character in "([{":
                brackets.append({"(": ")", "[": "]", "{": "}"}[character])
            elif brackets and character == brackets[-1]:
                brackets.pop()
            elif not brackets and character in "?;":
                boundaries.append(index + 1)
                interrogative |= character == "?"
        # A trailing directive belongs to the last question, not a new legal outcome.
        ends = boundaries[:-1] if interrogative else []
        ends.append(len(text))
        start = 0
        for determination_index, end in enumerate(ends):
            fragment = text[start:end]
            trimmed_start = start + len(fragment) - len(fragment.lstrip())
            trimmed_end = end - len(fragment) + len(fragment.rstrip())
            if fragment.strip():
                result.append(
                    {
                        "determination_id": f"q{question_index}:d{determination_index}",
                        "question_id": f"q{question_index}",
                        "question": text[trimmed_start:trimmed_end],
                        "start_char": trimmed_start,
                        "end_char": trimmed_end,
                    }
                )
            start = end
    return result


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
