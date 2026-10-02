import pytest

from onyx.asv3.scenario import initial_questions


@pytest.mark.parametrize("marker", [".", ")"])
def test_numbered_questions_keep_alternatives_and_multiline_text(marker: str) -> None:
    request = (
        "The machine is unused; this is a scenario fact.\n\n"
        f"1{marker} Does the exemption apply?\n"
        "What if permission is missing?\n"
        f"2{marker} Which amount applies?\n"
        f"3{marker} How is release and later settlement completed?"
    )
    assert initial_questions(request) == [
        "Does the exemption apply?\nWhat if permission is missing?",
        "Which amount applies?",
        "How is release and later settlement completed?",
    ]


@pytest.mark.parametrize(
    "question_text",
    [
        "Does article 16/1-b apply?",
        "2022. Historical context\n2024. A later amendment",
        "1. One list\n2. Its continuation\n1. Another list\n2. Its continuation",
        "1. One question\n3. A skipped number",
        "> 1. A quoted question\n> 2. Another quote\nCompare these sources.",
    ],
)
def test_ambiguous_numbering_keeps_the_full_request(question_text: str) -> None:
    assert initial_questions(question_text) == [question_text]
