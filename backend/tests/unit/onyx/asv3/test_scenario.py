import pytest

from onyx.asv3.scenario import initial_questions, question_determinations


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


@pytest.mark.parametrize(
    "question",
    [
        "Yeni izin ücretsiz mi? Önceki ödemenin iadesi mümkün mü?",
        "May the goods be released; which security and subsequent settlement apply?",
        "La licence est-elle gratuite ? Le paiement antérieur peut-il être remboursé ?",
    ],
)
def test_independent_parts_keep_literal_spans_and_parent(question: str) -> None:
    parts = question_determinations([question])
    assert len(parts) == 2
    assert [part["determination_id"] for part in parts] == ["q0:d0", "q0:d1"]
    assert all(
        part["question_id"] == "q0"
        and question[part["start_char"] : part["end_char"]] == part["question"]
        for part in parts
    )


@pytest.mark.parametrize(
    "question",
    [
        'Does the statement "May I export; refund?" prove permission?',
        "Does the rule (including 'A; B?') apply?",
        "Compare the sources; preserve their versions; use only the corpus.",
    ],
)
def test_quoted_punctuation_and_noninterrogative_prose_remain_intact(
    question: str,
) -> None:
    assert [part["question"] for part in question_determinations([question])] == [
        question
    ]


def test_trailing_corpus_directive_does_not_become_a_legal_question() -> None:
    parts = question_determinations(
        ["May it enter? Which amount applies?\nUse only the supplied corpus."]
    )
    assert len(parts) == 2
    assert parts[-1]["question"].endswith("Use only the supplied corpus.")
