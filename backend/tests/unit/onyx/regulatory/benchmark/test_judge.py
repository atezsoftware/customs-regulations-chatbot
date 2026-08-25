import json
from typing import cast
from unittest.mock import MagicMock, patch

from onyx.llm.models import ReasoningEffort
from onyx.regulatory.benchmark.judge import (
    _SYSTEM_PROMPT,
    _build_judge_payload,
    judge_benchmark_answer,
)


def test_judge_payload_preserves_complete_long_candidate_answer() -> None:
    section_seven = "## 7. İzinli Gönderici"
    candidate_answer = ("A" * 15_000) + section_seven

    payload = _build_judge_payload(
        question_snapshot={"prompt": "Question"},
        candidate_answer=candidate_answer,
        cited_sources=[],
        citation_recall=None,
        citation_precision=None,
    )

    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["candidate_answer"] == candidate_answer
    assert section_seven in serialized


def test_judge_payload_still_bounds_untrusted_large_fields() -> None:
    payload = _build_judge_payload(
        question_snapshot={"reference_answer": "R" * 60_000},
        candidate_answer="A" * 110_000,
        cited_sources=[{"text_excerpt": "S" * 7_000}] * 60,
        citation_recall=0.5,
        citation_precision=0.25,
    )

    bounded_answer = payload["candidate_answer"]
    bounded_question = cast(dict[str, object], payload["question"])
    bounded_sources = cast(list[dict[str, object]], payload["candidate_cited_sources"])
    assert isinstance(bounded_answer, str)
    assert isinstance(bounded_question, dict)
    assert isinstance(bounded_sources, list)

    bounded_reference = bounded_question["reference_answer"]
    first_source = bounded_sources[0]
    assert isinstance(bounded_reference, str)
    assert isinstance(first_source, dict)
    bounded_excerpt = first_source["text_excerpt"]
    assert isinstance(bounded_excerpt, str)

    assert len(bounded_answer) == 100_000
    assert len(bounded_reference) == 50_000
    assert len(bounded_sources) == 50
    assert len(bounded_excerpt) == 6_000


def test_judge_treats_reference_expectations_as_non_authoritative() -> None:
    normalized_prompt = " ".join(_SYSTEM_PROMPT.split())

    assert "evaluation expectations, not legal authorities" in normalized_prompt
    assert "Exact regulatory source text controls" in normalized_prompt
    assert "do not penalize a candidate for following" in normalized_prompt
    assert "do not credit a candidate merely for repeating" in normalized_prompt
    assert "selected expected citation does not prove" in normalized_prompt
    assert "mark that assessment unverifiable" in normalized_prompt
    assert "reference as an expected coverage target only" in normalized_prompt
    assert "whose supplied text supports the expected material proposition" in (
        normalized_prompt
    )
    assert "that is consistent with the controlling evidence" in normalized_prompt


def test_judge_has_enough_output_budget_for_complete_structured_review() -> None:
    llm = MagicMock()
    expected_result = MagicMock()

    with patch(
        "onyx.regulatory.benchmark.judge.generate_structured",
        return_value=expected_result,
    ) as generate:
        result = judge_benchmark_answer(
            llm,
            question_snapshot={"prompt": "Question"},
            candidate_answer="Answer",
            cited_sources=[],
            citation_recall=None,
            citation_precision=None,
        )

    assert result is expected_result
    assert generate.call_args.kwargs["max_tokens"] == 12_000
    assert generate.call_args.kwargs["reasoning_effort"] is ReasoningEffort.LOW
    assert generate.call_args.kwargs["max_attempts"] == 3
