from __future__ import annotations

import importlib
import json
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage


def _review_module() -> Any:
    return importlib.import_module("onyx.asv3.jev_answer_review")


def _evidence() -> list[Any]:
    module = _review_module()
    return [
        module.ReviewEvidence(
            citation=1,
            source_id="law-1",
            text="Muafiyet yalnız A ve B şartlarının birlikte sağlanması halinde uygulanır.",
            metadata={"article_no": "5"},
        )
    ]


def _repair_llm(content: str = "Düzeltilmiş cevap [1]") -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=1_048_576,
    )
    llm.invoke.return_value = ModelResponse(
        id="repair-1",
        created="0",
        choice=Choice(message=Message(content=content)),
        usage=Usage(
            completion_tokens=12,
            prompt_tokens=80,
            total_tokens=92,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    return llm


def _jev_transport(
    *, repair_probability: float, response_model: str = "jev-latest"
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://api.typesafe.ai/v1/systemone")
        assert request.headers["authorization"] == "Bearer test-typesafe-key"
        payload = json.loads(request.content)
        assert payload["model"] == "jev-latest"
        assert "candidate_answer" in payload["state"]
        assert set(payload["questions"]) == {
            "repair_required",
            "unsupported_material_claim",
            "missing_requested_scope",
            "citation_mismatch",
            "condition_loss",
        }
        answers = {
            name: {"type": "noul", "noul": 0.02} for name in payload["questions"]
        }
        answers["repair_required"]["noul"] = repair_probability
        if repair_probability >= 0.5:
            answers["condition_loss"]["noul"] = 0.91
        return httpx.Response(
            200,
            json={
                "model": response_model,
                "answers": answers,
                "usage": {"input_tokens": 120, "output_tokens": 5},
            },
        )

    return httpx.MockTransport(handler)


def test_clean_jev_review_keeps_candidate_without_repair_call() -> None:
    module = _review_module()
    repair_llm = _repair_llm()

    outcome = module.review_and_repair_answer(
        question="Muafiyet şartları nelerdir?",
        candidate_answer="A ve B birlikte gerekir [1].",
        evidence=_evidence(),
        repair_llm=repair_llm,
        api_key="test-typesafe-key",
        transport=_jev_transport(
            repair_probability=0.04, response_model="jev-2026-09-15"
        ),
    )

    assert outcome.answer == "A ve B birlikte gerekir [1]."
    assert outcome.review_completed is True
    assert outcome.repair_requested is False
    assert outcome.repair_applied is False
    assert outcome.review_input_tokens == 120
    assert outcome.review_output_tokens == 5
    repair_llm.invoke.assert_not_called()


def test_jev_repair_decision_invokes_gemini_once_and_records_usage() -> None:
    module = _review_module()
    repair_llm = _repair_llm()

    outcome = module.review_and_repair_answer(
        question="Muafiyet şartları nelerdir?",
        candidate_answer="Yalnız A gerekir [1].",
        evidence=_evidence(),
        repair_llm=repair_llm,
        api_key="test-typesafe-key",
        transport=_jev_transport(repair_probability=0.88),
    )

    assert outcome.answer == "Düzeltilmiş cevap [1]"
    assert outcome.review_completed is True
    assert outcome.repair_requested is True
    assert outcome.repair_applied is True
    assert outcome.defects == ["condition_loss"]
    assert outcome.review_scores["repair_required"] == pytest.approx(0.88)
    assert outcome.review_scores["condition_loss"] == pytest.approx(0.91)
    assert outcome.repair_input_tokens == 80
    assert outcome.repair_output_tokens == 12
    repair_llm.invoke.assert_called_once()
    invoke = repair_llm.invoke.call_args
    assert invoke.kwargs["use_streaming"] is False
    assert invoke.kwargs["provider_compatibility_attempts"] == 1


@pytest.mark.parametrize("failure", ["timeout", "malformed", "missing_key"])
def test_unavailable_jev_fails_open_without_repair(failure: str) -> None:
    module = _review_module()
    repair_llm = _repair_llm()

    def handler(_request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("slow")
        return httpx.Response(200, json={"model": "jev-latest", "answers": {}})

    outcome = module.review_and_repair_answer(
        question="Muafiyet şartları nelerdir?",
        candidate_answer="Özgün cevap [1].",
        evidence=_evidence(),
        repair_llm=repair_llm,
        api_key=None if failure == "missing_key" else "test-typesafe-key",
        transport=httpx.MockTransport(handler),
    )

    assert outcome.answer == "Özgün cevap [1]."
    assert outcome.review_completed is False
    assert outcome.repair_requested is False
    assert outcome.repair_applied is False
    assert outcome.failure_reason is not None
    repair_llm.invoke.assert_not_called()


@pytest.mark.parametrize(
    "repair_content",
    ["Yeni ve kayıtsız kaynak [99]", "Kaynak işaretleri tamamen kaldırıldı"],
)
def test_invalid_repair_citations_fail_open_to_original(repair_content: str) -> None:
    module = _review_module()
    repair_llm = _repair_llm(repair_content)

    outcome = module.review_and_repair_answer(
        question="Muafiyet şartları nelerdir?",
        candidate_answer="Özgün cevap [1].",
        evidence=_evidence(),
        repair_llm=repair_llm,
        api_key="test-typesafe-key",
        transport=_jev_transport(repair_probability=0.95),
    )

    assert outcome.answer == "Özgün cevap [1]."
    assert outcome.repair_requested is True
    assert outcome.repair_applied is False
    assert outcome.failure_reason == "invalid_repair_citations"
    repair_llm.invoke.assert_called_once()


def test_failed_gemini_repair_keeps_original_answer() -> None:
    module = _review_module()
    repair_llm = _repair_llm()
    repair_llm.invoke.side_effect = RuntimeError("provider unavailable")

    outcome = module.review_and_repair_answer(
        question="Muafiyet şartları nelerdir?",
        candidate_answer="Özgün cevap [1].",
        evidence=_evidence(),
        repair_llm=repair_llm,
        api_key="test-typesafe-key",
        transport=_jev_transport(repair_probability=0.95),
    )

    assert outcome.answer == "Özgün cevap [1]."
    assert outcome.review_completed is True
    assert outcome.repair_requested is True
    assert outcome.repair_applied is False
    assert outcome.failure_reason == "repair_failed"
    repair_llm.invoke.assert_called_once()


def test_missing_citable_evidence_skips_both_external_calls() -> None:
    module = _review_module()
    repair_llm = _repair_llm()

    def unexpected_transport(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("JEV must not run without citable evidence")

    outcome = module.review_and_repair_answer(
        question="Muafiyet şartları nelerdir?",
        candidate_answer="Kanıtsız aday cevap.",
        evidence=[],
        repair_llm=repair_llm,
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(unexpected_transport),
    )

    assert outcome.answer == "Kanıtsız aday cevap."
    assert outcome.review_completed is False
    assert outcome.failure_reason == "review_evidence_unavailable"
    repair_llm.invoke.assert_not_called()
