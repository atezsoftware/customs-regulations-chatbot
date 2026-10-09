from __future__ import annotations

import importlib
import json
from unittest.mock import MagicMock

from onyx.asv3.candidate_audit import CandidateAudit, CandidateAuditRecord
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.guardrails_v3 import (
    GuardrailsV3ReviewOutcome,
    NormalizedFinding,
)
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse, Usage


def _controller():
    return importlib.import_module("onyx.asv3.guardrails_v3_controller")


def _finding() -> NormalizedFinding:
    return NormalizedFinding(
        finding_id="finding-1",
        outcome_ids=["scope"],
        dimensions=["m1_coverage"],
        reason="omitted",
        review_question_ids=["m1.scope.coverage"],
        affected_answer_unit_ids=["s1"],
        draft_hash="a" * 64,
    )


def _llm(answer: str) -> MagicMock:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="vertex_ai",
        model_name="gemini-3.8-flash",
        temperature=0,
        max_input_tokens=1_048_576,
    )
    llm.invoke.return_value = ModelResponse(
        id="repair",
        created="0",
        choice=Choice(message=Message(content=answer)),
        usage=Usage(
            prompt_tokens=30,
            completion_tokens=10,
            total_tokens=40,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )
    return llm


def test_delivered_evidence_has_priority_and_patch_is_accepted_only_after_clean_recheck() -> (
    None
):
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="1", text="A ve B şarttır.")], context
    )
    ledger.record_delivery(
        "candidate", "asv3_coordinator", [{"citation": 1, "text": "A ve B şarttır."}]
    )
    initial = GuardrailsV3ReviewOutcome(
        review_completed=True, repair_requested=True, findings=[_finding()]
    )
    recheck = GuardrailsV3ReviewOutcome(review_completed=True)
    llm = _llm("A ve B şarttır [1].")

    outcome = _controller().finalize_guardrails_v3(
        candidate_answer="A şarttır [1].",
        initial_review=initial,
        ledger=ledger,
        context=context,
        repair_llm=llm,
        recheck=lambda *_args, **_kwargs: recheck,
    )

    assert outcome.answer == "A ve B şarttır [1]."
    assert outcome.action == "patch_delivered_evidence"
    assert outcome.repair_applied is True
    assert outcome.recheck_completed is True
    llm.invoke.assert_called_once()
    assert llm.invoke.call_args.kwargs["use_streaming"] is False
    assert llm.invoke.call_args.kwargs["provider_compatibility_attempts"] == 1


def test_failed_recheck_keeps_original_answer_after_only_one_repair_attempt() -> None:
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="1", text="A ve B şarttır.")], context
    )
    ledger.record_delivery(
        "candidate", "asv3_coordinator", [{"citation": 1, "text": "A ve B şarttır."}]
    )
    llm = _llm("A ve B şarttır [1].")
    outcome = _controller().finalize_guardrails_v3(
        candidate_answer="A şarttır [1].",
        initial_review=GuardrailsV3ReviewOutcome(
            review_completed=True, repair_requested=True, findings=[_finding()]
        ),
        ledger=ledger,
        context=context,
        repair_llm=llm,
        recheck=lambda *_args, **_kwargs: GuardrailsV3ReviewOutcome(
            review_completed=False, failure_reason="jev_invalid_review_response"
        ),
    )

    assert outcome.answer == "A şarttır [1]."
    assert outcome.repair_applied is False
    assert outcome.failure_reason == "recheck_failed"
    llm.invoke.assert_called_once()


def test_partial_evidence_is_not_sufficient_for_a_repair() -> None:
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="1", text="A ve B şarttır.")], context
    )
    ledger.record_delivery(
        "candidate", "asv3_coordinator", [{"citation": 1, "text": "A ve"}]
    )
    llm = _llm("A ve B şarttır [1].")

    outcome = _controller().finalize_guardrails_v3(
        candidate_answer="A şarttır [1].",
        initial_review=GuardrailsV3ReviewOutcome(
            review_completed=True, repair_requested=True, findings=[_finding()]
        ),
        ledger=ledger,
        context=context,
        repair_llm=llm,
        recheck=lambda *_args, **_kwargs: GuardrailsV3ReviewOutcome(
            review_completed=True
        ),
    )

    assert outcome.answer == "A şarttır [1]."
    assert outcome.action == "disclose_gap"
    assert outcome.failure_reason == "repair_evidence_undelivered"
    llm.invoke.assert_not_called()


def test_repair_cannot_remove_every_original_citation() -> None:
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="1", text="A ve B şarttır.")], context
    )
    ledger.record_delivery(
        "candidate", "asv3_coordinator", [{"citation": 1, "text": "A ve B şarttır."}]
    )
    llm = _llm("A ve B şarttır.")

    outcome = _controller().finalize_guardrails_v3(
        candidate_answer="A şarttır [1].",
        initial_review=GuardrailsV3ReviewOutcome(
            review_completed=True, repair_requested=True, findings=[_finding()]
        ),
        ledger=ledger,
        context=context,
        repair_llm=llm,
        recheck=lambda *_args, **_kwargs: GuardrailsV3ReviewOutcome(
            review_completed=True
        ),
    )

    assert outcome.answer == "A şarttır [1]."
    assert outcome.failure_reason == "invalid_repair_citations"


def test_repair_uses_only_the_configured_gemini_flash_model() -> None:
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="1", text="A ve B şarttır.")], context
    )
    ledger.record_delivery(
        "candidate", "asv3_coordinator", [{"citation": 1, "text": "A ve B şarttır."}]
    )
    llm = _llm("A ve B şarttır [1].")
    llm.config = llm.config.model_copy(update={"model_name": "gemini-3-pro"})

    outcome = _controller().finalize_guardrails_v3(
        candidate_answer="A şarttır [1].",
        initial_review=GuardrailsV3ReviewOutcome(
            review_completed=True, repair_requested=True, findings=[_finding()]
        ),
        ledger=ledger,
        context=context,
        repair_llm=llm,
        recheck=lambda *_args, **_kwargs: GuardrailsV3ReviewOutcome(
            review_completed=True
        ),
    )

    assert outcome.answer == "A şarttır [1]."
    assert outcome.failure_reason == "repair_unavailable"
    llm.invoke.assert_not_called()


def test_material_method_two_gap_runs_one_scoped_search_before_repair() -> None:
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="1", text="Ceza uygulanır.")],
        context,
    )
    ledger.record_delivery(
        "candidate", "asv3_coordinator", [{"citation": 1, "text": "Ceza uygulanır."}]
    )
    calls: list[dict[str, object]] = []

    def search(args: dict[str, object], _context: RunContext) -> ToolOutcome:
        calls.append(args)
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Scoped authority search completed",
            evidence=[
                EvidenceItem(
                    source_id="court",
                    chunk_id="2",
                    text="Mahkeme kararı ceza dayanağını iptal eder.",
                )
            ],
        )

    broker = MagicMock()
    broker.search_adapter = search
    llm = _llm("Ceza uygulanır; kararın etkisi ayrıca değerlendirilir [1][2].")
    initial = GuardrailsV3ReviewOutcome(
        review_completed=True,
        repair_requested=True,
        findings=[
            NormalizedFinding(
                finding_id="finding-court",
                outcome_ids=[],
                dimensions=["d1"],
                reason="missing",
                review_question_ids=["m2.d1.applicability", "m2.d1.treatment"],
                affected_answer_unit_ids=["s1"],
                draft_hash="a" * 64,
            )
        ],
    )

    outcome = _controller().finalize_guardrails_v3(
        question="Usulsüzlük cezasına ilişkin güncel hukukî sonucu nedir?",
        candidate_answer="Ceza uygulanır [1].",
        initial_review=initial,
        ledger=ledger,
        context=context,
        repair_llm=llm,
        broker=broker,
        recheck=lambda *_args, **_kwargs: GuardrailsV3ReviewOutcome(
            review_completed=True
        ),
    )

    assert outcome.action == "focused_search"
    assert outcome.repair_applied is True
    assert ledger.completely_delivered("guardrails-v3-focused-search") == {2}
    assert ledger.completely_delivered("guardrails-v3-repair") == {1, 2}
    repair_payload = json.loads(llm.invoke.call_args.args[0][0].content)
    assert repair_payload["evidence"] == [
        {
            "citation": 1,
            "source_id": "law",
            "chunk_id": "1",
            "text": "Ceza uygulanır.",
        },
        {
            "citation": 2,
            "source_id": "court",
            "chunk_id": "2",
            "text": "Mahkeme kararı ceza dayanağını iptal eder.",
        },
    ]
    assert calls == [
        {
            "query": "Usulsüzlük cezasına ilişkin güncel hukukî sonucu nedir?",
            "mode": "hybrid",
            "coverage_item": "m2.d1",
            "evidence_target": "review_gap:m2.d1",
            "expand_query": True,
        }
    ]


def test_excluded_candidate_is_hydrated_and_delivered_before_one_repair() -> None:
    context, ledger = RunContext(timeout_seconds=120), EvidenceLedger()
    audit = CandidateAudit(context, request="Kapsam?")
    audit.record(
        CandidateAuditRecord(
            search_run_id="search-1",
            candidate_id="law:7",
            source_id="law",
            chunk_id="7",
            lane="regulatory",
            mode="keyword",
            status="excluded",
            reason="rerank_below_selection",
            hydration_locator={
                "document_id": "law",
                "chunk_ind": 7,
                "regulatory_chunk_id": "canonical-7",
                "source_type": "user_file",
                "semantic_identifier": "Kanun",
                "blurb": "Madde 7",
            },
        )
    )
    broker = MagicMock()
    broker.hydrate_search_evidence.return_value = [
        EvidenceItem(source_id="law", chunk_id="7", text="A ve B şarttır.")
    ]
    llm = _llm("A ve B şarttır [1].")

    outcome = _controller().finalize_guardrails_v3(
        candidate_answer="A şarttır.",
        initial_review=GuardrailsV3ReviewOutcome(
            review_completed=True, repair_requested=True, findings=[_finding()]
        ),
        ledger=ledger,
        context=context,
        repair_llm=llm,
        candidate_audit=audit,
        broker=broker,
        recheck=lambda *_args, **_kwargs: GuardrailsV3ReviewOutcome(
            review_completed=True
        ),
    )

    assert outcome.action == "recover_audited_candidate"
    assert outcome.repair_applied is True
    assert broker.hydrate_search_evidence.call_args.args[0].document_id == "law"
    assert ledger.completely_delivered("guardrails-v3-recovery") == {1}
