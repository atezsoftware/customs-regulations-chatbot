from __future__ import annotations

import importlib
import json
from typing import Any

import httpx

from onyx.asv3.candidate_audit import CandidateAudit, CandidateAuditRecord
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate


def _module() -> Any:
    return importlib.import_module("onyx.asv3.guardrails_v3")


def _packet() -> Any:
    context = RunContext(run_id="v3-review", scope={"corpus": "authorized"})
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="law-a",
                chunk_id="1",
                text="İstisna, başvuru ile birlikte A ve B şartları sağlanırsa uygulanır.",
                question_ids=["q0"],
            ),
            EvidenceItem(
                source_id="law-b",
                chunk_id="2",
                text="İade için ayrıca süresi içinde ayrı bir talep gerekir.",
                question_ids=["q1"],
            ),
        ],
        context,
    )
    outcomes = OutcomeMap(
        ["İstisna şartları?", "İade usulü?"], context, factual_context="A ve B"
    )
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {
                        "outcome_id": "exception",
                        "question_ids": ["q0"],
                        "detail": "İstisna şartları",
                    },
                    {
                        "outcome_id": "refund",
                        "question_ids": ["q1"],
                        "detail": "İade usulü",
                    },
                ],
                "conditions": [
                    {
                        "condition_id": "both_conditions",
                        "outcome_ids": ["exception"],
                        "detail": "A ve B birlikte gerekir",
                        "witnesses": [{"citation": 1, "start_char": 0, "end_char": 20}],
                    }
                ],
                "resolutions": [
                    {
                        "outcome_id": "exception",
                        "status": "supported",
                        "condition_ids": ["both_conditions"],
                        "evidence_numbers": [1],
                    },
                    {
                        "outcome_id": "refund",
                        "status": "unresolved",
                        "gap": "İade talebinin süresi netleştirilemedi.",
                    },
                ],
            }
        ),
        ledger,
    )
    audit = CandidateAudit(context, request="İstisna ve iade usulü")
    audit.record(
        CandidateAuditRecord(
            search_run_id="search-1",
            candidate_id="law-b:2",
            source_id="law-b",
            chunk_id="2",
            lane="regulatory",
            mode="keyword",
            status="delivered",
            reason="hydrated_and_delivered",
            outcome_ids=["refund"],
        )
    )
    built = _module().build_frozen_review_packet(
        question="İstisna ve iade usulü nedir?",
        candidate_answer="İstisna için A ve B birlikte gerekir [1]. İade ayrıca değerlendirilmelidir.",
        outcome_map=outcomes,
        ledger=ledger,
        candidate_audit=audit,
    )
    assert built.packet is not None
    return built.packet


def test_frozen_packet_keeps_required_evidence_and_stable_hashes() -> None:
    packet = _packet()

    assert packet.version == 1
    assert [row.citation for row in packet.evidence] == [1, 2]
    assert packet.outcome_view["unassessed_outcome_ids"] == []
    assert packet.draft_hash == packet.candidate_answer_hash
    assert len(packet.packet_hash) == 64
    assert [row.outcome_ids for row in packet.candidate_audit] == [["refund"]]
    assert all(row.section_id.startswith("s") for row in packet.answer_units)


def test_required_packet_overflow_fails_open_without_silent_source_drop() -> None:
    context = RunContext(run_id="oversize", scope={"corpus": "authorized"})
    ledger = EvidenceLedger()
    ledger.add([EvidenceItem(source_id="law", text="x" * 300)], context)
    outcomes = OutcomeMap(["Kapsam?"], context)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {"outcome_id": "scope", "question_ids": ["q0"], "detail": "Kapsam"}
                ],
                "conditions": [
                    {
                        "condition_id": "rule",
                        "outcome_ids": ["scope"],
                        "detail": "Tam metin gerekir",
                        "witnesses": [{"citation": 1, "start_char": 0, "end_char": 20}],
                    }
                ],
            }
        ),
        ledger,
    )

    built = _module().build_frozen_review_packet(
        question="Kapsam?",
        candidate_answer="Kural uygulanır [1].",
        outcome_map=outcomes,
        ledger=ledger,
        candidate_audit=None,
        max_state_chars=100,
    )

    assert built.packet is None
    assert built.failure_reason == "review_packet_too_large"


def test_missing_outcome_map_is_a_defect_only_for_substantive_research() -> None:
    module = _module()
    ledger = EvidenceLedger()

    substantive = module.build_frozen_review_packet(
        question="Kapsam?",
        candidate_answer="Aday cevap.",
        outcome_map=None,
        ledger=ledger,
        candidate_audit=None,
    )
    exempt = module.build_frozen_review_packet(
        question="Merhaba",
        candidate_answer="Merhaba.",
        outcome_map=None,
        ledger=ledger,
        candidate_audit=None,
        requires_research=False,
    )

    assert substantive.failure_reason == "outcome_map_missing"
    assert exempt.failure_reason == "review_exempt"


def test_combined_jev_review_uses_exact_independent_questions_and_normalizes_findings() -> (
    None
):
    packet = _packet()

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "jev-latest"
        assert "m1.exception.evidence" in payload["questions"]
        assert "m2.d3.applicability" in payload["questions"]
        assert "m2.d3.treatment" in payload["questions"]
        assert all(
            question["type"] in {"choice", "noul"}
            for question in payload["questions"].values()
        )
        answers: dict[str, dict[str, object]] = {}
        for question_id, question in payload["questions"].items():
            if question["type"] == "noul":
                answers[question_id] = {"type": "noul", "noul": 0.04}
            elif question_id.endswith(".evidence"):
                answers[question_id] = {"type": "choice", "choice": "adequate"}
            elif question_id.endswith(".coverage"):
                answers[question_id] = {"type": "choice", "choice": "addressed"}
            elif question_id.endswith(".application"):
                answers[question_id] = {"type": "choice", "choice": "correct"}
            elif question_id.endswith(".conditions"):
                answers[question_id] = {"type": "choice", "choice": "preserved"}
            elif question_id.endswith(".applicability"):
                answers[question_id] = {"type": "choice", "choice": "not_applicable"}
            else:
                answers[question_id] = {"type": "choice", "choice": "adequate"}
        return httpx.Response(
            200,
            json={
                "model": "jev-2026-10-01",
                "answers": answers,
                "usage": {"input_tokens": 123, "output_tokens": 17},
            },
        )

    outcome = _module().review_frozen_packet(
        packet,
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(handler),
        timeout_seconds=5,
    )

    assert outcome.review_completed is True
    assert outcome.actual_model == "jev-2026-10-01"
    assert outcome.input_tokens == 123
    assert outcome.output_tokens == 17
    assert outcome.findings == []
    assert outcome.failure_reason is None


def test_malformed_jev_response_is_fail_open_not_clean_review() -> None:
    outcome = _module().review_frozen_packet(
        _packet(),
        api_key="test-typesafe-key",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200, json={"model": "jev-latest", "answers": {}}
            )
        ),
        timeout_seconds=5,
    )

    assert outcome.review_completed is False
    assert outcome.findings == []
    assert outcome.failure_reason == "jev_invalid_review_response"


def test_method_two_not_applicable_does_not_create_a_finding() -> None:
    packet = _packet()
    module = _module()

    finding = module.normalize_review_findings(
        packet,
        {
            "m2.d3.applicability": "relevant",
            "m2.d3.treatment": "partial",
        },
    )
    not_applicable = module.normalize_review_findings(
        packet,
        {
            "m2.d3.applicability": "not_applicable",
            "m2.d3.treatment": "missing",
        },
    )

    assert len(finding) == 1
    assert finding[0].dimensions == ["d3"]
    assert set(finding[0].outcome_ids) == {"exception", "refund"}
    assert not not_applicable
