"""Exercise source-lead omission through model delivery, dispatch and publication."""

from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import (
    LegalSourceReviews,
    related_source_reviews_enabled,
)
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import CapabilityCall, EvidenceItem, OutcomeStatus, RunContext
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.llm.model_response import ModelResponse
from onyx.prompts.asv3.research import LEGAL_DEPARTMENT_RESEARCH
from onyx.prompts.asv3.tuned import TUNED_LEGAL_DEPARTMENT_RESEARCH
from tests.unit.onyx.asv3.test_experimental_workflow import terminal_registry
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    navigation,
    review,
    seen,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_native_model_adapter import (
    adaptive_tool_view,
    last_payload,
    model,
    native_action,
)
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
    user_payload,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record


def tuned_context() -> tuple[RunContext, EvidenceLedger, LegalSourceReviews]:
    context, ledger, reviews = setup_reviews()
    context.services.update(
        research_profile="normal",
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        lean_native_mode=True,
        evidence=ledger,
        legal_source_reviews=reviews,
        legal_source_navigation=navigation,
    )
    return context, ledger, reviews


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_source_review_activation_is_isolated(profile: str) -> None:
    context = RunContext(services={"research_profile": profile})
    assert related_source_reviews_enabled(context) is (profile == "experimental")
    context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    assert related_source_reviews_enabled(context)


def test_actual_delivery_blocks_unread_lead_without_an_extra_model_call() -> None:
    context, ledger, reviews = tuned_context()
    selected = model()
    selected.invoke.return_value = native_action(
        "submit_answer",
        {"answer": "The rule certainly applies [1].", "basis": "originals"},
    )
    registry = terminal_registry([])
    adapter = ResearchModel(selected, context, lean_native_mode=True)
    decision = adapter.decide(
        adaptive_tool_view(original_evidence=[full_record(ledger, 1)]).model_copy(
            update={"tools": registry.definitions(context)}
        )
    )
    assert last_payload(selected)["related_source_navigation"]
    state = reviews.view(context, ledger, {1})
    assert state["pending_lead_ids"]
    rows = cast(list[dict[str, JsonValue]], state["reviews"])
    assert rows[0]["available_original_citations"] == []
    assert rows[0]["anchor_evidence_numbers"] == [1]
    assert (
        reviews.publication_gap(
            cast(str, decision.calls[0].arguments["answer"]),
            adapter.last_call_id or "",
            context,
            ledger,
        )
        is not None
    )
    # Removing the anchor citation must not erase a previously delivered lead.
    assert (
        reviews.publication_gap(
            "The rule still certainly applies [3].",
            adapter.last_call_id or "",
            context,
            ledger,
        )
        is not None
    )
    assert selected.invoke.call_count == 1


@pytest.mark.parametrize("status", ["examined", "not_material"])
def test_a_title_cannot_be_used_to_close_or_exclude_the_candidate(status: str) -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    context.services["last_model_call_id"] = "law-call"
    observed: list[dict[str, JsonValue]] = []
    outcome = terminal_registry(observed).dispatch(
        CapabilityCall(
            name="submit_answer",
            arguments={
                "answer": "The ordinary rule applies [1].",
                "basis": "originals",
                "_related_source_reviews": [review(status=status, witnesses=[])],
            },
        ),
        context,
    )
    assert outcome.status == OutcomeStatus.INVALID
    assert outcome.data["invalid_related_source_review"] is True
    assert observed == []
    assert reviews.view(context, ledger, {1})["pending_lead_ids"]


def test_examined_effect_and_limitations_need_their_own_substantive_citations() -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    deliver(ledger, "answer-call", [1, 2])
    reviews.apply([review()], "answer-call", context, ledger)
    for answer in ("Ordinary rule [1].", "Ordinary rule [1].\n\n## Decision [2]"):
        gap = reviews.publication_gap(answer, "answer-call", context, ledger)
        assert gap is not None
        missing = cast(
            list[dict[str, JsonValue]], gap.data["unretained_examined_source_effects"]
        )
        assert missing[0]["missing_answer_passages"] == ["effect", "limitations"]
        assert missing[0]["unbound_evidence_numbers"] == [2]
    assessment = review()
    retained = f"{assessment['effect']} {assessment['limitations']}"
    assert (
        reviews.publication_gap(
            f"The ordinary rule [1]. {retained} [2].",
            "answer-call",
            context,
            ledger,
        )
        is None
    )
    assert reviews.publication_gap("Merhaba!", "answer-call", context, ledger) is None


def test_unavailable_original_needs_its_precise_standalone_gap() -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    gap_text = "Bu hükme bağlı kararın yürürlük tarihi doğrulanamadı."
    reviews.apply(
        [
            review(
                status="unresolved", source_role="unknown", witnesses=[], gap=gap_text
            )
        ],
        "law-call",
        context,
        ledger,
    )
    assert (
        reviews.publication_gap("Supported remainder [1].", "law-call", context, ledger)
        is not None
    )
    assert (
        reviews.publication_gap(
            f"Supported remainder [1].\n\n{gap_text}", "law-call", context, ledger
        )
        is None
    )


def test_tuned_policy_replaces_one_section_without_case_specific_direction() -> None:
    context, _, _ = tuned_context()
    instruction = ResearchModel(
        model(), context, lean_native_mode=True
    )._research_instruction()
    assert TUNED_LEGAL_DEPARTMENT_RESEARCH in instruction
    before, _, rest = LEGAL_DEPARTMENT_RESEARCH.partition(
        "related_source_navigation contains"
    )
    _, _, after = rest.partition("Assess the strongest source-supported argument")
    assert TUNED_LEGAL_DEPARTMENT_RESEARCH.startswith(before)
    assert TUNED_LEGAL_DEPARTMENT_RESEARCH.endswith(after)
    assert instruction.count("Act as a careful legal department") == 1
    assert "_related_source_reviews" in TUNED_LEGAL_DEPARTMENT_RESEARCH
    assert len(TUNED_LEGAL_DEPARTMENT_RESEARCH) < len(LEGAL_DEPARTMENT_RESEARCH) + 400
    for forbidden in ("241", "royalti", "2026/72", "VAKA", "Anayasa"):
        policy = TUNED_LEGAL_DEPARTMENT_RESEARCH[len(before) : -len(after)]
        assert forbidden not in policy


@pytest.mark.parametrize("omit_examined_source", [False, True])
def test_runtime_rejects_premature_answer_then_reuses_read_originals(
    monkeypatch: pytest.MonkeyPatch, omit_examined_source: bool
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    law, candidate = (str(source.id) for source in broker.sources)
    for index, identity in enumerate((law, candidate)):
        broker.chunks[identity] = replace(
            broker.chunks[identity],
            text="A permit is required."
            if index == 0
            else "The judgment restricts this rule to approved transactions.",
            heading_path=("Example Law", "MADDE 17")
            if index == 0
            else ("Court decision", "Disposition"),
            metadata={"document_type": "kanun" if index == 0 else "court_decision"},
        )
    nav: list[dict[str, JsonValue]] = [
        {
            "anchor_source_id": law,
            "article_no": "17",
            "qualifier": None,
            "candidates": [
                {
                    "source_id": candidate,
                    "name": "Court decision on Example Law article 17",
                    "candidate_role": "judicial_candidate",
                }
            ],
        }
    ]
    pages: list[str] = []

    def page(source_id: str, _context: RunContext, **_options: Any) -> Any:
        pages.append(source_id)
        source = next(item for item in broker.sources if str(item.id) == source_id)
        return source, [broker.chunks[source_id]], False

    def related(
        item: EvidenceItem, _context: RunContext
    ) -> dict[str, JsonValue] | None:
        return nav[0] if item.source_id == law else None

    monkeypatch.setattr(broker, "page", page)
    monkeypatch.setattr(broker, "related_sources_for_evidence", related)
    monkeypatch.setattr(broker, "related_source_navigation", lambda: nav)
    calls = 0

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        footer = user_payload(arguments["prompt"][-1])
        if calls == 1:
            return response(
                calls=[("read_source_range", {"source_id": law, "_language": "tr"})]
            )
        if calls == 2:
            assert (
                footer["related_source_navigation"][0]["candidates"][0]["source_id"]
                == candidate
            )
            return response(
                calls=[
                    (
                        "submit_answer",
                        {
                            "answer": "The permit is always required [1].",
                            "basis": "originals",
                        },
                    )
                ]
            )
        if calls == 3:
            assert footer["publication_gap"]["pending_related_source_review"] is True
            assert kwargs["state_container"].answer_tokens is None
            return response(calls=[("read_source_range", {"source_id": candidate})])
        if calls == 4:
            originals = {
                row["source_id"]: row for row in delivered_originals(arguments)
            }
            decision = originals[candidate]
            state = cast(
                list[dict[str, JsonValue]], footer["related_source_reviews"]["reviews"]
            )
            assessment = {
                "lead_id": state[0]["lead_id"],
                "status": "examined",
                "source_role": "operative_text",
                "effect": "The judgment has restricted scope.",
                "limitations": "Approval is decisive.",
                "witnesses": [
                    {
                        "citation": decision["citation"],
                        "start_char": 0,
                        "end_char": len(str(decision["text"])),
                    }
                ],
                "gap": "",
            }
            answer = (
                "A permit is required [1]."
                if omit_examined_source
                else "A permit is required [1]. The judgment has restricted scope. Approval is decisive. [2]."
            )
            return response(
                calls=[
                    (
                        "submit_answer",
                        {
                            "answer": answer,
                            "basis": "originals",
                            "_related_source_reviews": [assessment],
                        },
                    )
                ]
            )
        assert calls == 5 and omit_examined_source
        assert footer["publication_gap"]["unretained_examined_source_effects"]
        assert pages == [law, candidate]
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": "A permit is required [1]. The judgment has restricted scope. Approval is decisive. [2].",
                        "basis": "originals",
                    },
                )
            ]
        )

    selected.invoke.side_effect = scripted
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    runtime.run_asv3_loop(**kwargs)
    assert calls == (5 if omit_examined_source else 4)
    assert pages == [law, candidate]
    assert checkpoints[-1]["publication_status"] == "found"
    assert "restricted scope" in kwargs["state_container"].answer_tokens
    assert set(checkpoints[-1]["evidence"]["included"]) == {1, 2}
