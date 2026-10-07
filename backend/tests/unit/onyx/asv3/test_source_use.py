"""Draft isolation, immutable coverage, delivery and cache boundaries."""

import json
from collections import Counter
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.source_metadata_transport import expand_source_metadata
from onyx.asv3.source_use import SourceUseIssue, SourceUseReviewer
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response
from tests.unit.onyx.asv3.test_runtime import delivered_originals, response, setup_run


def coordinator_delivery(
    ledger: EvidenceLedger, numbers: list[int], call_id: str = "coordinator"
) -> None:
    ledger.record_delivery(
        call_id,
        "asv3_coordinator",
        json.loads(ledger.serialize_records(numbers, max_chars=None)),
    )


def setup_review() -> tuple[EvidenceLedger, RunContext, ResearchModel]:
    ledger, context = original_ledger()
    context.services.update(
        asv3_workflow_variant="asv3_tuned",
        research_profile="normal",
        last_model_call_id="coordinator",
    )
    coordinator_delivery(ledger, [1, 2, 3])
    return ledger, context, ResearchModel(scripted_model(), context)


def model_payload(arguments: dict[str, Any]) -> dict[str, Any]:
    content = arguments["prompt"][1].content
    return json.loads(content if isinstance(content, str) else content[0].text)


def inventory_for(records: list[dict[str, Any]]) -> dict[str, Any]:
    row = records[0]
    return {
        "examined_citations": [r["citation"] for r in records],
        "requirements": [
            {
                "detail": "The operation depends on the operative exception.",
                "applicability": "Retain the conditional branch when its decisive fact is unknown.",
                "witnesses": [
                    {
                        "citation": row["citation"],
                        "witness_id": row["witness_spans"][0]["witness_id"],
                    }
                ],
            }
        ],
    }


def resolutions_for(payload: dict[str, Any]) -> list[dict[str, Any]]:
    units = payload["answer_units"]
    rows = []
    for req in payload["retained_requirements"]:
        numbers = {w["citation"] for w in req["witnesses"]}
        bound = [u["unit_id"] for u in units if numbers <= set(u["evidence_numbers"])]
        rows.append(
            {
                "requirement_id": req["requirement_id"],
                "status": "covered" if bound else "omitted",
                "answer_unit_ids": bound or [units[0]["unit_id"]],
                "explanation": "" if bound else "The operative detail is absent.",
            }
        )
    return rows


def review_for(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "examined_citations": payload["required_evidence_numbers"],
        "reviewed_answer_unit_ids": [u["unit_id"] for u in payload["answer_units"]],
        "resolutions": resolutions_for(payload),
        "issues": [],
    }


def scripted_reviews(
    model: ResearchModel, *, inventory_mutator: Any = None, review_mutator: Any = None
) -> Counter[str]:
    counts: Counter[str] = Counter()

    def script(**arguments: Any) -> Any:
        schema = (arguments.get("structured_response_format") or {}).get(
            "json_schema"
        ) or {}
        payload = model_payload(arguments)
        if (
            schema.get("name")
            or (
                "SourceUseReview" if "answer_units" in payload else "SourceUseInventory"
            )
        ) == "SourceUseInventory":
            counts["inventory"] += 1
            result = inventory_for(payload["original_evidence"])
            if inventory_mutator:
                inventory_mutator(result, payload)
        else:
            assert (schema.get("name") or "SourceUseReview") == "SourceUseReview"
            counts["review"] += 1
            result = review_for(payload)
            if review_mutator:
                review_mutator(result, payload)
        return text_response(result)

    cast(MagicMock, model.llm).invoke.side_effect = script
    return counts


@pytest.mark.parametrize("provider", ["vertex_ai", "openai", "anthropic"])
def test_complete_draft_blind_inventory_and_coordinator_identity(provider: str) -> None:
    ledger, context, model = setup_review()
    model.llm.config.model_provider = provider
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    answer = "The operation is allowed [2]."
    conversation = [
        {"role": "assistant", "content": "Earlier legal conclusion."},
        {"role": "user", "content": "A decisive supplied fact."},
    ]
    gap = reviewer.publication_gap(
        answer, "Actual request", "coordinator", conversation=conversation
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    assert (
        SourceUseIssue.model_validate(cast(list[Any], gap.data["source_use_gaps"])[0])
        .witnesses[0]
        .citation
        == 1
    )
    calls = cast(MagicMock, model.llm).invoke.call_args_list
    blind = model_payload(calls[0].kwargs)
    assert not {
        "answer_units",
        "draft",
        "retained_requirements",
        "issues",
    }.intersection(blind)
    assert blind["conversation"] == [conversation[1]]
    assert {r["citation"] for r in blind["original_evidence"]} == {1, 2, 3}
    for row in blind["original_evidence"]:
        original = ledger.get(row["citation"])
        assert original is not None and row["text"] == original.text
    assert context.services["last_model_call_id"] == "coordinator"
    assert model.last_call_id is not None
    assert ledger.completely_delivered(model.last_call_id) == {1, 2, 3}
    assert (
        ledger.delivery_flow(model.last_call_id) == LLMFlow.ASV3_SOURCE_USE_REVIEW.value
    )
    assert (
        reviewer.publication_gap(
            answer, "Actual request", "coordinator", conversation=conversation
        )
        == gap
    )
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize("change", ["answer", "facts", "evidence", "user_conversation"])
def test_inventory_reuse_is_independent_of_answer_but_bound_to_originals_and_facts(
    change: str,
) -> None:
    ledger, context, model = setup_review()
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    assert (
        reviewer.publication_gap("Allowed [1].", "Actual facts", "coordinator") is None
    )
    answer = "Allowed with proof [1]." if change == "answer" else "Allowed [1]."
    call_id = "coordinator"
    if change == "evidence":
        call_id = "next"
        coordinator_delivery(ledger, [1, 3], call_id)
    reviewer.publication_gap(
        answer,
        "Changed facts" if change == "facts" else "Actual facts",
        call_id,
        conversation=[{"role": "user", "content": "Another actual fact."}]
        if change == "user_conversation"
        else None,
    )
    assert counts == {"inventory": 1 if change == "answer" else 2, "review": 2}
    assert context.services["last_model_call_id"] == "coordinator"


@pytest.mark.parametrize(
    "defect", ["missing_citation", "unknown_witness", "duplicate_requirement"]
)
def test_inventory_failures_cannot_be_cached_or_approve(defect: str) -> None:
    ledger, context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        if defect == "missing_citation":
            result["examined_citations"].pop()
        elif defect == "unknown_witness":
            result["requirements"][0]["witnesses"][0]["witness_id"] = "invented"
        else:
            result["requirements"].append(result["requirements"][0])

    counts = scripted_reviews(model, inventory_mutator=mutate)
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap("Allowed [1].", "facts", "coordinator")
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 2}
    assert not reviewer._inventories
    assert reviewer.publication_gap("Allowed [1].", "facts", "coordinator") == gap
    assert counts == {"inventory": 2}
    assert context.services["last_model_call_id"] == "coordinator"


@pytest.mark.parametrize(
    "defect",
    [
        "missing_resolution",
        "renamed_requirement",
        "duplicate_resolution",
        "invented_unit",
        "missing_unit",
        "borrowed_citation",
        "missing_original",
        "assumed_exclusion",
    ],
)
def test_review_cannot_drop_retained_rules_or_borrow_neighboring_citations(
    defect: str,
) -> None:
    ledger, context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        if defect == "missing_resolution":
            result["resolutions"] = []
        elif defect == "renamed_requirement":
            row["requirement_id"] = "invented"
        elif defect == "duplicate_resolution":
            result["resolutions"].append(row)
        elif defect == "invented_unit":
            row["answer_unit_ids"] = ["invented"]
        elif defect == "missing_unit":
            result["reviewed_answer_unit_ids"].pop()
        elif defect == "missing_original":
            result["examined_citations"].pop()
        elif defect == "borrowed_citation":
            row["answer_unit_ids"] = [payload["answer_units"][-1]["unit_id"]]
        else:
            row.update(
                status="not_applicable",
                scenario_quote="The decisive condition is excluded.",
            )

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Allowed [1].\n\nAnother outcome [2].", "Actual facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}
    assert context.services["last_model_call_id"] == "coordinator"


def test_literal_user_exclusion_is_not_inferred_from_assistant_context() -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="not_applicable",
            scenario_quote="Excluded status",
            answer_unit_ids=[],
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap(
        "Allowed [1].",
        "facts",
        "coordinator",
        conversation=[{"role": "assistant", "content": "Excluded status"}],
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert (
        reviewer.publication_gap("Allowed [1].", "Excluded status", "coordinator")
        is None
    )
    assert counts == {"inventory": 2, "review": 3}


def test_new_issues_need_delivered_witnesses_and_current_units() -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        result["issues"] = [
            {
                "kind": "unsupported_claim",
                "answer_unit_ids": [payload["answer_units"][0]["unit_id"]],
                "detail": "Exact unsupported effect",
                "applicability": "Actual requested result",
                "witnesses": [{"citation": 100, "source_quote": "invented"}],
            }
        ]

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Allowed [1].", "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}


def test_failed_review_is_reused_without_approval_and_changed_answers_are_rechecked() -> (
    None
):
    ledger, context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["resolutions"][0]["requirement_id"] = "invented"

    counts = scripted_reviews(model, review_mutator=mutate)
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap("Allowed [1].", "facts", "coordinator")
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert reviewer.publication_gap("Allowed [1].", "facts", "coordinator") == gap
    assert counts == {"inventory": 1, "review": 2}
    changed = reviewer.publication_gap(
        "Allowed with proof [1].", "facts", "coordinator"
    )
    assert changed and changed.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 4}
    assert context.services["last_model_call_id"] == "coordinator"


def test_shared_metadata_and_full_text_survive_both_independent_phases() -> None:
    ledger, context, model = setup_review()
    item = ledger.get(2)
    assert item is not None
    item.source_id = "distinct-source"
    item.chunk_id = "another-original"
    item.metadata = {
        "title": "Actual source",
        "legal_dates": {"effective_start": "2025-01-01"},
    }
    ledger.add([item], context)
    coordinator_delivery(ledger, [2, 4], "next")
    scripted_reviews(model)
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Result [2] [4].", "facts", "next"
        )
        is None
    )
    for call in cast(MagicMock, model.llm).invoke.call_args_list:
        data = model_payload(call.kwargs)
        records = expand_source_metadata(
            {
                "original_metadata_catalogue": data["original_evidence"],
                "original_source_metadata": data["original_source_metadata"],
            }
        )
        for record in records:
            citation = record["citation"]
            assert isinstance(citation, int)
            original = ledger.get(citation)
            assert original is not None
            assert original.text == record["text"]
        metadata = next(row for row in records if row["citation"] == 4)["metadata"]
        assert isinstance(metadata, dict)
        assert metadata["legal_dates"] == {"effective_start": "2025-01-01"}


@pytest.mark.parametrize(
    "variant,profile,depth",
    [
        ("standard", "normal", 0),
        ("standard", "experimental", 0),
        ("standard", "deep", 0),
        ("asv3_tuned", "normal", 1),
    ],
)
def test_protected_workflows_and_workers_do_not_add_a_review(
    variant: str, profile: str, depth: int
) -> None:
    ledger, context, model = setup_review()
    context.services.update(asv3_workflow_variant=variant, research_profile=profile)
    context.depth = depth
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Legal answer [1].", "facts", "coordinator"
        )
        is None
    )
    cast(MagicMock, model.llm).invoke.assert_not_called()


def test_social_dialogue_does_not_add_a_review() -> None:
    ledger, _context, model = setup_review()
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Merhaba!", "Selam", "coordinator"
        )
        is None
    )
    cast(MagicMock, model.llm).invoke.assert_not_called()


def test_batched_publication_gaps_preserve_all_existing_acquisition_bindings() -> None:
    from onyx.asv3.models import ToolOutcome
    from onyx.asv3.source_use import combine_source_publication_gaps

    gaps = [
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Named source",
            data={
                "named_authority_gaps": [{"unit_id": "first"}],
                "instruction": "Use its own original",
            },
        ),
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Referenced original",
            data={
                "unread_cited_statute_references": [{"article": "7"}],
                "instruction": "Read the material reference",
            },
        ),
        ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Related source",
            data={
                "pending_related_source_review": True,
                "unread_related_sources": [{"lead_id": "known"}],
            },
        ),
    ]
    combined = combine_source_publication_gaps(gaps)
    assert combined and combined.status == OutcomeStatus.PARTIAL
    for gap in gaps:
        for key, value in gap.data.items():
            if key != "instruction":
                assert combined.data[key] == value
    assert len(cast(list[Any], combined.data["publication_check_instructions"])) == 3


def test_runtime_repairs_a_witnessed_omission_without_repeating_source_acquisition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import runtime

    kwargs, broker, selected, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    initial_script = selected.invoke.side_effect
    page = MagicMock(wraps=broker.page)
    monkeypatch.setattr(broker, "page", page)
    counts: Counter[str] = Counter()

    def script(**arguments: Any) -> Any:
        schema = (arguments.get("structured_response_format") or {}).get(
            "json_schema"
        ) or {}
        if schema.get("name") == "SourceUseInventory":
            counts["inventory"] += 1
            return text_response(
                inventory_for(model_payload(arguments)["original_evidence"])
            )
        if schema.get("name") == "SourceUseReview":
            counts["review"] += 1
            content = arguments["prompt"][1].content
            raw = content if isinstance(content, str) else content[0].text
            payload = json.loads(raw)
            records = payload["original_evidence"]
            units = payload["answer_units"]
            issues = []
            if counts["review"] == 1:
                original = records[-1]
                issues = [
                    {
                        "kind": "omitted_condition",
                        "answer_unit_ids": [units[0]["unit_id"]],
                        "witnesses": [
                            {
                                "citation": original["citation"],
                                "witness_id": original["witness_spans"][0][
                                    "witness_id"
                                ],
                            }
                        ],
                        "detail": "The material implementation condition must be carried.",
                        "applicability": "It affects the user's requested operation.",
                    }
                ]
            return text_response(
                {
                    "examined_citations": [row["citation"] for row in records],
                    "reviewed_answer_unit_ids": [row["unit_id"] for row in units],
                    "resolutions": resolutions_for(payload),
                    "issues": issues,
                }
            )
        counts["native"] += 1
        if counts["native"] <= 2:
            return initial_script(**arguments)
        assert counts["native"] == 3
        assert "source_use_gaps" in str(arguments["prompt"])
        originals = delivered_originals(arguments)
        assert len(originals) == 2
        return response("Tamir sonucu [1]; değiştirme sonucu ve uygulama koşulu [2].")

    selected.invoke.side_effect = script
    runtime.run_asv3_loop(**kwargs)
    assert counts == {"native": 3, "inventory": 1, "review": 2}
    assert page.call_count == 2
    assert (
        kwargs["state_container"].answer_tokens
        == "Tamir sonucu [[1]](https://example.test/law-0); değiştirme sonucu ve uygulama koşulu [[2]](https://example.test/law-1)."
    )
