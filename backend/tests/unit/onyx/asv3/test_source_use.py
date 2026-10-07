"""Publication must carry uncited operative conditions without replaying research."""

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.source_metadata_transport import expand_source_metadata
from onyx.asv3.source_use import SourceUseIssue, SourceUseReviewer
from onyx.asv3.witnesses import original_witness_spans
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


def review_payload(answer: str, *, omitted: bool = True) -> dict[str, Any]:
    ledger, _context = original_ledger()
    item = ledger.get(3)
    assert item is not None
    units = assertion_inventory(answer)
    return {
        "examined_citations": [1, 2, 3],
        "reviewed_answer_unit_ids": [unit["unit_id"] for unit in units],
        "source_assessments": source_assessments(
            json.loads(
                ledger.serialize_records(
                    [1, 2, 3], max_chars=None, include_witness_spans=True
                )
            ),
            units,
        ),
        "issues": [
            {
                "kind": "omitted_condition",
                "answer_unit_ids": [units[0]["unit_id"]],
                "witnesses": [
                    {
                        "citation": 3,
                        "witness_id": original_witness_spans(3, item.text)[0][
                            "witness_id"
                        ],
                    }
                ],
                "detail": "The operative exception is absent.",
                "applicability": "The user's requested operation depends on this exception.",
            }
        ]
        if omitted
        else [],
    }


def source_assessments(
    records: list[dict[str, Any]], units: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    first = {row["source_id"]: row for row in records}
    results = []
    for source_id, first_row in first.items():
        row = next(
            (
                record
                for record in records
                if record["source_id"] == source_id
                and any(
                    record["citation"] in unit["evidence_numbers"] for unit in units
                )
            ),
            first_row,
        )
        witness = [
            {
                "citation": row["citation"],
                "witness_id": row["witness_spans"][0]["witness_id"],
            }
        ]
        bound = [
            unit["unit_id"]
            for unit in units
            if row["citation"] in unit["evidence_numbers"]
        ]
        results.append(
            {
                "source_id": source_id,
                "requirements": [
                    {
                        "detail": "The operative rule",
                        "applicability": "It applies to the supplied facts",
                        "witnesses": witness,
                        "answer_unit_ids": bound,
                        "status": "covered",
                    }
                ]
                if bound
                else [],
                "exclusion": None
                if bound
                else {
                    "detail": "The original excludes the supplied scope",
                    "witnesses": witness,
                },
            }
        )
    return results


@pytest.mark.parametrize("provider", ["vertex_ai", "openai", "anthropic"])
def test_uncited_originals_are_delivered_and_coordinator_identity_preserved(
    provider: str,
) -> None:
    ledger, context, model = setup_review()
    model.llm.config.model_provider = provider
    answer = "The operation is permitted [1]."
    cast(MagicMock, model.llm).invoke.return_value = text_response(
        review_payload(answer)
    )
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap(
        answer, "An operation with an exception", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issues = gap.data["source_use_gaps"]
    assert isinstance(issues, list)
    assert SourceUseIssue.model_validate(issues[0]).witnesses[0].citation == 3
    assert context.services["last_model_call_id"] == "coordinator"
    assert model.last_call_id is not None
    assert ledger.completely_delivered(model.last_call_id) == {1, 2, 3}
    assert (
        ledger.delivery_flow(model.last_call_id) == LLMFlow.ASV3_SOURCE_USE_REVIEW.value
    )
    assert "coordinator" != model.last_call_id
    before = cast(MagicMock, model.llm).invoke.call_count
    assert (
        reviewer.publication_gap(
            answer, "An operation with an exception", "coordinator"
        )
        == gap
    )
    assert cast(MagicMock, model.llm).invoke.call_count == before


@pytest.mark.parametrize("change", ["answer", "facts", "evidence"])
def test_changed_candidate_facts_or_delivered_originals_cannot_reuse_approval(
    change: str,
) -> None:
    ledger, _context, model = setup_review()
    answer = "The operation is permitted [1]."
    first = review_payload(answer, omitted=False)
    cast(MagicMock, model.llm).invoke.return_value = text_response(first)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap(answer, "supplied facts", "coordinator") is None
    if change == "answer":
        answer = "The operation is permitted only if proved [1]."
    next_result = review_payload(answer)
    current_call = "coordinator"
    if change == "evidence":
        current_call = "coordinator-next"
        coordinator_delivery(ledger, [1, 3], current_call)
        next_result["examined_citations"] = [1, 3]
        next_result["source_assessments"] = source_assessments(
            json.loads(
                ledger.serialize_records(
                    [1, 3], max_chars=None, include_witness_spans=True
                )
            ),
            assertion_inventory(answer),
        )
    cast(MagicMock, model.llm).invoke.return_value = text_response(next_result)
    gap = reviewer.publication_gap(
        answer, "changed facts" if change == "facts" else "supplied facts", current_call
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    assert cast(MagicMock, model.llm).invoke.call_count == 2


@pytest.mark.parametrize(
    "defect",
    [
        "invented_witness",
        "uncited_unread_witness",
        "missing_unit",
        "missing_original",
        "invented_unit",
    ],
)
def test_unbound_or_incomplete_review_cannot_approve_publication(defect: str) -> None:
    ledger, _context, model = setup_review()
    answer = "The operation is permitted [1].\n\nA separate outcome [2]."
    payload = review_payload(answer)
    current_call = "coordinator"
    if defect == "invented_witness":
        payload["issues"][0]["witnesses"][0]["witness_id"] = "invented"
    elif defect == "uncited_unread_witness":
        current_call = "coordinator-next"
        coordinator_delivery(ledger, [1, 2], current_call)
        payload["examined_citations"] = [1, 2]
    elif defect == "missing_unit":
        payload["reviewed_answer_unit_ids"].pop()
    elif defect == "missing_original":
        payload["examined_citations"].pop()
    else:
        payload["issues"][0]["answer_unit_ids"] = ["invented"]
    cast(MagicMock, model.llm).invoke.return_value = text_response(payload)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        answer, "facts", current_call
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE


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


def test_complete_text_and_source_dates_survive_shared_metadata_transport() -> None:
    ledger, context, model = setup_review()
    second = ledger.get(2)
    assert second is not None
    second.source_id = "distinct-source-1"
    second.chunk_id = "another-exact-chunk"
    second.metadata = {
        "title": "Exact instrument",
        "legal_dates": {"effective_start": "2025-01-01"},
    }
    ledger.add([second], context)
    coordinator_delivery(ledger, [2, 4], "coordinator-next")
    payload = review_payload("Result [2].", omitted=False)
    payload["examined_citations"] = [2, 4]
    payload["source_assessments"] = source_assessments(
        json.loads(
            ledger.serialize_records([2, 4], max_chars=None, include_witness_spans=True)
        ),
        assertion_inventory("Result [2]."),
    )
    cast(MagicMock, model.llm).invoke.return_value = text_response(payload)
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Result [2].", "facts", "coordinator-next"
        )
        is None
    )
    prompt = cast(MagicMock, model.llm).invoke.call_args.kwargs["prompt"]
    content = prompt[1].content
    raw = content if isinstance(content, str) else content[0].text
    data = json.loads(raw)
    records = expand_source_metadata(
        {
            "original_metadata_catalogue": data["original_evidence"],
            "original_source_metadata": data["original_source_metadata"],
        }
    )
    for record in records:
        citation = record["citation"]
        assert isinstance(citation, int)
        item = ledger.get(citation)
        assert item is not None and record["text"] == item.text
    metadata = next(row for row in records if row["citation"] == 4)["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["legal_dates"] == {"effective_start": "2025-01-01"}


@pytest.mark.parametrize(
    "defect", ["missing_source", "wrong_source", "empty_assessment", "unbound_covered"]
)
def test_source_assessment_cannot_skip_a_source_or_approve_an_uncited_requirement(
    defect: str,
) -> None:
    ledger, _context, model = setup_review()
    answer = "Operation permitted [1]."
    payload = review_payload(answer, omitted=False)
    if defect == "missing_source":
        payload["source_assessments"].pop()
    elif defect == "wrong_source":
        payload["source_assessments"][2]["exclusion"]["witnesses"] = payload[
            "source_assessments"
        ][0]["requirements"][0]["witnesses"]
    elif defect == "empty_assessment":
        payload["source_assessments"][0]["requirements"] = []
        payload["source_assessments"][0]["exclusion"] = None
    else:
        witness = payload["source_assessments"][2]["exclusion"]["witnesses"]
        payload["source_assessments"][2] = {
            "source_id": "distinct-source-3",
            "exclusion": None,
            "requirements": [
                {
                    "status": "covered",
                    "detail": "The applicable exception",
                    "applicability": "It changes this outcome",
                    "witnesses": witness,
                    "answer_unit_ids": payload["reviewed_answer_unit_ids"],
                }
            ],
        }
    cast(MagicMock, model.llm).invoke.return_value = text_response(payload)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        answer, "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE


def test_source_requirement_omission_reaches_native_repair_even_without_separate_issue() -> (
    None
):
    ledger, _context, model = setup_review()
    answer = "Operation permitted [1]."
    payload = review_payload(answer, omitted=False)
    witness = payload["source_assessments"][2]["exclusion"]["witnesses"]
    payload["source_assessments"][2] = {
        "source_id": "distinct-source-3",
        "exclusion": None,
        "requirements": [
            {
                "status": "omitted",
                "detail": "Retain the conditional exception",
                "applicability": "The triggering fact is not supplied",
                "witnesses": witness,
                "answer_unit_ids": payload["reviewed_answer_unit_ids"],
            }
        ],
    }
    cast(MagicMock, model.llm).invoke.return_value = text_response(payload)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        answer, "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    assert len(cast(list[Any], gap.data["source_use_gaps"])) == 1
    assert (
        cast(list[Any], gap.data["source_use_gaps"])[0]["witnesses"][0]["citation"] == 3
    )


@pytest.mark.parametrize("defect", ["excluded_used_source", "uncited_neighbor"])
def test_requirement_coverage_cannot_borrow_another_answer_blocks_citation(
    defect: str,
) -> None:
    ledger, _context, model = setup_review()
    answer = "Operation permitted [1].\n\nA separate obligation [2]."
    payload = review_payload(answer, omitted=False)
    first = payload["source_assessments"][0]
    requirement = first["requirements"][0]
    if defect == "excluded_used_source":
        first["requirements"] = []
        first["exclusion"] = {
            "detail": "Outside the scope",
            "witnesses": requirement["witnesses"],
        }
    else:
        requirement["answer_unit_ids"] = payload["reviewed_answer_unit_ids"]
    cast(MagicMock, model.llm).invoke.return_value = text_response(payload)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        answer, "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE


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
                    "source_assessments": source_assessments(records, units),
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
    assert counts == {"native": 3, "review": 2}
    assert page.call_count == 2
    assert (
        kwargs["state_container"].answer_tokens
        == "Tamir sonucu [[1]](https://example.test/law-0); değiştirme sonucu ve uygulama koşulu [[2]](https://example.test/law-1)."
    )
