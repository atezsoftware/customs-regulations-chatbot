"""Draft isolation, immutable coverage, delivery and cache boundaries."""

import json
from collections import Counter
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from onyx.asv3.citation_numbers import strip_citation_markers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import OutcomeStatus, RunContext
from onyx.asv3.source_metadata_transport import expand_source_metadata
from onyx.asv3.source_use import (
    SourceUseCoverage,
    SourceUseIssue,
    SourceUseRequirement,
    SourceUseResolution,
    SourceUseReviewer,
    user_fact_spans,
)
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.asv3.test_citation_contract import original_ledger
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    seen,
    setup_reviews,
)
from tests.unit.onyx.asv3.test_model_adapter import scripted_model, text_response
from tests.unit.onyx.asv3.test_runtime import delivered_originals, response, setup_run
from tests.unit.onyx.asv3.test_tuned_focused_reads import judicial_chunk
from tests.unit.onyx.asv3.test_tuned_source_followthrough import tuned_context


def coordinator_delivery(
    ledger: EvidenceLedger, numbers: list[int], call_id: str = "coordinator"
) -> None:
    ledger.record_delivery(
        call_id,
        "asv3_coordinator",
        json.loads(ledger.serialize_records(numbers, max_chars=None)),
    )


@pytest.mark.parametrize("kind", ["kanun", "yönetmelik", "tebliğ"])
@pytest.mark.parametrize(
    "barrier", [None, "article", "qualifier", "date", "derived", "truncated"]
)
def test_uncited_sibling_navigation_requires_the_same_canonical_provision(
    kind: str, barrier: str | None
) -> None:
    initial, context, model = setup_review()
    originals = [initial.get(number) for number in (1, 2, 3)]
    for number in (1, 2):
        original = originals[number - 1]
        assert original is not None
        assert original.search_doc is not None
        assert original.chunk_id is not None
        original.search_doc.metadata["regulatory_chunk_id"] = original.chunk_id
        original.metadata.update(
            document_type=kind, heading_path=["Source title", "MADDE 17"]
        )
    second = originals[1]
    assert second is not None
    if barrier == "article":
        second.metadata["heading_path"] = ["Source title", "MADDE 18"]
    elif barrier == "qualifier":
        second.metadata["heading_path"] = ["Source title", "GEÇİCİ MADDE 17"]
    elif barrier == "date":
        second.metadata["read_as_of_date"] = "2025-01-01"
    elif barrier == "derived":
        second.metadata["derived"] = True
    elif barrier == "truncated":
        second.metadata["truncated"] = True
    ledger = EvidenceLedger()
    for original in originals:
        assert original is not None
        ledger.add([original], context)
    context.services["evidence"] = ledger
    coordinator_delivery(ledger, [1, 2, 3])
    scripted_reviews(model)
    SourceUseReviewer(model, ledger).publication_gap(
        "The primary rule applies [2].\n\nA different provision [3].",
        "Unknown qualifying fact",
        "coordinator",
    )
    payload = model_payload(cast(MagicMock, model.llm).invoke.call_args_list[-1].kwargs)
    candidates = payload["application_candidates"]
    if barrier is None:
        assert candidates[0]["answer_unit_ids"] == [
            payload["answer_units"][0]["unit_id"]
        ]
    else:
        assert candidates == []


def test_a_correct_detail_does_not_allow_an_unchecked_summary() -> None:
    ledger, _context, model = setup_review()

    def omit_summary(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        row["coverage"] = row["coverage"][1:]
        row["answer_unit_ids"] = row["answer_unit_ids"][1:]

    counts = scripted_reviews(model, review_mutator=omit_summary)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Unconditional summary [1].\n\nConditional detail [1].",
        "Decisive fact unknown",
        "coordinator",
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}


def test_local_misapplication_survives_an_aggregate_positive_resolution_without_retry() -> (
    None
):
    ledger, _context, model = setup_review()

    def negative_summary(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        binding = result["resolutions"][0]["coverage"][0]
        binding.update(
            status="misapplied",
            explanation="This summary applies the effect without its unknown qualifying fact.",
        )

    counts = scripted_reviews(model, review_mutator=negative_summary)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Unconditional summary [1].\n\nConditional detail [1].",
        "Decisive fact unknown",
        "coordinator",
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
    assert len(issues) == 1 and issues[0]["kind"] == "inconsistent_application"
    payload = model_payload(cast(MagicMock, model.llm).invoke.call_args_list[-1].kwargs)
    assert issues[0]["answer_unit_ids"] == [payload["answer_units"][0]["unit_id"]]
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize("status", ["covered", "misapplied"])
def test_grouped_bindings_keep_every_actual_application(status: str) -> None:
    ledger, _context, model = setup_review()

    def group(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        bindings = row["coverage"]
        binding = bindings[0]
        binding["additional_answer_unit_ids"] = [
            b["answer_unit_id"] for b in bindings[1:]
        ]
        binding["status"] = status
        if status == "misapplied":
            binding.update(
                explanation="Both applications omit the same decisive qualification.",
                witnesses=[],
            )
        row["coverage"] = [binding]

    counts = scripted_reviews(model, review_mutator=group)
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap(
        "The same application [1].\n\nThe same application [1].",
        "Unknown qualification",
        "coordinator",
    )
    if status == "covered":
        assert gap is None
    else:
        assert gap and gap.status == OutcomeStatus.PARTIAL
        issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
        assert len(issues) == 1
        assert len(issues[0]["answer_unit_ids"]) == 2
        assert issues[0]["kind"] == "inconsistent_application"
    assert counts == {"inventory": 1, "review": 1}


def test_grouped_binding_checks_each_units_inline_support() -> None:
    ledger, _context, model = setup_review()

    def group(result: dict[str, Any], payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        identities = [u["unit_id"] for u in payload["answer_units"]]
        row["answer_unit_ids"] = identities
        row["coverage"] = [
            {
                "answer_unit_id": identities[0],
                "additional_answer_unit_ids": identities[1:],
                "status": "covered",
                "witnesses": row["coverage"][0]["witnesses"],
            }
        ]

    counts = scripted_reviews(model, review_mutator=group)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Supported application [1].\n\nSupported application [2].",
        "facts",
        "coordinator",
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
    assert len(issues) == 1 and issues[0]["kind"] == "unsupported_claim"
    assert len(issues[0]["answer_unit_ids"]) == 1
    affected = cast(list[dict[str, Any]], gap.data["affected_answer_units"])
    assert issues[0]["answer_unit_ids"][0] == affected[0]["unit_id"]
    assert counts == {"inventory": 1, "review": 1}


def test_grouped_binding_cannot_count_an_application_twice() -> None:
    ledger, _context, model = setup_review()

    def duplicate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        binding = result["resolutions"][0]["coverage"][0]
        binding["additional_answer_unit_ids"] = [binding["answer_unit_id"]]

    counts = scripted_reviews(model, review_mutator=duplicate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Result [1].", "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}


def setup_review() -> tuple[EvidenceLedger, RunContext, ResearchModel]:
    originals, context = original_ledger()
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    for number in originals.citation_numbers():
        original = originals.get(number)
        assert original is not None and original.search_doc is not None
        original.source_id = "shared-source"
        original.search_doc.document_id = "shared-source"
        ledger.add([original], context)
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
                "option_kind": "none",
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
        candidates = next(
            (
                r["answer_unit_ids"]
                for r in payload["application_candidates"]
                if r["requirement_id"] == req["requirement_id"]
            ),
            [],
        )
        affected = bound or [units[0]["unit_id"]]
        rows.append(
            {
                "requirement_id": req["requirement_id"],
                "option_state": "not_an_option"
                if req["option_kind"] == "none"
                else "facts_unknown",
                "status": "covered" if bound else "omitted",
                "answer_unit_ids": affected,
                "explanation": "" if bound else "The operative detail is absent.",
                "coverage": [
                    {
                        "answer_unit_id": unit_id,
                        "compatible_counterexample": "",
                        "status": "covered"
                        if unit_id in bound
                        else "omitted"
                        if unit_id in affected
                        else "unaffected",
                        "explanation": ""
                        if unit_id in bound
                        else "The operative qualification is absent."
                        if unit_id in affected
                        else "This unit asserts a different effect.",
                        "witnesses": req["witnesses"],
                    }
                    for unit_id in dict.fromkeys([*candidates, *affected])
                ],
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
            for requirement in result["requirements"]:
                requirement.setdefault("option_kind", "none")
        else:
            assert (schema.get("name") or "SourceUseReview") == "SourceUseReview"
            counts["review"] += 1
            result = review_for(payload)
            if review_mutator:
                review_mutator(result, payload)
            for row in result["resolutions"]:
                row.setdefault("option_state", "not_an_option")
                for binding in row["coverage"]:
                    binding.setdefault("compatible_counterexample", "")
        return text_response(result)

    cast(MagicMock, model.llm).invoke.side_effect = script
    return counts


def test_citation_removal_preserves_all_other_wording() -> None:
    assert (
        strip_citation_markers(
            "Effect [1, 2]. [[3]] 【4】 ［5］ [unknown] Article 17 (2025)."
        )
        == "Effect .    [unknown] Article 17 (2025)."
    )


def test_citation_only_reuse_rebases_units_and_rechecks_inline_support() -> None:
    ledger, _context, model = setup_review()
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap("Allowed [1].", "facts", "coordinator") is None
    assert reviewer.publication_gap("Allowed [1,2].", "facts", "coordinator") is None
    gap = reviewer.publication_gap("Allowed [2].", "facts", "coordinator")
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
    assert len(issues) == 1 and issues[0]["kind"] == "unsupported_claim"
    assert issues[0]["witnesses"][0]["citation"] == 1
    current = cast(list[dict[str, Any]], gap.data["affected_answer_units"])
    assert issues[0]["answer_unit_ids"] == [current[0]["unit_id"]]
    assert current[0]["text"] == "Allowed [2]."
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize("change", ["meaning", "facts", "metadata"])
def test_citation_only_cache_never_reuses_changed_meaning_or_context(
    change: str,
) -> None:
    ledger, context, model = setup_review()
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap("Allowed [1].", "facts", "coordinator") is None
    if change == "metadata":
        updated = EvidenceLedger()
        for number in ledger.citation_numbers():
            original = ledger.get(number)
            assert original is not None
            original.metadata["read_as_of_date"] = "2025-01-01"
            updated.add([original], context)
        context.services["evidence"] = updated
        reviewer.ledger = updated
        coordinator_delivery(updated, [1, 2, 3], "next")
    reviewer.publication_gap(
        "Prohibited [1,2]." if change == "meaning" else "Allowed [1,2].",
        "different facts" if change == "facts" else "facts",
        "next" if change == "metadata" else "coordinator",
    )
    assert counts["review"] == 2


def test_citation_only_cache_requires_every_new_application_edge() -> None:
    ledger, _context, model = setup_review()
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    first = "Allowed [1].\n\nA separate effect [2]."
    assert reviewer.publication_gap(first, "facts", "coordinator") is None
    second = "Allowed [1].\n\nA separate effect [1,2]."
    reviewer.publication_gap(second, "facts", "coordinator")
    assert counts == {"inventory": 1, "review": 2}


@pytest.mark.parametrize("kind", ["unsupported_claim", "missing_original"])
def test_explicit_support_defects_are_reassessed_after_citation_changes(
    kind: str,
) -> None:
    ledger, _context, model = setup_review()

    def unsupported(result: dict[str, Any], payload: dict[str, Any]) -> None:
        result["issues"] = [
            {
                "kind": kind,
                "answer_unit_ids": [payload["answer_units"][0]["unit_id"]],
                "witnesses": payload["retained_requirements"][0]["witnesses"],
                "detail": "The asserted effect lacks operative support.",
                "applicability": "The asserted application.",
            }
        ]

    counts = scripted_reviews(model, review_mutator=unsupported)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap("Allowed [1].", "facts", "coordinator")
    assert reviewer.publication_gap("Allowed [1,2].", "facts", "coordinator")
    assert counts == {"inventory": 1, "review": 2}


def test_omitted_support_is_not_retained_after_new_inline_originals() -> None:
    ledger, _context, model = setup_review()
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap("Allowed [2].", "facts", "coordinator")
    assert reviewer.publication_gap("Allowed [1,2].", "facts", "coordinator") is None
    assert counts == {"inventory": 1, "review": 2}


@pytest.mark.parametrize("status", ["covered", "unaffected"])
def test_compatible_counterexample_cannot_be_approved_or_lost_in_grouping(
    status: str,
) -> None:
    ledger, _context, model = setup_review()

    def counterexample(result: dict[str, Any], payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        ids = [unit["unit_id"] for unit in payload["answer_units"]]
        row["answer_unit_ids"] = ids if status == "covered" else []
        if status == "unaffected":
            row.update(status="omitted", explanation="Unknown decisive qualification.")
        row["coverage"] = [
            {
                "answer_unit_id": ids[0],
                "additional_answer_unit_ids": ids[1:],
                "status": status,
                "explanation": "The asserted broad effect needs its qualification."
                if status == "unaffected"
                else "",
                "compatible_counterexample": "If the unknown qualification holds, the source gives a different consequence.",
                "witnesses": row["coverage"][0]["witnesses"],
            }
        ]

    counts = scripted_reviews(model, review_mutator=counterexample)
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap(
        "Unconditional application [1].\n\nUnconditional application [1].",
        "The decisive fact is unknown.",
        "coordinator",
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
    assert len(issues) == 1 and issues[0]["kind"] == "inconsistent_application"
    assert len(issues[0]["answer_unit_ids"]) == 2
    assert issues[0]["witnesses"][0]["citation"] == 1
    assert "different consequence" in issues[0]["detail"]
    assert counts == {"inventory": 1, "review": 1}


def test_inline_support_catalogue_addresses_only_each_units_actual_originals() -> None:
    ledger, _context, model = setup_review()
    scripted_reviews(model)
    SourceUseReviewer(model, ledger).publication_gap(
        "First application [1].\n\nDifferent application [2,3].", "facts", "coordinator"
    )
    payload = model_payload(cast(MagicMock, model.llm).invoke.call_args_list[-1].kwargs)
    assert {record["citation"] for record in payload["original_evidence"]} == {1, 2, 3}
    for unit, catalogue in zip(
        payload["answer_units"], payload["inline_support_catalogue"], strict=True
    ):
        assert catalogue["answer_unit_id"] == unit["unit_id"]
        assert {row["citation"] for row in catalogue["originals"]} == set(
            unit["evidence_numbers"]
        )
        for row in catalogue["originals"]:
            original = next(
                r
                for r in payload["original_evidence"]
                if r["citation"] == row["citation"]
            )
            assert row["witness_ids"] == [
                span["witness_id"] for span in original["witness_spans"]
            ]


def test_conditional_option_and_counterexample_fields_are_required() -> None:
    with pytest.raises(ValidationError, match="option_kind"):
        SourceUseRequirement.model_validate(
            {
                "detail": "A conditional reduction.",
                "applicability": "Before official detection.",
                "witnesses": [{"citation": 1, "witness_id": "source-witness"}],
            }
        )
    with pytest.raises(ValidationError, match="compatible_counterexample"):
        SourceUseCoverage.model_validate(
            {"answer_unit_id": "unit", "status": "unaffected"}
        )
    with pytest.raises(ValidationError, match="option_state"):
        SourceUseResolution.model_validate(
            {
                "requirement_id": "requirement",
                "status": "omitted",
                "answer_unit_ids": [],
                "coverage": [],
            }
        )


@pytest.mark.parametrize("status", ["not_applicable", "outside_request"])
@pytest.mark.parametrize(
    "state", ["available", "not_invoked_yet", "facts_unknown", "not_an_option"]
)
def test_non_invocation_cannot_exclude_a_source_bound_option_without_paid_retry(
    status: str, state: str
) -> None:
    ledger, _context, model = setup_review()

    def conditional(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["requirements"][0]["option_kind"] = "conditional_relief"

    def excluded(result: dict[str, Any], payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        row.update(
            status=status,
            option_state=state,
            answer_unit_ids=[],
            scenario_witness_ids=[payload["user_fact_spans"][0]["witness_id"]],
            explanation="The user has not invoked this source-supported route.",
        )
        for binding in row["coverage"]:
            binding.update(
                status="unaffected", explanation="This route has not been invoked."
            )

    counts = scripted_reviews(
        model, inventory_mutator=conditional, review_mutator=excluded
    )
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap(
        "The burden applies [1].\n\nThe same burden applies to the alternative [1].",
        "I have not submitted an application. Explain both outcomes.",
        "coordinator",
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
    units = cast(list[dict[str, Any]], gap.data["affected_answer_units"])
    assert len(issues) == 1 and issues[0]["kind"] == "omitted_condition"
    assert issues[0]["answer_unit_ids"] == [unit["unit_id"] for unit in units]
    assert issues[0]["witnesses"][0]["citation"] == 1
    assert "non-invocation" in issues[0]["detail"]
    rebased = reviewer.publication_gap(
        "The burden applies [1,2].\n\nThe same burden applies to the alternative [1,2].",
        "I have not submitted an application. Explain both outcomes.",
        "coordinator",
    )
    assert rebased and rebased.status == OutcomeStatus.PARTIAL
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize("barrier", ["literal", "missing_fact", "missing_explanation"])
def test_a_conditional_option_can_be_barred_only_with_an_actual_user_fact(
    barrier: str,
) -> None:
    ledger, _context, model = setup_review()

    def conditional(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["requirements"][0]["option_kind"] = "alternative_route"

    def excluded(result: dict[str, Any], payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        row.update(
            status="not_applicable",
            option_state="barred_by_explicit_fact",
            answer_unit_ids=[],
            scenario_witness_ids=[]
            if barrier == "missing_fact"
            else [payload["user_fact_spans"][0]["witness_id"]],
            explanation=""
            if barrier == "missing_explanation"
            else "The expressly completed deadline bars a new application too.",
        )
        for binding in row["coverage"]:
            binding.update(
                status="unaffected", explanation="The filing period expired."
            )

    counts = scripted_reviews(
        model, inventory_mutator=conditional, review_mutator=excluded
    )
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "The burden applies [1].", "The filing period expired.", "coordinator"
    )
    if barrier == "literal":
        assert gap is None
        assert counts == {"inventory": 1, "review": 1}
    else:
        assert gap and gap.status == OutcomeStatus.UNAVAILABLE


def test_a_supported_conditional_option_is_not_a_new_omission() -> None:
    ledger, _context, model = setup_review()

    def conditional(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["requirements"][0]["option_kind"] = "remedy"

    counts = scripted_reviews(model, inventory_mutator=conditional)
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "If the source condition holds, the remedy remains available [1].",
            "The decisive fact is not yet known.",
            "coordinator",
        )
        is None
    )
    assert counts == {"inventory": 1, "review": 1}


def test_an_option_for_a_different_effect_is_not_added_to_the_answer() -> None:
    ledger, _context, model = setup_review()

    def conditional(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["requirements"][0]["option_kind"] = "conditional_relief"

    def unrelated(result: dict[str, Any], payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="outside_request",
            option_state="not_related_to_asserted_effect",
            answer_unit_ids=[],
            coverage=[],
            explanation="The assessed ownership relief has no interaction with storage permission.",
            scenario_witness_ids=[payload["user_fact_spans"][0]["witness_id"]],
        )

    counts = scripted_reviews(
        model, inventory_mutator=conditional, review_mutator=unrelated
    )
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Storage permission [2].", "Explain storage permission.", "coordinator"
        )
        is None
    )
    assert counts == {"inventory": 1, "review": 1}


def test_distinct_summary_and_application_cannot_share_a_coverage_binding() -> None:
    ledger, _context, model = setup_review()

    def grouped(result: dict[str, Any], payload: dict[str, Any]) -> None:
        row = result["resolutions"][0]
        row["coverage"][0]["additional_answer_unit_ids"] = [
            payload["answer_units"][1]["unit_id"]
        ]
        row["coverage"] = row["coverage"][:1]

    counts = scripted_reviews(model, review_mutator=grouped)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "A qualified rule [1].\n\nAn unconditional application [1].",
        "facts",
        "coordinator",
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}


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
        "scenario",
        "conversation",
        "user_fact_spans",
    }.intersection(blind)
    assessment = model_payload(calls[1].kwargs)
    assert assessment["conversation"] == [conversation[1]]
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
def test_source_compilation_reuse_is_independent_of_answer_and_facts(
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
    assert counts == {"inventory": 2 if change == "evidence" else 1, "review": 2}
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
        "missing_original",
        "assumed_exclusion",
    ],
)
def test_review_cannot_drop_retained_rules_or_borrow_neighboring_citations(
    defect: str,
) -> None:
    ledger, context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
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
        else:
            row.update(
                status="not_applicable",
                scenario_witness_ids=["invented-fact"],
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

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="not_applicable",
            scenario_witness_ids=[payload["user_fact_spans"][0]["witness_id"]]
            if payload["scenario"] == "Excluded status"
            else ["assistant-fact"],
            answer_unit_ids=[],
            coverage=[
                {
                    **binding,
                    "status": "unaffected",
                    "explanation": "The supplied fact excludes this application.",
                }
                for binding in result["resolutions"][0]["coverage"]
            ],
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
    assert counts == {"inventory": 1, "review": 3}


def test_user_fact_selectors_preserve_every_character_and_reject_assistant_text() -> (
    None
):
    scenario = "Condition unknown.\n" + "A factual qualification; " * 95
    conversation = [
        {"role": "user", "content": "A distinct earlier user fact."},
        {"role": "assistant", "content": "An assumed decisive exclusion."},
    ]
    spans = user_fact_spans(scenario, conversation)
    texts = {"scenario": scenario, "conversation-0": conversation[0]["content"]}
    for reference, original in texts.items():
        selected = [s for s in spans if s["text_ref"] == reference]
        assert (
            "".join(
                original[cast(int, s["start_char"]) : cast(int, s["end_char"])]
                for s in selected
            )
            == original
        )
    assert {s["text_ref"] for s in spans} == texts.keys()
    assert not {
        s["witness_id"] for s in spans if s["text_ref"] == "scenario"
    }.intersection(s["witness_id"] for s in user_fact_spans(scenario + "!", []))


@pytest.mark.parametrize("defect", ["missing", "stale", "repeated"])
def test_exclusions_need_current_unique_user_fact_selectors(defect: str) -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        supplied = payload["user_fact_spans"][0]["witness_id"]
        selectors = (
            []
            if defect == "missing"
            else [user_fact_spans("An older fact", [])[0]["witness_id"]]
            if defect == "stale"
            else [supplied, supplied]
        )
        result["resolutions"][0].update(
            status="not_applicable",
            answer_unit_ids=[],
            coverage=[],
            scenario_witness_ids=selectors,
        )

    scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Allowed [1].", "A current actual exclusion", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE


def test_identical_user_context_is_deduplicated_without_rewriting_facts() -> None:
    ledger, _context, model = setup_review()
    scripted_reviews(model)
    SourceUseReviewer(model, ledger).publication_gap(
        "Allowed [1].",
        "Exact question",
        "coordinator",
        conversation=[
            {"role": "user", "content": "Exact question"},
            {"role": "user", "content": "A distinct earlier fact"},
            {"role": "user", "content": "A distinct earlier fact"},
            {"role": "user", "content": "exact question"},
        ],
    )
    payload = model_payload(cast(MagicMock, model.llm).invoke.call_args_list[1].kwargs)
    assert payload["scenario"] == "Exact question"
    assert [r["content"] for r in payload["conversation"]] == [
        "A distinct earlier fact",
        "exact question",
    ]


def test_required_coverage_and_stable_original_prefix_after_answer_edit() -> None:
    assert "coverage" in SourceUseResolution.model_json_schema()["required"]
    assert "scenario_quote" not in SourceUseResolution.model_json_schema()["properties"]
    ledger, _context, model = setup_review()
    scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    reviewer.publication_gap("Allowed [1].", "Actual facts", "coordinator")
    reviewer.publication_gap(
        "Allowed with qualification [1].", "Actual facts", "coordinator"
    )
    calls = cast(MagicMock, model.llm).invoke.call_args_list
    serialized = [
        c.kwargs["prompt"][1].content
        for c in calls
        if "answer_units" in model_payload(c.kwargs)
    ]
    assert len(serialized) == 2
    prefixes = [text.partition('"answer_units":')[0] for text in serialized]
    assert prefixes[0] == prefixes[1]
    assert '"original_evidence"' in prefixes[0]
    assert '"retained_requirements"' in prefixes[0]


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


def test_witnessed_omission_without_existing_block_is_repaired_without_review_retry() -> (
    None
):
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="omitted",
            answer_unit_ids=[],
            coverage=[],
            explanation="A material conditional branch needs its own new paragraph.",
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Another supported result [2].", "Actual facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issue = SourceUseIssue.model_validate(
        cast(list[Any], gap.data["source_use_gaps"])[0]
    )
    assert issue.kind == "omitted_condition" and not issue.answer_unit_ids
    assert issue.witnesses[0].citation == 1
    assert gap.data["affected_answer_units"] == []
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize("has_location", [True, False])
def test_omission_uses_retained_detail_without_repeating_explanation(
    has_location: bool,
) -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="omitted",
            answer_unit_ids=[payload["answer_units"][0]["unit_id"]]
            if has_location
            else [],
            explanation="",
            coverage=result["resolutions"][0]["coverage"] if has_location else [],
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Another supported result [2].", "Actual facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issue = SourceUseIssue.model_validate(
        cast(list[Any], gap.data["source_use_gaps"])[0]
    )
    retained = cast(list[Any], gap.data["retained_source_requirements"])[0]
    assert retained["detail"] in issue.detail
    assert issue.witnesses[0].citation == 1
    assert bool(issue.answer_unit_ids) is has_location
    assert counts == {"inventory": 1, "review": 1}


def test_misapplication_still_needs_its_changed_logic() -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="misapplied", explanation="", coverage=[]
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "An asserted effect [1].", "Actual facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}


@pytest.mark.parametrize(
    "kind", ["unsupported_claim", "inconsistent_application", "missing_original"]
)
def test_asserted_defects_still_require_their_actual_answer_unit(kind: str) -> None:
    with pytest.raises(ValueError, match="actual answer unit"):
        SourceUseIssue.model_validate(
            {
                "kind": kind,
                "answer_unit_ids": [],
                "witnesses": [{"citation": 1, "source_quote": "An original condition"}],
                "detail": "An actual asserted defect",
                "applicability": "Current facts",
            }
        )


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


def test_source_partition_keeps_every_original_and_reuses_unchanged_sources() -> None:
    ledger, context = original_ledger()
    context.services.update(
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
        last_model_call_id="coordinator",
    )
    coordinator_delivery(ledger, [1, 2, 3])
    model = ResearchModel(scripted_model(), context)
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap("Result [1, 2, 3].", "facts", "coordinator") is None
    assert counts == {"inventory": 3, "review": 3}
    calls = cast(MagicMock, model.llm).invoke.call_args_list
    blind = [
        model_payload(c.kwargs)
        for c in calls
        if "answer_units" not in model_payload(c.kwargs)
    ]
    assert len(blind) == 3
    assert {r["citation"] for data in blind for r in data["original_evidence"]} == {
        1,
        2,
        3,
    }
    assert all(
        len({r["source_id"] for r in data["original_evidence"]}) == 1 for data in blind
    )
    receipts = [
        next(iter(child._inventories.values()))[0]
        for child in reviewer._source_reviewers.values()
    ]
    assert len(set(receipts)) == 3
    assert all(len(ledger.completely_delivered(receipt)) == 1 for receipt in receipts)
    coordinator_delivery(ledger, [1, 2], "next")
    assert reviewer.publication_gap("Result [1, 2].", "facts", "next") is None
    assert counts == {"inventory": 3, "review": 5}
    assert context.services["last_model_call_id"] == "coordinator"


def test_application_scopes_keep_full_originals_zero_effect_sources_and_exact_reuse() -> (
    None
):
    ledger, context = original_ledger()
    context.services.update(
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
        last_model_call_id="coordinator",
    )
    coordinator_delivery(ledger, [1, 2, 3])
    model = ResearchModel(scripted_model(), context)
    third = ledger.get(3)
    assert third is not None

    def empty_background(result: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("inventory_source_id") == third.source_id:
            result["requirements"] = []

    counts = scripted_reviews(model, inventory_mutator=empty_background)
    reviewer = SourceUseReviewer(model, ledger)
    gap = reviewer.publication_gap(
        "Actual application [1].", "Unknown qualification", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    payloads = [
        model_payload(call.kwargs)
        for call in cast(MagicMock, model.llm).invoke.call_args_list
    ]
    scoped = [p for p in payloads if "assessment_source_ids" in p]
    assert len(scoped) == 3
    assert {r["citation"] for p in scoped for r in p["original_evidence"]} == {1, 2, 3}
    owned_requirements = [
        r["requirement_id"] for p in scoped for r in p["retained_requirements"]
    ]
    assert len(owned_requirements) == len(set(owned_requirements)) == 2
    for payload in scoped:
        own = payload["assessment_source_ids"]
        assert len(own) == 1
        originals = expand_source_metadata(
            {
                "original_metadata_catalogue": payload["original_evidence"],
                "original_source_metadata": payload["original_source_metadata"],
            }
        )
        assert {r["citation"] for r in originals} == {
            1,
            *[
                n
                for n in (1, 2, 3)
                if (item := ledger.get(n)) is not None and item.source_id in own
            ],
        }
        for row in originals:
            item = ledger.get(cast(int, row["citation"]))
            assert item is not None and row["text"] == item.text
        assert payload["scenario"] == "Unknown qualification"
        assert [u["text"] for u in payload["answer_units"]] == [
            "Actual application [1]."
        ]
    assert counts == {"inventory": 3, "review": 3}
    repeated = reviewer.publication_gap(
        "Actual application [1].", "Unknown qualification", "coordinator"
    )
    assert repeated and repeated.data == gap.data
    assert counts == {"inventory": 3, "review": 3}
    assert context.services["last_model_call_id"] == "coordinator"


def test_incomplete_source_check_does_not_erase_other_witnessed_defects() -> None:
    ledger, context = original_ledger()
    context.services.update(
        asv3_workflow_variant=ASV3_TUNED_VARIANT, research_profile="normal"
    )
    coordinator_delivery(ledger, [1, 2, 3])
    model = ResearchModel(scripted_model(), context)
    missing = ledger.get(3)
    assert missing is not None

    def incomplete(result: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload["assessment_source_ids"] == [missing.source_id]:
            result["resolutions"] = []

    counts = scripted_reviews(model, review_mutator=incomplete)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Unqualified application [1].", "Unknown qualification", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.PARTIAL
    assert gap.data["source_use_review_unavailable"] is True
    assert gap.data["incomplete_assessment_source_ids"] == [[missing.source_id]]
    issues = cast(list[dict[str, Any]], gap.data["source_use_gaps"])
    assert any(w["citation"] == 2 for issue in issues for w in issue["witnesses"])
    assert counts == {"inventory": 3, "review": 4}


def test_source_inventory_receives_only_its_recorded_anchor_context() -> None:
    context, ledger, reviews = setup_reviews()
    context.services.update(
        evidence=ledger,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
        legal_source_reviews=reviews,
    )
    seen(context, ledger, reviews)
    deliver(ledger, "coordinator", [1, 2, 3])
    model = ResearchModel(scripted_model(), context)
    counts = scripted_reviews(model)
    reviewer = SourceUseReviewer(model, ledger)
    assert reviewer.publication_gap("Result [1, 2, 3].", "facts", "coordinator") is None
    payloads = [
        model_payload(call.kwargs)
        for call in cast(MagicMock, model.llm).invoke.call_args_list
    ]
    blind = [p for p in payloads if "inventory_source_id" in p]
    linked = next(p for p in blind if p["inventory_source_id"] == "decision")
    assert linked["required_owned_evidence_numbers"] == [2]
    assert linked["required_evidence_numbers"] == [1, 2]
    assert {r["citation"] for r in linked["original_evidence"]} == {1, 2}
    for row in linked["original_evidence"]:
        original = ledger.get(row["citation"])
        assert original is not None and row["text"] == original.text
    assert linked["source_links"][0]["anchor_evidence_numbers"] == [1]
    assert not {"review", "effect", "limitations", "status"}.intersection(
        linked["source_links"][0]
    )
    unrelated = next(p for p in blind if p["inventory_source_id"] == "other-decision")
    assert unrelated["required_evidence_numbers"] == [3]
    assert unrelated["source_links"] == []
    assessment = next(
        p for p in payloads if p.get("assessment_source_ids") == ["decision"]
    )
    court_requirements = {
        r["requirement_id"]
        for r in assessment["retained_requirements"]
        if any(w["citation"] == 2 for w in r["witnesses"])
    }
    candidates = assessment["application_candidates"]
    assert court_requirements <= {r["requirement_id"] for r in candidates}
    assert {r["requirement_id"] for r in candidates} == {
        r["requirement_id"] for r in assessment["retained_requirements"]
    }
    assert all(
        r["answer_unit_ids"] == [assessment["answer_units"][0]["unit_id"]]
        for r in candidates
    )
    assert counts == {"inventory": 3, "review": 3}
    assert (
        reviewer.publication_gap("Changed result [1, 2, 3].", "facts", "coordinator")
        is None
    )
    assert counts == {"inventory": 3, "review": 6}


def test_linked_anchor_cannot_replace_target_inventory_witness() -> None:
    context, ledger, reviews = setup_reviews()
    context.services.update(
        evidence=ledger,
        asv3_workflow_variant=ASV3_TUNED_VARIANT,
        research_profile="normal",
        legal_source_reviews=reviews,
    )
    seen(context, ledger, reviews)
    deliver(ledger, "coordinator", [1, 2, 3])
    model = ResearchModel(scripted_model(), context)

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("inventory_source_id") == "decision":
            anchor = next(r for r in payload["original_evidence"] if r["citation"] == 1)
            result["requirements"][0]["witnesses"] = [
                {"citation": 1, "witness_id": anchor["witness_spans"][0]["witness_id"]}
            ]

    scripted_reviews(model, inventory_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Result [1, 2, 3].", "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE


@pytest.mark.parametrize("drop_effect", [False, True])
def test_disposition_body_effects_cannot_be_filtered_by_the_unseen_question(
    drop_effect: bool,
) -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    numbers = ledger.add(
        [
            judicial_chunk("V. HÜKÜM\nThe identified phrase is annulled.", 0),
            judicial_chunk("The connected qualification is also annulled.", 1),
        ],
        context,
    )
    deliver(ledger, "coordinator", [1, *numbers])
    model = ResearchModel(scripted_model(), context)

    def effects(result: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("inventory_source_id") != "decision":
            return
        assert not {"scenario", "conversation", "user_fact_spans"}.intersection(payload)
        dispositions = payload["disposition_originals"]
        assert {r["citation"] for r in dispositions} == set(numbers)
        result["requirements"] = [
            {
                "detail": "The source's own effect and qualification.",
                "applicability": "Restricted to the identified wording and scope.",
                "witnesses": [
                    {"citation": row["citation"], "witness_id": row["witness_ids"][0]}
                ],
            }
            for row in dispositions
        ]
        if drop_effect:
            result["requirements"].pop()

    scripted_reviews(model, inventory_mutator=effects)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        f"A result [1, {', '.join(map(str, numbers))}].",
        "The user asks only for the administrative procedure.",
        "coordinator",
    )
    if drop_effect:
        assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    else:
        assert gap is None


def test_application_navigation_keeps_summary_and_detail_for_the_same_original() -> (
    None
):
    ledger, _context, model = setup_review()
    counts = scripted_reviews(model)
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Unconditional summary [1].\n\nConditional detail [1].\n\nOther rule [2].",
            "Actual request",
            "coordinator",
        )
        is None
    )
    assessment = model_payload(
        cast(MagicMock, model.llm).invoke.call_args_list[-1].kwargs
    )
    assert assessment["application_candidates"] == [
        {
            "requirement_id": assessment["retained_requirements"][0]["requirement_id"],
            "answer_unit_ids": [u["unit_id"] for u in assessment["answer_units"][:2]],
        }
    ]
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize("text", ["V. HÜKÜM", "The applicant requests relief."])
def test_unknown_text_or_disposition_heading_does_not_manufacture_an_effect(
    text: str,
) -> None:
    context, ledger, reviews = tuned_context()
    seen(context, ledger, reviews)
    number = ledger.add([judicial_chunk(text, 0)], context)[0]
    deliver(ledger, "coordinator", [1, number])
    model = ResearchModel(scripted_model(), context)

    def extract(result: dict[str, Any], payload: dict[str, Any]) -> None:
        if payload.get("inventory_source_id") == "decision":
            assert payload["disposition_originals"] == []
            result["requirements"] = []

    scripted_reviews(model, inventory_mutator=extract)
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "A governing result [1].", "Actual facts", "coordinator"
        )
        is None
    )


@pytest.mark.parametrize(
    "defect", [None, "asserted_unit", "missing_scope", "invented_fact"]
)
def test_unrelated_background_needs_request_scope_without_claiming_legal_exclusion(
    defect: str | None,
) -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        result["resolutions"][0].update(
            status="outside_request",
            answer_unit_ids=[payload["answer_units"][0]["unit_id"]]
            if defect == "asserted_unit"
            else [],
            coverage=[],
            explanation=""
            if defect == "missing_scope"
            else "The request concerns storage permission; this original's ownership tax has no operative interaction with it.",
            scenario_witness_ids=["invented"]
            if defect == "invented_fact"
            else [payload["user_fact_spans"][0]["witness_id"]],
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Storage permission [2].",
        "Explain the storage permission procedure.",
        "coordinator",
    )
    if defect:
        assert gap and gap.status == OutcomeStatus.UNAVAILABLE
        assert counts == {"inventory": 1, "review": 2}
    else:
        assert gap is None
        assert counts == {"inventory": 1, "review": 1}


def test_coverage_can_use_the_resolved_governing_original() -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        unit = payload["answer_units"][0]
        governing = payload["original_evidence"][1]
        result["resolutions"][0].update(
            status="covered",
            answer_unit_ids=[unit["unit_id"]],
            coverage=[
                {
                    "answer_unit_id": unit["unit_id"],
                    "status": "covered",
                    "witnesses": [
                        {
                            "citation": governing["citation"],
                            "witness_id": governing["witness_spans"][0]["witness_id"],
                        }
                    ],
                }
            ],
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    assert (
        SourceUseReviewer(model, ledger).publication_gap(
            "Governing result [2].", "facts", "coordinator"
        )
        is None
    )
    assert counts == {"inventory": 1, "review": 1}


def test_missing_inline_support_is_a_targeted_gap_without_paid_format_retry() -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], payload: dict[str, Any]) -> None:
        unit = payload["answer_units"][-1]
        result["resolutions"][0].update(
            answer_unit_ids=[u["unit_id"] for u in payload["answer_units"]],
            coverage=[
                *result["resolutions"][0]["coverage"],
                {
                    "answer_unit_id": unit["unit_id"],
                    "status": "covered",
                    "witnesses": payload["retained_requirements"][0]["witnesses"],
                },
            ],
        )

    counts = scripted_reviews(model, review_mutator=mutate)
    reviewer = SourceUseReviewer(model, ledger)
    answer = "Rule [1].\n\nApplication [2]."
    gap = reviewer.publication_gap(answer, "facts", "coordinator")
    assert gap and gap.status == OutcomeStatus.PARTIAL
    issue = SourceUseIssue.model_validate(
        cast(list[Any], gap.data["source_use_gaps"])[0]
    )
    assert issue.kind == "unsupported_claim" and issue.witnesses[0].citation == 1
    assert "[1]" in issue.detail
    assert counts == {"inventory": 1, "review": 1}
    assert reviewer.publication_gap(answer, "facts", "coordinator") == gap
    assert counts == {"inventory": 1, "review": 1}


@pytest.mark.parametrize(
    "defect", ["unknown_witness", "missing_binding", "duplicate_binding"]
)
def test_coverage_bindings_cannot_forge_or_drop_unit_support(defect: str) -> None:
    ledger, _context, model = setup_review()

    def mutate(result: dict[str, Any], _payload: dict[str, Any]) -> None:
        coverage = result["resolutions"][0]["coverage"]
        if defect == "unknown_witness":
            coverage[0]["witnesses"][0]["witness_id"] = "forged"
        elif defect == "missing_binding":
            coverage.clear()
        else:
            coverage.append(coverage[0])

    counts = scripted_reviews(model, review_mutator=mutate)
    gap = SourceUseReviewer(model, ledger).publication_gap(
        "Result [1].", "facts", "coordinator"
    )
    assert gap and gap.status == OutcomeStatus.UNAVAILABLE
    assert counts == {"inventory": 1, "review": 2}


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
        for record in records:
            if record["citation"] == 4:
                metadata = record["metadata"]
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
    assert counts == {"native": 3, "inventory": 2, "review": 4}
    assert page.call_count == 2
    assert (
        kwargs["state_container"].answer_tokens
        == "Tamir sonucu [[1]](https://example.test/law-0); değiştirme sonucu ve uygulama koşulu [[2]](https://example.test/law-1)."
    )
