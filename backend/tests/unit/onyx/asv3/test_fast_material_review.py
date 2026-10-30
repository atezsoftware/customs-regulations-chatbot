"""Material review requires the actual writer source set and literal evidence."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from unittest.mock import Mock

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.fast_material_review import (
    FastMaterialFinding,
    FastMaterialReview,
    FastMaterialReviewResult,
)
from onyx.asv3.models import OutcomeStatus, RunContext, RunStopped
from tests.unit.onyx.asv3.test_shared_originals import original

QUESTION = "What consent and authority are required, and when must documents be filed?"
SOURCE = "Written consent AND no objection are required. An authorized signatory must prove authority within ten days."
ANSWER = "Either silence or written consent suffices [1]."


def setup() -> tuple[RunContext, EvidenceLedger]:
    context = RunContext(scope={"document_set": ["authorized"]})
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                "conditions", SOURCE, metadata={"legal_authority": "operative_text"}
            ),
            original("partial", "A partial original cannot be reviewed as complete."),
            original("not-delivered", "Unseen material cannot ground a finding."),
        ],
        context,
    )
    records = cast(
        list[dict[str, JsonValue]], json.loads(ledger.serialize_records([1, 2]))
    )
    records[1]["text"] = "A partial original"
    ledger.record_delivery("writer", "asv3_coordinator", records)
    return context, ledger


def result(
    payload: dict[str, JsonValue], **updates: object
) -> FastMaterialReviewResult:
    fields: dict[str, object] = {
        "status": "complete",
        "candidate_sha256": payload["candidate_sha256"],
        "source_set_sha256": payload["source_set_sha256"],
        "examined_citations": payload["required_evidence_numbers"],
        "findings": [],
        **updates,
    }
    return FastMaterialReviewResult.model_validate(fields)


def finding(**updates: object) -> FastMaterialFinding:
    return FastMaterialFinding.model_validate(
        {
            "kind": "condition",
            "detail": "Both cumulative requirements must be preserved.",
            "applicability": "The question asks whether silence suffices for this procedure.",
            "citation": 1,
            "source_quote": "Written consent AND no objection are required.",
            "answer_quote": "Either silence or written consent suffices",
            **updates,
        }
    )


def gap_status(gap: object) -> str:
    from onyx.asv3.models import ToolOutcome

    assert isinstance(gap, ToolOutcome)
    assert gap.status is OutcomeStatus.PARTIAL
    data = gap.data["asv3_fast_material_review"]
    assert isinstance(data, dict)
    return str(data["status"])


def test_complete_full_writer_originals_and_metadata_are_required_for_approval() -> (
    None
):
    context, ledger = setup()
    seen: list[dict[str, JsonValue]] = []

    def assess(payload: dict[str, JsonValue]) -> FastMaterialReviewResult:
        seen.append(copy.deepcopy(payload))
        return result(payload)

    review = FastMaterialReview(context, ledger, assess)
    assert review.publication_gap(ANSWER, "writer", QUESTION) is None
    payload = seen[0]
    assert payload["question"] == QUESTION and payload["candidate"] == ANSWER
    assert payload["required_evidence_numbers"] == [1]
    records = payload["original_evidence"]
    assert isinstance(records, list) and len(records) == 1
    row = records[0]
    assert isinstance(row, dict)
    assert (
        row["text"] == SOURCE
        and row["start_char"] == 0
        and row["end_char"] == len(SOURCE)
    )
    item = ledger.get(1)
    assert item is not None and item.search_doc is not None
    assert row["text_hash"] == item.text_hash
    assert (
        row["metadata"] == item.metadata
        and row["search_metadata"] == item.search_doc.metadata
    )
    assert review.publication_gap(ANSWER, "writer", QUESTION) is None
    ledger.record_delivery(
        "same-source-writer", "asv3_coordinator", [cast(dict[str, JsonValue], row)]
    )
    assert review.publication_gap(ANSWER, "same-source-writer", QUESTION) is None
    assert len(seen) == 1


def test_source_bound_condition_finding_is_a_repair_gap_and_new_body_is_reviewed() -> (
    None
):
    context, ledger = setup()
    assess = Mock(
        side_effect=lambda payload: result(
            payload, findings=[finding()] if payload["candidate"] == ANSWER else []
        )
    )
    review = FastMaterialReview(context, ledger, assess)
    gap = review.publication_gap(ANSWER, "writer", QUESTION)
    assert gap_status(gap) == "material_findings"
    assert gap is not None
    gap.data.clear()
    assert (
        gap_status(review.publication_gap(ANSWER, "writer", QUESTION))
        == "material_findings"
    )
    assert (
        review.publication_gap(
            "Written consent AND no objection are required [1].", "writer", QUESTION
        )
        is None
    )
    assert assess.call_count == 2
    assert (
        gap_status(
            review.publication_gap("A third distinct body [1].", "writer", QUESTION)
        )
        == "review_limit"
    )
    assert assess.call_count == 2


@pytest.mark.parametrize(
    "change",
    [
        {"candidate_sha256": "0" * 64},
        {"source_set_sha256": "0" * 64},
        {"examined_citations": []},
        {"examined_citations": [1, 1]},
        {"examined_citations": [1, 2]},
        {"findings": [finding(citation=3, source_quote="Unseen material")]},
        {"findings": [finding(source_quote="Invented quotation")]},
        {"findings": [finding(source_quote=" ")]},
        {"findings": [finding(answer_quote="Text outside candidate")]},
        {"findings": [finding(answer_quote="")]},
        {"findings": [finding(applicability=" ")]},
    ],
)
def test_unbound_or_missing_examination_never_approves(
    change: dict[str, object],
) -> None:
    context, ledger = setup()
    assess = Mock(side_effect=lambda payload: result(payload, **change))
    review = FastMaterialReview(context, ledger, assess)
    assert (
        gap_status(review.publication_gap(ANSWER, "writer", QUESTION))
        == "invalid_review_binding"
    )
    assert (
        gap_status(review.publication_gap(ANSWER, "writer", QUESTION))
        == "invalid_review_binding"
    )
    assert assess.call_count == 1


def test_material_omission_can_have_an_empty_answer_quote_only_with_a_literal_original() -> (
    None
):
    context, ledger = setup()

    def assess(payload: dict[str, JsonValue]) -> FastMaterialReviewResult:
        return result(payload, findings=[finding(kind="omission", answer_quote="")])

    review = FastMaterialReview(context, ledger, assess)
    assert (
        gap_status(review.publication_gap(ANSWER, "writer", QUESTION))
        == "material_findings"
    )


def test_uncertain_empty_result_is_not_approval() -> None:
    context, ledger = setup()
    review = FastMaterialReview(
        context, ledger, lambda payload: result(payload, status="uncertain")
    )
    assert gap_status(review.publication_gap(ANSWER, "writer", QUESTION)) == "uncertain"


def test_actual_question_and_writer_source_set_are_in_the_cache_key() -> None:
    context, ledger = setup()
    assess = Mock(side_effect=result)
    review = FastMaterialReview(context, ledger, assess)
    assert review.publication_gap(ANSWER, "writer", QUESTION) is None
    assert (
        review.publication_gap(ANSWER, "writer", "A distinct material question") is None
    )
    assert assess.call_count == 2
    rows = cast(
        list[dict[str, JsonValue]], json.loads(ledger.serialize_records([1, 3]))
    )
    ledger.record_delivery("new-writer", "asv3_coordinator", rows)
    assert (
        gap_status(review.publication_gap(ANSWER, "new-writer", QUESTION))
        == "review_limit"
    )
    assert assess.call_count == 2


def test_no_writer_receipt_or_no_canonical_original_does_not_call_the_checker() -> None:
    context, ledger = setup()
    assess = Mock(side_effect=result)
    review = FastMaterialReview(context, ledger, assess)
    assert (
        gap_status(review.publication_gap(ANSWER, "unknown-writer", QUESTION))
        == "missing_review_input"
    )
    assert (
        gap_status(review.publication_gap(ANSWER, None, QUESTION))
        == "missing_review_input"
    )
    assert assess.call_count == 0


def test_fresh_scope_and_cancellation_checks_run_even_for_cached_approval() -> None:
    context, ledger = setup()
    assess = Mock(side_effect=result)
    review = FastMaterialReview(context, ledger, assess)
    assert review.publication_gap(ANSWER, "writer", QUESTION) is None
    context.scope = {"document_set": ["different"]}
    with pytest.raises(PermissionError, match="scope changed"):
        review.publication_gap(ANSWER, "writer", QUESTION)
    context.scope = {"document_set": ["authorized"]}
    context.cancel()
    with pytest.raises(RunStopped, match="cancelled"):
        review.publication_gap(ANSWER, "writer", QUESTION)
    assert assess.call_count == 1


@pytest.mark.parametrize(
    "error",
    [
        ValueError("bad format"),
        RuntimeError("provider failed"),
        RunStopped("cancelled"),
    ],
)
def test_callback_errors_propagate_without_repeating_the_same_paid_assessment(
    error: Exception,
) -> None:
    context, ledger = setup()
    assess = Mock(side_effect=error)
    review = FastMaterialReview(context, ledger, assess)
    for _ in range(2):
        with pytest.raises(type(error)) as raised:
            review.publication_gap(ANSWER, "writer", QUESTION)
        assert raised.value is error
    assert assess.call_count == 1


def test_duplicate_concurrent_callers_share_one_semantic_assessment() -> None:
    context, ledger = setup()
    assess = Mock(side_effect=result)
    review = FastMaterialReview(context, ledger, assess)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: review.publication_gap(ANSWER, "writer", QUESTION), range(2)
            )
        )
    assert results == [None, None] and assess.call_count == 1


def test_result_model_rejects_missing_examination_and_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FastMaterialReviewResult.model_validate({"status": "complete", "findings": []})
    with pytest.raises(ValidationError):
        finding(citation=True)
