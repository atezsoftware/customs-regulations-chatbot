"""Citation syntax must preserve original delivery and fail-closed publication."""

import json
from unittest.mock import MagicMock

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import (
    QuestionVerification,
    ResearchModel,
    VerificationResult,
)
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.asv3.publication import publication_gap
from onyx.asv3.runtime import _evidence_record
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.tracing.flows import LLMFlow

CITATIONS = ["[1, 2]", "[[1]], [[2]]", "【1, 2】", "［1,2］"]


def original_ledger() -> tuple[EvidenceLedger, RunContext]:
    context = RunContext()
    ledger = EvidenceLedger()
    context.services["evidence"] = ledger
    for number in (1, 2, 3):
        text = f"Complete source {number}: condition AND exception; original tail {number}."
        ledger.add(
            [
                EvidenceItem(
                    source_id=f"distinct-source-{number}",
                    chunk_id=f"chunk-{number}",
                    text=text,
                    search_doc=SearchDoc(
                        document_id=f"distinct-source-{number}",
                        chunk_ind=0,
                        semantic_identifier=f"Law {number}",
                        link=f"https://example.test/law-{number}",
                        blurb=text,
                        source_type=DocumentSource.USER_FILE,
                        boost=0,
                        hidden=False,
                        metadata={},
                        match_highlights=[],
                    ),
                )
            ],
            context,
        )
    return ledger, context


def supported(numbers: list[int]) -> VerificationResult:
    return VerificationResult(
        status="supported",
        explanation="All actual original clauses support the answer.",
        required_conditions=[],
        missing_conditions=[],
        evidence_numbers=numbers,
        safe_to_publish=True,
        unsupported_claims=[],
        question_results=[
            QuestionVerification(
                question_id="q0",
                status="supported",
                evidence_numbers=numbers,
                missing_conditions=[],
            )
        ],
    )


def selected_model(context: RunContext) -> tuple[ResearchModel, MagicMock]:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="selected-model",
        temperature=0,
        max_input_tokens=12000,
    )
    llm.invoke.return_value = ModelResponse(
        id="review",
        created="0",
        choice=Choice(message=Message(content=supported([1, 2]).model_dump_json())),
    )
    return ResearchModel(llm, context), llm


@pytest.mark.parametrize("markers", CITATIONS)
def test_grouped_and_unicode_citations_deliver_every_distinct_source_original(
    markers: str,
) -> None:
    ledger, context = original_ledger()
    model, llm = selected_model(context)
    claim = f"Apply the complete conditions {markers}."
    evidence = _evidence_record(ledger, claim)
    records = json.loads(evidence)
    assert [record["citation"] for record in records] == [1, 2]
    assert {record["source_id"] for record in records} == {
        "distinct-source-1",
        "distinct-source-2",
    }
    for record in records:
        original = ledger.get(record["citation"])
        assert original is not None and record["text"] == original.text
    assert all(record["truncated"] is False for record in records)
    model.invoke_text(
        "Verify every original condition",
        json.dumps({"claim": claim, "scenario": "facts", "evidence": evidence}),
        LLMFlow.ASV3_VERIFICATION,
        max_tokens=1000,
    )
    assert model.last_call_id is not None
    assert ledger.completely_delivered(model.last_call_id) == {1, 2}
    supplied = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert json.loads(supplied["evidence"]) == records
    assert (
        publication_gap(
            claim,
            supported([1, 2]),
            ["question"],
            ledger,
            verification_call_id=model.last_call_id,
        )
        is None
    )


@pytest.mark.parametrize("markers", ["[1, 999]", "【1, 999】", "［1,999］"])
def test_grouped_unknown_member_is_rejected_despite_supported_review(
    markers: str,
) -> None:
    ledger, _context = original_ledger()
    gap = publication_gap(f"Rule {markers}", supported([1]), ["question"], ledger)
    assert gap is not None
    reasons = gap.data["gaps"]
    assert isinstance(reasons, list)
    assert "The draft contains unknown or non-citable source numbers." in reasons


def test_one_supported_question_cannot_cover_a_multi_question_request() -> None:
    ledger, _context = original_ledger()
    gap = publication_gap(
        "Main outcome [1]; alternative [2].",
        supported([1, 2]),
        ["Main outcome", "Alternative scenario", "Required later steps"],
        ledger,
    )
    assert gap is not None
    gaps = gap.data["gaps"]
    assert isinstance(gaps, list)
    assert any("complete question inventory" in str(item) for item in gaps)


@pytest.mark.parametrize("markers", CITATIONS)
def test_grouped_undelivered_member_is_rejected_even_if_review_only_mentions_first(
    markers: str,
) -> None:
    ledger, _context = original_ledger()
    first = ledger.get(1)
    assert first is not None
    ledger.record_delivery(
        "actual-review",
        LLMFlow.ASV3_VERIFICATION.value,
        [{"citation": 1, "text": first.text}],
    )
    gap = publication_gap(
        f"Rule {markers}",
        supported([1]),
        ["question"],
        ledger,
        verification_call_id="actual-review",
    )
    assert gap is not None
    reasons = gap.data["gaps"]
    assert isinstance(reasons, list)
    assert any(
        "not completely delivered" in str(reason) and "[2]" in str(reason)
        for reason in reasons
    )


@pytest.mark.parametrize("markers", CITATIONS)
def test_token_pressure_removes_supplementals_but_keeps_all_grouped_originals(
    markers: str,
) -> None:
    ledger, context = original_ledger()
    model, llm = selected_model(context)
    required = json.loads(ledger.serialize_records([1, 2], required=[1, 2]))
    model.invoke_text(
        "Verify complete grouped originals",
        json.dumps(
            {
                "claim": f"Rule {markers}",
                "scenario": "facts",
                "evidence": json.dumps(
                    [*required, {"citation": 3, "text": "SUPPLEMENTAL " * 10000}]
                ),
            }
        ),
        LLMFlow.ASV3_VERIFICATION,
        max_tokens=1000,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert json.loads(payload["evidence"]) == required
    assert payload["supplemental_evidence_omitted"] is True
    assert model.last_call_id and ledger.completely_delivered(model.last_call_id) == {
        1,
        2,
    }
    # A required member that cannot fit must stop, rather than disappear silently.
    with pytest.raises(RunStopped, match="Complete cited evidence"):
        model._fit(
            "Verify",
            json.dumps(
                {
                    "claim": f"Rule {markers}",
                    "evidence": json.dumps(
                        [required[0], {"citation": 2, "text": "REQUIRED " * 10000}]
                    ),
                }
            ),
            [],
            max_tokens=1000,
        )
    assert llm.invoke.call_count == 1


def test_token_pressure_keeps_originals_for_details_omitted_from_the_rewritten_answer() -> (
    None
):
    ledger, context = original_ledger()
    model, llm = selected_model(context)
    originals = json.loads(ledger.serialize_records([1, 2], required=[1, 2]))
    reference = {"draft": "Main result [1]; relevant later procedure [2]."}
    model.invoke_text(
        "Check omitted source-supported details",
        json.dumps(
            {
                "claim": "Main result [1].",
                "scenario": "facts",
                "preservation_reference": reference,
                "evidence": json.dumps(
                    [*originals, {"citation": 3, "text": "SUPPLEMENTAL " * 10000}]
                ),
            }
        ),
        LLMFlow.ASV3_VERIFICATION,
        max_tokens=1000,
    )
    payload = json.loads(llm.invoke.call_args.kwargs["prompt"][1].content)
    assert payload["preservation_reference"] == reference
    assert json.loads(payload["evidence"]) == originals
    assert model.last_call_id and ledger.completely_delivered(model.last_call_id) == {
        1,
        2,
    }
