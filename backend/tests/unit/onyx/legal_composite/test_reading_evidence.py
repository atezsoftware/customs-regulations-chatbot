"""Ordinal research supports remain bound to unchanged, actually delivered originals."""

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import cast

import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.asv3.witnesses import original_witness_spans
from onyx.legal_composite.acquisition import InvalidSourceAction
from onyx.legal_composite.draft_context import (
    decode_draft_context,
    encode_draft_context,
)
from onyx.legal_composite.models import (
    GapResolution,
    IssueResearchStep,
    MaterialDependencyRequest,
    SourceAction,
)
from onyx.legal_composite.reading_evidence import (
    IssueReadingResponse,
    ReadingRequirement,
    ReadingSupport,
    ReadingWitnessManifest,
    number_reading_witnesses,
    reading_witness_manifest,
    resolve_issue_reading,
)
from onyx.legal_composite.requirements import RequirementLedger, support_is_original
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.legal_composite.test_span_support import original, plan

CALL_ID = "synthetic-reading-call"
RULE = "A fictional application requires both approval and proof."
OTHER_RULE = "A fictional deadline begins on the notification date."


def evidence(
    *texts: str,
) -> tuple[EvidenceLedger, list[dict[str, JsonValue]], ReadingWitnessManifest]:
    ledger = EvidenceLedger()
    items = [original(text, f"original-{index}") for index, text in enumerate(texts, 1)]
    for item in items:
        assert item.search_doc is not None and item.chunk_id is not None
        item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    ledger.add(items, RunContext())
    raw: list[JsonValue] = json.loads(
        ledger.serialize_records(ledger.citation_numbers(), include_witness_spans=True)
    )
    records = cast(list[dict[str, JsonValue]], number_reading_witnesses(raw))
    ledger.record_delivery(CALL_ID, LLMFlow.LEGAL_COMPOSITE_RESEARCH.value, records)
    return ledger, records, reading_witness_manifest(records, call_id=CALL_ID)


def requirement(
    citation: int = 1, span_number: int = 1, *, identity: str = "approval_rule"
) -> ReadingRequirement:
    return ReadingRequirement(
        requirement_id=identity,
        need_id="approval",
        dimension="procedure",
        rule="Both fictional conditions must hold.",
        application="Apply each condition to the supplied facts.",
        supports=[ReadingSupport(citation=citation, span_number=span_number)],
    )


def response(*requirements: ReadingRequirement) -> IssueReadingResponse:
    return IssueReadingResponse(
        actions=[],
        ready_to_answer=True,
        remaining_gaps=[],
        requirements=list(requirements),
        issue_gaps={"approval": []},
    )


def test_resolves_exact_delivered_pair_without_mutating_reading_or_ledger() -> None:
    ledger, records, manifest = evidence(RULE, OTHER_RULE)
    value = response(requirement())
    submitted = value.model_dump(mode="json")
    originals = ledger.export()
    resolved = resolve_issue_reading(value, ledger, {1, 2}, manifest, call_id=CALL_ID)
    assert isinstance(resolved, IssueResearchStep)
    support = resolved.requirements[0].supports[0]
    assert support.quotation == RULE
    assert support.span_id == original_witness_spans(1, RULE)[0]["witness_id"]
    assert support_is_original(support, ledger, {1})
    assert value.model_dump(mode="json") == submitted
    assert ledger.export() == originals
    recorded = RequirementLedger(ledger)
    recorded.update(resolved.requirements, plan(), {1, 2})
    binding = recorded.export()[0]["original_bindings"]
    assert isinstance(binding, list) and binding[0]["start"] == 0
    assert binding[0]["text_hash"] == records[0]["text_hash"]


def test_numbering_is_local_idempotent_and_preserves_literal_metadata() -> None:
    ledger, records, _manifest = evidence("A" * 1_601, OTHER_RULE)
    raw = json.loads(ledger.serialize_records([1, 2], include_witness_spans=True))
    raw[0]["metadata"]["literal"] = {
        "span_number": 99,
        "witness_id": "not-a-selector",
        "nested": [{"citation": 2, "span_number": 7}],
    }
    unchanged = deepcopy(raw)
    numbered = number_reading_witnesses(raw)
    assert raw == unchanged and number_reading_witnesses(numbered) == numbered
    first = cast(dict[str, JsonValue], numbered[0])
    first_spans = cast(list[dict[str, JsonValue]], first["witness_spans"])
    second = cast(dict[str, JsonValue], numbered[1])
    second_spans = cast(list[dict[str, JsonValue]], second["witness_spans"])
    assert [span["span_number"] for span in first_spans] == [1, 2, 3]
    assert [span["span_number"] for span in second_spans] == [1]
    assert first["metadata"] == raw[0]["metadata"]
    assert first["text"] == records[0]["text"]
    assert first["text_hash"] == records[0]["text_hash"]


def test_identical_passage_at_different_offsets_keeps_selected_offset() -> None:
    text = "R" * 1_600
    ledger, _records, manifest = evidence(text)
    value = response(requirement(span_number=2))
    resolved = resolve_issue_reading(value, ledger, {1}, manifest, call_id=CALL_ID)
    recorded = RequirementLedger(ledger)
    recorded.update(resolved.requirements, plan(), {1})
    support = resolved.requirements[0].supports[0]
    assert support.quotation == "R" * 800
    assert support.span_id == original_witness_spans(1, text)[1]["witness_id"]
    bindings = recorded.export()[0]["original_bindings"]
    assert isinstance(bindings, list) and bindings[0]["start"] == 800
    assert bindings[0]["end"] == 1_600


def test_adjacent_span_selection_retains_complete_original_conditions() -> None:
    text = "A" * 790 + "AND B must hold; OTHERWISE the outcome is unavailable."
    ledger, _records, manifest = evidence(text)
    needed = requirement()
    needed.supports = [
        ReadingSupport(citation=1, span_number=1),
        ReadingSupport(citation=1, span_number=2),
    ]
    resolved = resolve_issue_reading(
        response(needed), ledger, {1}, manifest, call_id=CALL_ID
    )
    assert (
        "".join(support.quotation for support in resolved.requirements[0].supports)
        == text
    )
    assert all(
        support_is_original(support, ledger, {1})
        for support in resolved.requirements[0].supports
    )


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 0, -1])
@pytest.mark.parametrize("field", ["citation", "span_number"])
def test_support_indices_are_strict_positive_integers(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        ReadingSupport.model_validate({"citation": 1, "span_number": 1, field: value})


@pytest.mark.parametrize("field", ["quotation", "span_id", "witness_id"])
def test_support_transport_does_not_accept_alternate_or_literal_selectors(
    field: str,
) -> None:
    with pytest.raises(ValidationError):
        ReadingSupport.model_validate(
            {"citation": 1, "span_number": 1, field: "invented"}
        )


@pytest.mark.parametrize("citation,number", [(1, 2), (2, 1), (9, 1)])
def test_out_of_range_or_foreign_citation_is_not_silently_picked(
    citation: int, number: int
) -> None:
    ledger, _records, manifest = evidence(RULE)
    unchanged = ledger.export()
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        resolve_issue_reading(
            response(requirement(citation, number)),
            ledger,
            {1, 2, 9},
            manifest,
            call_id=CALL_ID,
        )
    assert ledger.export() == unchanged


def test_unfitted_or_previous_delivery_cannot_authorize_current_support() -> None:
    ledger, records, _manifest = evidence(RULE, OTHER_RULE)
    current_call = "only-first-original-fitted"
    fitted = records[:1]
    ledger.record_delivery(current_call, LLMFlow.LEGAL_COMPOSITE_RESEARCH.value, fitted)
    manifest = reading_witness_manifest(fitted, call_id=current_call)
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        resolve_issue_reading(
            response(requirement(citation=2)),
            ledger,
            {1, 2},
            manifest,
            call_id=current_call,
        )
    unfitted_manifest = reading_witness_manifest(records, call_id=current_call)
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        resolve_issue_reading(
            response(requirement(citation=2)),
            ledger,
            {1, 2},
            unfitted_manifest,
            call_id=current_call,
        )


def test_active_call_and_declared_delivery_must_both_match() -> None:
    ledger, _records, manifest = evidence(RULE)
    with pytest.raises(InvalidSourceAction, match="active delivery"):
        resolve_issue_reading(
            response(requirement()), ledger, {1}, manifest, call_id="another-call"
        )
    with pytest.raises(InvalidSourceAction, match="provided delivered witness"):
        resolve_issue_reading(
            response(requirement()), ledger, set(), manifest, call_id=CALL_ID
        )


@pytest.mark.parametrize("defect", ["text", "hash", "source", "chunk"])
def test_changed_original_fails_even_if_ordinal_is_in_range(
    defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _records, manifest = evidence(RULE)
    item = ledger.get(1)
    assert item is not None
    if defect == "text":
        item.text = OTHER_RULE
    elif defect == "hash":
        item.text_hash = "0" * 64
    elif defect == "source":
        item.source_id = "different-source"
    else:
        item.chunk_id = "different-chunk"
    monkeypatch.setattr(ledger, "get", lambda _citation: item.model_copy(deep=True))
    with pytest.raises(InvalidSourceAction, match="no longer matches"):
        resolve_issue_reading(
            response(requirement()), ledger, {1}, manifest, call_id=CALL_ID
        )


@pytest.mark.parametrize("flag", ["external", "derived", "untrusted", "truncated"])
def test_existing_canonical_original_flags_still_fail_closed(
    flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger, _records, manifest = evidence(RULE)
    item = ledger.get(1)
    assert item is not None
    item.metadata[flag] = True
    monkeypatch.setattr(ledger, "get", lambda _citation: item.model_copy(deep=True))
    with pytest.raises(InvalidSourceAction, match="canonical original"):
        resolve_issue_reading(
            response(requirement()), ledger, {1}, manifest, call_id=CALL_ID
        )


@pytest.mark.parametrize(
    "defect",
    ["number", "foreign_span", "range", "duplicate_record", "hash", "truncated"],
)
def test_manifest_rejects_tampered_numbered_catalogue(defect: str) -> None:
    _ledger, records, _manifest = evidence(RULE, OTHER_RULE)
    invalid = deepcopy(records)
    spans = cast(list[dict[str, JsonValue]], invalid[0]["witness_spans"])
    if defect == "number":
        spans[0]["span_number"] = 2
    elif defect == "foreign_span":
        other_spans = cast(list[dict[str, JsonValue]], invalid[1]["witness_spans"])
        spans[0]["witness_id"] = other_spans[0]["witness_id"]
    elif defect == "range":
        spans[0]["end_char"] = 1
    elif defect == "duplicate_record":
        invalid.append(invalid[0])
    elif defect == "hash":
        invalid[0]["text_hash"] = "0" * 64
    else:
        invalid[0]["truncated"] = True
    with pytest.raises(InvalidSourceAction, match="catalogue"):
        reading_witness_manifest(invalid, call_id=CALL_ID)


def test_manifest_is_immutable_and_forged_or_duplicate_entries_are_rechecked() -> None:
    ledger, _records, manifest = evidence(RULE)
    with pytest.raises(FrozenInstanceError):
        setattr(manifest.witnesses[0], "witness_id", "invented")
    duplicate = replace(manifest, witnesses=manifest.witnesses + manifest.witnesses)
    with pytest.raises(InvalidSourceAction, match="catalogue"):
        resolve_issue_reading(
            response(requirement()), ledger, {1}, duplicate, call_id=CALL_ID
        )
    foreign = replace(manifest.witnesses[0], witness_id="invented")
    forged = replace(manifest, witnesses=(foreign,))
    with pytest.raises(InvalidSourceAction, match="catalogue"):
        resolve_issue_reading(
            response(requirement()), ledger, {1}, forged, call_id=CALL_ID
        )


def test_one_invalid_requirement_cannot_partially_change_existing_requirement_ledger() -> (
    None
):
    ledger, _records, manifest = evidence(RULE)
    recorded = RequirementLedger(ledger)
    good = resolve_issue_reading(
        response(requirement()), ledger, {1}, manifest, call_id=CALL_ID
    )
    recorded.update(good.requirements, plan(), {1})
    frozen = recorded.export()
    pending = response(
        requirement(identity="new-valid"), requirement(span_number=8, identity="bad")
    )
    with pytest.raises(InvalidSourceAction):
        resolved = resolve_issue_reading(
            pending, ledger, {1}, manifest, call_id=CALL_ID
        )
        recorded.update(resolved.requirements, plan(), {1})
    assert recorded.export() == frozen


def test_non_support_reading_fields_and_defaults_preserve_existing_step_semantics() -> (
    None
):
    ledger, _records, manifest = evidence(RULE)
    value = response(requirement())
    value.actions = [
        SourceAction(
            need_ids=["approval"],
            tool="read_provision",
            arguments={"source_id": "observed", "article": "7"},
        )
    ]
    value.ready_to_answer = False
    value.remaining_gaps = ["Missing operative continuation"]
    value.reconsider_citations = [1]
    value.gap_resolutions = [
        GapResolution(
            need_id="approval", gap="prior exact gap", requirement_ids=["approval_rule"]
        )
    ]
    value.material_dependencies = [
        MaterialDependencyRequest(
            need_ids=["approval"],
            origin_citation=1,
            instrument_name="Fictional rules",
            article="7",
            reason="Could alter the actual outcome",
        )
    ]
    value.issue_gaps = {"approval": ["Missing operative continuation"]}
    value.requirements[0].supersedes_requirement_ids = ["obsolete-rule"]
    value.requirements[0].missing_user_facts = ["approval status"]
    resolved = resolve_issue_reading(value, ledger, {1}, manifest, call_id=CALL_ID)
    submitted = value.model_dump(mode="json")
    accepted = resolved.model_dump(mode="json")
    assert {key: item for key, item in accepted.items() if key != "requirements"} == {
        key: item for key, item in submitted.items() if key != "requirements"
    }
    assert {
        key: item
        for key, item in accepted["requirements"][0].items()
        if key != "supports"
    } == {
        key: item
        for key, item in submitted["requirements"][0].items()
        if key != "supports"
    }
    empty = resolve_issue_reading(response(), ledger, {1}, manifest, call_id=CALL_ID)
    assert empty.model_dump(mode="json") == IssueResearchStep.model_validate(
        response().model_dump(mode="json")
    ).model_dump(mode="json")


def test_constructed_response_is_strictly_revalidated_before_host_translation() -> None:
    ledger, _records, manifest = evidence(RULE)
    invalid = response(requirement())
    invalid.requirements[0].supports[0] = ReadingSupport.model_construct(
        citation=1, span_number=True
    )
    with pytest.raises(InvalidSourceAction, match="transport schema"):
        resolve_issue_reading(invalid, ledger, {1}, manifest, call_id=CALL_ID)


def test_actual_lossless_draft_codec_preserves_all_numbered_witness_fields() -> None:
    _ledger, records, _manifest = evidence(RULE)
    payload: dict[str, JsonValue] = {
        "original_evidence": cast(list[JsonValue], records),
        "draft": {
            "answer": "Heading\n\nLiteral legal passage [1]",
            "sections": [
                {
                    "section_id": "s",
                    "need_ids": ["approval"],
                    "text": "Heading\n\nLiteral legal passage [1]",
                    "claim_ids": ["c"],
                }
            ],
            "claims": [
                {
                    "claim_id": "c",
                    "section_id": "s",
                    "need_ids": ["approval"],
                    "answer_excerpt": "Literal legal passage [1]",
                    "supports": [],
                }
            ],
            "requirements": [],
        },
    }
    frozen = deepcopy(payload)
    encoded = encode_draft_context(payload)
    assert encoded is not None
    assert encoded.payload["original_evidence"] == payload["original_evidence"]
    assert decode_draft_context(encoded.payload) == payload == frozen
