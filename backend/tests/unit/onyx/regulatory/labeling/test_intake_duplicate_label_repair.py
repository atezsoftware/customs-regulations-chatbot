"""A model chooses original duplicate quotes; unique assignments stay fixed."""

import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.duplicate_ack_evidence import DuplicateAckEvidence
from onyx.regulatory.labeling.intake_duplicate_label_repair import (
    DUPLICATE_REPAIR_KEY,
    DuplicateLabelRepair,
    DuplicateLabelRepairAudit,
    DuplicateSourceBinding,
    build_duplicate_label_repair,
    build_duplicate_repair_audit,
    known_duplicate_shape,
    parse_duplicate_output,
    validate_duplicate_selection,
)
from onyx.regulatory.labeling.intake_quote_repair import build_quote_repair
from onyx.regulatory.labeling.intake_quote_resolution import (
    resolve_intake_labeling_response,
)
from onyx.regulatory.labeling.provider import (
    MAX_RESPONSE_BYTES,
    LabelingOutcome,
    TaxonomyDefinition,
)


@pytest.fixture
def frozen() -> dict[str, Any]:
    return json.loads(
        (
            Path(__file__).parent / "fixtures/legacy_duplicate_label_context.json"
        ).read_text()
    )


def repair_for(
    frozen: dict[str, Any], response: str | None = None
) -> DuplicateLabelRepair:
    return build_duplicate_label_repair(
        base_request=VertexBatchRequest.model_validate(frozen["base_request"]),
        rejected_response=frozen["base_response"] if response is None else response,
        text=frozen["target_text"],
        chunk_id=frozen["target_chunk_id"],
        source_hash=frozen["target_source_hash"],
        taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
        provider_binding={"transport": "vertex_direct_v1", "model": "gemini-3.5-flash"},
    )


def selections(repair: DuplicateLabelRepair, choice: int = 0) -> str:
    return json.dumps(
        {
            "labels": [
                {"label_id": label, "candidate_id": values[choice].candidate_id}
                for label, values in repair.candidates.items()
            ],
            "abstained": False,
        }
    )


def ack_line(request: VertexBatchRequest, response: str) -> str:
    return json.dumps(
        {
            "key": request.request_hash,
            "request": request.to_generate_content_request(),
            "response": {
                "candidates": [
                    {"finishReason": "STOP", "content": {"parts": [{"text": response}]}}
                ]
            },
        },
        ensure_ascii=False,
    )


def pointer(
    repair: DuplicateLabelRepair, response: str, *, base: bool
) -> DuplicateAckEvidence:
    request = repair.base_request if base else repair.request
    binding, submission = "a" * 64, "regulatory-labeling-" + ("b" if base else "c") * 64
    line = ack_line(request, response)
    return DuplicateAckEvidence(
        file_id=f"regulatory-labeling-direct/{binding}/{submission.removeprefix('regulatory-labeling-')}/{request.request_hash}.json",
        binding_fingerprint=binding,
        submission_key=submission,
        request_hash=request.request_hash,
        blob_sha256=sha256(line.encode()).hexdigest(),
        blob_byte_count=len(line.encode()),
        line_sha256=sha256(line.encode()).hexdigest(),
        line_byte_count=len(line.encode()),
        response_sha256=sha256(response.encode()).hexdigest(),
    )


def source_binding(frozen: dict[str, Any]) -> DuplicateSourceBinding:
    return DuplicateSourceBinding(
        document_set_id=15,
        user_file_id=str(uuid4()),
        source_asset_id=str(uuid4()),
        intake_id=str(uuid4()),
        package_id=str(uuid4()),
        owner_id=str(uuid4()),
        source_file_id="local-test-source",
        original_file_id="local-test-copy",
        raw_sha256=frozen["source_pdf_sha256"],
        frozen_context_sources_sha256="e" * 64,
        byte_count=frozen["source_pdf_bytes"],
    )


def test_actual_five_assignments_keep_three_unique_quotes_and_two_original_candidates(
    frozen: dict[str, Any],
) -> None:
    before = deepcopy(frozen)
    repair = repair_for(frozen)
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
    assert known_duplicate_shape(frozen["base_response"], taxonomy)
    assert set(repair.candidates) == {"SUB.TRD.PRODUCT_COMPLIANCE"}
    assert len(repair.candidates["SUB.TRD.PRODUCT_COMPLIANCE"]) == 2
    assert len(repair.retained_quotes) == 3
    original = LabelingOutcome.model_validate_json(frozen["base_response"])
    for entry in original.labels:
        resolved, _ = resolve_intake_labeling_response(
            LabelingOutcome(labels=[entry], abstained=False).model_dump_json(),
            text=frozen["target_text"],
            taxonomy=taxonomy,
        )
        if entry.label_id in repair.retained_quotes:
            assert (
                repair.retained_quotes[entry.label_id]
                == resolved.labels[0].evidence_quote
            )
    for candidate in repair.candidates["SUB.TRD.PRODUCT_COMPLIANCE"]:
        assert candidate.canonical_quote in frozen["target_text"]
        assert (
            candidate.original_quote_sha256
            == sha256(
                original.labels[candidate.original_index].evidence_quote.encode()
            ).hexdigest()
        )
    old, new = json.loads(repair.base_request.prompt), json.loads(repair.request.prompt)
    assert new["target"] == old["target"]
    assert new["interpretation_context"] == old["interpretation_context"]
    assert new["taxonomy"] == old["taxonomy"]
    assert set(new) - set(old) == {DUPLICATE_REPAIR_KEY}
    assert repair.request.generation_config is not None
    assert repair.base_request.generation_config is not None
    assert {
        k: v
        for k, v in repair.request.generation_config.items()
        if k != "responseJsonSchema"
    } == {
        k: v
        for k, v in repair.base_request.generation_config.items()
        if k != "responseJsonSchema"
    }
    assert (
        sha256(frozen["actual_base_cache_blob"].encode()).hexdigest()
        == frozen["actual_base_cache_blob_sha256"]
    )
    assert frozen == before
    with pytest.raises(ValueError, match="known, unique"):
        build_quote_repair(
            base_request=repair.base_request,
            rejected_response=frozen["base_response"],
            text=frozen["target_text"],
            chunk_id=frozen["target_chunk_id"],
            source_hash=frozen["target_source_hash"],
            taxonomy=taxonomy,
        )


@pytest.mark.parametrize("choice", [0, 1])
def test_model_selection_preserves_exact_unique_set_and_all_fixed_quotes(
    frozen: dict[str, Any], choice: int
) -> None:
    repair = repair_for(frozen)
    outcome = validate_duplicate_selection(
        selections(repair, choice),
        repair=repair,
        text=frozen["target_text"],
        taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
    )
    quotes = {entry.label_id: entry.evidence_quote for entry in outcome.labels}
    assert len(quotes) == 4 and not outcome.abstained
    assert all(
        quotes[label] == quote for label, quote in repair.retained_quotes.items()
    )
    assert (
        quotes["SUB.TRD.PRODUCT_COMPLIANCE"]
        == repair.candidates["SUB.TRD.PRODUCT_COMPLIANCE"][choice].canonical_quote
    )


@pytest.mark.parametrize(
    "change",
    [
        "unknown",
        "empty",
        "unique",
        "abstained",
        "blank",
        "unmatched",
        "oversize",
        "schema",
    ],
)
def test_original_invalid_results_never_permit_selection(
    frozen: dict[str, Any], change: str
) -> None:
    value = json.loads(frozen["base_response"])
    if change == "unknown":
        value["labels"][0]["label_id"] = "unknown"
    elif change == "empty":
        value["labels"] = []
    elif change == "unique":
        value["labels"].pop(1)
    elif change == "abstained":
        value["abstained"] = True
    elif change in {"blank", "unmatched"}:
        value["labels"][0]["evidence_quote"] = (
            " " if change == "blank" else "İlgisiz ve kaynakta bulunmayan alıntı."
        )
    elif change == "schema":
        value["unexpected"] = True
    response = (
        " " * (MAX_RESPONSE_BYTES + 1) if change == "oversize" else json.dumps(value)
    )
    with pytest.raises(ValueError):
        repair_for(frozen, response)


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "duplicate",
        "unknown",
        "unique-ID",
        "candidate",
        "quote",
        "empty",
        "abstained",
        "bool",
        "oversize",
    ],
)
def test_selection_cannot_change_quotes_labels_or_group_count(
    frozen: dict[str, Any], change: str
) -> None:
    repair = repair_for(frozen)
    value = json.loads(selections(repair))
    if change in {"missing", "empty"}:
        value["labels"] = []
    elif change == "duplicate":
        value["labels"].append(deepcopy(value["labels"][0]))
    elif change == "unknown":
        value["labels"][0]["label_id"] = "unknown"
    elif change == "unique-ID":
        value["labels"][0]["label_id"] = next(iter(repair.retained_quotes))
    elif change == "candidate":
        value["labels"][0]["candidate_id"] = "fabricated-candidate"
    elif change == "quote":
        value["labels"][0]["evidence_quote"] = frozen["target_text"]
    elif change == "abstained":
        value["abstained"] = True
    elif change == "bool":
        value["abstained"] = 0
    response = (
        " " * (MAX_RESPONSE_BYTES + 1) if change == "oversize" else json.dumps(value)
    )
    with pytest.raises(ValueError):
        validate_duplicate_selection(
            response,
            repair=repair,
            text=frozen["target_text"],
            taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
        )


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "context",
        "taxonomy",
        "retained",
        "candidate",
        "generation",
        "rejected",
    ],
)
def test_frozen_authority_cannot_drift_even_when_candidate_ids_look_valid(
    frozen: dict[str, Any], change: str
) -> None:
    repair = repair_for(frozen)
    response, text = selections(repair), frozen["target_text"]
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
    if change == "source":
        text += " changed"
    elif change == "context":
        prompt = json.loads(repair.base_request.prompt)
        prompt["interpretation_context"] += " changed"
        repair = repair.model_copy(
            update={
                "base_request": repair.base_request.model_copy(
                    update={"prompt": json.dumps(prompt)}
                )
            }
        )
    elif change == "taxonomy":
        taxonomy = taxonomy.model_copy(update={"name": taxonomy.name + " changed"})
    elif change == "retained":
        repair.retained_quotes[next(iter(repair.retained_quotes))] = "changed"
    elif change == "candidate":
        repair.candidates["SUB.TRD.PRODUCT_COMPLIANCE"][0] = repair.candidates[
            "SUB.TRD.PRODUCT_COMPLIANCE"
        ][0].model_copy(update={"canonical_quote": "changed"})
    elif change == "generation":
        assert repair.request.generation_config is not None
        repair.request.generation_config["maxOutputTokens"] = 1
    else:
        repair = repair.model_copy(
            update={"rejected_response": repair.rejected_response + " "}
        )
    with pytest.raises(ValueError):
        validate_duplicate_selection(
            response, repair=repair, text=text, taxonomy=taxonomy
        )


def test_multiple_groups_require_one_own_candidate_per_original_group(
    frozen: dict[str, Any],
) -> None:
    value = json.loads(frozen["base_response"])
    value["labels"].append(deepcopy(value["labels"][2]))
    repair = repair_for(frozen, json.dumps(value))
    assert len(repair.candidates) == 2 and len(repair.retained_quotes) == 2
    taxonomy = TaxonomyDefinition.model_validate(frozen["taxonomy"])
    outcome = validate_duplicate_selection(
        selections(repair), repair=repair, text=frozen["target_text"], taxonomy=taxonomy
    )
    assert len(outcome.labels) == 4
    value = json.loads(selections(repair))
    value["labels"].pop()
    with pytest.raises(ValueError):
        validate_duplicate_selection(
            json.dumps(value),
            repair=repair,
            text=frozen["target_text"],
            taxonomy=taxonomy,
        )


def test_audit_records_only_pointers_and_exact_small_responses(
    frozen: dict[str, Any],
) -> None:
    repair = repair_for(frozen)
    response = selections(repair)
    audit = build_duplicate_repair_audit(
        repair=repair,
        selection_response=response,
        base_ack=pointer(repair, repair.rejected_response, base=True),
        repair_ack=pointer(repair, response, base=False),
        source_binding=source_binding(frozen),
        text=frozen["target_text"],
        taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
    )
    assert audit.rejected_response == frozen["base_response"]
    assert audit.attempt == 1 and len(audit.outcome.labels) == 4
    assert "actual_base_cache_blob" not in audit.model_dump_json()
    for field, value in [
        ("attempt", 2),
        ("attempt", True),
        ("version", True),
        ("version", 2),
        ("method", "intake_abstention_reconsideration"),
    ]:
        data = audit.model_dump(mode="json")
        data[field] = value
        with pytest.raises(ValueError):
            DuplicateLabelRepairAudit.model_validate(data)
    data = pointer(repair, response, base=False).model_dump()
    data["response_sha256"] = "f" * 64
    with pytest.raises(ValueError):
        build_duplicate_repair_audit(
            repair=repair,
            selection_response=response,
            base_ack=audit.base_ack,
            repair_ack=DuplicateAckEvidence.model_validate(data),
            source_binding=audit.source_binding,
            text=frozen["target_text"],
            taxonomy=TaxonomyDefinition.model_validate(frozen["taxonomy"]),
        )


@pytest.mark.parametrize("change", ["echo", "key", "stop", "extra", "missing", "size"])
def test_actual_complete_response_echo_and_byte_caps_are_required(
    frozen: dict[str, Any], change: str
) -> None:
    repair = repair_for(frozen)
    value = json.loads(ack_line(repair.request, selections(repair)))
    if change == "echo":
        value["request"]["contents"][0]["parts"][0]["text"] += " changed"
    elif change == "key":
        value["key"] = "f" * 64
    elif change == "stop":
        value["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
    line = json.dumps(value)
    lines = (
        []
        if change == "missing"
        else [line, line]
        if change == "extra"
        else [" " * (MAX_RESPONSE_BYTES + 1)]
        if change == "size"
        else [line]
    )
    with pytest.raises(ValueError):
        parse_duplicate_output(lines, request=repair.request, on_progress=lambda: None)
