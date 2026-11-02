"""Source-bound single-attempt semantic repair contracts."""

import json
from typing import Any

import pytest

from onyx.regulatory.labeling.intake_abstention_repair import (
    parse_abstention_repair_output,
)
from tests.unit.onyx.regulatory.labeling.abstention_repair_test_helpers import (
    BINDING,
    TEXT,
    audit,
    build,
    line,
    response,
    taxonomy,
    validate_audit,
)


@pytest.mark.parametrize(
    "change", ["key", "echo", "finish", "candidates", "error", "duplicate", "missing"]
)
def test_provider_envelope_must_prove_one_complete_correlated_repair(
    change: str,
) -> None:
    repair = build()
    payload = json.loads(line(repair, response()))
    if change == "key":
        payload["key"] = "f" * 64
    elif change == "echo":
        payload["request"] = repair.base_request.to_generate_content_request()
    elif change == "finish":
        payload["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
    elif change == "candidates":
        payload["response"]["candidates"] *= 2
    elif change == "error":
        payload["error"] = {"code": 500}
    lines = [json.dumps(payload)]
    if change == "duplicate":
        lines *= 2
    elif change == "missing":
        lines = [" "]
    with pytest.raises(ValueError):
        parse_abstention_repair_output(lines, repair=repair, on_progress=lambda: None)


@pytest.mark.parametrize(
    "field",
    [
        "canonical_text_sha256",
        "context_text_sha256",
        "taxonomy_sha256",
        "provider_binding_sha256",
        "repair_request_hash",
        "output_response_sha256",
        "output_row_sha256",
        "rejected_response_sha256",
        "base_request_hash",
    ],
)
def test_stored_hash_tampering_is_rejected(field: str) -> None:
    repair = build()
    receipt = audit(repair, response()).model_dump(mode="json")
    receipt[field] = "f" * 64
    with pytest.raises(ValueError):
        validate_audit(receipt, repair)


@pytest.mark.parametrize(
    "change",
    ["target", "source", "taxonomy", "binding", "model", "labels", "assignments"],
)
def test_current_data_and_accepted_assignments_must_still_match(change: str) -> None:
    repair = build()
    receipt = audit(repair, response())
    values: dict[str, Any] = {}
    if change == "target":
        values["chunk_id"] = "other"
    elif change == "source":
        values["text"] = TEXT + "!"
    elif change == "taxonomy":
        values["taxonomy"] = taxonomy().model_copy(update={"name": "other"})
    elif change in {"binding", "model"}:
        values["provider_binding"] = {
            **BINDING,
            "project" if change == "binding" else "model": "changed",
        }
    elif change == "labels":
        values["labels"] = []
    else:
        values["assignments"] = [{"label_id": "origin", "evidence_quote": "different"}]
    with pytest.raises(ValueError):
        validate_audit(receipt, repair, **values)
