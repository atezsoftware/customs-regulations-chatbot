import json

import pytest
from pydantic import ValidationError

from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchContractError,
    VertexBatchRequest,
    build_vertex_jsonl,
)
from onyx.regulatory.labeling.provider import (
    TaxonomyDefinition,
    build_labeling_request,
    parse_labeling_batch_output,
    validate_labeling_response,
)


def _taxonomy() -> TaxonomyDefinition:
    return TaxonomyDefinition.model_validate(
        {
            "name": "Fixture",
            "labels": [
                {"id": "a", "name": "A", "description": "Applies to target A"},
                {"id": "b", "name": "B", "description": "Applies to target B"},
            ],
        }
    )


def _request(text: str = "Target A") -> VertexBatchRequest:
    return build_labeling_request(
        chunk_id="existing-id",
        text=text,
        context="Context B",
        taxonomy=_taxonomy(),
        source_hash="a" * 64,
    )


def test_structured_batch_request_binds_schema_context_source_and_identity() -> None:
    request = _request()
    wire = json.loads(build_vertex_jsonl([request]))
    config = wire["request"]["generationConfig"]
    assert config["responseMimeType"] == "application/json"
    assert not {"temperature", "topP", "topK", "thinkingBudget"} & config.keys()
    assert "systemInstruction" in wire["request"]
    assert request.request_hash != _request("Different target").request_hash
    payload = json.loads(request.prompt)
    assert payload["target"]["chunk_id"] == "existing-id"
    assert payload["target"]["text"] == "Target A"
    assert payload["interpretation_context"] == "Context B"
    enum = config["responseJsonSchema"]["properties"]["labels"]["items"]["properties"][
        "label_id"
    ]["enum"]
    assert enum == ["a", "b"]


def test_duplicate_and_empty_taxonomy_cannot_start_labeling() -> None:
    with pytest.raises(ValidationError):
        TaxonomyDefinition(name="Empty", labels=[])
    label = {"id": "a", "name": "A", "description": "A criterion"}
    with pytest.raises(ValidationError):
        TaxonomyDefinition.model_validate(
            {"name": "duplicate", "labels": [label, label]}
        )


@pytest.mark.parametrize(
    "response",
    [
        {
            "labels": [{"label_id": "unknown", "evidence_quote": "Target A"}],
            "abstained": False,
        },
        {
            "labels": [{"label_id": "a", "evidence_quote": "Context B"}],
            "abstained": False,
        },
        {
            "labels": [{"label_id": "a", "evidence_quote": "Target A"}],
            "abstained": True,
        },
        {
            "labels": [{"label_id": "a", "evidence_quote": "Target A"}] * 2,
            "abstained": False,
        },
        {"labels": [], "abstained": "false"},
    ],
)
def test_response_rejects_unknown_context_only_duplicate_or_inconsistent_labels(
    response: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        validate_labeling_response(
            json.dumps(response), text="Target A", taxonomy=_taxonomy()
        )


def test_valid_outcome_requires_exact_source_evidence() -> None:
    outcome = validate_labeling_response(
        '{"labels":[{"label_id":"a","evidence_quote":"Target A"}],"abstained":false}',
        text="This is Target A.",
        taxonomy=_taxonomy(),
    )
    assert outcome.labels[0].label_id == "a"
    assert not outcome.abstained


def _line(key: str, finish: str = "STOP", *, thought: bool = False) -> str:
    parts = ([{"text": "private reasoning", "thought": True}] if thought else []) + [
        {"text": '{"labels":[],"abstained":false}'}
    ]
    return json.dumps(
        {
            "key": key,
            "response": {
                "candidates": [{"finishReason": finish, "content": {"parts": parts}}]
            },
        }
    )


def test_batch_correlation_rejects_duplicates_and_unexpected_keys() -> None:
    key = _request().request_hash
    with pytest.raises(VertexBatchContractError, match="duplicate"):
        parse_labeling_batch_output(_line(key) + "\n" + _line(key), [key])
    with pytest.raises(VertexBatchContractError, match="unexpected"):
        parse_labeling_batch_output(_line("b" * 64), [key])


def test_batch_does_not_accept_truncated_json_or_include_thoughts() -> None:
    key = _request().request_hash
    failed = parse_labeling_batch_output(_line(key, "MAX_TOKENS"), [key])
    assert failed[key].error is not None
    accepted = parse_labeling_batch_output(_line(key, thought=True), [key])
    assert accepted[key].context == '{"labels":[],"abstained":false}'


def test_partial_output_preserves_missing_items_for_explicit_failure_handling() -> None:
    key = _request().request_hash
    outcomes = parse_labeling_batch_output(
        _line(key), [key, "b" * 64], require_complete=False
    )
    assert set(outcomes) == {key}


def test_structured_request_hash_changes_with_generation_policy() -> None:
    left = VertexBatchRequest(prompt="same", generation_config={"maxOutputTokens": 1})
    right = VertexBatchRequest(prompt="same", generation_config={"maxOutputTokens": 2})
    assert left.request_hash != right.request_hash
