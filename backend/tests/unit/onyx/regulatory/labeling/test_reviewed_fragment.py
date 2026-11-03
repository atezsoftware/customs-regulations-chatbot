"""Exact reviewed source structure, rather than numeric text alone, admits rows."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from onyx.regulatory.labeling.reviewed_fragment import (
    ReviewedFragmentProof,
    ReviewedFragmentSource,
    build_reviewed_fragment_proof,
    require_base_ack_line,
)

HOME = Path(__file__).parent / "fixtures"
FROZEN = json.loads((HOME / "reviewed_blank_form_rows_context.json").read_text())
RAW = (HOME / "reviewed-blank-form-source.pdf").read_bytes()
TARGET = FROZEN["target_chunk_id"]
FILE = UUID(FROZEN["sources"][0]["file_id"])
ABSTAINED = '{"labels":[],"abstained":true}'


def sources() -> list[ReviewedFragmentSource]:
    return [ReviewedFragmentSource(**deepcopy(s)) for s in FROZEN["sources"]]


def proof(values: list[ReviewedFragmentSource] | None = None) -> ReviewedFragmentProof:
    return build_reviewed_fragment_proof(
        raw_pdf=RAW,
        source_asset_id=UUID("00000000-0000-0000-0000-000000000001"),
        user_file_id=FILE,
        chunk_id=TARGET,
        sources=values if values is not None else sources(),
    )


def test_actual_all_nine_reviewed_sources_qualify_only_shareholder_rows() -> None:
    result = proof()
    assert result.profile_kind == "blank_form_rows"
    assert result.review_basis == "content_addressed_public_source_visual_review"
    assert len(result.reviewed_page_sha256) == 2
    assert result.source_byte_count == 82928
    values = sources()
    with pytest.raises(ValueError, match="shareholder"):
        build_reviewed_fragment_proof(
            raw_pdf=RAW,
            source_asset_id=uuid4(),
            user_file_id=FILE,
            chunk_id=values[7].chunk_id,
            sources=values,
        )


@pytest.mark.parametrize("index", range(9))
@pytest.mark.parametrize(
    "change", ["text", "metadata", "heading", "type", "order", "active", "file"]
)
def test_every_current_context_member_is_bound(index: int, change: str) -> None:
    values = sources()
    source = values[index]
    if change == "text":
        values[index] = replace(source, current_text=source.current_text + "x")
    elif change == "metadata":
        source.current_snapshot["chunk_metadata"]["manual_drift"] = True
    elif change == "heading":
        source.current_snapshot["heading_path"] = ["different"]
    elif change == "type":
        source.current_snapshot["chunk_type"] = "paragraph"
    elif change == "order":
        source.current_snapshot["position"] = 99
    elif change == "active":
        values[index] = replace(source, active=False)
    else:
        values[index] = replace(source, file_id=str(uuid4()))
    with pytest.raises(ValueError):
        proof(values)


@pytest.mark.parametrize(
    "change",
    [
        "cell",
        "signature",
        "footnote",
        "own_fact",
        "own_scope",
        "structural",
        "missing",
        "duplicate",
        "raw",
    ],
)
def test_closed_source_and_metadata_variations_reject(change: str) -> None:
    values = sources()
    target = values[4]
    if change in {"cell", "signature", "footnote"}:
        index = 4 if change == "cell" else 8
        source = values[index]
        if change == "signature":
            source.snapshot["heading_path"] = source.current_snapshot[
                "heading_path"
            ] = ["EK – 1"]
        else:
            text = source.text + " X"
            values[index] = replace(
                source,
                text=text,
                current_text=text,
                canonical_text_sha256=context_hash(text),
            )
    elif change in {"own_fact", "own_scope", "structural"}:
        for snapshot in [target.snapshot, target.current_snapshot]:
            fields = snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"]
            if change == "own_fact":
                fields["annex_title"]["source_chunk_id"] = values[7].chunk_id
            elif change == "own_scope":
                fields["annex_title"]["evidence_scope"] = "heading"
            else:
                fields["table_index"]["source"] = "llm"
    elif change == "missing":
        values.pop()
    elif change == "duplicate":
        values[-1] = values[0]
    if change == "raw":
        with pytest.raises(ValueError, match="retained source"):
            build_reviewed_fragment_proof(
                raw_pdf=RAW + b"x",
                source_asset_id=uuid4(),
                user_file_id=FILE,
                chunk_id=TARGET,
                sources=values,
            )
    else:
        with pytest.raises(ValueError):
            proof(values)


def test_qualification_is_id_neutral_but_binds_current_ids() -> None:
    values = sources()
    renamed = {s.chunk_id: f"test-{i}" for i, s in enumerate(values)}
    updated = []
    for source in values:
        for snapshot in [source.snapshot, source.current_snapshot]:
            if "context_member_ids" in snapshot:
                snapshot["context_member_ids"] = [
                    renamed[k] for k in snapshot["context_member_ids"]
                ]
            for fact in snapshot["chunk_metadata"]["intake_legal_metadata"][
                "fields"
            ].values():
                if fact.get("source_chunk_id") in renamed:
                    fact["source_chunk_id"] = renamed[fact["source_chunk_id"]]
        updated.append(replace(source, chunk_id=renamed[source.chunk_id]))
    result = build_reviewed_fragment_proof(
        raw_pdf=RAW,
        source_asset_id=uuid4(),
        user_file_id=FILE,
        chunk_id=renamed[TARGET],
        sources=updated,
    )
    assert result.transcription_sha256 == proof().transcription_sha256
    assert result.context_sources_sha256 != proof().context_sources_sha256


@pytest.mark.parametrize(
    "change",
    [None, "nonempty", "not_abstained", "finish", "request", "key", "error", "status"],
)
def test_only_genuine_correlated_stop_empty_base_ack_is_authority(
    change: str | None,
) -> None:
    request = VertexBatchRequest.model_validate(FROZEN["base_request"])
    response = (
        ABSTAINED
        if change not in {"nonempty", "not_abstained"}
        else '{"labels":[],"abstained":false}'
    )
    row: dict[str, Any] = {
        "key": request.request_hash,
        "request": request.to_generate_content_request(),
        "response": {
            "candidates": [
                {"finishReason": "STOP", "content": {"parts": [{"text": response}]}}
            ]
        },
    }
    if change == "finish":
        row["response"]["candidates"][0]["finishReason"] = "MAX_TOKENS"
    if change == "request":
        row["request"] = {}
    if change == "key":
        row["key"] = "0" * 64
    if change == "error":
        row["error"] = {"message": "failure"}
    if change == "status":
        row["status"] = "failed"
    taxonomy = TaxonomyDefinition.model_validate(
        {
            "name": "Test",
            "labels": [{"id": "duty", "name": "Duty", "description": "A stated duty."}],
        }
    )
    if change:
        with pytest.raises(ValueError):
            require_base_ack_line(
                json.dumps(row), request, text=sources()[4].text, taxonomy=taxonomy
            )
    else:
        assert (
            require_base_ack_line(
                json.dumps(row), request, text=sources()[4].text, taxonomy=taxonomy
            )
            == ABSTAINED
        )
