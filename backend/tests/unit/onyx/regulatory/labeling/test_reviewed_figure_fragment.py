"""A reviewed diagram tail keeps its datum without inventing a classification."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchRequest
from onyx.regulatory.labeling.intake_abstention_repair import build_abstention_repair
from onyx.regulatory.labeling.provider import TaxonomyDefinition
from onyx.regulatory.labeling.reviewed_fragment import (
    ReviewedFigureFragmentProof,
    ReviewedFragmentSource,
    build_reviewed_figure_proof,
    require_base_ack_line,
)

HOME = Path(__file__).parent / "fixtures"
FROZEN = json.loads((HOME / "reviewed_figure_caption_context.json").read_text())
RAW = (HOME / "reviewed-figure-caption-source.pdf").read_bytes()
TARGET = FROZEN["target_chunk_id"]
FILE = UUID(FROZEN["sources"][0]["file_id"])


def sources() -> list[ReviewedFragmentSource]:
    return [ReviewedFragmentSource(**deepcopy(s)) for s in FROZEN["sources"]]


def proof(
    values: list[ReviewedFragmentSource] | None = None,
) -> ReviewedFigureFragmentProof:
    return build_reviewed_figure_proof(
        raw_pdf=RAW,
        source_asset_id=uuid4(),
        user_file_id=FILE,
        chunk_id=TARGET,
        sources=values if values is not None else sources(),
    )


def test_actual_public_figure_and_both_actual_stop_empty_requests() -> None:
    result = proof()
    assert (
        result.source_byte_count == 4526449 and result.profile_kind == "figure_caption"
    )
    base = FROZEN["base_cache_blob"]
    semantic = FROZEN["semantic_cache_blob"]
    request = VertexBatchRequest.model_validate(base["request_payload"])
    taxonomy = TaxonomyDefinition.model_validate(json.loads(request.prompt)["taxonomy"])
    text = next(s.text for s in sources() if s.chunk_id == TARGET)
    response = require_base_ack_line(
        base["line"], request, text=text, taxonomy=taxonomy
    )
    repair = build_abstention_repair(
        base_request=request,
        rejected_response=response,
        text=text,
        chunk_id=TARGET,
        source_hash=context_hash(text),
        taxonomy=taxonomy,
        provider_binding={"model": "gemini-3.5-flash", "transport": "vertex_direct_v1"},
    )
    assert repair.request == VertexBatchRequest.model_validate(
        semantic["request_payload"]
    )
    assert (
        repair.request.request_hash
        == "90e4f43d76daaa249c1eae7238da52484bf0433f45a9f53d3035fc24a8c19ecb"
    )
    assert (
        require_base_ack_line(
            semantic["line"], repair.request, text=text, taxonomy=taxonomy
        )
        == response
    )


@pytest.mark.parametrize("index", range(27))
@pytest.mark.parametrize("field", ["text", "metadata", "heading", "type", "active"])
def test_every_current_figure_context_member_is_bound(index: int, field: str) -> None:
    values = sources()
    s = values[index]
    if field == "text":
        values[index] = replace(s, current_text=s.current_text + "x")
    elif field == "metadata":
        s.current_snapshot["chunk_metadata"]["changed"] = True
    elif field == "heading":
        s.current_snapshot["heading_path"] = ["changed"]
    elif field == "type":
        s.current_snapshot["chunk_type"] = "changed"
    else:
        values[index] = replace(s, active=False)
    with pytest.raises(ValueError):
        proof(values)


@pytest.mark.parametrize(
    "change",
    [
        "datum",
        "caption",
        "operative",
        "diagram",
        "next_clause",
        "scope",
        "witness",
        "paragraph",
        "missing",
        "raw",
        "foreign",
    ],
)
def test_closed_reviewed_figure_and_honest_metadata_roles_reject_variations(
    change: str,
) -> None:
    values = sources()
    target = next(s for s in values if s.chunk_id == TARGET)
    if change in {"datum", "caption", "operative", "diagram", "next_clause"}:
        victim = (
            target
            if change in {"datum", "caption", "operative"}
            else next(
                s
                for s in values
                if s.snapshot["position"] == (44 if change == "diagram" else 46)
            )
        )
        text = (
            target.text.replace("48", "49")
            if change == "datum"
            else victim.text + " changed"
        )
        values[values.index(victim)] = replace(
            victim,
            text=text,
            current_text=text,
            canonical_text_sha256=context_hash(text),
        )
    elif change in {"scope", "witness", "paragraph"}:
        for snapshot in [target.snapshot, target.current_snapshot]:
            fields = snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"]
            if change == "scope":
                fields["section_title"]["evidence_scope"] = "body"
            elif change == "witness":
                fields["annex_title"]["source_chunk_id"] = TARGET
            else:
                fields["paragraph_no"]["source"] = "manual"
    elif change == "missing":
        values.pop()
    elif change == "foreign":
        values[0] = replace(values[0], file_id=str(uuid4()))
    if change == "raw":
        with pytest.raises(ValueError):
            build_reviewed_figure_proof(
                raw_pdf=RAW + b"x",
                source_asset_id=uuid4(),
                user_file_id=FILE,
                chunk_id=TARGET,
                sources=values,
            )
    else:
        with pytest.raises(ValueError):
            proof(values)
