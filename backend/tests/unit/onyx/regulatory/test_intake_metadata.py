"""Metadata evidence must describe this new document and retain complete coverage."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from google.auth.credentials import Credentials
from google.genai import types as genai_types

from onyx.file_store.file_store import FileStore
from onyx.regulatory.document_intake.metadata import (
    LegalMetadataFact,
    MetadataChunk,
    build_metadata_request,
    metadata_evidence_scope,
    metadata_input_hash,
    validate_metadata_response,
)
from onyx.regulatory.indexing_jobs.models import (
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.labeling import vertex_direct
from onyx.tracing.flows import LLMFlow
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _Store


def _sources() -> list[MetadataChunk]:
    return [
        MetadataChunk(
            chunk_id="intro",
            text="Resmî Gazete Tarihi: 26.10.2026\n5000 sayılı Gümrük Kanunu",
            heading_path=[],
            chunk_type="preamble",
        ),
        MetadataChunk(
            chunk_id="article",
            text="MADDE 1 — Yeni gümrük kuralı uygulanır.",
            heading_path=["MADDE 1"],
            chunk_type="article",
            structural_metadata={"article_no": "1"},
        ),
        MetadataChunk(
            chunk_id="closing",
            text="Yürürlük MADDE 99 — Bu Kanun yayımı tarihinde yürürlüğe girer. Bu Kanun 31 Aralık 2030 tarihinde yürürlükten kalkar. 4000 sayılı Kanun 1 Ocak 2027 tarihinde yürürlükten kaldırılmıştır.",
            heading_path=["Yürürlük", "MADDE 99"],
            chunk_type="article",
        ),
    ]


def _response(
    field: str = "validity_start_date", value: Any = "2026-10-26", **overrides: Any
) -> str:
    fact = {
        "field": field,
        "value": value,
        "source_chunk_id": "closing",
        "evidence_quote": "Bu Kanun yayımı tarihinde yürürlüğe girer.",
        "basis": "relative_publication",
        "reference_chunk_id": "intro",
        "reference_quote": "Resmî Gazete Tarihi: 26.10.2026",
        **overrides,
    }
    return json.dumps(
        {"chunks": [{"chunk_id": "article", "fields": [fact]}], "warnings": []},
        ensure_ascii=False,
    )


def test_full_document_including_closing_clause_is_in_each_target_window() -> None:
    chunks = _sources()
    request = build_metadata_request(chunks, ["article"])
    payload = json.loads(request.prompt)
    assert payload["target_chunk_ids"] == ["article"]
    assert payload["complete_document"][-1]["text"] == chunks[-1].text
    assert (
        request.generation_config
        and request.generation_config["maxOutputTokens"] == 32768
    )
    assert "No approval-date fallback" in (request.system_instruction or "")


def test_serving_schema_matches_working_labeling_pattern_without_nested_grammar_bounds() -> (
    None
):
    request = build_metadata_request(_sources(), ["article"])
    assert request.generation_config
    serialized = json.dumps(request.generation_config["responseJsonSchema"])
    for unsupported in ('"$defs"', '"$ref"', '"maxItems"', '"minItems"', '"default"'):
        assert unsupported not in serialized
    assert '"additionalProperties": false' in serialized


def test_verified_heading_can_ground_annex_identity_without_weakening_date_evidence() -> (
    None
):
    chunk = MetadataChunk(
        chunk_id="annex",
        text="Belge tablosu.",
        heading_path=["Chunk İnceleme Yerel Testi EK-1"],
        chunk_type="table",
    )
    fact = LegalMetadataFact(
        field="appendix_label",
        value="EK-1",
        source_chunk_id="annex",
        evidence_quote=chunk.heading_path[0],
    )
    assert metadata_evidence_scope(fact, {chunk.chunk_id: chunk}) == "heading"
    outcome = {
        "chunks": [{"chunk_id": "annex", "fields": [fact.model_dump()]}],
        "warnings": [],
    }
    assert (
        validate_metadata_response(
            json.dumps(outcome), chunks=[chunk], target_ids=[chunk.chunk_id]
        )
        .chunks[0]
        .fields[0]
        .value
        == "EK-1"
    )
    dated = chunk.model_copy(update={"heading_path": ["2027-01-01"]})
    with pytest.raises(ValueError, match="exact quote"):
        metadata_evidence_scope(
            LegalMetadataFact(
                field="validity_start_date",
                value="2027-01-01",
                source_chunk_id="annex",
                evidence_quote="2027-01-01",
            ),
            {dated.chunk_id: dated},
        )


def test_publication_relative_commencement_uses_quoted_own_publication_anchor() -> None:
    result = validate_metadata_response(
        _response(), chunks=_sources(), target_ids=["article"]
    )
    assert result.chunks[0].fields[0].value == "2026-10-26"


def test_explicit_own_end_accepts_turkish_month_without_reusing_old_repeal() -> None:
    result = validate_metadata_response(
        _response(
            "validity_end_date",
            "2030-12-31",
            basis="explicit",
            evidence_quote="Bu Kanun 31 Aralık 2030 tarihinde yürürlükten kalkar.",
        ),
        chunks=_sources(),
        target_ids=["article"],
    )
    assert result.chunks[0].fields[0].value == "2030-12-31"


@pytest.mark.parametrize(
    "change",
    [
        {"evidence_quote": "Uydurma yürürlük tarihi."},
        {"source_chunk_id": "outside-document"},
        {"value": "2026-10-27"},
        {"reference_chunk_id": "outside-document"},
        {"reference_quote": "Resmî Gazete Tarihi: 27.10.2026"},
        {"value": "26.10.2026"},
    ],
)
def test_invalid_or_unquoted_metadata_dates_are_rejected(
    change: dict[str, Any],
) -> None:
    with pytest.raises(ValueError):
        validate_metadata_response(
            _response(**change), chunks=_sources(), target_ids=["article"]
        )


def test_old_law_repeal_does_not_end_new_law() -> None:
    with pytest.raises(ValueError, match="another law"):
        validate_metadata_response(
            _response(
                "validity_end_date",
                "2027-01-01",
                basis="explicit",
                evidence_quote="4000 sayılı Kanun 1 Ocak 2027 tarihinde yürürlükten kaldırılmıştır.",
            ),
            chunks=_sources(),
            target_ids=["article"],
        )


def test_document_number_must_match_its_exact_quoted_identity() -> None:
    with pytest.raises(ValueError, match="contradicts"):
        validate_metadata_response(
            _response(
                "document_number",
                "4000",
                basis="explicit",
                source_chunk_id="intro",
                evidence_quote="5000 sayılı Gümrük Kanunu",
            ),
            chunks=_sources(),
            target_ids=["article"],
        )


def test_own_temporary_provision_can_state_end_without_repeating_document_name() -> (
    None
):
    chunks = [
        MetadataChunk(
            chunk_id="temporary",
            text="Bu düzenleme 31 Aralık 2030 tarihine kadar geçerlidir.",
            heading_path=[],
            chunk_type="article",
        )
    ]
    result = validate_metadata_response(
        json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": "temporary",
                        "fields": [
                            {
                                "field": "validity_end_date",
                                "value": "2030-12-31",
                                "source_chunk_id": "temporary",
                                "evidence_quote": chunks[0].text,
                                "basis": "explicit",
                            }
                        ],
                    }
                ]
            }
        ),
        chunks=chunks,
        target_ids=["temporary"],
    )
    assert result.chunks[0].fields[0].value == "2030-12-31"


@pytest.mark.parametrize(
    "result",
    [
        {"chunks": [{"chunk_id": "outside", "fields": []}]},
        {
            "chunks": [
                {"chunk_id": "article", "fields": []},
                {"chunk_id": "article", "fields": []},
            ]
        },
    ],
)
def test_exact_target_coverage_rejects_foreign_or_duplicate_chunks(
    result: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="exact requested chunks"):
        validate_metadata_response(
            json.dumps(result), chunks=_sources(), target_ids=["article"]
        )


def test_unknown_metadata_stays_blank_without_date_fallback() -> None:
    result = validate_metadata_response(
        '{"chunks":[{"chunk_id":"article","fields":[]}],"warnings":["Yürürlük belirsiz"]}',
        chunks=_sources(),
        target_ids=["article"],
    )
    assert result.chunks[0].fields == []


def test_source_text_or_heading_changes_invalidate_metadata_input_receipt() -> None:
    chunks = _sources()
    initial = metadata_input_hash(chunks)
    assert (
        metadata_input_hash(
            [
                chunks[0].model_copy(
                    update={"structural_metadata": {"article_no": "different"}}
                ),
                *chunks[1:],
            ]
        )
        == initial
    )
    assert (
        metadata_input_hash(
            [chunks[0].model_copy(update={"text": "changed"}), *chunks[1:]]
        )
        != initial
    )


def test_large_complete_source_uses_metadata_budget_without_legacy_768k_truncation() -> (
    None
):
    chunk = MetadataChunk(
        chunk_id="large", text="a" * (900 * 1024), heading_path=[], chunk_type="article"
    )
    request = build_metadata_request([chunk], ["large"])
    assert json.loads(request.prompt)["complete_document"][0]["text"] == chunk.text


def test_metadata_transport_caches_large_complete_echo_with_its_own_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = MetadataChunk(
        chunk_id="large",
        text="x" * (1300 * 1024),
        heading_path=[],
        chunk_type="article",
    )
    request = build_metadata_request([chunk], ["large"])
    store, client = _Store(), MagicMock()
    client.models.generate_content.return_value = genai_types.GenerateContentResponse.model_validate(
        {
            "modelVersion": "gemini-3.5-flash",
            "candidates": [
                {
                    "finishReason": "STOP",
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "text": '{"chunks":[{"chunk_id":"large","fields":[]}],"warnings":[]}'
                            }
                        ],
                    },
                }
            ],
        }
    )
    flows: list[LLMFlow] = []

    @contextmanager
    def traced(**kwargs: Any) -> Iterator[Any]:
        flows.append(kwargs["flow"])
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    gateway = vertex_direct.LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=1,
            model_name="gemini-3.5-flash",
            project="test",
            location="global",
            authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
        ),
        binding_sha256="a" * 64,
        credential_identity="test@example.com",
        credential_json_provider=lambda: None,
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=lambda _credentials: client,
        store=cast(FileStore, store),
        max_request_bytes=8 * 1024 * 1024,
        max_response_bytes=16 * 1024 * 1024,
        flow=LLMFlow.REGULATORY_INTAKE_METADATA,
    )
    key = "regulatory-labeling-" + "b" * 64
    state = gateway.submit(
        [request], submission_key=key, max_jsonl_bytes=8 * 1024 * 1024
    )
    assert state.output_uri
    assert len(next(gateway.read_results(state.output_uri)).encode()) > 1024 * 1024
    assert (
        gateway.submit([request], submission_key=key, max_jsonl_bytes=8 * 1024 * 1024)
        == state
    )
    client.models.generate_content.assert_called_once()
    assert flows == [LLMFlow.REGULATORY_INTAKE_METADATA]


def test_metadata_job_uses_bounded_240_second_gateway_with_complete_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import uuid4

    from onyx.regulatory.document_intake import metadata_job

    session = MagicMock()
    batch = SimpleNamespace(created_by=uuid4(), labeling_model_configuration_id=7)
    sources = _sources()
    binding = SimpleNamespace(fingerprint="f" * 64)
    gateway = MagicMock()
    gateway.submit.side_effect = RuntimeError("Stop before provider inference")
    resolver = MagicMock(return_value=gateway)

    @contextmanager
    def session_scope() -> Iterator[Any]:
        yield session

    monkeypatch.setattr(metadata_job, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(
        metadata_job.repository,
        "load_metadata_work",
        lambda *_args: (batch, [(uuid4(), sources)]),
    )
    monkeypatch.setattr(
        metadata_job.repository, "heartbeat_metadata", lambda *_args: None
    )
    monkeypatch.setattr(
        metadata_job,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(metadata_job, "resolve_labeling_gateway", resolver)

    with pytest.raises(RuntimeError, match="Stop before provider inference"):
        metadata_job.run_intake_metadata(
            intake_id=uuid4(), lease_token=uuid4(), tenant_id="public"
        )

    assert resolver.call_args.kwargs["direct_request_timeout_seconds"] == 240
    assert (
        resolver.call_args.kwargs["direct_flow"] == LLMFlow.REGULATORY_INTAKE_METADATA
    )
    payload = json.loads(gateway.submit.call_args.args[0][0].prompt)
    assert [item["text"] for item in payload["complete_document"]] == [
        item.text for item in sources
    ]
    assert payload["target_chunk_ids"] == [item.chunk_id for item in sources]
