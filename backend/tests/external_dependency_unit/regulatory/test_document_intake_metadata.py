"""Metadata enrichment and manual corrections remain within the new intake scope."""

import datetime
import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from google.auth.credentials import Credentials
from google.genai import types as genai_types
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as intake
from onyx.db import document_intake_metadata as metadata
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    User,
)
from onyx.file_store.file_store import FileStore
from onyx.regulatory.document_intake import metadata_job
from onyx.regulatory.document_intake.metadata import (
    METADATA_AUDIT_KEY,
    TARGETS_PER_REQUEST,
    ChunkLegalMetadata,
    LegalMetadataFact,
    LegalMetadataResult,
    metadata_input_hash,
)
from onyx.regulatory.indexing_jobs.models import (
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.indexing_jobs.vertex_batch import VertexBatchContractError
from onyx.regulatory.labeling import vertex_direct
from onyx.tracing.flows import LLMFlow
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _admitted,
    _existing_rows,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as _intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as _intake_session,
)
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _Store

intake_owner = _intake_owner
intake_session = _intake_session


def _ready_metadata(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> tuple[
    RegulatoryDocumentIntake,
    list[RegulatoryDocumentIntakeFile],
    UUID,
    RegulatoryChunk,
    RegulatoryChunk,
]:
    user, target, model = owner
    batch, members = _admitted(session, user, target, model)
    token = uuid4()
    batch.lease_token = token
    for member in members:
        member.status = "chunked"
    chunks = list(
        session.scalars(
            select(RegulatoryChunk).where(
                RegulatoryChunk.user_file_id == members[0].user_file_id
            )
        )
    )
    chunks[0].text = "MADDE 1 — Yeni belge kuralı."
    chunks[0].chunk_metadata = {
        **chunks[0].chunk_metadata,
        "article_no": "1",
        "clause_label": "a",
        "source_asset_ids": [str(members[0].source_asset_id)],
    }
    closing = RegulatoryChunk(
        id=f"rc_meta_{uuid4().hex}",
        user_file_id=members[0].user_file_id,
        text="Yürürlük MADDE 99 — Bu Kanun 1 Ocak 2027 tarihinde yürürlüğe girer.",
        position=1,
        projection_ordinal=1,
        heading_path=["Yürürlük", "MADDE 99"],
        chunk_type="article",
        chunk_metadata={},
        status="active",
        source="indexed",
    )
    session.add(closing)
    session.flush()
    return batch, members, token, chunks[0], closing


def _apply(
    session: Session,
    batch_id: UUID,
    token: UUID,
    document_id: UUID,
    rows: list[RegulatoryChunk],
    evidence: RegulatoryChunk,
) -> None:
    chunks = metadata.metadata_chunks(rows)
    result = LegalMetadataResult(
        chunks=[
            ChunkLegalMetadata(
                chunk_id=row.id,
                fields=[
                    LegalMetadataFact(
                        field="validity_start_date",
                        value="2027-01-01",
                        source_chunk_id=evidence.id,
                        evidence_quote="Bu Kanun 1 Ocak 2027 tarihinde yürürlüğe girer.",
                    )
                ],
            )
            for row in rows
        ]
    )
    metadata.apply_metadata_results(
        session,
        intake_id=batch_id,
        lease_token=token,
        document_id=document_id,
        expected_input_sha256=metadata_input_hash(chunks),
        results=[("a" * 64, result)],
    )


def test_metadata_sets_scoped_dates_with_exact_cross_chunk_evidence_and_retry_receipts(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, token, article, closing = _ready_metadata(
        intake_session, intake_owner
    )
    annex_before = _existing_rows(intake_session, [members[1].user_file_id])
    _apply(intake_session, batch.id, token, members[0].id, [article, closing], closing)
    assert article.validity_start_date == datetime.date(2027, 1, 1)
    audit = article.chunk_metadata[METADATA_AUDIT_KEY]
    assert audit["model"] == "gemini-3.5-flash"
    assert audit["fields"]["validity_start_date"]["source_chunk_id"] == closing.id
    assert audit["fields"]["article_no"]["source"] == "structural"
    assert article.chunk_metadata["article_no"] == "1"
    assert _existing_rows(intake_session, [members[1].user_file_id]) == annex_before
    _batch, work = metadata.load_metadata_work(intake_session, batch.id, token)
    assert [identifier for identifier, _chunks in work] == [members[1].id]


def test_manual_patch_preserves_provenance_and_text_edit_invalidates_cross_chunk_llm_dates(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, token, article, closing = _ready_metadata(
        intake_session, intake_owner
    )
    _apply(intake_session, batch.id, token, members[0].id, [article, closing], closing)
    source_before = deepcopy(article.chunk_metadata["source_asset_ids"])
    updates = metadata.manual_edit_metadata(
        article,
        {
            "chunk_metadata": {"article_no": "2", "clause_label": None},
            "validity_start_date": datetime.date(2028, 1, 1),
        },
    )
    for key, value in updates.items():
        setattr(article, key, value)
    closing.text = "Yürürlük artık belirlenemiyor."
    metadata.invalidate_metadata_file_context([article, closing])
    assert article.chunk_metadata["article_no"] == "2"
    assert "clause_label" not in article.chunk_metadata
    assert article.chunk_metadata["source_asset_ids"] == source_before
    assert article.validity_start_date == datetime.date(2028, 1, 1)
    assert closing.validity_start_date is None
    assert closing.chunk_metadata[METADATA_AUDIT_KEY]["fields"]["validity_start_date"][
        "stale"
    ]
    assert article.chunk_metadata[METADATA_AUDIT_KEY]["input_stale"]


def test_full_form_noop_values_do_not_misrepresent_model_dates_as_manual(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, token, article, closing = _ready_metadata(
        intake_session, intake_owner
    )
    _apply(intake_session, batch.id, token, members[0].id, [article, closing], closing)
    updates = metadata.manual_edit_metadata(
        article,
        {
            "text": "MADDE 1 — Düzeltilmiş kural.",
            "validity_start_date": article.validity_start_date,
            "validity_end_date": None,
            "chunk_metadata": {"article_no": "1"},
        },
    )
    assert "chunk_metadata" not in updates
    article.text = updates["text"]
    metadata.invalidate_metadata_file_context([article, closing])
    assert article.validity_start_date is None
    assert (
        article.chunk_metadata[METADATA_AUDIT_KEY]["fields"]["validity_start_date"][
            "source"
        ]
        == "llm"
    )


@pytest.mark.parametrize(
    "key",
    [
        "source_asset_ids",
        "source_links",
        "projection_ordinal",
        "chunk_variant",
        "hierarchy_root_id",
        METADATA_AUDIT_KEY,
    ],
)
def test_manual_metadata_cannot_change_source_graph_projection_or_audit(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    key: str,
) -> None:
    _batch, _members, _token, article, _closing = _ready_metadata(
        intake_session, intake_owner
    )
    before = deepcopy(article.chunk_metadata)
    with pytest.raises(ValueError, match="protected"):
        metadata.manual_edit_metadata(article, {"chunk_metadata": {key: "injected"}})
    assert article.chunk_metadata == before


def test_changed_source_hash_or_wrong_document_blocks_metadata_without_admission(
    intake_session: Session, intake_owner: tuple[User, DocumentSet, ModelConfiguration]
) -> None:
    batch, members, token, article, closing = _ready_metadata(
        intake_session, intake_owner
    )
    chunks = metadata.metadata_chunks([article, closing])
    result = LegalMetadataResult(
        chunks=[
            ChunkLegalMetadata(chunk_id=row.id, fields=[]) for row in [article, closing]
        ]
    )
    article.text += " edit during extraction"
    with pytest.raises(intake.IntakeConflictError, match="changed"):
        metadata.apply_metadata_results(
            intake_session,
            intake_id=batch.id,
            lease_token=token,
            document_id=members[0].id,
            expected_input_sha256=metadata_input_hash(chunks),
            results=[("a" * 64, result)],
        )
    assert METADATA_AUDIT_KEY not in article.chunk_metadata


def test_real_cached_gateway_reuses_acknowledged_window_after_failure_and_applies_whole_file(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, members, token, article, _closing = _ready_metadata(
        intake_session, intake_owner
    )
    monkeypatch.setattr(metadata_job, "METADATA_CONCURRENT_WINDOWS", 1)
    for position in range(2, TARGETS_PER_REQUEST + 1):
        intake_session.add(
            RegulatoryChunk(
                id=f"rc_meta_window_{uuid4().hex}",
                user_file_id=article.user_file_id,
                text=f"MADDE {position} — Kural.",
                position=position,
                projection_ordinal=position,
                heading_path=[f"MADDE {position}"],
                chunk_type="article",
                chunk_metadata={},
                status="active",
                source="indexed",
            )
        )
    intake_session.flush()
    binding = LabelingProviderBinding(
        model_configuration_id=batch.labeling_model_configuration_id,
        provider_id=1,
        project="test",
        location="global",
        authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
        credential_identity="test@example.com",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    store, sdk = _Store(), MagicMock()
    calls = 0
    prompts: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        nonlocal calls
        calls += 1
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        prompts.append(payload)
        if calls == 2:
            raise RuntimeError("temporary provider failure")
        outcome = {
            "chunks": [
                {"chunk_id": identifier, "fields": []}
                for identifier in payload["target_chunk_ids"]
            ],
            "warnings": [],
        }
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": json.dumps(outcome)}],
                        },
                    }
                ]
            }
        )

    sdk.models.generate_content.side_effect = generate
    gateway = vertex_direct.LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=batch.labeling_model_configuration_id,
            model_name=binding.model,
            project="test",
            location="global",
            authentication_mode=binding.authentication_mode,
        ),
        binding_sha256=binding.fingerprint,
        credential_identity=binding.credential_identity,
        credential_json_provider=lambda: None,
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=lambda _credentials: sdk,
        store=cast(FileStore, store),
        flow=LLMFlow.REGULATORY_INTAKE_METADATA,
        max_request_bytes=8 * 1024 * 1024,
        max_response_bytes=16 * 1024 * 1024,
    )

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield intake_session

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    monkeypatch.setattr(metadata_job, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(
        metadata_job,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        metadata_job, "resolve_labeling_gateway", lambda *_args, **_kwargs: gateway
    )
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    with pytest.raises(VertexBatchContractError):
        metadata_job.run_intake_metadata(
            intake_id=batch.id, lease_token=token, tenant_id="public"
        )
    assert METADATA_AUDIT_KEY not in article.chunk_metadata
    assert calls == 2
    metadata_job.run_intake_metadata(
        intake_id=batch.id, lease_token=token, tenant_id="public"
    )
    assert calls == 4

    assert (
        len(prompts[0]["complete_document"])
        == len(prompts[1]["complete_document"])
        == TARGETS_PER_REQUEST + 1
    )
    assert [len(prompt["target_chunk_ids"]) for prompt in prompts] == [
        TARGETS_PER_REQUEST,
        1,
        1,
        1,
    ]
    assert article.chunk_metadata[METADATA_AUDIT_KEY]["model"] == "gemini-3.5-flash"
    metadata_job.run_intake_metadata(
        intake_id=batch.id, lease_token=token, tenant_id="public"
    )
    assert calls == 4


@pytest.mark.parametrize("repair_quote", [False, True], ids=["valid", "quote-repair"])
def test_adaptive_split_keeps_file_unapplied_until_all_cached_leaves_resume(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
    repair_quote: bool,
) -> None:
    from onyx.regulatory.document_intake.metadata import build_metadata_request

    batch, members, token, article, closing = _ready_metadata(
        intake_session, intake_owner
    )
    if repair_quote:
        article.text += "\nYeni TEBLİĞ"
    monkeypatch.setattr(metadata_job, "METADATA_CONCURRENT_WINDOWS", 1)
    for position in range(2, 4):
        intake_session.add(
            RegulatoryChunk(
                id=f"rc_meta_split_{uuid4().hex}",
                user_file_id=article.user_file_id,
                text=f"MADDE {position} — Kural.",
                position=position,
                projection_ordinal=position,
                heading_path=[f"MADDE {position}"],
                chunk_type="article",
                chunk_metadata={},
                status="active",
                source="indexed",
            )
        )
    intake_session.flush()
    rows = list(
        intake_session.scalars(
            select(RegulatoryChunk)
            .where(RegulatoryChunk.user_file_id == article.user_file_id)
            .order_by(RegulatoryChunk.position, RegulatoryChunk.id)
        )
    )
    chunks = metadata.metadata_chunks(rows)
    parent_ids = [chunk.chunk_id for chunk in chunks]
    parent_hash = build_metadata_request(chunks, parent_ids).request_hash
    expected_leaves = {
        identifier: build_metadata_request(chunks, group).request_hash
        for group in (parent_ids[:2], parent_ids[2:])
        for identifier in group
    }
    binding = LabelingProviderBinding(
        model_configuration_id=batch.labeling_model_configuration_id,
        provider_id=1,
        project="test",
        location="global",
        authentication_mode=VertexAuthenticationMode.SERVICE_ACCOUNT_JSON,
        credential_identity="test@example.com",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    store, sdk = _Store(), MagicMock()
    payloads: list[dict[str, Any]] = []

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        payloads.append(payload)
        if len(payloads) == (4 if repair_quote else 3):
            raise RuntimeError("synthetic second-child interruption")
        targets = payload["target_chunk_ids"]
        main_document = len(payload["complete_document"]) == 4
        fields = (
            [
                {
                    "field": "validity_start_date",
                    "value": "2027-01-01",
                    "source_chunk_id": closing.id,
                    "evidence_quote": "Bu Kanun 1 Ocak 2027 tarihinde yürürlüğe girer.",
                }
            ]
            if main_document
            else []
        )
        if repair_quote and targets == parent_ids[:2]:
            repaired = "validation_feedback" in payload
            fields.append(
                {
                    "field": "document_title",
                    "value": "Yeni TEBLİĞ",
                    "source_chunk_id": article.id,
                    "evidence_quote": "Yeni TEBLİĞ" if repaired else "Yeni TEBLIĞ",
                }
            )
        outcome = {
            "chunks": [
                {"chunk_id": identifier, "fields": fields} for identifier in targets
            ],
            "warnings": [],
        }
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "MAX_TOKENS" if len(targets) == 4 else "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": json.dumps(outcome)}],
                        },
                    }
                ]
            }
        )

    sdk.models.generate_content.side_effect = generate
    gateway = vertex_direct.LabelingVertexDirectGateway(
        config=VertexBatchConfig(
            model_configuration_id=batch.labeling_model_configuration_id,
            model_name=binding.model,
            project="test",
            location="global",
            authentication_mode=binding.authentication_mode,
        ),
        binding_sha256=binding.fingerprint,
        credential_identity=binding.credential_identity,
        credential_json_provider=lambda: None,
        credentials_factory=lambda: cast(Credentials, object()),
        client_factory=lambda _credentials: sdk,
        store=cast(FileStore, store),
        flow=LLMFlow.REGULATORY_INTAKE_METADATA,
        max_request_bytes=8 * 1024 * 1024,
        max_response_bytes=16 * 1024 * 1024,
    )
    submit = MagicMock(wraps=gateway.submit)
    monkeypatch.setattr(gateway, "submit", submit)

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield intake_session

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    monkeypatch.setattr(metadata_job, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(
        metadata_job,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: binding,
    )
    monkeypatch.setattr(
        metadata_job, "resolve_labeling_gateway", lambda *_args, **_kwargs: gateway
    )
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    with pytest.raises(VertexBatchContractError):
        metadata_job.run_intake_metadata(
            intake_id=batch.id, lease_token=token, tenant_id="public"
        )
    assert len(payloads) == (4 if repair_quote else 3)
    if repair_quote:
        repair_request = submit.call_args_list[2].args[0][0]
        repaired_hash = repair_request.request_hash
        assert repaired_hash not in {parent_hash, expected_leaves[parent_ids[0]]}
        assert payloads[2]["validation_feedback"]["repair_attempt"] == 1
        for identifier in parent_ids[:2]:
            expected_leaves[identifier] = repaired_hash
    intake_session.expire_all()
    assert all(
        METADATA_AUDIT_KEY not in row.chunk_metadata and row.validity_start_date is None
        for row in rows
    )
    annex = intake_session.scalar(
        select(RegulatoryChunk).where(
            RegulatoryChunk.user_file_id == members[1].user_file_id
        )
    )
    assert annex is not None and METADATA_AUDIT_KEY not in annex.chunk_metadata
    metadata_job.run_intake_metadata(
        intake_id=batch.id, lease_token=token, tenant_id="public"
    )
    assert [len(payload["target_chunk_ids"]) for payload in payloads] == (
        [4, 2, 2, 2, 2, 1] if repair_quote else [4, 2, 2, 2, 1]
    )
    assert all(len(payload["complete_document"]) == 4 for payload in payloads[:-1])
    assert all(row.validity_start_date == datetime.date(2027, 1, 1) for row in rows)
    assert all(
        row.chunk_metadata[METADATA_AUDIT_KEY]["request_hash"]
        == expected_leaves[row.id]
        != parent_hash
        for row in rows
    )
    if repair_quote:
        for row in rows[:2]:
            assert row.chunk_metadata["document_title"] == "Yeni TEBLİĞ"
            assert row.chunk_metadata[METADATA_AUDIT_KEY]["fields"]["document_title"][
                "evidence_quote"
            ] == "Yeni TEBLİĞ"
    metadata_job.run_intake_metadata(
        intake_id=batch.id, lease_token=token, tenant_id="public"
    )
    assert len(payloads) == (6 if repair_quote else 5)
