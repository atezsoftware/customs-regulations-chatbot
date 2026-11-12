"""Real ZIP requests and frozen source graphs admit separate, owned documents."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4
from zipfile import ZipFile

import pytest
from google.auth.credentials import Credentials
from google.genai import types as genai_types
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as repository
from onyx.db import document_intake_metadata as metadata_repository
from onyx.db import regulatory_labeling as labeling
from onyx.db.amendment_sources import list_source_assets
from onyx.db.document_intake_sources import create_zip_source_package
from onyx.db.labeling_configuration import LabelingProviderBinding
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryLabelingRun,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.file_store.file_store import FileStore
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake import metadata_job
from onyx.regulatory.document_intake.metadata import METADATA_AUDIT_KEY
from onyx.regulatory.document_intake.source_graph import ZIP_INPUT_KIND
from onyx.regulatory.indexing_jobs.models import (
    VertexAuthenticationMode,
    VertexBatchConfig,
)
from onyx.regulatory.labeling import orchestrator, vertex_direct
from onyx.regulatory.labeling.provider import LabelDefinition, TaxonomyDefinition
from onyx.tracing.flows import LLMFlow
from shared_configs.contextvars import get_current_tenant_id
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_markdown import (
    native_markdown_runtime as native_markdown_runtime,
)
from tests.unit.onyx.regulatory.labeling.test_vertex_direct import _Store


def _inference_runtime(
    session: Session,
    model: ModelConfiguration,
    store: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Replace provider transport while running metadata and labeling coordinators."""
    binding = LabelingProviderBinding(
        model_configuration_id=model.id,
        provider_id=model.llm_provider.id,
        project="mixed-zip-test",
        location="global",
        authentication_mode=VertexAuthenticationMode.WORKLOAD_IDENTITY,
        credential_identity="fixture",
        model="gemini-3.5-flash",
        transport="vertex_direct_v1",
    )
    sdk = MagicMock()

    def generate(**kwargs: Any) -> genai_types.GenerateContentResponse:
        payload = json.loads(kwargs["contents"][0].parts[0].text)
        result = (
            {
                "covered_targets": payload["target_indices"],
                "shared_fields": [],
                "assignments": [],
                "warnings": [],
            }
            if "target_indices" in payload
            else (
                {
                    "chunks": [
                        {"chunk_id": identifier, "fields": []}
                        for identifier in payload["target_chunk_ids"]
                    ],
                    "warnings": [],
                }
                if "target_chunk_ids" in payload
                else {
                    "labels": [
                        {
                            "label_id": "customs",
                            "evidence_quote": payload["target"]["text"],
                        }
                    ],
                    "abstained": False,
                }
            )
        )
        return genai_types.GenerateContentResponse.model_validate(
            {
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "role": "model",
                            "parts": [{"text": json.dumps(result)}],
                        },
                    }
                ]
            }
        )

    sdk.models.generate_content.side_effect = generate
    receipts = _Store()

    def gateway(flow: LLMFlow) -> vertex_direct.LabelingVertexDirectGateway:
        return vertex_direct.LabelingVertexDirectGateway(
            config=VertexBatchConfig(
                model_configuration_id=model.id,
                model_name=binding.model,
                project=binding.project,
                location=binding.location,
                authentication_mode=binding.authentication_mode,
            ),
            binding_sha256=binding.fingerprint,
            credential_identity=binding.credential_identity,
            credential_json_provider=lambda: None,
            credentials_factory=lambda: cast(Credentials, object()),
            client_factory=lambda _credentials: sdk,
            store=cast(FileStore, receipts),
            flow=flow,
        )

    metadata_gateway = gateway(LLMFlow.REGULATORY_INTAKE_METADATA)
    label_gateway = gateway(LLMFlow.REGULATORY_LABELING_BATCH)

    @contextmanager
    def session_scope() -> Iterator[Session]:
        yield session

    @contextmanager
    def traced(**_kwargs: Any) -> Iterator[Any]:
        yield SimpleNamespace(span_data=SimpleNamespace(usage=None))

    from onyx.db import document_intake_background_labels

    for module in (repository, metadata_job, document_intake_background_labels):
        monkeypatch.setattr(
            module,
            "resolve_intake_labeling_provider_binding",
            lambda *_a, **_k: binding,
        )
    for module in (metadata_job, orchestrator):
        monkeypatch.setattr(module, "get_session_with_current_tenant", session_scope)
    monkeypatch.setattr(metadata_repository, "get_default_file_store", lambda: store)
    monkeypatch.setattr(
        metadata_job, "resolve_labeling_gateway", lambda *_a, **_k: metadata_gateway
    )
    monkeypatch.setattr(
        orchestrator, "resolve_labeling_gateway", lambda *_a, **_k: label_gateway
    )
    monkeypatch.setattr(vertex_direct, "traced_llm_call", traced)
    settings = labeling.get_label_settings(session)
    settings.taxonomy = labeling.get_or_create_taxonomy(
        session,
        taxonomy=TaxonomyDefinition(
            name="ZIP labels",
            labels=[LabelDefinition(id="customs", name="Gümrük", description="Gümrük")],
        ),
        created_by_id=None,
    )
    session.commit()
    return sdk


def test_mixed_zip_source_worker_and_admission_preserve_real_members_and_roots(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, target, model = intake_owner
    store = native_markdown_runtime
    sdk = _inference_runtime(intake_session, model, store, monkeypatch)
    files = {
        "main.html": "<html><body><h1>Yeni Mevzuat</h1><p>MADDE 1 — ZIP_ROOT_KURAL beyan usulüdür.</p><p><a href='ekler/ek-1.md'>EK-1 belgeler</a></p></body></html>".encode(),
        "ekler/ek-1.md": "# EK-1\n\n1. ZIP_ANNEX_KURAL fatura ibraz edilir.\n".encode(),
        "independent.md": "# Bağımsız Kanun\n\nMADDE 1 — ZIP_OTHER_KURAL süre otuz gündür.\n".encode(),
    }
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        for path, raw in files.items():
            archive.writestr(path, raw)
    raw_zip = output.getvalue()
    key = uuid4()
    package, created = create_zip_source_package(
        intake_session,
        user=owner,
        environment=_ENVIRONMENT,
        idempotency_key=key,
        display_name="mixed.zip",
        content=raw_zip,
        store=store,
    )
    assert created and package.document_set_id == target.id
    assert package.input_spec["kind"] == ZIP_INPUT_KIND
    assert package.input_spec["input_sha256"] == sha256(raw_zip).hexdigest()
    same, repeated = create_zip_source_package(
        intake_session,
        user=owner,
        environment=_ENVIRONMENT,
        idempotency_key=key,
        display_name="mixed.zip",
        content=raw_zip,
        store=store,
    )
    assert same.id == package.id and not repeated
    package_id = package.id
    intake_session.commit()
    source_job.run_source_package(package_id=package_id, environment=_ENVIRONMENT)
    intake_session.expire_all()
    package = intake_session.get_one(AmendmentSourcePackage, package_id)
    assert package.status == "ready" and not package.issues
    assert package.input_file_id is not None and package.manifest_file_id is not None
    with store.read_file(package.input_file_id) as stream:
        assert stream.read() == raw_zip
    with store.read_file(package.manifest_file_id) as stream:
        manifest = json.loads(stream.read())
    assert manifest["input_sha256"] == sha256(raw_zip).hexdigest()
    assert len(manifest["intake_graph"]["root_hashes"]) == 2
    assert len(manifest["intake_graph"]["primary_parent_by_hash"]) == 1
    assets = list_source_assets(intake_session, package_id)
    assert len(assets) == 3 and "application/zip" not in {
        asset.mime_type for asset in assets
    }
    for asset in assets:
        assert asset.sha256 == sha256(files[asset.display_name]).hexdigest()
        with store.read_file(asset.file_id) as stream:
            assert stream.read() == files[asset.display_name]
    batch, _created = repository.create_intake(
        intake_session,
        source_package_id=package_id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    batch_id = batch.id
    intake_session.commit()
    assert (
        intake_job.run_document_intake(
            intake_id=batch_id, tenant_id=get_current_tenant_id()
        )
        is None
    )
    intake_session.expire_all()
    batch = intake_session.get_one(type(batch), batch_id)
    members = repository.intake_files(intake_session, batch)
    assert len(members) == 3
    roots = [member for member in members if member.role == "root"]
    annex = next(member for member in members if member.role == "annex")
    assert len(roots) == 2 and annex.parent_document_id in {
        member.id for member in roots
    }
    for member in members:
        asset = intake_session.get_one(RegulatorySourceAsset, member.source_asset_id)
        file = intake_session.get_one(UserFile, member.user_file_id)
        assert file.file_id != asset.file_id
        with store.read_file(file.file_id) as stream:
            assert stream.read() == files[asset.display_name]
        text = "\n".join(
            intake_session.scalars(
                select(RegulatoryChunk.text).where(
                    RegulatoryChunk.user_file_id == file.id
                )
            )
        )
        if asset.display_name == "main.html":
            assert "ZIP_ROOT_KURAL" in text and "ZIP_ANNEX_KURAL" not in text
        if member == annex:
            assert "ZIP_ANNEX_KURAL" in text
        chunks = list(
            intake_session.scalars(
                select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
            )
        )
        assert chunks and all(
            METADATA_AUDIT_KEY in chunk.chunk_metadata for chunk in chunks
        )

    assert (batch.status, batch.stage, batch.background_label_status) == (
        "ready",
        "review",
        "pending",
    )
    assert batch.labeling_run_id is None
    from onyx.db.document_intake_background_labels import start_background_labels

    assert start_background_labels(intake_session, batch) is not None
    intake_session.commit()
    run = intake_session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
    for _step in range(20):
        intake_session.expire_all()
        run = intake_session.get_one(RegulatoryLabelingRun, batch.labeling_run_id)
        if run.status not in {"queued", "running"}:
            break
        orchestrator.run_labeling_step(
            run.id, run.lease_generation, get_current_tenant_id()
        )
    assert run.status == "completed", run.error
    assert set(run.file_ids) == {str(member.user_file_id) for member in members}
    assert sdk.models.generate_content.call_count >= 6
    repository.refresh_intake(intake_session, batch)
    assert batch.status == "ready" and batch.review_sha256
    deliveries = repository.approve_intake(intake_session, batch, batch.review_sha256)
    assert {file_id for file_id, _token in deliveries} == {
        member.user_file_id for member in members
    }
    intake_session.commit()
