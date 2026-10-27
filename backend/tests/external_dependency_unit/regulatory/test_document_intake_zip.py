"""Real ZIP requests and frozen source graphs admit separate, owned documents."""

import json
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from uuid import uuid4
from zipfile import ZipFile

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.db import document_intake as repository
from onyx.db.amendment_sources import list_source_assets
from onyx.db.document_intake_sources import create_zip_source_package
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes import job as source_job
from onyx.regulatory.document_intake import job as intake_job
from onyx.regulatory.document_intake.source_graph import ZIP_INPUT_KIND
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


def test_mixed_zip_source_worker_and_admission_preserve_real_members_and_roots(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    native_markdown_runtime: PostgresBackedFileStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, target, model = intake_owner
    store = native_markdown_runtime
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
    batch_id, model_id = batch.id, model.id
    intake_session.commit()
    monkeypatch.setattr(
        repository,
        "resolve_intake_labeling_provider_binding",
        lambda *_args, **_kwargs: SimpleNamespace(
            model="gemini-3.5-flash",
            model_dump=lambda **_options: {"model_configuration_id": model_id},
        ),
    )
    monkeypatch.setattr(
        "onyx.regulatory.document_intake.metadata_job.run_intake_metadata",
        lambda **_kwargs: None,
    )
    assert (
        intake_job.run_document_intake(intake_id=batch_id, tenant_id="public")
        is not None
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
