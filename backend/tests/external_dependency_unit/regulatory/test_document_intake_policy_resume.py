"""A committed reprepare file resumes from its frozen encoder policy."""

import json
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import partial
from hashlib import sha256
from io import BytesIO
from typing import Literal
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db import document_intake_normalization as normalization
from onyx.db import document_intake_reprepare as reprepare
from onyx.db.models import (
    DocumentSet,
    ModelConfiguration,
    RegulatoryDocumentIntakeReprepare,
    RegulatoryFilePublication,
    RegulatorySourceAsset,
    SearchSettings,
    User,
    UserFile,
)
from onyx.db.regulatory_writer_publication import canonical_scope_digest
from onyx.file_processing.original_ingestion import OriginalIngestionReceipt
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake.models import (
    IntakePreparationPolicy,
    IntakeSourceAsset,
)
from onyx.regulatory.document_intake.preparation import load_verified_intake_documents
from onyx.regulatory.indexing import documents_to_regulatory_chunks
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from shared_configs.enums import EmbeddingProvider
from tests.external_dependency_unit.regulatory.test_document_intake import _ENVIRONMENT
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)
from tests.external_dependency_unit.regulatory.test_document_intake_job import (
    _verified_html_package,
)
from tests.external_dependency_unit.regulatory.test_document_intake_preparation_policy import (
    google_settings as google_settings,
)


@pytest.mark.parametrize("initial_version", ["3", "4"])
def test_committed_file_reload_keeps_target_policy_after_process_interruption(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    google_settings: SearchSettings,
    monkeypatch: pytest.MonkeyPatch,
    initial_version: Literal["3", "4"],
) -> None:
    session = intake_session
    owner, target, model = intake_owner
    package = _verified_html_package(session, owner, target)
    batch, _ = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=owner,
        environment=_ENVIRONMENT,
    )
    policy = IntakePreparationPolicy()
    batch.preparation_policy = (
        policy.model_dump(mode="json") if initial_version == "4" else None
    )
    assets = list(
        session.scalars(
            select(RegulatorySourceAsset).where(
                RegulatorySourceAsset.package_id == package.id
            )
        )
    )
    views = [IntakeSourceAsset.model_validate(asset) for asset in assets]
    by_id = {view.id: view for view in views}
    store = PostgresBackedFileStore()
    monkeypatch.setattr(
        store, "read_file", partial(store.read_file, db_session=session)
    )
    monkeypatch.setattr(normalization, "get_default_file_store", lambda: store)
    assert package.manifest_file_id is not None
    with store.read_file(package.manifest_file_id) as stream:
        manifest = json.load(stream)
    for item in manifest["assets"]:
        asset = next(asset for asset in assets if asset.sha256 == item["sha256"])
        assert asset.text_file_id is not None
        with store.read_file(asset.text_file_id) as stream:
            prepared = stream.read().decode()
        item.update(
            text=prepared,
            **{
                key: getattr(asset, key)
                for key in ("mime_type", "display_name", "original_url", "final_url")
            },
        )
    encoded = json.dumps(manifest).encode()
    package.manifest_file_id = store.save_file(
        BytesIO(encoded),
        "verified-manifest.json",
        FileOrigin.OTHER,
        "application/json",
        db_session=session,
    )
    package.manifest_sha256 = sha256(encoded).hexdigest()
    links = [SourceLink.model_validate(link) for link in manifest["links"]]
    members = intake.admit_intake_assets(
        session, batch, assets, links, root_hash=manifest["assets"][0]["sha256"]
    )
    tokenizer = TiktokenTokenizer("text-embedding-3-large")
    for member in members:
        file = session.get_one(UserFile, member.user_file_id)
        documents = load_verified_intake_documents(
            package_id=package.id,
            asset=by_id[member.source_asset_id],
            user_file_id=file.id,
            links=links,
            package_assets=views,
            store=store,
            chunking_version=initial_version,
            preparation_policy=policy if initial_version == "4" else None,
        )
        documents_to_regulatory_chunks(
            documents, session, tokenizer, enable_contextual_rag=True
        )
        generation = compute_regulatory_chunk_generation_hash(
            embedding_provider=EmbeddingProvider.GOOGLE,
            embedding_model_name=google_settings.model_name,
            enable_contextual_rag=True,
            intake_chunking_version=initial_version,
            intake_preparation_policy=policy if initial_version == "4" else None,
        )
        file.regulatory_chunk_generation_hash = generation
        session.add(
            RegulatoryFilePublication(
                user_file_id=file.id,
                scope_key="isolated-resume-test",
                owner_id=batch.id,
                fencing_token=1,
                lease_expires_at=datetime.now(timezone.utc) - timedelta(minutes=1),
                original_ingestion_receipt=OriginalIngestionReceipt(
                    **documents.original_extraction.model_dump(),
                    canonical_sha256=canonical_scope_digest(session, file.id),
                    generation_hash=generation,
                ).model_dump(mode="json"),
            )
        )
    batch.status, batch.stage = "failed", "metadata"
    session.flush()
    preview = reprepare.reprepare_snapshot(session, batch)
    request = reprepare.request_reprepare(
        session,
        batch,
        expected_sha256=preview["expected_sha256"],
        idempotency_key=batch.id,
    )
    token = reprepare.claim_reprepare(session, batch.id, request.id)
    assert token is not None
    member = members[0]
    file = session.get_one(UserFile, member.user_file_id)
    target_policy = IntakePreparationPolicy(chunking_version="5")
    target = load_verified_intake_documents(
        package_id=package.id,
        asset=by_id[member.source_asset_id],
        user_file_id=file.id,
        links=links,
        package_assets=views,
        store=store,
        chunking_version="5",
        preparation_policy=target_policy,
    )
    documents_to_regulatory_chunks(
        target, session, tokenizer, enable_contextual_rag=True
    )
    generation = compute_regulatory_chunk_generation_hash(
        embedding_provider=EmbeddingProvider.GOOGLE,
        embedding_model_name=google_settings.model_name,
        enable_contextual_rag=True,
        intake_chunking_version="5",
        intake_preparation_policy=target_policy,
    )
    file.regulatory_chunk_generation_hash = generation
    canonical = canonical_scope_digest(session, file.id)
    entries = deepcopy(request.documents)
    entries[str(member.id)].update(
        status="prepared",
        after_receipt=OriginalIngestionReceipt(
            **target.original_extraction.model_dump(),
            canonical_sha256=canonical,
            generation_hash=generation,
        ).model_dump(mode="json"),
    )
    request.documents = entries
    batch_id, request_id, member_id, file_id = batch.id, request.id, member.id, file.id
    session.commit()
    session.expire_all()
    loaded = reprepare.prepared_reprepare_documents(
        session, batch_id, request_id, token, member_id
    )
    assert loaded.original_extraction == target.original_extraction
    assert loaded[0].doc_metadata is not None
    assert loaded[0].doc_metadata["regulatory_document_intake"][
        "preparation_policy"
    ] == target_policy.model_dump(mode="json")
    assert canonical_scope_digest(session, file_id) == canonical
    assert (
        session.get_one(RegulatoryDocumentIntakeReprepare, request_id).documents
        == entries
    )
