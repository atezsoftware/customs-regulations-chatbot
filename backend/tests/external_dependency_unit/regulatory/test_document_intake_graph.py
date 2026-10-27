"""ZIP forests retain independent files and bind ownership to frozen evidence."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace
from typing import Any, cast
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db import document_intake as intake
from onyx.db.models import (
    AmendmentSourcePackage,
    DocumentSet,
    DocumentSet__UserFile,
    ModelConfiguration,
    RegulatoryChunk,
    RegulatoryDocumentIntake,
    RegulatorySourceAsset,
    User,
    UserFile,
)
from onyx.file_store.file_store import FileStore
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.document_intake import job, metadata_job, reprepare_job
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import build_intake_document
from onyx.regulatory.document_intake.source_graph import (
    ZIP_INPUT_KIND,
    IntakeGraph,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    _ENVIRONMENT,
    _admitted,
    _package,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_owner as intake_owner,
)
from tests.external_dependency_unit.regulatory.test_document_intake import (
    intake_session as intake_session,
)


def _zip_package(
    session: Session, owner: tuple[User, DocumentSet, ModelConfiguration]
) -> tuple[
    RegulatoryDocumentIntake,
    list[RegulatorySourceAsset],
    list[SourceLink],
    IntakeGraph,
    bytes,
]:
    user, target, model = owner
    package, assets, links = _package(session, user, target)
    store = PostgresBackedFileStore()
    raw = b"A separate independent regulation."
    identifier = store.save_file(
        BytesIO(raw),
        "independent.html",
        FileOrigin.OTHER,
        "text/html",
        db_session=session,
    )
    extra = RegulatorySourceAsset(
        package_id=package.id,
        sha256=sha256(raw).hexdigest(),
        file_id=identifier,
        text_file_id=identifier,
        text_sha256=sha256(raw).hexdigest(),
        mime_type="text/html",
        display_name="independent.html",
        byte_count=len(raw),
    )
    session.add(extra)
    session.flush()
    root, annex = assets
    assets = [annex, extra, root]
    # Navigation back to a root must not turn that root into an attachment.
    links.append(
        SourceLink(
            parent_asset_hash=annex.sha256,
            target_asset_hash=root.sha256,
            source_field="backlink",
            label="Ana belge",
            kind="embedded",
        )
    )
    graph = IntakeGraph(
        version=1,
        root_hashes=[root.sha256, extra.sha256],
        primary_parent_by_hash={annex.sha256: root.sha256},
    )
    archive = b"frozen archive input, not a document asset"
    payload = json.dumps(
        {
            "status": "ready",
            "issues": [],
            "assets": [{"sha256": a.sha256, "mime_type": a.mime_type} for a in assets],
            "links": [link.model_dump(mode="json") for link in links],
            "intake_graph": graph.model_dump(mode="json"),
            "input_sha256": sha256(archive).hexdigest(),
        }
    ).encode()
    package.input_spec = {
        "kind": ZIP_INPUT_KIND,
        "input_sha256": sha256(archive).hexdigest(),
    }
    package.input_file_id = store.save_file(
        BytesIO(archive),
        "documents.zip",
        FileOrigin.OTHER,
        "application/zip",
        db_session=session,
    )
    package.manifest_file_id = store.save_file(
        BytesIO(payload),
        "source-manifest.json",
        FileOrigin.OTHER,
        "application/json",
        db_session=session,
    )
    package.manifest_sha256 = sha256(payload).hexdigest()
    package.asset_count = len(assets)
    package.total_bytes = sum(a.byte_count for a in assets)
    session.flush()
    batch, created = intake.create_intake(
        session,
        source_package_id=package.id,
        idempotency_key=uuid4(),
        labeling_model_configuration_id=model.id,
        user=user,
        environment=_ENVIRONMENT,
    )
    assert created
    return batch, assets, links, graph, payload


def test_zip_forest_keeps_independent_roots_annex_backlinks_and_replay_ids(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, assets, links, graph, raw = _zip_package(intake_session, intake_owner)
    members = intake.admit_intake_assets(
        intake_session,
        batch,
        assets,
        links,
        root_hash=assets[0].sha256,
        intake_graph=graph,
        source_manifest_bytes=raw,
    )
    by_asset = {member.source_asset_id: member for member in members}
    by_hash = {asset.sha256: by_asset[asset.id] for asset in assets}
    assert len(members) == 3
    assert {digest for digest, m in by_hash.items() if m.role == "root"} == set(
        graph.root_hashes
    )
    for child, parent in graph.primary_parent_by_hash.items():
        assert by_hash[child].role == "annex"
        assert by_hash[child].parent_document_id == by_hash[parent].id
    assert all(
        by_hash[digest].parent_document_id is None for digest in graph.root_hashes
    )
    assert set(intake_session.scalars(select(DocumentSet__UserFile.user_file_id))) == {
        m.user_file_id for m in members
    }
    before = {(m.id, m.user_file_id) for m in members}
    replay = intake.admit_intake_assets(
        intake_session,
        batch,
        list(reversed(assets)),
        links,
        root_hash=assets[0].sha256,
        intake_graph=graph,
        source_manifest_bytes=raw,
    )
    assert {(m.id, m.user_file_id) for m in replay} == before
    assert len(list(intake_session.scalars(select(UserFile.id)))) == 3
    for asset in assets:
        view = IntakeSourceAsset.model_validate(asset)
        doc = build_intake_document(
            package_id=batch.source_package_id,
            asset=view,
            user_file_id=by_asset[asset.id].user_file_id,
            text="Complete legal source text.",
            links=links,
            package_assets=[IntakeSourceAsset.model_validate(a) for a in assets],
            parent_asset_hashes=graph.parents_for(asset.sha256),
        )
        assert doc.doc_metadata is not None
        context = doc.doc_metadata["regulatory_document_intake"]
        assert context["parent_asset_hashes"] == list(graph.parents_for(asset.sha256))
        assert len(context["source_links"]) == sum(
            link.target_asset_hash == asset.sha256 for link in links
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing-graph",
        "forged-graph",
        "hash",
        "links",
        "stored-source",
        "kind",
        "archive-sha",
    ],
)
def test_zip_admission_rejects_forged_or_changed_evidence_before_creating_files(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    change: str,
) -> None:
    batch, assets, links, graph, raw = _zip_package(intake_session, intake_owner)
    supplied_graph: IntakeGraph | None = graph
    if change == "missing-graph":
        supplied_graph = None
    elif change == "forged-graph":
        supplied_graph = IntakeGraph(
            version=1, root_hashes=[a.sha256 for a in assets], primary_parent_by_hash={}
        )
    elif change == "hash":
        raw = raw + b" "
    elif change == "links":
        links = links[:-1]
    elif change == "stored-source":
        original = assets[0]
        assets[0] = RegulatorySourceAsset(
            **{
                key: getattr(original, key)
                for key in (
                    "id",
                    "package_id",
                    "sha256",
                    "mime_type",
                    "byte_count",
                    "text_file_id",
                    "text_sha256",
                    "original_url",
                    "final_url",
                    "display_name",
                )
            },
            file_id="forged-source-pointer",
        )
    else:
        package = intake_session.get_one(
            AmendmentSourcePackage, batch.source_package_id
        )
        package.input_spec = (
            {"kind": "ordinary"}
            if change == "kind"
            else {**package.input_spec, "input_sha256": "8" * 64}
        )
        intake_session.flush()
    with pytest.raises(intake.IntakeConflictError):
        intake.admit_intake_assets(
            intake_session,
            batch,
            assets,
            links,
            root_hash=assets[0].sha256,
            intake_graph=supplied_graph,
            source_manifest_bytes=raw,
        )
    assert not list(intake_session.scalars(select(UserFile.id)))
    assert not intake.intake_files(intake_session, batch)


@pytest.mark.parametrize("change", ["role", "parent", "relationships"])
def test_zip_replay_rejects_changed_member_graph(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    change: str,
) -> None:
    batch, assets, links, graph, raw = _zip_package(intake_session, intake_owner)
    members = intake.admit_intake_assets(
        intake_session,
        batch,
        assets,
        links,
        root_hash=assets[0].sha256,
        intake_graph=graph,
        source_manifest_bytes=raw,
    )
    member = next(m for m in members if m.role == "annex")
    if change == "role":
        member.role = "root"
    elif change == "parent":
        member.parent_document_id = None
    else:
        member.relationships = []
    intake_session.flush()
    before = {(m.id, m.user_file_id) for m in members}
    with pytest.raises(intake.IntakeConflictError, match="ownership|source links"):
        intake.admit_intake_assets(
            intake_session,
            batch,
            assets,
            links,
            root_hash=assets[0].sha256,
            intake_graph=graph,
            source_manifest_bytes=raw,
        )
    assert {
        (m.id, m.user_file_id) for m in intake.intake_files(intake_session, batch)
    } == before


def test_review_name_hash_is_legacy_compatible_and_protects_new_display_names(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
) -> None:
    batch, members = _admitted(intake_session, *intake_owner)
    file = intake_session.get_one(UserFile, members[0].user_file_id)
    legacy = intake.review_digest(intake_session, batch)
    file.name = "Legacy rename with no naming marker"
    intake_session.flush()
    assert intake.review_digest(intake_session, batch) == legacy
    chunk = intake_session.scalar(
        select(RegulatoryChunk).where(RegulatoryChunk.user_file_id == file.id)
    )
    assert chunk is not None
    chunk.chunk_metadata = {
        **chunk.chunk_metadata,
        "intake_display_name": {"version": 1, "title": file.name},
    }
    intake_session.flush()
    protected = intake.review_digest(intake_session, batch)
    source_before = deepcopy(chunk.chunk_metadata)
    file.name = "Another displayed legal title"
    intake_session.flush()
    assert intake.review_digest(intake_session, batch) != protected
    assert chunk.chunk_metadata == source_before


def test_initial_and_reprepare_jobs_keep_frozen_zip_parent_scope(
    intake_session: Session,
    intake_owner: tuple[User, DocumentSet, ModelConfiguration],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch, assets, _links, graph, raw = _zip_package(intake_session, intake_owner)
    calls: list[dict[str, Any]] = []
    store = cast(FileStore, SimpleNamespace(read_file=lambda _key: BytesIO(raw)))

    @contextmanager
    def scope() -> Iterator[Session]:
        yield intake_session

    monkeypatch.setattr(job, "get_session_with_current_tenant", scope)
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(
        job, "clone_intake_original", lambda **kw: f"owned-{kw['asset'].id}"
    )
    monkeypatch.setattr(job, "prepare_intake_file", lambda **kw: calls.append(kw))
    monkeypatch.setattr(metadata_job, "run_intake_metadata", lambda **_kw: None)
    monkeypatch.setattr(intake, "start_intake_labeling", lambda *_a: uuid4())
    assert job.run_document_intake(intake_id=batch.id, tenant_id="public") is not None
    first_ids = {m.user_file_id for m in intake.intake_files(intake_session, batch)}
    assert len(calls) == 3
    assert {kw["asset"].sha256: kw["parent_asset_hashes"] for kw in calls} == {
        a.sha256: graph.parents_for(a.sha256) for a in assets
    }
    calls.clear()
    request_id = uuid4()
    token = batch.lease_token
    assert token is not None
    monkeypatch.setattr(reprepare_job, "get_session_with_current_tenant", scope)
    monkeypatch.setattr(reprepare_job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(
        reprepare_job, "prepare_intake_file", lambda **kw: calls.append(kw)
    )
    monkeypatch.setattr(reprepare_job, "run_intake_metadata", lambda **_kw: None)
    # Existing reprepare tests cover publication/receipt claim guards separately.
    monkeypatch.setattr(reprepare_job.repository, "claim_reprepare", lambda *_a: token)
    monkeypatch.setattr(
        reprepare_job.repository,
        "latest_reprepare",
        lambda *_a: SimpleNamespace(id=request_id, documents={}),
    )
    monkeypatch.setattr(reprepare_job.repository, "finish_reprepare", lambda *_a: None)
    assert (
        reprepare_job.run_document_intake_reprepare(
            intake_id=batch.id, request_id=request_id, tenant_id="public"
        )
        is not None
    )
    assert {
        m.user_file_id for m in intake.intake_files(intake_session, batch)
    } == first_ids
    assert len(calls) == 3
    assert {kw["asset"].sha256: kw["parent_asset_hashes"] for kw in calls} == {
        a.sha256: graph.parents_for(a.sha256) for a in assets
    }
