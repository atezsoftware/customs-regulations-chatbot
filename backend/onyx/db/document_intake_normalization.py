"""Versioned, retained canonical-input evidence for source-preserving reprepare."""

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict
from sqlalchemy import select
from sqlalchemy.orm import Session

from onyx.connectors.models import Document
from onyx.db.document_intake_graph import (
    require_frozen_zip_intake_graph,
    require_frozen_zip_members,
)
from onyx.db.file_record import FileRecordNotFoundError
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryDocumentIntake,
    RegulatoryDocumentIntakeFile,
    RegulatorySourceAsset,
)
from onyx.db.regulatory_canonical_revisions import get_canonical_revisions
from onyx.document_index.publication_models import publication_digest
from onyx.file_processing.original_ingestion import (
    LoadedUserFileDocuments,
    OriginalIngestionReceipt,
    documents_sha256,
)
from onyx.file_store.file_store import get_default_file_store
from onyx.regulatory.amendments.annexes.evidence import read_source_graph
from onyx.regulatory.document_intake.models import IntakeSourceAsset
from onyx.regulatory.document_intake.preparation import load_verified_intake_documents
from onyx.regulatory.document_intake.source_graph import (
    ZIP_INPUT_KIND,
    parse_intake_graph,
)

NORMALIZATION_WITNESS_KEY = "intake_normalization_witness"


class NormalizationDerivation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    formatter_version: Literal[2, 3] = 3
    source_context_sha256: str
    raw_sha256: str
    prepared_text_sha256: str
    original_plaintext_sha256: str
    original_documents_sha256: str
    normalized_plaintext_sha256: str
    normalized_documents_sha256: str
    normalized_documents: list[dict[str, Any]]
    witness_sha256: str = ""


def _read_frozen_source(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    original: OriginalIngestionReceipt,
    *,
    lock: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    package = session.get_one(
        AmendmentSourcePackage,
        batch.source_package_id,
        populate_existing=True,
        with_for_update=lock,
    )
    if (
        package.status != "ready"
        or package.issues
        or package.created_by != batch.created_by
        or package.document_set_id != batch.document_set_id
        or not package.manifest_file_id
        or not package.manifest_sha256
    ):
        raise ValueError("The frozen normalization source package changed")
    query = (
        select(RegulatorySourceAsset)
        .where(RegulatorySourceAsset.package_id == package.id)
        .order_by(RegulatorySourceAsset.id)
        .execution_options(populate_existing=True)
    )
    if lock:
        query = query.with_for_update()
    assets = list(session.scalars(query))
    members = list(
        session.scalars(
            select(RegulatoryDocumentIntakeFile)
            .where(RegulatoryDocumentIntakeFile.intake_id == batch.id)
            .order_by(RegulatoryDocumentIntakeFile.id)
            .execution_options(populate_existing=True)
        )
    )
    by_id = {asset.id: asset for asset in assets}
    if (
        member.source_asset_id not in by_id
        or {row.source_asset_id for row in members} != set(by_id)
        or len(members) != len(assets)
        or by_id[member.source_asset_id].sha256 != original.raw_sha256
    ):
        raise ValueError("The frozen normalization source asset changed")
    store = get_default_file_store()
    links = read_source_graph(
        store,
        manifest_file_id=package.manifest_file_id,
        manifest_sha256=package.manifest_sha256,
        assets=assets,
    )
    with store.read_file(package.manifest_file_id) as stream:
        manifest_bytes = stream.read(150 * 1024 * 1024 + 1)
    if hashlib.sha256(manifest_bytes).hexdigest() != package.manifest_sha256:
        raise ValueError("The frozen normalization manifest changed")
    manifest = json.loads(manifest_bytes)
    frozen_assets = {item["sha256"]: item for item in manifest["assets"]}
    for asset in assets:
        frozen = frozen_assets[asset.sha256]
        if (
            not isinstance(frozen.get("text"), str)
            or hashlib.sha256(frozen["text"].encode()).hexdigest() != asset.text_sha256
            or any(
                frozen.get(key) != getattr(asset, key)
                for key in ("mime_type", "display_name", "original_url", "final_url")
            )
        ):
            raise ValueError(
                "The prepared normalization source differs from its manifest"
            )
    for row in members:
        digest = by_id[row.source_asset_id].sha256
        if row.relationships != [
            link.model_dump(mode="json")
            for link in links
            if link.parent_asset_hash == digest or link.target_asset_hash == digest
        ]:
            raise ValueError("The frozen normalization source relationships changed")
    graph = None
    if package.input_spec.get("kind") == ZIP_INPUT_KIND:
        graph = parse_intake_graph(
            manifest,
            input_kind=ZIP_INPUT_KIND,
            asset_hashes=[asset.sha256 for asset in assets],
            links=links,
        )
        require_frozen_zip_intake_graph(
            session,
            package,
            assets,
            links,
            intake_graph=graph,
            manifest_bytes=manifest_bytes,
        )
        assert graph is not None
        require_frozen_zip_members(members, assets, links, graph)
    views = [IntakeSourceAsset.model_validate(asset) for asset in assets]
    asset = next(view for view in views if view.id == member.source_asset_id)
    context = {
        "package_id": str(package.id),
        "manifest_file_id": package.manifest_file_id,
        "manifest_sha256": package.manifest_sha256,
        "input_spec": package.input_spec,
        "assets": [view.model_dump(mode="json") for view in views],
        "members": [
            {
                "id": str(row.id),
                "user_file_id": str(row.user_file_id),
                "source_asset_id": str(row.source_asset_id),
                "parent_document_id": str(row.parent_document_id)
                if row.parent_document_id
                else None,
                "role": row.role,
                "relationships": row.relationships,
            }
            for row in members
        ],
        "original_file_id": original.file_id,
    }
    arguments = dict(
        package_id=package.id,
        asset=asset,
        user_file_id=member.user_file_id,
        links=links,
        package_assets=views,
        store=store,
        original_file_id=original.file_id,
        parent_asset_hashes=graph.parents_for(asset.sha256)
        if graph is not None
        else None,
    )
    return context, arguments


def _frozen_source(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    original: OriginalIngestionReceipt,
    *,
    lock: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        return _read_frozen_source(session, batch, member, original, lock=lock)
    except (KeyError, TypeError, FileRecordNotFoundError) as error:
        raise ValueError("The retained normalization source is incomplete") from error


def derive_normalization(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    original: OriginalIngestionReceipt,
    documents: LoadedUserFileDocuments,
) -> dict[str, Any]:
    context, arguments = _frozen_source(session, batch, member, original, lock=True)
    try:
        rebuilt = load_verified_intake_documents(**arguments)
    except FileRecordNotFoundError as error:
        raise ValueError("The retained normalization source is missing") from error
    if rebuilt.original_extraction != documents.original_extraction or (
        documents_sha256(rebuilt) != documents_sha256(documents)
    ):
        raise ValueError(
            "Reprepared documents differ from the verified source formatter"
        )
    proof = NormalizationDerivation(
        source_context_sha256=publication_digest(context),
        raw_sha256=original.raw_sha256,
        prepared_text_sha256=arguments["asset"].text_sha256,
        original_plaintext_sha256=original.plaintext_sha256,
        original_documents_sha256=original.documents_sha256,
        normalized_plaintext_sha256=rebuilt.original_extraction.plaintext_sha256,
        normalized_documents_sha256=rebuilt.original_extraction.documents_sha256,
        normalized_documents=[
            document.model_dump(mode="json", exclude={"doc_updated_at"})
            for document in rebuilt
        ],
    )
    payload = proof.model_dump(mode="json", exclude={"witness_sha256"})
    return {**payload, "witness_sha256": publication_digest(payload)}


def validated_normalized_documents(
    session: Session,
    batch: RegulatoryDocumentIntake,
    member: RegulatoryDocumentIntakeFile,
    original: OriginalIngestionReceipt,
    receipt: OriginalIngestionReceipt,
    derivation: Any,
    revision_ids: Sequence[str],
) -> LoadedUserFileDocuments:
    """Validate the sealed artifact without executing a later formatter version."""
    proof = NormalizationDerivation.model_validate(derivation)
    if proof.witness_sha256 != publication_digest(
        proof.model_dump(mode="json", exclude={"witness_sha256"})
    ):
        raise ValueError("The retained normalization witness changed")
    revisions = get_canonical_revisions(
        session, [UUID(value) for value in revision_ids]
    )
    snapshots = sorted(
        [revision.snapshot for revision in revisions.values()],
        key=lambda row: (row.position, row.id),
    )
    if (
        not snapshots
        or len(snapshots) != len(revision_ids)
        or any(
            row.user_file_id != str(member.user_file_id)
            or row.metadata.get(NORMALIZATION_WITNESS_KEY) != proof.witness_sha256
            for row in snapshots
        )
        or publication_digest([row.model_dump(mode="json") for row in snapshots])
        != receipt.canonical_sha256
    ):
        raise ValueError(
            "The normalization witness differs from its immutable revisions"
        )
    context, arguments = _frozen_source(session, batch, member, original)
    asset = arguments["asset"]
    if (
        proof.source_context_sha256 != publication_digest(context)
        or proof.raw_sha256 != original.raw_sha256
        or proof.prepared_text_sha256 != asset.text_sha256
        or proof.original_plaintext_sha256 != original.plaintext_sha256
        or proof.original_documents_sha256 != original.documents_sha256
        or proof.normalized_plaintext_sha256 != receipt.plaintext_sha256
        or proof.normalized_documents_sha256 != receipt.documents_sha256
    ):
        raise ValueError("The retained normalization source or receipt changed")
    # Verify both retained streams even when their storage pointers still match.
    from onyx.regulatory.amendments.annexes.sources import (
        MAX_ASSET_BYTES,
        MAX_TEXT_CHARS,
    )
    from onyx.regulatory.document_intake.preparation import _read_verified

    store = arguments["store"]
    try:
        raw = _read_verified(store, asset.file_id, asset.sha256, limit=MAX_ASSET_BYTES)
        if len(raw) != asset.byte_count:
            raise ValueError("The retained normalization source byte count changed")
        _read_verified(store, original.file_id, asset.sha256, limit=MAX_ASSET_BYTES)
        _read_verified(
            store, asset.text_file_id, asset.text_sha256, limit=MAX_TEXT_CHARS * 4
        )
    except FileRecordNotFoundError as error:
        raise ValueError("The retained normalization source is missing") from error
    documents = LoadedUserFileDocuments(
        [Document.model_validate(value) for value in proof.normalized_documents],
        file_id=original.file_id,
        raw_sha256=original.raw_sha256,
    )
    if (
        len(documents) != 1
        or documents[0].id != str(member.user_file_id)
        or documents[0].file_id != original.file_id
        or documents.original_extraction.documents_sha256 != receipt.documents_sha256
        or documents.original_extraction.plaintext_sha256 != receipt.plaintext_sha256
    ):
        raise ValueError("The retained normalized document artifact changed")
    return documents
