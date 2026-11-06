"""Durable, revision-locked Markdown originals with explicit upload provenance."""

import datetime
import hashlib
import json
import unicodedata
from collections.abc import Sequence
from io import BytesIO
from pathlib import PurePosixPath
from urllib.parse import unquote, urlsplit
from uuid import UUID, uuid4

from sqlalchemy import event, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db.amendment_sources import create_source_package
from onyx.db.document_intake import IntakeConflictError, resolve_intake_target
from onyx.db.models import (
    AmendmentSourcePackage,
    RegulatoryMarkdownDraft,
    RegulatoryMarkdownDraftDocument,
    RegulatoryMarkdownDraftUpload,
    RegulatorySourceAsset,
    User,
)
from onyx.file_store.file_store import (
    FileStore,
    S3BackedFileStore,
    get_default_file_store,
)
from onyx.file_store.postgres_file_store import PostgresBackedFileStore
from onyx.regulatory.amendments.annexes.models import (
    AcquiredAsset,
    AcquisitionResult,
    SourceLink,
)
from onyx.regulatory.document_intake.markdown_draft_models import (
    MarkdownDraftDocumentSnapshot,
    MarkdownDraftSnapshot,
)
from onyx.regulatory.document_intake.markdown_package import (
    MAX_MARKDOWN_DOCUMENT_BYTES,
    MAX_MARKDOWN_DOCUMENTS,
    MAX_MARKDOWN_LINKS,
    MAX_MARKDOWN_PACKAGE_BYTES,
    MarkdownPackageError,
    _decode,
    _markdown_references,
    _path,
    _target,
)
from onyx.regulatory.document_intake.source_graph import MARKDOWN_DRAFT_INPUT_KIND
from onyx.utils.logger import setup_logger

logger = setup_logger()


def create_draft(
    session: Session, *, user: User, environment: str, idempotency_key: UUID
) -> RegulatoryMarkdownDraft:
    target = resolve_intake_target(session, user)
    session.execute(
        insert(RegulatoryMarkdownDraft)
        .values(
            id=uuid4(),
            created_by=user.id,
            document_set_id=target.id,
            environment=environment,
            idempotency_key=idempotency_key,
        )
        .on_conflict_do_nothing(constraint="uq_markdown_draft_request")
    )
    return session.scalars(
        select(RegulatoryMarkdownDraft).where(
            RegulatoryMarkdownDraft.created_by == user.id,
            RegulatoryMarkdownDraft.document_set_id == target.id,
            RegulatoryMarkdownDraft.environment == environment,
            RegulatoryMarkdownDraft.idempotency_key == idempotency_key,
        )
    ).one()


def get_draft(
    session: Session, draft_id: UUID, *, user: User, environment: str
) -> RegulatoryMarkdownDraft:
    target = resolve_intake_target(session, user)
    draft = session.scalar(
        select(RegulatoryMarkdownDraft).where(
            RegulatoryMarkdownDraft.id == draft_id,
            RegulatoryMarkdownDraft.created_by == user.id,
            RegulatoryMarkdownDraft.document_set_id == target.id,
            RegulatoryMarkdownDraft.environment == environment,
        )
    )
    if draft is None:
        raise ValueError(
            "Markdown draft does not belong to this user, document set and environment"
        )
    return draft


def list_drafts(
    session: Session, *, user: User, environment: str
) -> list[RegulatoryMarkdownDraft]:
    target = resolve_intake_target(session, user)
    return list(
        session.scalars(
            select(RegulatoryMarkdownDraft)
            .where(
                RegulatoryMarkdownDraft.created_by == user.id,
                RegulatoryMarkdownDraft.document_set_id == target.id,
                RegulatoryMarkdownDraft.environment == environment,
            )
            .order_by(RegulatoryMarkdownDraft.updated_at.desc())
        )
    )


def draft_documents(
    session: Session, draft: RegulatoryMarkdownDraft
) -> list[RegulatoryMarkdownDraftDocument]:
    return list(
        session.scalars(
            select(RegulatoryMarkdownDraftDocument)
            .where(
                RegulatoryMarkdownDraftDocument.draft_id == draft.id,
                RegulatoryMarkdownDraftDocument.removed.is_(False),
            )
            .order_by(
                RegulatoryMarkdownDraftDocument.created_at,
                RegulatoryMarkdownDraftDocument.id,
            )
        )
    )


def draft_snapshot(
    session: Session, draft: RegulatoryMarkdownDraft
) -> MarkdownDraftSnapshot:
    # Read revision and tree under one lock, including immediately after commit.
    session.refresh(draft, with_for_update={"read": True})
    snapshot = MarkdownDraftSnapshot.model_validate(draft)
    snapshot.documents = [
        MarkdownDraftDocumentSnapshot.model_validate(row)
        for row in draft_documents(session, draft)
    ]
    return snapshot


def _lock(session: Session, draft: RegulatoryMarkdownDraft) -> None:
    session.refresh(draft, with_for_update=True)


def _require_mutable(draft: RegulatoryMarkdownDraft, expected_revision: int) -> None:
    if draft.status != "draft" or draft.revision != expected_revision:
        raise IntakeConflictError(
            "The Markdown draft changed or is already prepared; reload before editing"
        )


def _touch(draft: RegulatoryMarkdownDraft) -> None:
    draft.revision += 1
    draft.updated_at = datetime.datetime.now(datetime.timezone.utc)


def _save_blob(
    session: Session,
    store: FileStore,
    raw: bytes,
    name: str,
    mime: str,
    metadata: dict[str, str],
) -> str:
    # Postgres FileStore commits its supplied session. Restrict that commit to a savepoint.
    if isinstance(store, PostgresBackedFileStore):
        with Session(
            session.connection(), join_transaction_mode="create_savepoint"
        ) as blob_session:
            return store.save_file(
                BytesIO(raw),
                name,
                FileOrigin.OTHER,
                mime,
                file_metadata=metadata,
                db_session=blob_session,
            )
    # External storage keeps its committed FileRecord so rollback cleanup can find the object.
    if isinstance(store, S3BackedFileStore):
        file_id = store.save_file(
            BytesIO(raw),
            name,
            FileOrigin.OTHER,
            mime,
            file_metadata=metadata,
            cleanup_on_metadata_failure=True,
        )
    else:
        file_id = store.save_file(
            BytesIO(raw), name, FileOrigin.OTHER, mime, file_metadata=metadata
        )
    pending = session.info.setdefault("markdown_draft_external_blobs", [])
    pending.append((store, file_id))
    if not session.info.get("markdown_draft_blob_hooks"):
        session.info["markdown_draft_blob_hooks"] = True
        event.listen(session, "after_rollback", _cleanup_rollback_blobs)
        event.listen(session, "after_commit", _forget_committed_blobs)
    return file_id


def _cleanup_rollback_blobs(session: Session) -> None:
    for store, file_id in session.info.pop("markdown_draft_external_blobs", []):
        try:
            store.delete_file(file_id, error_on_missing=False)
        except Exception:
            logger.exception("Markdown draft rollback blob cleanup failed: %s", file_id)


def _forget_committed_blobs(session: Session) -> None:
    session.info.pop("markdown_draft_external_blobs", None)


def _read_blob(session: Session, store: FileStore, file_id: str, limit: int) -> bytes:
    stream = (
        store.read_file(file_id, db_session=session)
        if isinstance(store, PostgresBackedFileStore)
        else store.read_file(file_id)
    )
    with stream:
        return stream.read(limit + 1)


def _validate_upload(name: str, raw: bytes) -> None:
    if (
        len(name) > 255
        or "/" in name
        or "\\" in name
        or PurePosixPath(name).suffix.lower() != ".md"
    ):
        raise ValueError(
            "Upload a Markdown .md file with a plain filename of at most 255 characters"
        )
    _path(name)
    references, _title = _markdown_references(_decode(raw, name), name)
    if len(references) > MAX_MARKDOWN_LINKS:
        raise MarkdownPackageError("markdown_link_limit", name)
    # A local target may arrive in a later upload; validate its syntax and resource type now.
    for reference in references:
        try:
            _target(reference, name, {name})
        except MarkdownPackageError as error:
            if error.code != "missing_local_markdown_target":
                raise


def add_documents(
    session: Session,
    draft: RegulatoryMarkdownDraft,
    *,
    files: Sequence[tuple[str, bytes]],
    expected_revision: int,
    idempotency_key: UUID,
    parent_document_id: UUID | None = None,
    store: FileStore | None = None,
) -> RegulatoryMarkdownDraft:
    _lock(session, draft)
    if not files or len(files) > MAX_MARKDOWN_DOCUMENTS:
        raise ValueError(
            f"Upload between 1 and {MAX_MARKDOWN_DOCUMENTS} Markdown documents"
        )
    descriptors = [
        {
            "name": name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
        for name, raw in files
    ]
    request_hash = hashlib.sha256(
        json.dumps(
            {
                "files": descriptors,
                "parent_document_id": str(parent_document_id)
                if parent_document_id
                else None,
                "expected_revision": expected_revision,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    receipt = session.get(RegulatoryMarkdownDraftUpload, (draft.id, idempotency_key))
    if receipt is not None:
        if receipt.request_hash != request_hash:
            raise IntakeConflictError(
                "The idempotency key belongs to a different Markdown upload"
            )
        return draft
    _require_mutable(draft, expected_revision)
    existing = draft_documents(session, draft)
    if parent_document_id is not None and parent_document_id not in {
        row.id for row in existing
    }:
        raise ValueError(
            "The attachment parent must be a document in this same Markdown draft"
        )
    if len(existing) + len(files) > MAX_MARKDOWN_DOCUMENTS:
        raise ValueError(
            f"A Markdown draft may contain at most {MAX_MARKDOWN_DOCUMENTS} documents"
        )
    if (
        sum(row.byte_count for row in existing) + sum(len(raw) for _, raw in files)
        > MAX_MARKDOWN_PACKAGE_BYTES
    ):
        raise MarkdownPackageError("markdown_package_byte_limit")
    hashes = {row.sha256 for row in existing}
    for (name, raw), descriptor in zip(files, descriptors):
        _validate_upload(name, raw)
        if descriptor["sha256"] in hashes:
            raise MarkdownPackageError("duplicate_markdown_content", name)
        hashes.add(str(descriptor["sha256"]))
    file_store = store or get_default_file_store()
    for (name, raw), descriptor in zip(files, descriptors):
        identifier = uuid4()
        file_id = _save_blob(
            session,
            file_store,
            raw,
            name,
            "text/markdown",
            {
                "markdown_draft_id": str(draft.id),
                "draft_document_id": str(identifier),
                "sha256": str(descriptor["sha256"]),
            },
        )
        session.add(
            RegulatoryMarkdownDraftDocument(
                id=identifier,
                draft_id=draft.id,
                parent_document_id=parent_document_id,
                display_name=name,
                file_id=file_id,
                byte_count=len(raw),
                sha256=str(descriptor["sha256"]),
            )
        )
    _touch(draft)
    session.add(
        RegulatoryMarkdownDraftUpload(
            draft_id=draft.id,
            idempotency_key=idempotency_key,
            request_hash=request_hash,
            resulting_revision=draft.revision,
        )
    )
    session.flush()
    return draft


def remove_document(
    session: Session,
    draft: RegulatoryMarkdownDraft,
    document_id: UUID,
    *,
    expected_revision: int,
) -> RegulatoryMarkdownDraft:
    _lock(session, draft)
    _require_mutable(draft, expected_revision)
    documents = draft_documents(session, draft)
    if document_id not in {row.id for row in documents}:
        raise ValueError("Document does not belong to this Markdown draft")
    removed = {document_id}
    while True:
        descendants = {row.id for row in documents if row.parent_document_id in removed}
        if descendants <= removed:
            break
        removed.update(descendants)
    for document in documents:
        if document.id in removed:
            document.removed = True
    _touch(draft)
    session.flush()
    return draft


def cleanup_removed_documents(
    session: Session, draft: RegulatoryMarkdownDraft, *, store: FileStore | None = None
) -> None:
    """Retained tombstones make failed storage deletion retryable after commit."""
    file_store = store or get_default_file_store()
    rows = list(
        session.scalars(
            select(RegulatoryMarkdownDraftDocument).where(
                RegulatoryMarkdownDraftDocument.draft_id == draft.id,
                RegulatoryMarkdownDraftDocument.removed.is_(True),
                RegulatoryMarkdownDraftDocument.file_id.is_not(None),
            )
        )
    )
    for row in rows:
        assert row.file_id is not None
        if isinstance(file_store, PostgresBackedFileStore):
            with Session(
                session.connection(), join_transaction_mode="create_savepoint"
            ) as blob_session:
                file_store.delete_file(
                    row.file_id, error_on_missing=False, db_session=blob_session
                )
        else:
            file_store.delete_file(row.file_id, error_on_missing=False)
        row.file_id = None
    session.flush()


def prepare_draft(
    session: Session,
    draft: RegulatoryMarkdownDraft,
    *,
    expected_revision: int,
    store: FileStore | None = None,
) -> AmendmentSourcePackage:
    _lock(session, draft)
    if draft.status == "prepared":
        if draft.revision not in {expected_revision, expected_revision + 1}:
            raise IntakeConflictError(
                "The Markdown draft revision changed before preparation"
            )
        assert draft.source_package_id is not None
        return session.get_one(AmendmentSourcePackage, draft.source_package_id)
    _require_mutable(draft, expected_revision)
    documents = draft_documents(session, draft)
    if not documents:
        raise ValueError("Upload at least one Markdown document before preparing")
    file_store = store or get_default_file_store()
    by_id = {row.id: row for row in documents}
    roots = [row for row in documents if row.parent_document_id is None]
    ordered = list(roots)
    for parent in ordered:
        ordered.extend(row for row in documents if row.parent_document_id == parent.id)
    if len(ordered) != len(documents):
        raise ValueError("The Markdown draft attachment tree is incomplete")
    assets: list[AcquiredAsset] = []
    references_by_id = {}
    for document in ordered:
        if document.file_id is None:
            raise ValueError("A staged Markdown original is missing")
        raw = _read_blob(
            session, file_store, document.file_id, MAX_MARKDOWN_DOCUMENT_BYTES
        )
        if (
            len(raw) != document.byte_count
            or hashlib.sha256(raw).hexdigest() != document.sha256
        ):
            raise ValueError("A staged Markdown original changed before preparation")
        _validate_upload(document.display_name, raw)
        text = _decode(raw, document.display_name)
        references, _title = _markdown_references(text, document.display_name)
        references_by_id[document.id] = references
        assets.append(
            AcquiredAsset(
                sha256=document.sha256,
                content=raw,
                mime_type="text/markdown",
                display_name=document.display_name,
                text=text,
            )
        )
    if (
        sum(len(references) for references in references_by_id.values())
        > MAX_MARKDOWN_LINKS
    ):
        raise MarkdownPackageError("markdown_link_limit")
    links = [
        SourceLink(
            parent_asset_hash=by_id[row.parent_document_id].sha256,
            target_asset_hash=row.sha256,
            source_field=f"markdown-draft:{row.parent_document_id}:{row.id}",
            label=row.display_name,
            kind="embedded",
        )
        for row in ordered
        if row.parent_document_id is not None
    ]
    # Resolve optional body links only within one root family. Explicit attachments need no body link.
    for document in ordered:
        ancestor = document
        while ancestor.parent_document_id is not None:
            ancestor = by_id[ancestor.parent_document_id]
        family_ids = {ancestor.id}
        for member in ordered:
            if member.parent_document_id in family_ids:
                family_ids.add(member.id)
        paths: dict[str, list[RegulatoryMarkdownDraftDocument]] = {}
        for member in ordered:
            if member.id in family_ids:
                paths.setdefault(member.display_name, []).append(member)
        for ordinal, reference in enumerate(references_by_id[document.id], 1):
            try:
                target = _target(reference, document.display_name, set(paths))
            except MarkdownPackageError as error:
                if error.code != "missing_local_markdown_target":
                    raise
                # The chooser strips folders; resolve only an explicit family's unique basename.
                decoded_path = unquote(
                    urlsplit(reference.url).path, encoding="utf-8", errors="strict"
                )
                if ".." in PurePosixPath(decoded_path).parts or any(
                    unicodedata.category(character).startswith("C")
                    for character in decoded_path
                ):
                    raise
                basename = PurePosixPath(decoded_path).name
                if basename not in paths:
                    raise
                target = basename
            if target is not None:
                candidates = paths[target]
                if len(candidates) != 1:
                    raise ValueError(
                        "A local Markdown body link has an ambiguous filename within its document family"
                    )
                linked = candidates[0]
                links.append(
                    SourceLink(
                        parent_asset_hash=document.sha256,
                        target_asset_hash=linked.sha256,
                        requested_url=target,
                        original_url=reference.literal_url,
                        source_field=f"markdown:{document.id}:link:{ordinal}",
                        label=reference.label or target,
                        kind="internal" if linked.id == document.id else "embedded",
                    )
                )
    upload_documents = [
        {
            "id": str(row.id),
            "display_name": row.display_name,
            "parent_document_id": str(row.parent_document_id)
            if row.parent_document_id
            else None,
            "sha256": row.sha256,
            "byte_count": row.byte_count,
            "file_id": row.file_id,
        }
        for row in ordered
    ]
    frozen_input = {
        "draft_id": str(draft.id),
        "revision": draft.revision,
        "created_by": str(draft.created_by),
        "document_set_id": draft.document_set_id,
        "environment": draft.environment,
        "documents": upload_documents,
    }
    input_bytes = json.dumps(frozen_input, sort_keys=True, ensure_ascii=False).encode()
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    package, created = create_source_package(
        session,
        document_set_id=draft.document_set_id,
        environment=draft.environment,
        idempotency_key=f"markdown-draft:{draft.id}",
        request_hash=input_hash,
        input_spec={
            "kind": MARKDOWN_DRAFT_INPUT_KIND,
            "preparation_method": "staged_markdown",
            "display_name": "Markdown documents",
            "input_sha256": input_hash,
            "draft_id": str(draft.id),
            "draft_revision": draft.revision,
        },
        created_by=draft.created_by,
    )
    if not created:
        raise IntakeConflictError("The Markdown draft already has a source preparation")
    result = AcquisitionResult(status="ready", assets=assets, links=links)
    manifest = result.model_dump(mode="json")
    manifest.update(
        preparation_method="staged_markdown",
        input_sha256=input_hash,
        markdown_draft=frozen_input,
        intake_graph={
            "version": 1,
            "root_hashes": [row.sha256 for row in roots],
            "primary_parent_by_hash": {
                row.sha256: by_id[row.parent_document_id].sha256
                for row in ordered
                if row.parent_document_id is not None
            },
        },
    )
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode()
    for document, asset in zip(ordered, assets):
        text_bytes = asset.text.encode()
        text_id = _save_blob(
            session,
            file_store,
            text_bytes,
            "prepared-markdown.txt",
            "text/plain",
            {"source_package_id": str(package.id)},
        )
        assert document.file_id is not None
        session.add(
            RegulatorySourceAsset(
                package_id=package.id,
                sha256=asset.sha256,
                file_id=document.file_id,
                text_file_id=text_id,
                text_sha256=hashlib.sha256(text_bytes).hexdigest(),
                mime_type=asset.mime_type,
                display_name=asset.display_name,
                byte_count=len(asset.content),
            )
        )
    package.input_file_id = _save_blob(
        session,
        file_store,
        input_bytes,
        "markdown-draft-input.json",
        "application/json",
        {"source_package_id": str(package.id)},
    )
    package.manifest_file_id = _save_blob(
        session,
        file_store,
        manifest_bytes,
        "source-manifest.json",
        "application/json",
        {"source_package_id": str(package.id)},
    )
    package.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    package.status, package.asset_count = "ready", len(assets)
    package.total_bytes = sum(row.byte_count for row in ordered)
    draft.source_package_id, draft.status = package.id, "prepared"
    _touch(draft)
    session.flush()
    return package
