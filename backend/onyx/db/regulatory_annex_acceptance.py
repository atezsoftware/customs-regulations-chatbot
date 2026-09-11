"""Durable ownership for the fixed fictional annex release canary."""

import datetime
import hashlib
from collections.abc import Mapping
from typing import Literal, cast
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import KVStore, User, UserFile


class CreationIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["chat", "markdown"]
    marker: str
    artifact_id: UUID | None = None


class RetainedArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str
    id: str
    index_name: str | None = None
    index_uuid: str | None = None


class CanaryRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    release_sha: str
    run_id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    file_id: UUID = Field(default_factory=uuid4)
    document_set_id: int | None = None
    pat_id: int | None = None
    package_id: UUID | None = None
    batch_id: int | None = None
    review_id: UUID | None = None
    chat_ids: list[UUID] = Field(default_factory=list)
    markdown_file_ids: list[UUID] = Field(default_factory=list)
    creation_intents: list[CreationIntent] = Field(default_factory=list, max_length=4)
    retained_objects: list[RetainedArtifact] = Field(
        default_factory=list, max_length=512
    )
    vision_roles: list[dict[str, str | int]] = Field(
        default_factory=list, max_length=100
    )
    phase: str = "reserved"
    created_at: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    evidence: dict[str, str | int | bool] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return "regulatory_annex_acceptance:" + self.release_sha

    @property
    def name(self) -> str:
        return "FICTIONAL ANNEX CANARY " + str(self.run_id)


def verify_dev_configuration() -> dict[str, object]:
    from onyx.db.engine.sql_engine import SqlEngine
    from onyx.db.enums import IndexModelStatus
    from onyx.db.models import SearchSettings
    from onyx.db.regulatory_annex_dev_cutover import configured_indices
    from onyx.indexing.contextual_settings import require_contextual_rag_llm
    from onyx.llm.factory import get_default_llm, get_default_llm_with_vision

    indices = configured_indices()
    with SqlEngine.scoped_engine(pool_size=2, max_overflow=0):
        default = get_default_llm()
        vision = get_default_llm_with_vision()
        if vision is None:
            raise ValueError("configured_vision_model_required")
        contextual: list[dict[str, object]] = []
        with get_session_with_current_tenant() as session:
            session.execute(text("SET TRANSACTION READ ONLY"))
            for settings in session.scalars(
                select(SearchSettings).where(
                    SearchSettings.index_name.in_(indices),
                    SearchSettings.status.in_(
                        [IndexModelStatus.PRESENT, IndexModelStatus.FUTURE]
                    ),
                )
            ):
                model = require_contextual_rag_llm(settings)
                contextual.append(
                    {
                        "index_name": settings.index_name,
                        "contextual_enabled": model is not None,
                        "model_provider": model.config.model_provider
                        if model
                        else None,
                        "model_name": model.config.model_name if model else None,
                    }
                )
    return {
        "database": "customs-regulations-dev",
        "indices": indices,
        "contextual": contextual,
        "default_provider": default.config.model_provider,
        "default_model": default.config.model_name,
        "vision_provider": vision.config.model_provider,
        "vision_model": vision.config.model_name,
    }


def reserve_canary(release_sha: str, *, admin_email: str) -> CanaryRun:
    from onyx.auth.users import is_user_admin

    key = "regulatory_annex_acceptance:" + release_sha
    with get_session_with_current_tenant() as session:
        session.execute(text("SELECT pg_advisory_xact_lock(818297340019)"))
        saved = session.get(KVStore, key)
        if saved is not None:
            return CanaryRun.model_validate(saved.value)
        user = session.scalar(select(User).where(User.__table__.c.email == admin_email))
        if user is None or not user.is_active or not is_user_admin(user):
            raise ValueError("exact_active_canary_admin_required")
        run = CanaryRun(release_sha=release_sha, user_id=user.id)
        session.add(KVStore(key=key, value=run.model_dump(mode="json")))
        session.commit()
        return run


def save_canary(run: CanaryRun) -> None:
    with get_session_with_current_tenant() as session:
        row = session.get(KVStore, run.key, with_for_update=True)
        if row is None or CanaryRun.model_validate(row.value).run_id != run.run_id:
            raise ValueError("canary_ownership_missing")
        row.value = run.model_dump(mode="json")
        session.commit()


def issue_canary_token(run: CanaryRun) -> str:
    from onyx.db.enums import Permission
    from onyx.db.pat import create_pat, revoke_pat

    with get_session_with_current_tenant() as session:
        row = session.get(KVStore, run.key, with_for_update=True)
        if row is None or CanaryRun.model_validate(row.value).run_id != run.run_id:
            raise ValueError("canary_ownership_missing")
        if run.pat_id is not None:
            revoke_pat(session, run.pat_id, run.user_id)
        pat, raw = create_pat(
            session,
            run.user_id,
            run.name,
            1,
            scopes=[
                Permission.FULL_ADMIN_PANEL_ACCESS,
                Permission.BASIC_ACCESS,
                Permission.READ_CHAT,
                Permission.WRITE_CHAT,
            ],
        )
        pat.expires_at = datetime.datetime.now(
            datetime.timezone.utc
        ) + datetime.timedelta(minutes=20)
        run.pat_id = pat.id
        row.value = run.model_dump(mode="json")
        session.commit()
        return raw


def revoke_canary_token(run: CanaryRun) -> None:
    from onyx.db.pat import revoke_pat

    if run.pat_id is not None:
        with get_session_with_current_tenant() as session:
            revoke_pat(session, run.pat_id, run.user_id)
            session.commit()


def bootstrap_canary_file(run: CanaryRun, content: bytes) -> None:
    """Bind only the pre-reserved fictional original; do not fabricate index state."""
    import io

    from onyx.configs.constants import FileOrigin
    from onyx.db.document_set import insert_document_set
    from onyx.db.enums import UserFileStatus
    from onyx.file_store.file_store import get_default_file_store
    from onyx.server.features.document_set.models import DocumentSetCreationRequest

    if run.document_set_id is None:
        with get_session_with_current_tenant() as session:
            from onyx.db.models import DocumentSet

            document_set = session.scalar(
                select(DocumentSet).where(DocumentSet.name == run.name)
            )
            if document_set is None:
                document_set, _ = insert_document_set(
                    DocumentSetCreationRequest(
                        name=run.name,
                        description="Owned fictional release canary; retained audit scope.",
                        cc_pair_ids=[],
                        is_public=False,
                        users=[],
                    ),
                    run.user_id,
                    session,
                )
            if document_set.user_id != run.user_id or document_set.is_public:
                raise ValueError("canary_private_scope_ownership_mismatch")
            run.document_set_id = document_set.id
            row = session.get(KVStore, run.key, with_for_update=True)
            if row is None:
                raise ValueError("canary_ownership_missing")
            row.value = run.model_dump(mode="json")
            session.commit()
    store = get_default_file_store()
    storage_id = "annex-canary-original-" + str(run.file_id)
    existing_size = store.get_file_size(storage_id)
    if existing_size is None:
        store.save_file(
            io.BytesIO(content),
            display_name="Temsili Oran Yonetmeligi.pdf",
            file_origin=FileOrigin.OTHER,
            file_type="application/pdf",
            file_id=storage_id,
        )
    else:
        with store.read_file(storage_id) as stream:
            if (
                hashlib.sha256(stream.read()).digest()
                != hashlib.sha256(content).digest()
            ):
                raise ValueError("canary_original_bytes_changed")
    with get_session_with_current_tenant() as session:
        from onyx.db.document_set import link_user_file_to_document_set

        file = session.get(UserFile, run.file_id)
        if file is None:
            file = UserFile(
                id=run.file_id,
                user_id=run.user_id,
                file_id=storage_id,
                name="Temsili Oran Yonetmeligi.pdf",
                file_type="application/pdf",
                status=UserFileStatus.PROCESSING,
            )
            session.add(file)
            session.flush()
            link_user_file_to_document_set(session, run.document_set_id, file)
        elif file.user_id != run.user_id or file.file_id != storage_id:
            raise ValueError("canary_file_ownership_mismatch")
        session.commit()


def canary_file_state(run: CanaryRun) -> dict[str, object]:
    from onyx.db.models import RegulatoryChunk

    with get_session_with_current_tenant() as session:
        file = session.get(UserFile, run.file_id)
        rows = list(
            session.scalars(
                select(RegulatoryChunk).where(
                    RegulatoryChunk.user_file_id == run.file_id
                )
            ).all()
        )
        return {
            "file_exists": file is not None,
            "status": file.status.value if file else None,
            "canonical": [
                {
                    "id": row.id,
                    "text": row.text,
                    "position": row.position,
                    "start": str(row.validity_start_date),
                    "end": str(row.validity_end_date),
                }
                for row in rows
            ],
        }


def canary_index_names() -> list[str]:
    from onyx.db.enums import IndexModelStatus
    from onyx.db.models import SearchSettings

    with get_session_with_current_tenant() as session:
        return list(
            session.scalars(
                select(SearchSettings.index_name).where(
                    SearchSettings.status.in_(
                        [IndexModelStatus.PRESENT, IndexModelStatus.FUTURE]
                    )
                )
            ).all()
        )


def cleanup_empty_canary_scope(run: CanaryRun) -> None:
    from onyx.db.document_set import delete_document_set
    from onyx.db.models import AmendmentSourcePackage, DocumentSet
    from onyx.file_store.file_store import get_default_file_store

    get_default_file_store().delete_file(
        "annex-canary-original-" + str(run.file_id), error_on_missing=False
    )
    if run.document_set_id is None:
        return
    with get_session_with_current_tenant() as session:
        scope = session.get(DocumentSet, run.document_set_id)
        if scope is None:
            return
        if scope.name != run.name or scope.user_id != run.user_id or scope.is_public:
            raise ValueError("canary_cleanup_scope_ownership_mismatch")
        retained_source = session.scalar(
            select(AmendmentSourcePackage.id)
            .where(AmendmentSourcePackage.document_set_id == scope.id)
            .limit(1)
        )
        if scope.user_files:
            raise ValueError("canary_cleanup_scope_still_has_files")
        if retained_source is not None:
            run.evidence["retained_source_scope"] = scope.id
            return
        delete_document_set(scope, session)


def matching_creation_ids(run: CanaryRun, intent: CreationIntent) -> list[UUID]:
    """Resolve only the exact persisted marker inside its authorized owner scope."""
    from onyx.db.models import ChatSession, DocumentSet, DocumentSet__UserFile

    with get_session_with_current_tenant() as session:
        if intent.kind == "chat":
            if intent.marker not in {
                run.name + " / dated 2026-09-09",
                run.name + " / dated 2026-09-10",
                run.name + " / markdown",
            }:
                raise ValueError("canary_chat_intent_marker_mismatch")
            query = select(ChatSession.id).where(
                ChatSession.user_id == run.user_id,
                ChatSession.description == intent.marker,
            )
        else:
            if intent.marker != "ANNEXCANARY" + run.run_id.hex + ".md":
                raise ValueError("canary_upload_intent_marker_mismatch")
            query = (
                select(UserFile.id)
                .join(
                    DocumentSet__UserFile,
                    DocumentSet__UserFile.user_file_id == UserFile.id,
                )
                .join(
                    DocumentSet, DocumentSet.id == DocumentSet__UserFile.document_set_id
                )
                .where(
                    UserFile.user_id == run.user_id,
                    UserFile.name == intent.marker,
                    DocumentSet.id == run.document_set_id,
                    DocumentSet.user_id == run.user_id,
                    DocumentSet.name == run.name,
                    DocumentSet.is_public.is_(False),
                )
            )
        return list(session.scalars(query.limit(2)).all())


def recover_creation_intents(run: CanaryRun) -> bool:
    complete = True
    for intent in run.creation_intents:
        if intent.artifact_id is None:
            matches = matching_creation_ids(run, intent)
            if len(matches) != 1:
                complete = False
                continue
            intent.artifact_id = matches[0]
        identifiers = run.chat_ids if intent.kind == "chat" else run.markdown_file_ids
        if intent.artifact_id not in identifiers:
            identifiers.append(intent.artifact_id)
    save_canary(run)
    return complete


def _read_canary_worker_failure(
    session: Session, run: CanaryRun
) -> tuple[str, str] | None:
    from onyx.db.models import AmendmentBatch, AmendmentSourcePackage, DocumentSet

    if run.batch_id is None:
        return None
    saved = session.get(KVStore, run.key)
    if saved is None:
        raise ValueError("canary_worker_failure_run_missing")
    stored = CanaryRun.model_validate(saved.value)
    fields = (
        "run_id",
        "release_sha",
        "user_id",
        "file_id",
        "document_set_id",
        "batch_id",
        "package_id",
    )
    if any(getattr(stored, field) != getattr(run, field) for field in fields):
        raise ValueError("canary_worker_failure_run_mismatch")
    batch = session.get(AmendmentBatch, run.batch_id)
    scope = (
        session.get(DocumentSet, run.document_set_id) if run.document_set_id else None
    )
    package = (
        session.get(AmendmentSourcePackage, run.package_id) if run.package_id else None
    )
    if (
        batch is None
        or scope is None
        or scope.is_public
        or package is None
        or package.document_set_id != run.document_set_id
        or package.created_by != run.user_id
        or scope.user_id != run.user_id
        or scope.name != run.name
        or batch.created_by != run.user_id
        or batch.document_set_id != run.document_set_id
        or run.package_id is None
        or batch.source_package_id != run.package_id
        or batch.user_file_ids != [str(run.file_id)]
    ):
        raise ValueError("canary_worker_failure_scope_mismatch")
    if batch.status != "failed":
        return None
    key = f"regulatory_amendment_failure:{batch.id}:{batch.lease_generation}"
    receipt = session.get(KVStore, key)
    if receipt is None:
        return None
    expected = {
        "batch_id": batch.id,
        "lease_generation": batch.lease_generation,
        "document_set_id": run.document_set_id,
        "created_by": str(run.user_id),
        "source_package_id": str(run.package_id),
        "user_file_ids": [str(run.file_id)],
    }
    if not isinstance(receipt.value, Mapping):
        raise ValueError("canary_worker_failure_receipt_mismatch")
    value = cast(Mapping[str, object], receipt.value)
    if any(
        value.get(field) != expected_value for field, expected_value in expected.items()
    ):
        raise ValueError("canary_worker_failure_receipt_mismatch")
    detail = value.get("detail")
    if not isinstance(detail, str) or len(detail) > 4000:
        raise ValueError("canary_worker_failure_detail_invalid")
    return key, detail


def read_canary_worker_failure(run: CanaryRun) -> tuple[str, str] | None:
    with get_session_with_current_tenant() as session:
        session.execute(text("SET TRANSACTION READ ONLY"))
        return _read_canary_worker_failure(session, run)


def retained_canary_audit(run: CanaryRun) -> list[RetainedArtifact]:
    from onyx.db.models import (
        AmendmentSourcePackage,
        AnnexChangeSet,
        DocumentSet,
        PersonalAccessToken,
        RegulatoryFilePublication,
        RegulatorySourceAsset,
    )

    retained: list[RetainedArtifact] = []
    with get_session_with_current_tenant() as session:
        record = session.get(KVStore, run.key)
        if (
            record is None
            or CanaryRun.model_validate(record.value).run_id != run.run_id
        ):
            raise ValueError("canary_audit_run_ownership_missing")
        retained.append(RetainedArtifact(kind="run_record", id=run.key))
        failure = _read_canary_worker_failure(session, run)
        if failure is not None:
            retained.append(
                RetainedArtifact(kind="worker_failure_receipt", id=failure[0])
            )
        tokens = list(
            session.scalars(
                select(PersonalAccessToken.id)
                .where(
                    PersonalAccessToken.user_id == run.user_id,
                    PersonalAccessToken.name == run.name,
                )
                .limit(21)
            )
        )
        if len(tokens) > 20:
            raise ValueError("canary_audit_token_bound_exceeded")
        retained.extend(
            RetainedArtifact(kind="pat_audit", id=str(identifier))
            for identifier in tokens
        )
        scope = (
            session.get(DocumentSet, run.document_set_id)
            if run.document_set_id
            else None
        )
        if scope is not None:
            if (
                scope.name != run.name
                or scope.user_id != run.user_id
                or scope.is_public
            ):
                raise ValueError("canary_audit_scope_ownership_mismatch")
            retained.append(RetainedArtifact(kind="private_scope", id=str(scope.id)))
            packages = list(
                session.scalars(
                    select(AmendmentSourcePackage)
                    .where(
                        AmendmentSourcePackage.document_set_id == scope.id,
                        AmendmentSourcePackage.created_by == run.user_id,
                    )
                    .limit(3)
                )
            )
            if len(packages) > 2:
                raise ValueError("canary_audit_package_bound_exceeded")
            for package in packages:
                retained.append(
                    RetainedArtifact(kind="source_package", id=str(package.id))
                )
                for identifier in (package.input_file_id, package.manifest_file_id):
                    if identifier:
                        retained.append(
                            RetainedArtifact(kind="source_blob", id=identifier)
                        )
                assets = list(
                    session.scalars(
                        select(RegulatorySourceAsset)
                        .where(
                            RegulatorySourceAsset.package_id == package.id,
                        )
                        .limit(22)
                    )
                )
                if len(assets) > 21:
                    raise ValueError("canary_audit_asset_bound_exceeded")
                for asset in assets:
                    retained.append(
                        RetainedArtifact(kind="source_asset", id=str(asset.id))
                    )
                    for identifier in (asset.file_id, asset.text_file_id):
                        if identifier:
                            retained.append(
                                RetainedArtifact(kind="source_blob", id=identifier)
                            )
        reviews = (
            list(
                session.scalars(
                    select(AnnexChangeSet)
                    .where(
                        AnnexChangeSet.batch_id == run.batch_id,
                        AnnexChangeSet.user_file_id == run.file_id,
                    )
                    .limit(11)
                )
            )
            if run.batch_id is not None
            else []
        )
        if len(reviews) > 10:
            raise ValueError("canary_audit_review_bound_exceeded")
        for review in reviews:
            retained.append(RetainedArtifact(kind="review", id=str(review.id)))
            for evidence in review.review_payload.get("evidence", []):
                retained.append(
                    RetainedArtifact(kind="review_evidence", id=str(evidence["id"]))
                )
                retained.append(
                    RetainedArtifact(kind="review_blob", id=str(evidence["file_id"]))
                )
        owners = session.scalars(
            select(RegulatoryFilePublication).where(
                RegulatoryFilePublication.user_file_id.in_(
                    [run.file_id, *run.markdown_file_ids]
                ),
            )
        )
        retained.extend(
            RetainedArtifact(kind="publication_owner", id=str(owner.user_file_id))
            for owner in owners
        )
    unique = {(item.kind, item.id): item for item in retained}
    if len(unique) > 500:
        raise ValueError("canary_audit_total_bound_exceeded")
    return list(unique.values())
