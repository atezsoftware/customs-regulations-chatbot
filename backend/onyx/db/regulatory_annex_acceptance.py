"""Durable ownership for the fixed fictional annex release canary."""

import datetime
import hashlib
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, text

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import KVStore, User, UserFile


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
        if retained_source is not None:
            run.evidence["retained_source_scope"] = scope.id
            return
        if scope.user_files:
            raise ValueError("canary_cleanup_scope_still_has_files")
        delete_document_set(scope, session)
