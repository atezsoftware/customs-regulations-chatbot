import datetime
import threading
import time
from uuid import UUID

import sqlalchemy as sa
from celery import Celery, Task, shared_task
from redis.exceptions import LockNotOwnedError, RedisError
from redis.lock import Lock as RedisLock
from sqlalchemy import select

from onyx.access.access import build_access_for_user_files, get_access_for_user_files
from onyx.access.models import DocumentAccess
from onyx.background.celery.apps.app_base import task_logger
from onyx.background.celery.celery_redis import (
    celery_get_broker_client,
    celery_get_queue_length,
)
from onyx.background.celery.celery_utils import httpx_init_vespa_pool
from onyx.background.celery.tasks.shared.RetryDocumentIndex import RetryDocumentIndex
from onyx.configs import app_configs
from onyx.configs.app_configs import (
    CELERY_WORKER_USER_FILE_PROCESSING_CONCURRENCY,
    DEFER_USER_FILE_INDEXING,
    DISABLE_VECTOR_DB,
    MANAGED_VESPA,
    VESPA_CLOUD_CERT_PATH,
    VESPA_CLOUD_KEY_PATH,
)
from onyx.configs.constants import (
    CELERY_GENERIC_BEAT_LOCK_TIMEOUT,
    CELERY_USER_FILE_DELETE_TASK_EXPIRES,
    CELERY_USER_FILE_PROCESSING_LOCK_TIMEOUT,
    CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
    CELERY_USER_FILE_PROJECT_SYNC_LOCK_TIMEOUT,
    CELERY_USER_FILE_PROJECT_SYNC_TASK_EXPIRES,
    USER_FILE_DELETE_MAX_QUEUE_DEPTH,
    USER_FILE_PROCESSING_MAX_QUEUE_DEPTH,
    USER_FILE_PROJECT_SYNC_MAX_QUEUE_DEPTH,
    OnyxCeleryPriority,
    OnyxCeleryQueues,
    OnyxCeleryTask,
    OnyxRedisLocks,
)
from onyx.connectors.file.connector import LocalFileConnector
from onyx.connectors.models import Document
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import UserFileStatus
from onyx.db.models import SearchSettings, UserFile
from onyx.db.port_attempt import port_backfill_has_pending_work
from onyx.db.port_orphan_candidate import record_port_orphan_candidates_for_user_file
from onyx.db.regulatory_chunks import (
    get_chunk_counts_for_files,
    get_chunks_for_file,
    has_regulatory_chunks_for_file,
)
from onyx.db.regulatory_indexing_jobs import (
    get_regulatory_indexing_job,
    request_user_file_deletion_cleanup,
)
from onyx.db.search_settings import (
    active_secondary_port_target,
    get_active_search_settings,
    get_active_search_settings_list,
)
from onyx.db.user_file import (
    fetch_document_set_names_for_user_files,
    fetch_persona_ids_for_user_files,
    fetch_user_files_with_access_relationships,
    fetch_user_project_ids_for_user_files,
    lock_completed_user_file_for_projection,
    mark_user_file_reconcile_pending,
)
from onyx.document_index.factory import get_all_document_indices
from onyx.document_index.interfaces_new import (
    IndexingMetadata,
    MetadataUpdateRequest,
    SecondaryIndexDocumentMissingError,
)
from onyx.file_processing.user_file_loader import load_user_file_documents
from onyx.file_store.file_store import get_default_file_store
from onyx.file_store.staging import delete_files_best_effort
from onyx.file_store.utils import (
    store_user_file_plaintext,
    user_file_id_to_plaintext_file_name,
)
from onyx.httpx.httpx_pool import HttpxPool
from onyx.indexing.adapters.user_file_indexing_adapter import (
    UserFileDeletingSkip,
    UserFileIndexingAdapter,
)
from onyx.indexing.contextual_settings import effective_contextual_rag_enabled
from onyx.indexing.embedder import DefaultIndexingEmbedder
from onyx.indexing.indexing_pipeline import (
    process_image_sections,
    run_indexing_pipeline,
)
from onyx.redis.redis_pool import get_redis_client
from onyx.redis.tenant_redis_client import TenantRedisClient
from onyx.regulatory.indexing import RegulatoryIndexingChunker
from onyx.regulatory.indexing_jobs.configuration import (
    compute_regulatory_chunk_generation_hash,
)
from onyx.regulatory.indexing_jobs.preparation import (
    prepare_regulatory_indexing_job_from_chunks,
)
from onyx.regulatory.projection import (
    _project_rows_to_search_settings,
    project_user_file_to_index,
)
from onyx.utils.variable_functionality import global_version

_USER_FILE_PROJECT_SYNC_QUEUE_TARGET_DEPTH = max(
    1, CELERY_WORKER_USER_FILE_PROCESSING_CONCURRENCY * 2
)
_REDIS_LOCK_HEARTBEAT_STOP_TIMEOUT_SECONDS = 5.0


class _RedisLockHeartbeat:
    """Renew a finite Redis lease while one project-sync operation is active."""

    def __init__(
        self,
        lock: RedisLock,
        *,
        lock_timeout_seconds: int,
        operation_id: str,
    ) -> None:
        self._lock = lock
        self._operation_id = operation_id
        self._refresh_interval_seconds = max(1.0, lock_timeout_seconds / 3)
        self._retry_interval_seconds = min(5.0, self._refresh_interval_seconds)
        self._stop_event = threading.Event()
        self._lost_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name=f"project-sync-lock-heartbeat-{self._operation_id}",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        wait_seconds = self._refresh_interval_seconds
        while not self._stop_event.wait(wait_seconds):
            try:
                if not self._lock.reacquire():
                    self._lost_event.set()
                    task_logger.error(
                        "Project-sync lock renewal returned false for user_file_id=%s",
                        self._operation_id,
                    )
                    return
                wait_seconds = self._refresh_interval_seconds
            except LockNotOwnedError:
                self._lost_event.set()
                task_logger.exception(
                    "Project-sync lock lease was lost for user_file_id=%s",
                    self._operation_id,
                )
                return
            except RedisError:
                # Redis outages also prevent the producer from observing or
                # enqueuing work. Retry promptly so a recovered connection can
                # renew the lease before its TTL elapses.
                task_logger.warning(
                    "Could not renew project-sync lock for user_file_id=%s; retrying",
                    self._operation_id,
                    exc_info=True,
                )
                wait_seconds = self._retry_interval_seconds

    def ensure_owned(self) -> None:
        if self._lost_event.is_set():
            raise RuntimeError(
                f"Project-sync lock lease lost for user file {self._operation_id}"
            )
        try:
            owned = self._lock.owned()
        except RedisError as e:
            self._lost_event.set()
            raise RuntimeError(
                f"Could not verify project-sync lock for user file {self._operation_id}"
            ) from e
        if not owned:
            self._lost_event.set()
            raise RuntimeError(
                f"Project-sync lock lease lost for user file {self._operation_id}"
            )

    def stop(self) -> bool:
        self._stop_event.set()
        if self._thread is None:
            return True
        self._thread.join(timeout=_REDIS_LOCK_HEARTBEAT_STOP_TIMEOUT_SECONDS)
        return not self._thread.is_alive()


def _as_uuid(value: str | UUID) -> UUID:
    """Return a UUID, accepting either a UUID or a string-like value."""
    return value if isinstance(value, UUID) else UUID(str(value))


def _user_file_lock_key(user_file_id: str | UUID) -> str:
    return f"{OnyxRedisLocks.USER_FILE_PROCESSING_LOCK_PREFIX}:{user_file_id}"


def _user_file_queued_key(user_file_id: str | UUID) -> str:
    """Key that exists while a process_single_user_file task is sitting in the queue.

    The beat generator sets this with a TTL equal to CELERY_USER_FILE_PROCESSING_TASK_EXPIRES
    before enqueuing and the worker deletes it as its first action.  This prevents
    the beat from adding duplicate tasks for files that already have a live task
    in flight.
    """
    return f"{OnyxRedisLocks.USER_FILE_QUEUED_PREFIX}:{user_file_id}"


def user_file_project_sync_lock_key(user_file_id: str | UUID) -> str:
    return f"{OnyxRedisLocks.USER_FILE_PROJECT_SYNC_LOCK_PREFIX}:{user_file_id}"


def _user_file_project_sync_queued_key(user_file_id: str | UUID) -> str:
    return f"{OnyxRedisLocks.USER_FILE_PROJECT_SYNC_QUEUED_PREFIX}:{user_file_id}"


def _user_file_delete_lock_key(user_file_id: str | UUID) -> str:
    return f"{OnyxRedisLocks.USER_FILE_DELETE_LOCK_PREFIX}:{user_file_id}"


def _user_file_delete_queued_key(user_file_id: str | UUID) -> str:
    """Key that exists while a delete_single_user_file task is sitting in the queue.

    The beat generator sets this with a TTL equal to CELERY_USER_FILE_DELETE_TASK_EXPIRES
    before enqueuing and the worker deletes it as its first action.  This prevents
    the beat from adding duplicate tasks for files that already have a live task
    in flight.
    """
    return f"{OnyxRedisLocks.USER_FILE_DELETE_QUEUED_PREFIX}:{user_file_id}"


def get_user_file_project_sync_queue_depth(celery_app: Celery) -> int:
    redis_celery = celery_get_broker_client(celery_app)
    return celery_get_queue_length(
        OnyxCeleryQueues.USER_FILE_PROJECT_SYNC, redis_celery
    )


def enqueue_user_file_project_sync_task(
    *,
    celery_app: Celery,
    redis_client: TenantRedisClient,
    user_file_id: str | UUID,
    tenant_id: str,
    priority: OnyxCeleryPriority = OnyxCeleryPriority.HIGH,
) -> bool:
    """Enqueue unless this file already has queued or actively running work."""
    queued_key = _user_file_project_sync_queued_key(user_file_id)
    active_lock_key = user_file_project_sync_lock_key(user_file_id)

    if redis_client.exists(active_lock_key):
        return False

    # NX+EX gives us atomic dedupe and a self-healing TTL.
    queued_guard_set = redis_client.set(
        queued_key,
        1,
        nx=True,
        ex=CELERY_USER_FILE_PROJECT_SYNC_TASK_EXPIRES,
    )
    if not queued_guard_set:
        return False

    # Close the race where a previously queued worker acquired its lock after
    # the first active-lock check but before this producer installed the guard.
    if redis_client.exists(active_lock_key):
        redis_client.delete(queued_key)
        return False

    try:
        celery_app.send_task(
            OnyxCeleryTask.PROCESS_SINGLE_USER_FILE_PROJECT_SYNC,
            kwargs={"user_file_id": str(user_file_id), "tenant_id": tenant_id},
            queue=OnyxCeleryQueues.USER_FILE_PROJECT_SYNC,
            priority=priority,
            expires=CELERY_USER_FILE_PROJECT_SYNC_TASK_EXPIRES,
        )
    except Exception:
        # Roll back the queued guard if task publish fails.
        redis_client.delete(queued_key)
        raise

    return True


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_USER_FILE_PROCESSING,
    soft_time_limit=300,
    bind=True,
    ignore_result=True,
)
def check_user_file_processing(self: Task, *, tenant_id: str) -> None:
    """Scan for user files with PROCESSING status and enqueue per-file tasks.

    Three mechanisms prevent queue runaway:

    1. **Queue depth backpressure** – if the broker queue already has more than
       USER_FILE_PROCESSING_MAX_QUEUE_DEPTH items we skip this beat cycle
       entirely.  Workers are clearly behind; adding more tasks would only make
       the backlog worse.

    2. **Per-file queued guard** – before enqueuing a task we set a short-lived
       Redis key (TTL = CELERY_USER_FILE_PROCESSING_TASK_EXPIRES).  If that key
       already exists the file already has a live task in the queue, so we skip
       it.  The worker deletes the key the moment it picks up the task so the
       next beat cycle can re-enqueue if the file is still PROCESSING.

    3. **Task expiry** – every enqueued task carries an `expires` value equal to
       CELERY_USER_FILE_PROCESSING_TASK_EXPIRES.  If a task is still sitting in
       the queue after that deadline, Celery discards it without touching the DB.
       This is a belt-and-suspenders defence: even if the guard key is lost (e.g.
       Redis restart), stale tasks evict themselves rather than piling up forever.
    """
    task_logger.info("check_user_file_processing - Starting")

    redis_client = get_redis_client(tenant_id=tenant_id)
    lock: RedisLock = redis_client.lock(
        OnyxRedisLocks.USER_FILE_PROCESSING_BEAT_LOCK,
        timeout=CELERY_GENERIC_BEAT_LOCK_TIMEOUT,
    )

    # Do not overlap generator runs
    if not lock.acquire(blocking=False):
        return None

    enqueued = 0
    skipped_guard = 0
    try:
        # --- Protection 1: queue depth backpressure ---
        r_celery = celery_get_broker_client(self.app)
        queue_len = celery_get_queue_length(
            OnyxCeleryQueues.USER_FILE_PROCESSING, r_celery
        )
        if queue_len > USER_FILE_PROCESSING_MAX_QUEUE_DEPTH:
            task_logger.warning(
                f"check_user_file_processing - Queue depth {queue_len} exceeds "
                f"{USER_FILE_PROCESSING_MAX_QUEUE_DEPTH}, skipping enqueue for "
                f"tenant={tenant_id}"
            )
            return None

        with get_session_with_current_tenant() as db_session:
            user_file_ids = (
                db_session.execute(
                    select(UserFile.id).where(
                        UserFile.status == UserFileStatus.PROCESSING
                    )
                )
                .scalars()
                .all()
            )

            for user_file_id in user_file_ids:
                # --- Protection 2: per-file queued guard ---
                queued_key = _user_file_queued_key(user_file_id)
                guard_set = redis_client.set(
                    queued_key,
                    1,
                    ex=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
                    nx=True,
                )
                if not guard_set:
                    skipped_guard += 1
                    continue

                # --- Protection 3: task expiry ---
                # If task submission fails, clear the guard immediately so the
                # next beat cycle can retry enqueuing this file.
                try:
                    self.app.send_task(
                        OnyxCeleryTask.PROCESS_SINGLE_USER_FILE,
                        kwargs={
                            "user_file_id": str(user_file_id),
                            "tenant_id": tenant_id,
                        },
                        queue=OnyxCeleryQueues.USER_FILE_PROCESSING,
                        priority=OnyxCeleryPriority.HIGH,
                        expires=CELERY_USER_FILE_PROCESSING_TASK_EXPIRES,
                    )
                except Exception:
                    redis_client.delete(queued_key)
                    raise
                enqueued += 1

    finally:
        if lock.owned():
            lock.release()

    task_logger.info(
        f"check_user_file_processing - Enqueued {enqueued} skipped_guard={skipped_guard} tasks for tenant={tenant_id}"
    )
    return None


def _process_user_file_without_vector_db(
    user_file_id: str | UUID,
    documents: list[Document],
) -> None:
    """Process a user file when the vector DB is disabled.

    Extracts raw text and computes a token count, stores the plaintext in
    the file store, and marks the file as COMPLETED.  Skips embedding and
    the indexing pipeline entirely.

    Opens its own short DB session only for the final status write, so the
    caller does not need to hold a session open during the text/token work.
    """
    from onyx.llm.factory import get_default_llm, get_llm_tokenizer_encode_func

    user_file_uuid = _as_uuid(user_file_id)

    # Combine section text from all document sections. Tabular sections are
    # file-backed and materialize their staged CSV on demand.
    text_parts: list[str] = []
    for doc in documents:
        for section in doc.sections:
            text = section.materialize_text()
            if text:
                text_parts.append(text)
    combined_text = " ".join(text_parts)

    # Compute token count using the user's default LLM tokenizer
    try:
        llm = get_default_llm()
        encode = get_llm_tokenizer_encode_func(llm)
        token_count: int | None = len(encode(combined_text))
    except Exception:
        task_logger.warning(
            f"_process_user_file_without_vector_db - Failed to compute token count for {user_file_uuid}, falling back to None"
        )
        token_count = None

    # Persist plaintext for fast FileReaderTool loads (no DB session needed)
    store_user_file_plaintext(
        user_file_id=user_file_uuid,
        plaintext_content=combined_text,
    )

    # Short session only for the status write
    with get_session_with_current_tenant() as db_session:
        uf = db_session.get(UserFile, user_file_uuid)
        if uf is None:
            return
        if uf.status != UserFileStatus.DELETING:
            uf.status = UserFileStatus.COMPLETED
        uf.token_count = token_count
        uf.chunk_count = 0  # no chunks without vector DB
        uf.last_project_sync_at = datetime.datetime.now(datetime.timezone.utc)
        db_session.add(uf)
        db_session.commit()

    task_logger.info(
        f"_process_user_file_without_vector_db - Completed id={user_file_uuid} tokens={token_count}"
    )


def _load_user_file_documents(
    user_file_id: str,
    file_id: str,
    file_name: str | None,
    tenant_id: str,
) -> tuple[list[Document], list[str]]:
    return load_user_file_documents(
        user_file_id=str(user_file_id),
        file_id=file_id,
        file_name=file_name,
        tenant_id=tenant_id,
        connector_factory=LocalFileConnector,
    )


def _chunk_user_file_without_indexing(
    user_file_id: str,
    documents: list[Document],
    tenant_id: str,
) -> None:
    """Phase one: write the file's regulatory chunks, index nothing.

    Uses the same chunker the indexing path uses, so a later index pass cannot
    shift chunk boundaries. Deliberately does not touch the embedder, the
    contextual retrieval model, or any document index -- chunking needs none of
    them, which is what lets an operator review chunks on a deployment that has
    no contextualization model configured yet.
    """

    with get_session_with_current_tenant() as db_session:
        user_file = db_session.get(UserFile, _as_uuid(user_file_id))
        if user_file is None or user_file.status == UserFileStatus.DELETING:
            task_logger.info(
                f"_chunk_user_file_without_indexing - user file {user_file_id} is gone "
                "or being deleted; skipping"
            )
            return

        search_settings_list = get_active_search_settings_list(db_session)
        current_search_settings = next(
            (ss for ss in search_settings_list if ss.status.is_current()),
            None,
        )
        if current_search_settings is None:
            raise RuntimeError(
                "_chunk_user_file_without_indexing - No current search settings found "
                f"for tenant={tenant_id}"
            )

        # The embedder is built only for its tokenizer: chunk sizing is measured
        # in embedding tokens, so the boundaries must be computed against it.
        embedding_model = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=current_search_settings,
        )
        contextual_rag_enabled = effective_contextual_rag_enabled(
            current_search_settings
        )
        chunker = RegulatoryIndexingChunker(
            db_session=db_session,
            tokenizer=embedding_model.embedding_model.tokenizer,
            enable_contextual_rag=contextual_rag_enabled,
        )
        # The same Document -> IndexingDocument conversion the pipeline performs,
        # so the chunker sees exactly the sections it would see when indexing.
        indexable_documents = process_image_sections(documents)
        # Writes the chunk rows; the emitted DocAwareChunks belong to the
        # indexing phase and are discarded here.
        chunker.chunk(indexable_documents)

        rows = get_chunks_for_file(db_session, _as_uuid(user_file_id))
        if not rows:
            raise RuntimeError(
                f"_chunk_user_file_without_indexing - produced no chunks for {user_file_id}"
            )

        if user_file.status != UserFileStatus.DELETING:
            user_file.status = UserFileStatus.CHUNKED
        user_file.chunk_count = len(rows)
        user_file.regulatory_chunk_generation_hash = (
            compute_regulatory_chunk_generation_hash(
                embedding_provider=current_search_settings.provider_type,
                embedding_model_name=current_search_settings.model_name,
                enable_contextual_rag=contextual_rag_enabled,
            )
        )
        db_session.add(user_file)
        # The session does not commit on exit, and the chunk rows were written
        # into this same transaction.
        db_session.commit()

    text = " ".join(
        section_text
        for document in documents
        for section in document.sections
        if (section_text := section.materialize_text())
    )
    store_user_file_plaintext(
        user_file_id=_as_uuid(user_file_id), plaintext_content=text
    )

    task_logger.info(
        f"_chunk_user_file_without_indexing - Chunked id={user_file_id} "
        f"chunks={len(rows)}; awaiting an explicit index request"
    )


def index_user_file_impl(*, user_file_id: str, tenant_id: str) -> None:
    """Phase two: project already-written chunk rows into the search index."""

    try:
        with get_session_with_current_tenant() as db_session:
            user_file = db_session.get(UserFile, _as_uuid(user_file_id))
            if user_file is None or user_file.status == UserFileStatus.DELETING:
                task_logger.info(
                    f"index_user_file_impl - user file {user_file_id} is gone or being "
                    "deleted; skipping"
                )
                return

            if app_configs.REGULATORY_BATCH_INDEXING_ENABLED:
                job_id = prepare_regulatory_indexing_job_from_chunks(
                    user_file.id,
                    tenant_id,
                    db_session,
                )
                chunk_count = user_file.chunk_count or 0
            else:
                job_id = None
                chunk_count = project_user_file_to_index(
                    db_session, user_file, tenant_id, include_chunked=True
                )
                if user_file.status != UserFileStatus.DELETING:
                    user_file.status = UserFileStatus.COMPLETED
                db_session.add(user_file)
                db_session.commit()
        if job_id is not None:
            _enqueue_durable_regulatory_indexing(
                job_id=job_id,
                tenant_id=tenant_id,
                user_file_id=user_file_id,
            )
    except Exception:
        task_logger.exception(
            f"index_user_file_impl - Failed to index user file {user_file_id}"
        )
        with get_session_with_current_tenant() as db_session:
            user_file = db_session.get(UserFile, _as_uuid(user_file_id))
            if user_file is not None and user_file.status != UserFileStatus.DELETING:
                user_file.status = UserFileStatus.FAILED
                db_session.add(user_file)
                db_session.commit()
        raise

    task_logger.info(
        f"index_user_file_impl - Indexed id={user_file_id} chunks={chunk_count}"
    )


def _process_user_file_with_indexing(
    user_file_id: str,
    documents: list[Document],
    tenant_id: str,
) -> None:
    """Process a user file through the full indexing pipeline (vector DB path).

    Opens its own DB session for the indexing pipeline.  The caller should
    not hold an open session when calling this function.
    """
    # 20 is the documented default for httpx max_keepalive_connections
    if MANAGED_VESPA:
        httpx_init_vespa_pool(
            20, ssl_cert=VESPA_CLOUD_CERT_PATH, ssl_key=VESPA_CLOUD_KEY_PATH
        )
    else:
        httpx_init_vespa_pool(20)

    with get_session_with_current_tenant() as db_session:
        user_file = db_session.get(UserFile, _as_uuid(user_file_id))
        if user_file is None or user_file.status == UserFileStatus.DELETING:
            task_logger.info(
                f"_process_user_file_with_indexing - user file {user_file_id} is gone or "
                "being deleted; skipping indexing (the delete owns removal)"
            )
            return
        search_settings_list = get_active_search_settings_list(db_session)
        current_search_settings = next(
            (ss for ss in search_settings_list if ss.status.is_current()),
            None,
        )
        if current_search_settings is None:
            raise RuntimeError(
                f"_process_user_file_with_indexing - No current search settings found for tenant={tenant_id}"
            )
        embedding_model = DefaultIndexingEmbedder.from_db_search_settings(
            search_settings=current_search_settings,
        )
        document_indices = get_all_document_indices(
            current_search_settings,
            None,
            httpx_client=HttpxPool.get("vespa"),
        )
        adapter = UserFileIndexingAdapter(
            tenant_id=tenant_id,
            db_session=db_session,
        )
        # User files go through the structure-aware regulatory chunker: chunk
        # rows land in Postgres (source of truth) within this same session's
        # transaction, and the pipeline projects them into Elasticsearch.
        regulatory_chunker = RegulatoryIndexingChunker(
            db_session=db_session,
            tokenizer=embedding_model.embedding_model.tokenizer,
            enable_contextual_rag=effective_contextual_rag_enabled(
                current_search_settings
            ),
        )
        try:
            index_pipeline_result = run_indexing_pipeline(
                embedder=embedding_model,
                document_indices=document_indices,
                ignore_time_skip=True,
                db_session=db_session,
                tenant_id=tenant_id,
                document_batch=documents,
                request_id=None,
                adapter=adapter,
                chunker=regulatory_chunker,
                search_settings_override=current_search_settings,
            )
        except UserFileDeletingSkip:
            # File began deleting mid-pipeline — the delete owns removal; skip cleanly
            # rather than fail. (The early-out above catches the already-deleting case.)
            task_logger.info(
                f"_process_user_file_with_indexing - user file {user_file_id} began "
                "deleting mid-indexing; skipping"
            )
            return

    task_logger.info(
        f"_process_user_file_with_indexing - Indexing pipeline completed ={index_pipeline_result}"
    )

    if (
        index_pipeline_result.failures
        or index_pipeline_result.total_docs != len(documents)
        or index_pipeline_result.total_chunks == 0
    ):
        task_logger.error(
            f"_process_user_file_with_indexing - Indexing pipeline failed id={user_file_id}"
        )
        with get_session_with_current_tenant() as db_session:
            uf = db_session.get(UserFile, _as_uuid(user_file_id))
            if uf is not None and uf.status != UserFileStatus.DELETING:
                uf.status = UserFileStatus.FAILED
                db_session.add(uf)
                db_session.commit()
        raise RuntimeError(f"Indexing pipeline failed for user file {user_file_id}")

    _dual_write_new_file_to_secondary(user_file_id, tenant_id)


def _enqueue_durable_regulatory_indexing(
    *,
    job_id: UUID,
    tenant_id: str,
    user_file_id: str,
) -> None:
    """Emit a broker hint for a DB-backed durable regulatory job."""

    from onyx.background.celery.apps.client import celery_app
    from onyx.background.celery.tasks.regulatory_indexing.tasks import (
        enqueue_regulatory_indexing_step,
    )
    from onyx.regulatory.indexing_jobs.orchestrator import OrchestrationDeliveryKind

    with get_session_with_current_tenant() as db_session:
        job = get_regulatory_indexing_job(db_session, job_id)
        if job is None:
            raise RuntimeError("prepared regulatory indexing job disappeared")
        expected_generation = job.lease_generation

    try:
        enqueue_regulatory_indexing_step(
            celery_app,
            job_id=job_id,
            expected_generation=expected_generation,
            tenant_id=tenant_id,
            delivery_kind=OrchestrationDeliveryKind.NORMAL,
        )
    except Exception as error:
        # PostgreSQL is the durable scheduler. The recovery scan will redeliver
        # the committed job if this short-lived broker hint cannot be published.
        task_logger.warning(
            "durable regulatory indexing enqueue failed id=%s error_type=%s",
            user_file_id,
            error.__class__.__name__,
        )


def _index_user_file_to_secondary(
    user_file_id: str | UUID,
    secondary: SearchSettings,
    tenant_id: str,
) -> bool:
    """Project canonical PostgreSQL chunks into one FUTURE search setting."""
    user_file_uuid = _as_uuid(user_file_id)
    user_file_id_str = str(user_file_uuid)
    with get_session_with_current_tenant() as db_session:
        # Callers resolve `secondary` in a separate, already-closed session, so it arrives
        # detached. Re-bind before from_db_search_settings reads its cloud_provider-backed
        # properties (api_key/api_url/api_version/deployment_name), which would otherwise
        # lazy-load and raise DetachedInstanceError.
        bound_secondary = db_session.get(SearchSettings, secondary.id)
        if bound_secondary is None:
            raise RuntimeError(
                f"secondary search settings gone for user file {user_file_id}"
            )
        # Don't resurrect a file already being deleted into the target index — the delete
        # owns removing it, and the port orphan sweep can't remove these non-port chunks.
        # (the adapter's DELETING skip re-checks under the row lock to close the race.)
        user_file = lock_completed_user_file_for_projection(db_session, user_file_uuid)
        if user_file is None:
            task_logger.info(
                f"_index_user_file_to_secondary - user file {user_file_id} is gone or "
                "not completed; skipping secondary write"
            )
            return False
        rows = get_chunks_for_file(db_session, user_file_uuid)
        if not rows:
            raise RuntimeError(
                f"No regulatory chunks found for secondary projection {user_file_id}"
            )
        project_ids = fetch_user_project_ids_for_user_files(
            [user_file_id_str], db_session
        )
        persona_ids = fetch_persona_ids_for_user_files([user_file_id_str], db_session)
        document_set_names = fetch_document_set_names_for_user_files(
            [user_file_id_str], db_session
        )
        user_file_access = get_access_for_user_files([user_file_id_str], db_session)
        new_chunk_count = len(rows)
        indexing_metadata = IndexingMetadata(
            doc_id_to_chunk_cnt_diff={
                user_file_id_str: IndexingMetadata.ChunkCounts(
                    old_chunk_cnt=max(user_file.chunk_count or 0, new_chunk_count),
                    new_chunk_cnt=new_chunk_count,
                )
            }
        )
        _project_rows_to_search_settings(
            user_file=user_file,
            rows=rows,
            search_settings=bound_secondary,
            tenant_id=tenant_id,
            project_ids=project_ids,
            persona_ids=persona_ids,
            document_set_names=document_set_names,
            user_file_access=user_file_access,
            indexing_metadata=indexing_metadata,
        )
        return True


def _index_legacy_user_file_to_secondary(
    user_file_id: str,
    documents: list[Document],
    secondary: SearchSettings,
    tenant_id: str,
) -> None:
    """Preserve the ordinary user-file fallback for legacy non-regulatory rows."""

    with get_session_with_current_tenant() as db_session:
        bound_secondary = db_session.get(SearchSettings, secondary.id)
        user_file = db_session.get(UserFile, _as_uuid(user_file_id))
        if bound_secondary is None:
            raise RuntimeError(
                f"secondary search settings gone for user file {user_file_id}"
            )
        if user_file is None or user_file.status == UserFileStatus.DELETING:
            return
        embedder = DefaultIndexingEmbedder.from_db_search_settings(bound_secondary)
        document_indices = get_all_document_indices(
            bound_secondary,
            None,
            httpx_client=HttpxPool.get("vespa"),
        )
        result = run_indexing_pipeline(
            embedder=embedder,
            document_indices=document_indices,
            ignore_time_skip=True,
            index_to_secondary=True,
            db_session=db_session,
            tenant_id=tenant_id,
            document_batch=documents,
            request_id=None,
            adapter=UserFileIndexingAdapter(
                tenant_id=tenant_id,
                db_session=db_session,
            ),
            search_settings_override=bound_secondary,
        )
    if (
        result.failures
        or result.total_docs != len(documents)
        or result.total_chunks == 0
    ):
        raise RuntimeError(
            f"legacy secondary index write incomplete for user file {user_file_id}: "
            f"{result}"
        )


def _dual_write_new_file_to_secondary(user_file_id: str, tenant_id: str) -> None:
    """During a reindex-port, also index a freshly-processed file into the secondary target so
    it isn't missing at swap. Target resolved fresh (catches a file crossing kickoff). Isolated:
    a failure only flags the file for the reconciler, never touching live status."""
    with get_session_with_current_tenant() as db_session:
        secondary = active_secondary_port_target(db_session)
    if secondary is None:
        return
    try:
        _index_user_file_to_secondary(user_file_id, secondary, tenant_id)
    except Exception as e:
        task_logger.exception(
            f"_dual_write_new_file_to_secondary - failed id={user_file_id}; "
            f"flagging for reconcile - {e.__class__.__name__}"
        )
        with get_session_with_current_tenant() as db_session:
            mark_user_file_reconcile_pending(db_session, _as_uuid(user_file_id))


def _supply_user_file_to_secondary(user_file_id: str, tenant_id: str) -> bool:
    """Reconcile a missing FUTURE document from canonical PostgreSQL chunks."""
    with get_session_with_current_tenant() as db_session:
        secondary = active_secondary_port_target(db_session)
        user_file = db_session.get(UserFile, _as_uuid(user_file_id))
        file_id = user_file.file_id if user_file is not None else None
        file_name = user_file.name if user_file is not None else None
        has_regulatory_chunks = bool(
            get_chunks_for_file(db_session, _as_uuid(user_file_id))
        )
    if secondary is None or user_file is None:
        return False

    # Fully isolated: any failure keeps the flag and never propagates into the
    # sync task. Regulatory files use PostgreSQL rows; only a legacy file with
    # no regulatory rows falls back to its uploaded blob.
    staged_csv_ids: list[str] = []
    try:
        if has_regulatory_chunks:
            return _index_user_file_to_secondary(user_file_id, secondary, tenant_id)
        else:
            if file_id is None:
                return False
            documents, staged_csv_ids = _load_user_file_documents(
                user_file_id,
                file_id,
                file_name,
                tenant_id,
            )
            _index_legacy_user_file_to_secondary(
                user_file_id,
                documents,
                secondary,
                tenant_id,
            )
        return True
    except Exception as e:
        task_logger.exception(
            f"_supply_user_file_to_secondary - failed id={user_file_id} "
            f"- {e.__class__.__name__}"
        )
        return False
    finally:
        delete_files_best_effort(
            staged_csv_ids,
            context=f"legacy user-file secondary supply cleanup uf={user_file_id}",
        )


def _sync_metadata_and_reconcile_secondary(
    retry_indices: list[RetryDocumentIndex],
    update_request: MetadataUpdateRequest,
    user_file_id: str,
    tenant_id: str,
    force_content_reconcile: bool = False,
) -> tuple[bool, bool]:
    """Apply the metadata update to every index; if the secondary is still porting and lacks
    the doc, supply its content instead. Returns (secondary consistent, canonical
    regulatory projection completed by this call)."""
    secondary_missing = False
    for retry_index in retry_indices:
        try:
            retry_index.update([update_request])
        except SecondaryIndexDocumentMissingError:
            task_logger.debug(
                f"user_file={user_file_id} missing from a still-porting index; "
                "supplying content."
            )
            secondary_missing = True
    if not secondary_missing and not force_content_reconcile:
        return True, False

    with get_session_with_current_tenant() as db_session:
        canonical_regulatory_file = has_regulatory_chunks_for_file(
            db_session, _as_uuid(user_file_id)
        )
    supplied = _supply_user_file_to_secondary(user_file_id, tenant_id)
    return supplied, supplied and canonical_regulatory_file


def process_user_file_impl(
    *, user_file_id: str, tenant_id: str, redis_locking: bool
) -> None:
    """Core implementation for processing a single user file.

    When redis_locking=True, acquires a per-file Redis lock and clears the
    queued-key guard (Celery path).  When redis_locking=False, skips all Redis
    operations (BackgroundTask path).
    """
    task_logger.info(f"process_user_file_impl - Starting id={user_file_id}")
    start = time.monotonic()

    file_lock: RedisLock | None = None
    if redis_locking:
        redis_client = get_redis_client(tenant_id=tenant_id)
        redis_client.delete(_user_file_queued_key(user_file_id))
        file_lock = redis_client.lock(
            _user_file_lock_key(user_file_id),
            timeout=CELERY_USER_FILE_PROCESSING_LOCK_TIMEOUT,
        )
        if file_lock is not None and not file_lock.acquire(blocking=False):
            task_logger.info(
                f"process_user_file_impl - Lock held, skipping user_file_id={user_file_id}"
            )
            return

    documents: list[Document] = []
    try:
        # Short read session: fetch what we need from UserFile then release the
        # connection before the slow file-I/O and indexing pipeline phases.
        with get_session_with_current_tenant() as db_session:
            uf = db_session.get(UserFile, _as_uuid(user_file_id))
            if not uf:
                task_logger.warning(
                    f"process_user_file_impl - UserFile not found id={user_file_id}"
                )
                return

            if uf.status not in (
                UserFileStatus.PROCESSING,
                UserFileStatus.INDEXING,
            ):
                task_logger.info(
                    f"process_user_file_impl - Skipping id={user_file_id} status={uf.status}"
                )
                return

            file_id = uf.file_id
            file_name = uf.name
        # DB connection returned to pool here; file I/O and indexing run without it.

        try:
            documents, staged_csv_ids = _load_user_file_documents(
                user_file_id, file_id, file_name, tenant_id
            )
            try:
                if DISABLE_VECTOR_DB:
                    _process_user_file_without_vector_db(
                        user_file_id=user_file_id,
                        documents=documents,
                    )
                elif DEFER_USER_FILE_INDEXING:
                    _chunk_user_file_without_indexing(
                        user_file_id=user_file_id,
                        documents=documents,
                        tenant_id=tenant_id,
                    )
                else:
                    _process_user_file_with_indexing(
                        user_file_id=user_file_id,
                        documents=documents,
                        tenant_id=tenant_id,
                    )
            finally:
                delete_files_best_effort(
                    staged_csv_ids,
                    context=f"user-file tabular staging cleanup uf={user_file_id}",
                )
        except Exception as e:
            task_logger.exception(
                f"process_user_file_impl - Error processing file id={user_file_id} - {e.__class__.__name__}"
            )
            with get_session_with_current_tenant() as db_session:
                current_user_file = db_session.get(UserFile, _as_uuid(user_file_id))
                if (
                    current_user_file
                    and current_user_file.status != UserFileStatus.DELETING
                ):
                    current_user_file.status = UserFileStatus.FAILED
                    db_session.add(current_user_file)
                    db_session.commit()
            return

        elapsed = time.monotonic() - start
        task_logger.info(
            f"process_user_file_impl - Finished id={user_file_id} docs={len(documents)} elapsed={elapsed:.2f}s"
        )
    except Exception as e:
        with get_session_with_current_tenant() as db_session:
            uf = db_session.get(UserFile, _as_uuid(user_file_id))
            if uf:
                if uf.status != UserFileStatus.DELETING:
                    uf.status = UserFileStatus.FAILED
                db_session.add(uf)
                db_session.commit()

        task_logger.exception(
            f"process_user_file_impl - Error processing file id={user_file_id} - {e.__class__.__name__}"
        )
        raise
    finally:
        if file_lock is not None and file_lock.owned():
            file_lock.release()


@shared_task(
    name=OnyxCeleryTask.PROCESS_SINGLE_USER_FILE,
    bind=True,
    ignore_result=True,
)
def process_single_user_file(
    self: Task,  # noqa: ARG001
    *,
    user_file_id: str,
    tenant_id: str,
) -> None:
    process_user_file_impl(
        user_file_id=user_file_id, tenant_id=tenant_id, redis_locking=True
    )


@shared_task(
    name=OnyxCeleryTask.INDEX_SINGLE_USER_FILE,
    bind=True,
    ignore_result=True,
)
def index_single_user_file(
    self: Task,  # noqa: ARG001
    *,
    user_file_id: str,
    tenant_id: str,
) -> None:
    index_user_file_impl(user_file_id=user_file_id, tenant_id=tenant_id)


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_USER_FILE_DELETE,
    soft_time_limit=300,
    bind=True,
    ignore_result=True,
)
def check_for_user_file_delete(self: Task, *, tenant_id: str) -> None:
    """Scan for user files with DELETING status and enqueue per-file tasks.

    Three mechanisms prevent queue runaway (mirrors check_user_file_processing):

    1. **Queue depth backpressure** – if the broker queue already has more than
       USER_FILE_DELETE_MAX_QUEUE_DEPTH items we skip this beat cycle entirely.

    2. **Per-file queued guard** – before enqueuing a task we set a short-lived
       Redis key (TTL = CELERY_USER_FILE_DELETE_TASK_EXPIRES).  If that key
       already exists the file already has a live task in the queue, so we skip
       it.  The worker deletes the key the moment it picks up the task so the
       next beat cycle can re-enqueue if the file is still DELETING.

    3. **Task expiry** – every enqueued task carries an `expires` value equal to
       CELERY_USER_FILE_DELETE_TASK_EXPIRES.  If a task is still sitting in
       the queue after that deadline, Celery discards it without touching the DB.
    """
    task_logger.info("check_for_user_file_delete - Starting")
    redis_client = get_redis_client(tenant_id=tenant_id)
    lock: RedisLock = redis_client.lock(
        OnyxRedisLocks.USER_FILE_DELETE_BEAT_LOCK,
        timeout=CELERY_GENERIC_BEAT_LOCK_TIMEOUT,
    )
    if not lock.acquire(blocking=False):
        return None

    enqueued = 0
    skipped_guard = 0
    try:
        # --- Protection 1: queue depth backpressure ---
        # NOTE: must use the broker's Redis client (not redis_client) because
        # Celery queues live on a separate Redis DB with CELERY_SEPARATOR keys.
        r_celery = celery_get_broker_client(self.app)
        queue_len = celery_get_queue_length(OnyxCeleryQueues.USER_FILE_DELETE, r_celery)
        if queue_len > USER_FILE_DELETE_MAX_QUEUE_DEPTH:
            task_logger.warning(
                f"check_for_user_file_delete - Queue depth {queue_len} exceeds "
                f"{USER_FILE_DELETE_MAX_QUEUE_DEPTH}, skipping enqueue for "
                f"tenant={tenant_id}"
            )
            return None

        with get_session_with_current_tenant() as db_session:
            user_file_ids = (
                db_session.execute(
                    select(UserFile.id).where(
                        UserFile.status == UserFileStatus.DELETING
                    )
                )
                .scalars()
                .all()
            )
            for user_file_id in user_file_ids:
                # --- Protection 2: per-file queued guard ---
                queued_key = _user_file_delete_queued_key(user_file_id)
                guard_set = redis_client.set(
                    queued_key,
                    1,
                    ex=CELERY_USER_FILE_DELETE_TASK_EXPIRES,
                    nx=True,
                )
                if not guard_set:
                    skipped_guard += 1
                    continue

                # --- Protection 3: task expiry ---
                try:
                    self.app.send_task(
                        OnyxCeleryTask.DELETE_SINGLE_USER_FILE,
                        kwargs={
                            "user_file_id": str(user_file_id),
                            "tenant_id": tenant_id,
                        },
                        queue=OnyxCeleryQueues.USER_FILE_DELETE,
                        priority=OnyxCeleryPriority.HIGH,
                        expires=CELERY_USER_FILE_DELETE_TASK_EXPIRES,
                    )
                except Exception:
                    redis_client.delete(queued_key)
                    raise
                enqueued += 1
    finally:
        if lock.owned():
            lock.release()

    task_logger.info(
        f"check_for_user_file_delete - Enqueued {enqueued} tasks, skipped_guard={skipped_guard} for tenant={tenant_id}"
    )
    return None


def delete_user_file_impl(
    *, user_file_id: str, tenant_id: str, redis_locking: bool
) -> None:
    """Core implementation for deleting a single user file.

    When redis_locking=True, acquires a per-file Redis lock (Celery path).
    When redis_locking=False, skips Redis operations (BackgroundTask path).
    """
    task_logger.info(f"delete_user_file_impl - Starting id={user_file_id}")

    file_lock: RedisLock | None = None
    if redis_locking:
        redis_client = get_redis_client(tenant_id=tenant_id)
        # Clear the queued guard so the beat can re-enqueue if deletion fails
        # and the file remains in DELETING status.
        redis_client.delete(_user_file_delete_queued_key(user_file_id))
        file_lock = redis_client.lock(
            _user_file_delete_lock_key(user_file_id),
            timeout=CELERY_GENERIC_BEAT_LOCK_TIMEOUT,
        )
        if file_lock is not None and not file_lock.acquire(blocking=False):
            task_logger.info(
                f"delete_user_file_impl - Lock held, skipping user_file_id={user_file_id}"
            )
            return

    try:
        with get_session_with_current_tenant() as db_session:
            deletion_plan = request_user_file_deletion_cleanup(
                db_session,
                user_file_id=_as_uuid(user_file_id),
                now=datetime.datetime.now(datetime.timezone.utc),
            )
        if not deletion_plan.ready_to_delete:
            from onyx.background.celery.apps.client import celery_app
            from onyx.background.celery.tasks.regulatory_indexing.tasks import (
                enqueue_regulatory_indexing_step,
            )
            from onyx.regulatory.indexing_jobs.orchestrator import (
                OrchestrationDeliveryKind,
            )

            for delivery in deletion_plan.deliveries:
                try:
                    enqueue_regulatory_indexing_step(
                        celery_app,
                        job_id=delivery.job_id,
                        expected_generation=delivery.expected_generation,
                        tenant_id=tenant_id,
                        delivery_kind=OrchestrationDeliveryKind.NORMAL,
                    )
                except Exception as error:
                    task_logger.warning(
                        "durable regulatory cancellation enqueue failed "
                        "user_file_id=%s job_id=%s error_type=%s",
                        user_file_id,
                        delivery.job_id,
                        error.__class__.__name__,
                    )
            task_logger.info(
                "delete_user_file_impl - Waiting for durable cancellation id=%s",
                user_file_id,
            )
            return

        skip_vespa = DISABLE_VECTOR_DB
        retry_document_indices: list[RetryDocumentIndex] = []
        chunk_count_from_db: int | None = None
        file_id: str = ""

        if not skip_vespa:
            if MANAGED_VESPA:
                httpx_init_vespa_pool(
                    20, ssl_cert=VESPA_CLOUD_CERT_PATH, ssl_key=VESPA_CLOUD_KEY_PATH
                )
            else:
                httpx_init_vespa_pool(20)

        # Phase 1: short read session — extract everything needed for slow I/O
        with get_session_with_current_tenant() as db_session:
            user_file = db_session.get(UserFile, _as_uuid(user_file_id))
            if not user_file:
                task_logger.info(
                    f"delete_user_file_impl - User file not found id={user_file_id}"
                )
                return

            file_id = user_file.file_id
            chunk_count_from_db = user_file.chunk_count

            if not skip_vespa:
                active_search_settings = get_active_search_settings(db_session)
                document_indices = get_all_document_indices(
                    search_settings=active_search_settings.primary,
                    secondary_search_settings=active_search_settings.secondary,
                    httpx_client=HttpxPool.get("vespa"),
                )
                retry_document_indices = [
                    RetryDocumentIndex(document_index)
                    for document_index in document_indices
                ]

                # Record the deletion before the index delete (below) so a racing port's
                # sweep removes any chunk its create-only copy resurrects. No-op when no
                # port targets this file.
                if user_file.user_id is not None:
                    recorded = record_port_orphan_candidates_for_user_file(
                        db_session,
                        port_user_id=user_file.user_id,
                        document_id=str(user_file.id),
                        primary=active_search_settings.primary,
                        secondary=active_search_settings.secondary,
                    )
                    if recorded:
                        db_session.commit()

        # Phase 2: vector DB deletes + file store deletes (no DB session held).
        # Pass the DB chunk count when known; otherwise None, which each document
        # index resolves itself (Vespa fans out to find chunks, Elasticsearch deletes
        # by document id). This keeps the path backend-agnostic.
        if not skip_vespa:
            chunk_count: int | None = (
                chunk_count_from_db
                if chunk_count_from_db is not None and chunk_count_from_db > 0
                else None
            )
            for retry_document_index in retry_document_indices:
                retry_document_index.delete(
                    user_file_id,
                    chunk_count=chunk_count,
                )

        file_store = get_default_file_store()
        try:
            file_store.delete_file(file_id)
            file_store.delete_file(
                user_file_id_to_plaintext_file_name(_as_uuid(user_file_id))
            )
        except Exception as e:
            task_logger.exception(
                f"delete_user_file_impl - Error deleting file id={user_file_id} - {e.__class__.__name__}"
            )

        # Phase 3: short write session — remove the DB record
        with get_session_with_current_tenant() as db_session:
            user_file = db_session.get(UserFile, _as_uuid(user_file_id))
            if user_file is not None:
                db_session.delete(user_file)
                db_session.commit()
        task_logger.info(f"delete_user_file_impl - Completed id={user_file_id}")
    except Exception as e:
        task_logger.exception(
            f"delete_user_file_impl - Error processing file id={user_file_id} - {e.__class__.__name__}"
        )
        raise
    finally:
        if file_lock is not None and file_lock.owned():
            file_lock.release()


@shared_task(
    name=OnyxCeleryTask.DELETE_SINGLE_USER_FILE,
    bind=True,
    ignore_result=True,
)
def process_single_user_file_delete(
    self: Task,  # noqa: ARG001
    *,
    user_file_id: str,
    tenant_id: str,
) -> None:
    delete_user_file_impl(
        user_file_id=user_file_id, tenant_id=tenant_id, redis_locking=True
    )


@shared_task(
    name=OnyxCeleryTask.CHECK_FOR_USER_FILE_PROJECT_SYNC,
    soft_time_limit=300,
    bind=True,
    ignore_result=True,
)
def check_for_user_file_project_sync(self: Task, *, tenant_id: str) -> None:
    """Scan for user files needing project sync and enqueue per-file tasks."""
    task_logger.info("Starting")

    redis_client = get_redis_client(tenant_id=tenant_id)
    lock: RedisLock = redis_client.lock(
        OnyxRedisLocks.USER_FILE_PROJECT_SYNC_BEAT_LOCK,
        timeout=CELERY_GENERIC_BEAT_LOCK_TIMEOUT,
    )

    if not lock.acquire(blocking=False):
        return None

    enqueued = 0
    skipped_guard = 0
    try:
        queue_depth = get_user_file_project_sync_queue_depth(self.app)
        queue_limit = min(
            USER_FILE_PROJECT_SYNC_MAX_QUEUE_DEPTH,
            _USER_FILE_PROJECT_SYNC_QUEUE_TARGET_DEPTH,
        )
        if queue_depth >= queue_limit:
            task_logger.warning(
                f"Queue depth {queue_depth} reached "
                f"{queue_limit}, skipping enqueue for tenant={tenant_id}"
            )
            return None
        available_queue_slots = queue_limit - queue_depth

        with get_session_with_current_tenant() as db_session:
            user_file_ids = (
                db_session.execute(
                    select(UserFile.id).where(
                        sa.or_(
                            sa.and_(
                                UserFile.status == UserFileStatus.COMPLETED,
                                sa.or_(
                                    UserFile.needs_project_sync.is_(True),
                                    UserFile.needs_persona_sync.is_(True),
                                    UserFile.needs_document_set_sync.is_(True),
                                    # re-enqueue un-reconciled files so the reconciler retries
                                    UserFile.secondary_reconcile_pending.is_(True),
                                ),
                            ),
                            sa.and_(
                                UserFile.status == UserFileStatus.FAILED,
                                UserFile.needs_document_set_sync.is_(True),
                            ),
                        )
                    )
                )
                .scalars()
                .all()
            )

            for user_file_id in user_file_ids:
                if enqueued >= available_queue_slots:
                    break
                if not enqueue_user_file_project_sync_task(
                    celery_app=self.app,
                    redis_client=redis_client,
                    user_file_id=user_file_id,
                    tenant_id=tenant_id,
                    priority=OnyxCeleryPriority.HIGH,
                ):
                    skipped_guard += 1
                    continue
                enqueued += 1
    finally:
        if lock.owned():
            lock.release()

    task_logger.info(
        f"Enqueued {enqueued} Skipped guard {skipped_guard} tasks for tenant={tenant_id}"
    )
    return None


def project_sync_user_file_impl(
    *, user_file_id: str, tenant_id: str, redis_locking: bool
) -> None:
    """Sync mutable metadata or remove residual indexed chunks for a failed file.

    When redis_locking=True, acquires and renews a per-file Redis lock, then
    clears the queued-key guard (Celery path). When redis_locking=False, skips
    Redis operations (BackgroundTask path).
    """
    task_logger.info(f"project_sync_user_file_impl - Starting id={user_file_id}")

    file_lock: RedisLock | None = None
    lock_heartbeat: _RedisLockHeartbeat | None = None
    if redis_locking:
        redis_client = get_redis_client(tenant_id=tenant_id)
        file_lock = redis_client.lock(
            user_file_project_sync_lock_key(user_file_id),
            timeout=CELERY_USER_FILE_PROJECT_SYNC_LOCK_TIMEOUT,
            thread_local=False,
        )
        if file_lock is not None and not file_lock.acquire(blocking=False):
            task_logger.info(
                f"project_sync_user_file_impl - Lock held, skipping user_file_id={user_file_id}"
            )
            return
        redis_client.delete(_user_file_project_sync_queued_key(user_file_id))
        lock_heartbeat = _RedisLockHeartbeat(
            file_lock,
            lock_timeout_seconds=CELERY_USER_FILE_PROJECT_SYNC_LOCK_TIMEOUT,
            operation_id=user_file_id,
        )

    try:
        if lock_heartbeat is not None:
            lock_heartbeat.start()

        # Phase 1: short read session — extract all data needed for Vespa, then
        # release the connection before the network-bound update calls.
        retry_document_indices: list[RetryDocumentIndex] = []
        project_ids: list[int] = []
        persona_ids: list[int] = []
        document_set_names: set[str] = set()
        file_id_str: str = ""
        chunk_count: int | None = None
        access: DocumentAccess | None = None
        force_content_reconcile = False
        reconcile_target_settings_id: int | None = None
        failed_document_set_sync = False
        failed_file_delete_chunk_count: int | None = None
        skip_vespa = DISABLE_VECTOR_DB

        if not skip_vespa:
            if MANAGED_VESPA:
                httpx_init_vespa_pool(
                    20, ssl_cert=VESPA_CLOUD_CERT_PATH, ssl_key=VESPA_CLOUD_KEY_PATH
                )
            else:
                httpx_init_vespa_pool(20)

        with get_session_with_current_tenant() as db_session:
            user_files = fetch_user_files_with_access_relationships(
                [user_file_id],
                db_session,
                eager_load_groups=global_version.is_ee_version(),
            )
            user_file = user_files[0] if user_files else None
            if not user_file:
                task_logger.info(
                    f"project_sync_user_file_impl - User file not found id={user_file_id}"
                )
                return

            completed_metadata_sync = user_file.status == UserFileStatus.COMPLETED and (
                user_file.needs_project_sync
                or user_file.needs_persona_sync
                or user_file.needs_document_set_sync is True
                or user_file.secondary_reconcile_pending
            )
            failed_document_set_sync = (
                user_file.status == UserFileStatus.FAILED
                and user_file.needs_document_set_sync is True
            )
            if not completed_metadata_sync and not failed_document_set_sync:
                task_logger.info(
                    "project_sync_user_file_impl - No pending work for "
                    f"user_file_id={user_file_id}; skipping stale task"
                )
                return

            if not skip_vespa:
                active_search_settings = get_active_search_settings(db_session)
                force_content_reconcile = bool(
                    user_file.secondary_reconcile_pending
                    and active_search_settings.secondary is not None
                )
                if force_content_reconcile:
                    assert active_search_settings.secondary is not None
                    reconcile_target_settings_id = active_search_settings.secondary.id
                # INSTANT-promoted primary still backfilling: defer updates to
                # not-yet-ported files, else the create-only port reinstalls a stale ACL.
                primary_backfill_in_progress = (
                    active_search_settings.primary.port_backfill_source_id is not None
                    and port_backfill_has_pending_work(
                        db_session, active_search_settings.primary.id
                    )
                )
                document_indices = get_all_document_indices(
                    search_settings=active_search_settings.primary,
                    secondary_search_settings=active_search_settings.secondary,
                    httpx_client=HttpxPool.get("vespa"),
                    primary_backfill_in_progress=primary_backfill_in_progress,
                )
                retry_document_indices = [
                    RetryDocumentIndex(document_index)
                    for document_index in document_indices
                ]

                file_id_str = str(user_file.id)
                if failed_document_set_sync:
                    regulatory_chunk_count = get_chunk_counts_for_files(
                        db_session, [user_file.id]
                    ).get(user_file.id, 0)
                    effective_chunk_count = max(
                        user_file.chunk_count or 0, regulatory_chunk_count
                    )
                    failed_file_delete_chunk_count = effective_chunk_count or None
                else:
                    project_ids = [project.id for project in user_file.projects]
                    persona_ids = [p.id for p in user_file.assistants if not p.deleted]
                    document_set_names = {
                        document_set.name
                        for document_set in user_file.document_sets
                        if not document_set.is_deleting
                    }
                    chunk_count = user_file.chunk_count
                    access_map = build_access_for_user_files([user_file])
                    access = access_map.get(file_id_str)
        # DB connection returned to pool here; index update calls run without it.

        if lock_heartbeat is not None:
            lock_heartbeat.ensure_owned()

        # Phase 2: index update calls (no DB session held)
        secondary_consistent = True
        canonical_projection_completed = False
        if not skip_vespa:
            if failed_document_set_sync:
                for retry_document_index in retry_document_indices:
                    retry_document_index.delete(
                        user_file_id,
                        chunk_count=failed_file_delete_chunk_count,
                    )
            else:
                update_request = MetadataUpdateRequest(
                    document_ids=[file_id_str],
                    doc_id_to_chunk_cnt={
                        file_id_str: chunk_count if chunk_count is not None else -1
                    },
                    access=access if access is not None else None,
                    document_sets=document_set_names,
                    project_ids=set(project_ids),
                    persona_ids=set(persona_ids),
                )
                (
                    secondary_consistent,
                    canonical_projection_completed,
                ) = _sync_metadata_and_reconcile_secondary(
                    retry_document_indices,
                    update_request,
                    user_file_id,
                    tenant_id,
                    force_content_reconcile=force_content_reconcile,
                )

        lost_lock_after_canonical_projection = False
        if lock_heartbeat is not None:
            try:
                lock_heartbeat.ensure_owned()
            except RuntimeError:
                # The canonical writer has already completed its exact
                # delete-and-replace operation under a PostgreSQL row lock. A
                # long process pause can still expire the renewable Redis lease.
                # Accept only that completed, target-scoped content operation;
                # never reacquire the Redis lock or clear unrelated sync flags.
                if not (
                    canonical_projection_completed
                    and reconcile_target_settings_id is not None
                ):
                    raise
                lost_lock_after_canonical_projection = True
                task_logger.warning(
                    "Project-sync lock was lost after exact canonical projection; "
                    "accepting content only for user_file_id=%s target=%s",
                    user_file_id,
                    reconcile_target_settings_id,
                )

        task_logger.info(f"project_sync_user_file_impl - User file id={user_file_id}")

        # Phase 3: short write session — mark sync as done
        with get_session_with_current_tenant() as db_session:
            user_file = db_session.get(UserFile, _as_uuid(user_file_id))
            if user_file is not None:
                if failed_document_set_sync:
                    user_file.needs_document_set_sync = False
                    db_session.add(user_file)
                    db_session.commit()
                    return
                active_reconcile_target_id: int | None = None
                if not skip_vespa and reconcile_target_settings_id is not None:
                    active_reconcile_target = active_secondary_port_target(db_session)
                    if active_reconcile_target is not None:
                        active_reconcile_target_id = active_reconcile_target.id
                reconcile_target_changed = (
                    active_reconcile_target_id is not None
                    and active_reconcile_target_id != reconcile_target_settings_id
                )
                if lost_lock_after_canonical_projection:
                    # Another task may now own project/persona synchronization.
                    # Only settle the canonical content bit, and only if the
                    # target generation is still the one we projected.
                    user_file.secondary_reconcile_pending = (
                        not secondary_consistent or reconcile_target_changed
                    ) and user_file.status == UserFileStatus.COMPLETED
                    db_session.add(user_file)
                    db_session.commit()
                    return
                user_file.needs_project_sync = False
                user_file.needs_persona_sync = False
                user_file.needs_document_set_sync = False
                user_file.last_project_sync_at = datetime.datetime.now(
                    datetime.timezone.utc
                )
                # Flag only a portable (COMPLETED) file — a non-portable one is never ported,
                # so its flag would never reconcile (leave it clear instead).
                user_file.secondary_reconcile_pending = (
                    not secondary_consistent or reconcile_target_changed
                ) and user_file.status == UserFileStatus.COMPLETED
                db_session.add(user_file)
                db_session.commit()

    except Exception as e:
        task_logger.exception(
            f"project_sync_user_file_impl - Error syncing project for file id={user_file_id} - {e.__class__.__name__}"
        )
        raise
    finally:
        heartbeat_stopped = (
            lock_heartbeat.stop() if lock_heartbeat is not None else True
        )
        if not heartbeat_stopped:
            task_logger.error(
                "Project-sync lock heartbeat did not stop for user_file_id=%s; "
                "leaving the lease to expire",
                user_file_id,
            )
        elif file_lock is not None:
            try:
                if file_lock.owned():
                    file_lock.release()
            except RedisError:
                task_logger.warning(
                    "Could not release project-sync lock for user_file_id=%s; "
                    "the lease will expire",
                    user_file_id,
                    exc_info=True,
                )


@shared_task(
    name=OnyxCeleryTask.PROCESS_SINGLE_USER_FILE_PROJECT_SYNC,
    bind=True,
    ignore_result=True,
)
def process_single_user_file_project_sync(
    self: Task,  # noqa: ARG001
    *,
    user_file_id: str,
    tenant_id: str,
) -> None:
    project_sync_user_file_impl(
        user_file_id=user_file_id, tenant_id=tenant_id, redis_locking=True
    )
