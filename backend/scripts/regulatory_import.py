"""Import regulatory source files from a separately operated full backend image.

This command deliberately reuses the user-file ingestion and indexing services used by
the application. It does not run migrations and it does not require a Celery worker.
"""

import argparse
import json
import mimetypes
import os
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, TypedDict
from uuid import UUID

from alembic.config import Config
from alembic.script import ScriptDirectory
from fastapi import UploadFile
from sqlalchemy.orm import Session
from starlette.datastructures import Headers

from onyx.background.celery.tasks.user_file_processing.tasks import (
    process_user_file_impl,
)
from onyx.configs.app_configs import (
    DISABLE_VECTOR_DB,
    ENABLE_ELASTICSEARCH_INDEXING_FOR_ONYX,
)
from onyx.db.elasticsearch_migration import get_elasticsearch_retrieval_state
from onyx.db.engine.sql_engine import SqlEngine, get_session_with_current_tenant
from onyx.db.enums import UserFileStatus
from onyx.db.models import Project__UserFile, User, UserFile, UserProject
from onyx.db.projects import check_project_write_access, create_user_files
from onyx.db.regulatory_chunks import get_chunks_for_file
from onyx.db.schema_version import get_database_alembic_heads
from onyx.db.search_settings import get_active_search_settings
from onyx.db.users import get_user_by_email
from onyx.document_index.elasticsearch.client import ElasticsearchIndexClient
from onyx.document_index.elasticsearch.schema import (
    CHUNK_INDEX_FIELD_NAME,
    DOCUMENT_ID_FIELD_NAME,
    REGULATORY_CHUNK_ID_FIELD_NAME,
    TENANT_ID_FIELD_NAME,
    DocumentSchema,
)
from onyx.file_processing.import_capability import ensure_document_import_available
from shared_configs.configs import MULTI_TENANT, POSTGRES_DEFAULT_SCHEMA
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_ELASTICSEARCH_VISIBILITY_ATTEMPTS = 10
_ELASTICSEARCH_VISIBILITY_INTERVAL_S = 0.5
_ELASTICSEARCH_PROJECTION_BATCH_SIZE = 250


class ImportResult(TypedDict):
    path: str
    status: str
    user_file_id: str | None
    chunk_count: int
    detail: str | None


@dataclass(frozen=True)
class ImportTarget:
    """Stable identifiers that remain valid after the preflight session closes."""

    user_id: UUID
    project_id: int


@dataclass(frozen=True)
class ElasticsearchTarget:
    """Active index metadata copied out of the database preflight session."""

    index_name: str
    embedding_dimension: int
    tenant_id: str
    multitenant: bool


@dataclass(frozen=True)
class RegulatoryProjectionIdentity:
    """Small canonical identity needed for post-index visibility checks."""

    regulatory_chunk_id: str
    chunk_index: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Parse and index regulatory files using the full importer runtime "
            "against the same services as production."
        )
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Files or directories")
    parser.add_argument("--user-email", required=True, help="Owner/admin user email")
    project_group = parser.add_mutually_exclusive_group(required=True)
    project_group.add_argument("--project-id", type=int)
    project_group.add_argument("--project-name")
    parser.add_argument(
        "--tenant-id",
        default=POSTGRES_DEFAULT_SCHEMA,
        help="PostgreSQL tenant schema (default: configured public schema)",
    )
    parser.add_argument(
        "--recursive",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recursively include files under directory arguments (default: true)",
    )
    parser.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Allow another active file with the same name in the target directory",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON result manifest path",
    )
    return parser


def _resolve_paths(paths: Sequence[Path], recursive: bool) -> list[Path]:
    resolved_files: set[Path] = set()
    for input_path in paths:
        resolved_path = input_path.expanduser().resolve()
        if resolved_path.is_file():
            resolved_files.add(resolved_path)
            continue
        if not resolved_path.is_dir():
            raise ValueError(f"Import path does not exist: {input_path}")

        candidates = resolved_path.rglob("*") if recursive else resolved_path.glob("*")
        resolved_files.update(path for path in candidates if path.is_file())

    if not resolved_files:
        raise ValueError("No files found in the supplied paths")
    return sorted(resolved_files, key=lambda path: str(path))


def _duplicate_file_names(paths: Sequence[Path]) -> list[str]:
    counts = Counter(path.name for path in paths)
    return sorted(name for name, count in counts.items() if count > 1)


def _ensure_files_readable(paths: Sequence[Path]) -> None:
    """Fail before creating any records if an input cannot be read."""

    for path in paths:
        try:
            with path.open("rb") as file_handle:
                file_handle.read(1)
        except OSError as exc:
            raise ValueError(f"Import path is not readable: {path}: {exc}") from exc


def _get_application_alembic_heads() -> frozenset[str]:
    """Read the migration heads shipped in the exact importer image."""

    config = Config(str(_BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(_BACKEND_ROOT / "alembic"))
    heads = frozenset(ScriptDirectory.from_config(config).get_heads())
    if not heads:
        raise ValueError("Importer image contains no application Alembic heads")
    return heads


def _ensure_database_schema_current(db_session: Session) -> None:
    """Fail closed when code and the selected tenant schema are not identical."""

    application_heads = _get_application_alembic_heads()
    database_heads = get_database_alembic_heads(db_session)
    if application_heads == database_heads:
        return

    expected = ", ".join(sorted(application_heads))
    actual = ", ".join(sorted(database_heads)) or "<none>"
    raise ValueError(
        "Database schema does not match the importer image's Alembic heads "
        f"(image={expected}; database={actual}). Deploy the exact matching image "
        "or migrate production before importing."
    )


def _resolve_elasticsearch_target(
    db_session: Session, *, tenant_id: str
) -> ElasticsearchTarget:
    """Resolve the active, writable Elasticsearch projection without provider calls."""

    if DISABLE_VECTOR_DB:
        raise ValueError("Document indexing is disabled (DISABLE_VECTOR_DB=true)")
    if not ENABLE_ELASTICSEARCH_INDEXING_FOR_ONYX:
        raise ValueError("Elasticsearch indexing is not enabled for this deployment")
    if not get_elasticsearch_retrieval_state(db_session):
        raise ValueError(
            "Elasticsearch is not the active retrieval index for this tenant"
        )

    search_settings = get_active_search_settings(db_session).primary
    embedding_dimension = search_settings.reduced_dimension or search_settings.model_dim
    if embedding_dimension <= 0:
        raise ValueError("Active search settings have an invalid embedding dimension")
    if not search_settings.index_name:
        raise ValueError("Active search settings have no Elasticsearch index name")
    return ElasticsearchTarget(
        index_name=search_settings.index_name,
        embedding_dimension=embedding_dimension,
        tenant_id=tenant_id,
        multitenant=MULTI_TENANT,
    )


def _ensure_elasticsearch_ready(target: ElasticsearchTarget) -> None:
    """Prove cluster reachability and read-only compatibility of the live index."""

    expected_schema = DocumentSchema.get_document_schema(
        target.embedding_dimension,
        target.multitenant,
    )
    with ElasticsearchIndexClient(
        index_name=target.index_name,
        emit_metrics=False,
    ) as client:
        if not client.ping():
            raise ValueError("Elasticsearch readiness probe failed")
        if not client.index_exists():
            raise ValueError(
                f"Active Elasticsearch index does not exist: {target.index_name}"
            )
        if not client.validate_index(expected_schema):
            raise ValueError(
                "Active Elasticsearch index mapping is incompatible with this importer "
                f"image: {target.index_name}"
            )


def _projection_batch_query(
    *,
    document_id: str,
    identities: Sequence[RegulatoryProjectionIdentity],
    target: ElasticsearchTarget,
) -> dict[str, object]:
    identity_clauses = [
        {
            "bool": {
                "filter": [
                    {
                        "term": {
                            REGULATORY_CHUNK_ID_FIELD_NAME: {
                                "value": identity.regulatory_chunk_id
                            }
                        }
                    },
                    {"term": {CHUNK_INDEX_FIELD_NAME: {"value": identity.chunk_index}}},
                ]
            }
        }
        for identity in identities
    ]
    filters: list[dict[str, object]] = [
        {"term": {DOCUMENT_ID_FIELD_NAME: {"value": document_id}}}
    ]
    if target.multitenant:
        filters.append({"term": {TENANT_ID_FIELD_NAME: {"value": target.tenant_id}}})
    return {
        "size": len(identities),
        "_source": False,
        "query": {
            "bool": {
                "filter": filters,
                "should": identity_clauses,
                "minimum_should_match": 1,
            }
        },
    }


def _projection_is_visible(
    client: ElasticsearchIndexClient,
    *,
    document_id: str,
    identities: Sequence[RegulatoryProjectionIdentity],
    target: ElasticsearchTarget,
) -> bool:
    document_filters: list[dict[str, object]] = [
        {"term": {DOCUMENT_ID_FIELD_NAME: {"value": document_id}}}
    ]
    if target.multitenant:
        document_filters.append(
            {"term": {TENANT_ID_FIELD_NAME: {"value": target.tenant_id}}}
        )
    actual_document_count = client.count_by_query(
        {"query": {"bool": {"filter": document_filters}}}
    )
    if actual_document_count != len(identities):
        return False

    for offset in range(0, len(identities), _ELASTICSEARCH_PROJECTION_BATCH_SIZE):
        batch = identities[offset : offset + _ELASTICSEARCH_PROJECTION_BATCH_SIZE]
        matching_ids = client.search_for_document_ids(
            _projection_batch_query(
                document_id=document_id,
                identities=batch,
                target=target,
            )
        )
        if len(matching_ids) != len(batch) or len(set(matching_ids)) != len(batch):
            return False
    return True


def _ensure_projection_visible(
    *,
    document_id: str,
    identities: Sequence[RegulatoryProjectionIdentity],
    target: ElasticsearchTarget,
) -> None:
    """Refresh once, then poll a bounded number of times for the exact projection."""

    if not identities:
        raise ValueError("Cannot verify an empty regulatory projection")

    last_error: Exception | None = None
    with ElasticsearchIndexClient(
        index_name=target.index_name,
        emit_metrics=False,
    ) as client:
        client.refresh_index()
        for attempt in range(_ELASTICSEARCH_VISIBILITY_ATTEMPTS):
            try:
                if _projection_is_visible(
                    client,
                    document_id=document_id,
                    identities=identities,
                    target=target,
                ):
                    return
                last_error = None
            except Exception as exc:
                last_error = exc
            if attempt + 1 < _ELASTICSEARCH_VISIBILITY_ATTEMPTS:
                time.sleep(_ELASTICSEARCH_VISIBILITY_INTERVAL_S)

    suffix = (
        f"; last Elasticsearch error={type(last_error).__name__}: {last_error}"
        if last_error is not None
        else ""
    )
    raise ValueError(
        "Imported regulatory projection did not become exactly visible in the "
        f"active Elasticsearch index after {_ELASTICSEARCH_VISIBILITY_ATTEMPTS} checks "
        f"(document_id={document_id}, expected_chunks={len(identities)}){suffix}"
    )


def _resolve_import_target(
    db_session: Session,
    *,
    user_email: str,
    project_id: int | None,
    project_name: str | None,
) -> ImportTarget:
    user = get_user_by_email(user_email, db_session)
    if user is None:
        raise ValueError(f"User not found: {user_email}")

    project_query = db_session.query(UserProject)
    if project_id is not None:
        project = project_query.filter(UserProject.id == project_id).one_or_none()
    else:
        project = project_query.filter(
            UserProject.user_id == user.id,
            UserProject.name == project_name,
        ).one_or_none()
    if project is None:
        identifier = project_id if project_id is not None else project_name
        raise ValueError(f"Directory not found: {identifier}")
    if not check_project_write_access(project.id, user, db_session):
        raise ValueError(f"User cannot write to directory: {project.id}")
    return ImportTarget(user_id=user.id, project_id=project.id)


def _active_file_with_name_exists(
    db_session: Session, *, project_id: int, file_name: str
) -> bool:
    return (
        db_session.query(UserFile.id)
        .join(Project__UserFile, Project__UserFile.user_file_id == UserFile.id)
        .filter(
            Project__UserFile.project_id == project_id,
            UserFile.name == file_name,
            UserFile.status != UserFileStatus.DELETING,
        )
        .first()
        is not None
    )


def _as_upload(path: Path, file_handle: BinaryIO) -> UploadFile:
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return UploadFile(
        file=file_handle,
        size=path.stat().st_size,
        filename=path.name,
        headers=Headers({"content-type": content_type}),
    )


def _import_one_file(
    path: Path,
    *,
    target: ImportTarget,
    tenant_id: str,
    elasticsearch_target: ElasticsearchTarget,
) -> ImportResult:
    with path.open("rb") as file_handle:
        upload = _as_upload(path, file_handle)
        with get_session_with_current_tenant() as db_session:
            user = db_session.get(User, target.user_id)
            if user is None:
                raise ValueError(
                    f"Import user disappeared after preflight: {target.user_id}"
                )
            categorized = create_user_files(
                files=[upload],
                project_id=target.project_id,
                user=user,
                db_session=db_session,
            )
            rejected_details = [item.reason for item in categorized.rejected_files]
            user_file_ids = [user_file.id for user_file in categorized.user_files]
            indexable_file_ids = [
                user_file.id for user_file in categorized.indexable_files
            ]

    if rejected_details:
        return ImportResult(
            path=str(path),
            status="rejected",
            user_file_id=None,
            chunk_count=0,
            detail="; ".join(rejected_details),
        )
    if len(indexable_file_ids) != 1:
        return ImportResult(
            path=str(path),
            status="failed",
            user_file_id=(str(user_file_ids[0]) if user_file_ids else None),
            chunk_count=0,
            detail="File was stored but not accepted for indexing",
        )

    user_file_uuid = indexable_file_ids[0]
    user_file_id = str(user_file_uuid)
    process_user_file_impl(
        user_file_id=user_file_id,
        tenant_id=tenant_id,
        redis_locking=False,
    )

    with get_session_with_current_tenant() as db_session:
        user_file = db_session.get(UserFile, user_file_uuid)
        chunks = get_chunks_for_file(db_session, user_file_uuid)
        status = user_file.status if user_file is not None else None
        recorded_chunk_count = user_file.chunk_count if user_file is not None else None
        projection_identities = [
            RegulatoryProjectionIdentity(
                regulatory_chunk_id=chunk.id,
                chunk_index=chunk.position,
            )
            for chunk in chunks
        ]

    if (
        status != UserFileStatus.COMPLETED
        or not chunks
        or recorded_chunk_count != len(chunks)
    ):
        return ImportResult(
            path=str(path),
            status="failed",
            user_file_id=user_file_id,
            chunk_count=len(chunks),
            detail=(
                "Post-import verification failed "
                f"(status={status}, recorded_chunks={recorded_chunk_count}, "
                f"postgres_chunks={len(chunks)})"
            ),
        )

    try:
        _ensure_projection_visible(
            document_id=user_file_id,
            identities=projection_identities,
            target=elasticsearch_target,
        )
    except Exception as exc:
        return ImportResult(
            path=str(path),
            status="failed",
            user_file_id=user_file_id,
            chunk_count=len(chunks),
            detail=f"Elasticsearch post-import verification failed: {exc}",
        )

    return ImportResult(
        path=str(path),
        status="completed",
        user_file_id=user_file_id,
        chunk_count=len(chunks),
        detail=None,
    )


def _prepare_manifest_path(path: Path) -> Path:
    """Resolve and prove manifest writability before any remote writes occur."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", dir=path.parent):
        pass
    return path


def _write_manifest(path: Path, results: Sequence[ImportResult]) -> None:
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            dir=path.parent,
            delete=False,
        ) as manifest_file:
            temporary_path = Path(manifest_file.name)
            json.dump(list(results), manifest_file, ensure_ascii=False, indent=2)
            manifest_file.write("\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> int:
    ensure_document_import_available()
    files = _resolve_paths(args.paths, args.recursive)
    _ensure_files_readable(files)

    duplicate_input_names = _duplicate_file_names(files)
    if duplicate_input_names and not args.allow_duplicate:
        raise ValueError(
            "Input paths contain duplicate destination file names: "
            + ", ".join(duplicate_input_names)
        )
    manifest_path = (
        _prepare_manifest_path(args.manifest) if args.manifest is not None else None
    )

    SqlEngine.set_app_name("regulatory_importer")
    with SqlEngine.scoped_engine(pool_size=2, max_overflow=0):
        tenant_token = CURRENT_TENANT_ID_CONTEXTVAR.set(args.tenant_id)
        try:
            with get_session_with_current_tenant() as db_session:
                _ensure_database_schema_current(db_session)
                target = _resolve_import_target(
                    db_session,
                    user_email=args.user_email,
                    project_id=args.project_id,
                    project_name=args.project_name,
                )
                elasticsearch_target = _resolve_elasticsearch_target(
                    db_session,
                    tenant_id=args.tenant_id,
                )
                duplicate_names = [
                    path.name
                    for path in files
                    if _active_file_with_name_exists(
                        db_session,
                        project_id=target.project_id,
                        file_name=path.name,
                    )
                ]
            if duplicate_names and not args.allow_duplicate:
                duplicate_list = ", ".join(sorted(set(duplicate_names)))
                raise ValueError(
                    "Active files with the same name already exist in the directory: "
                    + duplicate_list
                )
            _ensure_elasticsearch_ready(elasticsearch_target)

            results: list[ImportResult] = []
            for path in files:
                try:
                    result = _import_one_file(
                        path,
                        target=target,
                        tenant_id=args.tenant_id,
                        elasticsearch_target=elasticsearch_target,
                    )
                except Exception as exc:
                    result = ImportResult(
                        path=str(path),
                        status="failed",
                        user_file_id=None,
                        chunk_count=0,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)

            if manifest_path is not None:
                _write_manifest(manifest_path, results)
            return (
                0 if all(result["status"] == "completed" for result in results) else 1
            )
        finally:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(tenant_token)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    try:
        return run(parser.parse_args(argv))
    except Exception as exc:
        print(
            f"Importer preflight failed: {type(exc).__name__}: {exc}", file=sys.stderr
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
