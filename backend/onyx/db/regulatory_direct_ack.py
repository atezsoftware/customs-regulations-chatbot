"""Read bounded direct-gateway ACKs from their retained PostgreSQL objects."""

from sqlalchemy import cast, func, literal, select
from sqlalchemy.dialects.postgresql import OID
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from onyx.db.models import FileContent, FileRecord
from onyx.file_store.postgres_file_store import POSTGRES_BUCKET_SENTINEL


def read_direct_ack(session: Session, file_id: str, *, max_bytes: int) -> bytes:
    if (
        not file_id.startswith("regulatory-labeling-direct/")
        or not 0 < max_bytes <= 8 * 1024 * 1024
    ):
        raise ValueError("The direct-gateway ACK read scope is invalid")
    record = session.get(FileRecord, file_id, populate_existing=True)
    content = session.get(FileContent, file_id, populate_existing=True)
    if (
        record is None
        or content is None
        or record.bucket_name != POSTGRES_BUCKET_SENTINEL
        or record.object_key != str(content.lobj_oid)
        or not 0 < content.file_size <= max_bytes
    ):
        raise ValueError("The direct-gateway ACK PostgreSQL identity or size changed")
    try:
        raw = session.scalar(
            select(
                func.pg_catalog.lo_get(
                    cast(literal(content.lobj_oid), OID), 0, max_bytes + 1
                )
            )
        )
    except DBAPIError as error:
        raise ValueError(
            "The direct-gateway ACK PostgreSQL bytes are unavailable"
        ) from error
    if raw is None or not 0 < len(raw) == content.file_size <= max_bytes:
        raise ValueError("The direct-gateway ACK PostgreSQL byte count changed")
    return bytes(raw)
