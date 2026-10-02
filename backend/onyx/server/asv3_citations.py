"""Owned saved-run previews for verified original-file evidence."""

from fastapi import APIRouter, Depends
from pydantic import JsonValue
from sqlalchemy.orm import Session

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.auth.permissions import require_permission
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusScopeUnavailable
from onyx.db.asv3_runs import load_asv3_checkpoint
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission
from onyx.db.models import User
from onyx.db.search_settings import get_current_search_settings
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.natural_language_processing.utils import get_tokenizer
from onyx.server.documents.models import ChunkInfo

router = APIRouter(prefix="/asv3")


def native_item_from_checkpoint(
    checkpoint: dict[str, JsonValue], number: int
) -> tuple[EvidenceItem, IndexFilters]:
    scope = checkpoint.get("scope")
    evidence = checkpoint.get("evidence")
    if number < 1 or not isinstance(scope, dict) or not isinstance(evidence, dict):
        raise ValueError("Invalid citation checkpoint")
    if evidence.get("version") != 1:
        raise ValueError("Unsupported citation ledger version")
    records = evidence.get("records")
    if not isinstance(records, list):
        raise ValueError("Missing citation ledger")
    record = next(
        (
            row
            for row in records
            if isinstance(row, dict) and row.get("citation") == number
        ),
        None,
    )
    if record is None:
        raise ValueError("Citation not found")
    item = EvidenceItem.model_validate(record.get("item"))
    if (
        item.chunk_id is not None
        or not item.metadata.get("derived")
        or not item.metadata.get("source_sha256")
        or item.search_doc is None
        or item.search_doc.chunk_ind != -number
        or item.search_doc.document_id != item.source_id
    ):
        raise ValueError("Citation is not verified native evidence")
    return item, IndexFilters.model_validate(scope)


@router.get("/citation/{assistant_message_id}/{citation_num}")
def get_native_citation(
    assistant_message_id: int,
    citation_num: int,
    user: User = Depends(require_permission(Permission.BASIC_ACCESS)),
    db_session: Session = Depends(get_session),
) -> ChunkInfo:
    try:
        checkpoint = load_asv3_checkpoint(
            message_id=assistant_message_id, user_id=user.id
        )
        if checkpoint is None:
            raise ValueError("Citation checkpoint not found")
        item, filters = native_item_from_checkpoint(checkpoint, citation_num)
        context = RunContext(scope=filters.model_dump(mode="json"), timeout_seconds=30)
        CorpusBroker(user, filters).revalidate_evidence([item], context)
    except (PermissionError, ValueError, CorpusScopeUnavailable) as error:
        raise OnyxError(OnyxErrorCode.NOT_FOUND, "Citation not available") from error
    settings = get_current_search_settings(db_session)
    encode = get_tokenizer(
        provider_type=settings.provider_type, model_name=settings.model_name
    ).encode
    return ChunkInfo(content=item.text, num_tokens=len(encode(item.text)))
