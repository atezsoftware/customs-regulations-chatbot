"""Fixed read-only ownership lookups for retained DEV acceptance diagnostics."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from onyx.db.regulatory_annex_acceptance import CanaryRun


def load_source861_canary(session: "Session") -> "CanaryRun | None":
    from onyx.db.models import KVStore
    from onyx.db.regulatory_annex_acceptance import CanaryRun

    saved = session.get(
        KVStore,
        "regulatory_annex_acceptance:8612398f20e5d3d03d154fffa196c5bf951b1862",
    )
    return CanaryRun.model_validate(saved.value) if saved is not None else None


def chat50792_error_category(value: str | None) -> str:
    """Classify retained errors without returning response text or unknown values."""
    if not value:
        return "empty"
    bounded = value[:50000]
    for marker, category in (
        (
            "Selected Document Sets are outside this agent's knowledge scope.",
            "agent_document_set_scope",
        ),
        ("User does not have access to document sets:", "document_set_access"),
        (
            "A source changed during this response. Please try again.",
            "publication_changed",
        ),
        ("No search settings configured", "search_configuration_missing"),
    ):
        if marker in bounded:
            return category
    return "no_known_error"


def load_chat50792_diagnostic(session: "Session") -> dict[str, object]:
    from uuid import UUID

    from onyx.db.chat import get_chat_messages_by_session
    from onyx.db.models import (
        AmendmentBatch,
        ChatSession,
        DocumentSet,
        KVStore,
        Persona,
    )
    from onyx.db.regulatory_annex_acceptance import CanaryRun

    release = "50792ae3d877577c4dafcf577bc0027b593370d9"
    run_id = UUID("97819935-8cd6-4bcf-aeee-04797d17e864")
    chat_id = UUID("01668c44-1692-469c-be15-93d38fa8a858")
    file_id = UUID("df0750d8-ddc3-4303-91f1-1aa9e3d43620")
    output: dict[str, object] = {
        "status": "scope_refused",
        "scope_failure": "run_missing",
    }
    saved = session.get(KVStore, "regulatory_annex_acceptance:" + release)
    if saved is None:
        return output
    run = CanaryRun.model_validate(saved.value)
    marker = "FICTIONAL ANNEX CANARY " + str(run_id) + " / dated 2026-09-09"
    if (
        run.release_sha != release
        or run.user_id != UUID("7e0d56bc-6f9c-4cec-b29b-5a8c9ec2b844")
        or run.run_id != run_id
        or run.file_id != file_id
        or run.document_set_id != 24
        or run.batch_id != 49
        or run.phase != "cleaned"
        or run.chat_ids != [chat_id]
        or not any(
            intent.kind == "chat"
            and intent.marker == marker
            and intent.artifact_id == chat_id
            for intent in run.creation_intents
        )
    ):
        output["scope_failure"] = "run_ownership"
        return output
    scope = session.get(DocumentSet, 24)
    batch = session.get(AmendmentBatch, 49)
    chat = session.get(ChatSession, chat_id)
    if (
        scope is None
        or scope.is_public
        or scope.user_id != run.user_id
        or scope.name != run.name
        or batch is None
        or batch.document_set_id != 24
        or batch.created_by != run.user_id
        or batch.user_file_ids != [str(file_id)]
        or batch.source_package_id != run.package_id
        or chat is None
        or chat.user_id != run.user_id
        or chat.description != marker
    ):
        output["scope_failure"] = "chat_ownership"
        return output
    persona = (
        session.get(Persona, chat.persona_id) if chat.persona_id is not None else None
    )
    default = session.get(Persona, 0)
    scopes = list(persona.document_sets) if persona is not None else []
    defaults = list(default.document_sets) if default is not None else []
    if len(scopes) > 1000 or len(defaults) > 1000:
        output["scope_failure"] = "evidence_limit"
        return output
    messages = get_chat_messages_by_session(
        chat_id,
        run.user_id,
        session,
        skip_permission_check=True,
        prefetch_top_two_level_tool_calls=True,
        prefetch_message_details=True,
    )
    if len(messages) > 64:
        output["scope_failure"] = "evidence_limit"
        return output
    tool_rows: list[dict[str, object]] = []
    message_rows: list[dict[str, object]] = []
    seen: set[int] = set()
    for message in messages:
        documents = message.search_docs or []
        receipt = message.publication_read
        message_rows.append(
            {
                "message_id": message.id,
                "assistant": message.message_type.value == "assistant",
                "document_count": len(documents),
                "owned_document_count": sum(
                    doc.document_id == str(file_id) for doc in documents
                ),
                "citation_count": len(message.citations or {}),
                "has_error": bool(message.error),
                "error_category": chat50792_error_category(message.error),
                "publication_read_present": receipt is not None,
                "publication_finalized": isinstance(receipt, dict)
                and receipt.get("finalized") is True,
            }
        )
        pending = list(message.tool_calls or [])
        while pending:
            tool = pending.pop()
            if tool.id in seen:
                continue
            if len(seen) >= 64 or tool.chat_session_id != chat_id:
                return {"status": "scope_refused", "scope_failure": "evidence_limit"}
            seen.add(tool.id)
            pending.extend(tool.tool_call_children or [])
            tool_rows.append(
                {
                    "tool_call_id": tool.id,
                    "tool_id": tool.tool_id,
                    "result_count": len(tool.search_docs or []),
                    "owned_result_count": sum(
                        doc.document_id == str(file_id)
                        for doc in tool.search_docs or []
                    ),
                    "error_category": chat50792_error_category(tool.tool_call_response),
                }
            )
    return {
        "status": "read",
        "assistant_id": chat.persona_id,
        "assistant_present": persona is not None,
        "assistant_scope_count": len(scopes),
        "assistant_owned_scope_included": any(
            item.id == 24 and item.name == run.name for item in scopes
        ),
        "default_present": default is not None,
        "default_scope_count": len(defaults),
        "default_owned_scope_included": any(
            item.id == 24 and item.name == run.name for item in defaults
        ),
        "chat_deleted": chat.deleted,
        "messages": message_rows,
        "tools": tool_rows,
    }
