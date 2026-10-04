"""Carry authorized originals between questions without restoring a previous run."""

from collections.abc import Callable
from datetime import date

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.db.asv3_corpus import CorpusScopeUnavailable


def session_research_checkpoint(
    context: RunContext, question: str
) -> dict[str, JsonValue]:
    memory = context.services.get("session_research")
    result: dict[str, JsonValue] = dict(memory) if isinstance(memory, dict) else {}
    requests = result.get("requests", [])
    result["requests"] = [
        *(
            [value for value in requests if isinstance(value, str)]
            if isinstance(requests, list)
            else []
        ),
        question,
    ]
    return result


def retain_session_research(
    previous: dict[str, JsonValue] | None,
    context: RunContext,
    ledger: EvidenceLedger,
    revalidate: Callable[[list[EvidenceItem], RunContext], None],
) -> None:
    if previous is None:
        return
    remembered = previous.get("session_research")
    prior_requests = (
        remembered.get("requests", []) if isinstance(remembered, dict) else []
    )
    requests = (
        [value for value in prior_requests if isinstance(value, str)]
        if isinstance(prior_requests, list)
        else []
    )
    request = previous.get("request")
    if isinstance(request, str) and request not in requests:
        requests.append(request)
    memory: dict[str, JsonValue] = {
        "requests": requests,
        "reused_evidence_numbers": [],
        "source_gaps": [],
        "status": "scope_changed",
    }
    context.services["session_research"] = memory
    if previous.get("scope") != context.scope:
        return
    evidence = previous.get("evidence")
    records = evidence.get("records", []) if isinstance(evidence, dict) else []
    if not isinstance(records, list):
        return
    groups: dict[tuple[str, str], list[EvidenceItem]] = {}
    gaps: list[JsonValue] = []
    for row in records:
        if not isinstance(row, dict):
            continue
        try:
            item = EvidenceItem.model_validate(row.get("item"))
        except ValueError:
            gaps.append({"status": "invalid_retained_original"})
            continue
        if item.metadata.get("external"):
            gaps.append(
                {
                    "source_id": item.source_id,
                    "status": "fresh_authorized_read_required",
                }
            )
            continue
        # Earlier question bindings are not outcomes of the new question.
        item.question_ids = []
        if item.metadata.get("read_as_of_date"):
            item.metadata["read_as_of_date"] = str(
                context.scope.get("as_of_date") or date.today().isoformat()
            )
        if item.chunk_id is None and item.metadata.get("source_sha256"):
            # Native previews bind the new message and newly allocated citation number.
            item.search_doc = None
        key = (item.source_id, str(item.metadata.get("read_as_of_date", "")))
        groups.setdefault(key, []).append(item)
    retained: list[int] = []
    originals = [item for items in groups.values() for item in items]
    if originals:
        try:
            revalidate(originals, context)
        except (PermissionError, CorpusScopeUnavailable):
            pass
        else:
            memory.update(
                status="revalidated",
                reused_evidence_numbers=ledger.add(originals, context),
                source_gaps=gaps,
            )
            return
    for items in groups.values():
        context.check_active()
        try:
            revalidate(items, context)
        except (PermissionError, CorpusScopeUnavailable):
            for item in items:
                if len(items) > 1:
                    try:
                        revalidate([item], context)
                    except (PermissionError, CorpusScopeUnavailable):
                        pass
                    else:
                        retained.extend(ledger.add([item], context))
                        continue
                gaps.append(
                    {
                        "source_id": item.source_id,
                        "chunk_id": item.chunk_id,
                        "status": "retained_original_unavailable",
                    }
                )
            continue
        retained.extend(ledger.add(items, context))
    memory.update(
        status="revalidated",
        reused_evidence_numbers=retained,
        source_gaps=gaps,
    )
