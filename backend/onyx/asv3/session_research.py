"""Carry authorized originals between questions without restoring a previous run."""

from collections.abc import Callable
from datetime import date

from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.asv3.outcome_map import OutcomeCondition, OutcomeMap, RequestedOutcome
from onyx.db.asv3_corpus import CorpusScopeUnavailable

_NAVIGATION_NOTICE = (
    "Prior model-selected outcomes and source-bound candidates are navigation only; "
    "they are not approval or completion of the current question. Reassess the "
    "current facts, original text, date and applicability."
)


def _compact_outcome_navigation(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    outcomes: list[JsonValue] = []
    known_outcomes: set[str] = set()
    rows = data.get("outcomes", [])
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            outcome = RequestedOutcome.model_validate(
                {
                    "outcome_id": row.get("outcome_id"),
                    "question_ids": ["prior"],
                    "detail": row.get("detail"),
                    "decisive_facts": row.get("decisive_facts", []),
                }
            )
        except ValueError:
            continue
        if outcome.outcome_id in known_outcomes or not outcome.detail.strip():
            continue
        known_outcomes.add(outcome.outcome_id)
        outcomes.append(outcome.model_dump(mode="json", exclude={"question_ids"}))
    conditions: list[JsonValue] = []
    rows = data.get("conditions", [])
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            condition = OutcomeCondition.model_validate(row)
        except ValueError:
            continue
        bound = [
            identity for identity in condition.outcome_ids if identity in known_outcomes
        ]
        if bound:
            conditions.append(
                {**condition.model_dump(mode="json"), "outcome_ids": bound}
            )
    open_gaps: list[JsonValue] = []
    rows = data.get("open_gaps", [])
    resolution_rows = data.get("resolutions", [])
    if isinstance(resolution_rows, list):
        rows = [
            *(rows if isinstance(rows, list) else []),
            *(
                row
                for row in resolution_rows
                if isinstance(row, dict) and row.get("status") == "unresolved"
            ),
        ]
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        identity, gap = row.get("outcome_id"), row.get("gap")
        if (
            isinstance(identity, str)
            and identity in known_outcomes
            and isinstance(gap, str)
            and gap.strip()
        ):
            item: dict[str, JsonValue] = {"outcome_id": identity, "gap": gap}
            if item not in open_gaps:
                open_gaps.append(item)
    return {
        "outcomes": outcomes,
        "conditions": conditions,
        "open_gaps": open_gaps,
        "notice": _NAVIGATION_NOTICE,
    }


def _remap_outcome_navigation(
    data: dict[str, JsonValue],
    records: list[JsonValue],
    retained: list[int],
    ledger: EvidenceLedger,
) -> dict[str, JsonValue]:
    navigation = _compact_outcome_navigation(data)
    available = {
        item.identity: (number, item)
        for number in retained
        if (item := ledger.get(number)) is not None
    }
    prior: dict[int, EvidenceItem] = {}
    ambiguous: set[int] = set()
    for row in records:
        if not isinstance(row, dict) or type(row.get("citation")) is not int:
            continue
        citation = row["citation"]
        assert isinstance(citation, int)
        try:
            original = EvidenceItem.model_validate(row.get("item"))
        except ValueError:
            continue
        if citation in prior:
            ambiguous.add(citation)
        else:
            prior[citation] = original
    valid_conditions: list[JsonValue] = []
    rows = navigation["conditions"]
    assert isinstance(rows, list)
    for row in rows:
        condition = OutcomeCondition.model_validate(row)
        witnesses: list[JsonValue] = []
        for witness in condition.witnesses:
            original = prior.get(witness.citation)
            remapped = (
                available.get(original.identity) if original is not None else None
            )
            if (
                original is None
                or witness.citation in ambiguous
                or remapped is None
                or not witness.start_char < witness.end_char <= len(original.text)
                or witness.end_char > len(remapped[1].text)
            ):
                break
            witnesses.append(
                {**witness.model_dump(mode="json"), "citation": remapped[0]}
            )
        else:
            valid_conditions.append(
                {**condition.model_dump(mode="json"), "witnesses": witnesses}
            )
    navigation["conditions"] = valid_conditions
    return navigation


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
    outcomes = context.services.get("outcome_map")
    navigation = outcomes.view() if isinstance(outcomes, OutcomeMap) else None
    if navigation is not None and navigation.get("outcomes"):
        result["outcome_navigation"] = _compact_outcome_navigation(navigation)
    else:
        previous_navigation = result.get("prior_outcomes")
        if isinstance(previous_navigation, dict):
            result["outcome_navigation"] = _compact_outcome_navigation(
                previous_navigation
            )
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
    prior_navigation = (
        remembered.get("outcome_navigation") if isinstance(remembered, dict) else None
    )
    if isinstance(prior_navigation, dict):
        memory["prior_outcomes"] = _remap_outcome_navigation(
            prior_navigation, [], [], ledger
        )
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
            retained = ledger.add(originals, context)
            memory.update(
                status="revalidated",
                reused_evidence_numbers=retained,
                source_gaps=gaps,
            )
            if isinstance(prior_navigation, dict):
                memory["prior_outcomes"] = _remap_outcome_navigation(
                    prior_navigation,
                    records,
                    retained,
                    ledger,
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
    if isinstance(prior_navigation, dict):
        memory["prior_outcomes"] = _remap_outcome_navigation(
            prior_navigation, records, retained, ledger
        )
