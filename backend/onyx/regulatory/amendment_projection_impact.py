"""Limit amendment publication to changed legal intervals and actual consumers."""

import json
from collections import defaultdict
from collections.abc import Callable
from datetime import date
from time import monotonic
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict

from onyx.llm.interfaces import LLM
from onyx.prompts.regulatory.amendment_impact import CONTEXT_IMPACT_AUDIT
from onyx.regulatory.amendments.annexes.context_dependencies import (
    context_hash,
    contextual_model_fingerprint,
)
from onyx.regulatory.amendments.annexes.models import (
    AnnexCanonicalSnapshot,
    AnnexTemporalProjection,
)
from onyx.regulatory.amendments.annexes.selective_impact import (
    recover_source_membership,
    source_ids,
)
from onyx.regulatory.structured_llm import generate_structured
from onyx.regulatory.writer_publication_models import (
    AmendmentConsumer,
    AmendmentContextEvidence,
    AmendmentImpactReport,
)
from onyx.tracing.flows import LLMFlow
from onyx.utils.logger import setup_logger
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel

logger = setup_logger()
Window = tuple[date, date]


def find_source_consumers(
    rows: list[AnnexCanonicalSnapshot], source_chunk_ids: set[str]
) -> list[AmendmentConsumer]:
    """Read-only usage report; textual occurrence alone never authorizes a rewrite."""
    by_id = {r.id: r for r in rows}
    if len(by_id) != len(rows) or len({r.user_file_id for r in rows}) != 1:
        raise ValueError("source usage requires unique identities in one file")
    if not source_chunk_ids <= by_id.keys():
        raise ValueError("selected source outside canonical snapshot")
    recovered = recover_source_membership(rows)
    reverse: dict[str, set[str]] = defaultdict(set)
    invalid: set[str] = set()
    for row in rows:
        members = recovered.get(row.id, source_ids(row))
        if (
            row.metadata.get("chunk_variant") == "hierarchical_aggregate"
            and not members
        ):
            raise ValueError("aggregate source membership unavailable: " + row.id)
        for source in members:
            if source not in by_id or source == row.id:
                invalid.add(row.id)
            reverse[source].add(row.id)
    found: list[AmendmentConsumer] = []
    for source in sorted(source_chunk_ids):
        visited = {source}
        pending = [(source, [source])]
        while pending:
            identifier, path = pending.pop(0)
            for consumer in sorted(reverse[identifier]):
                if consumer in invalid:
                    raise ValueError("derived source missing or cyclic: " + consumer)
                if consumer in path:
                    raise ValueError("cyclic derived source membership")
                if consumer in visited:
                    continue
                visited.add(consumer)
                next_path = [*path, consumer]
                row = by_id[consumer]
                found.append(
                    AmendmentConsumer(
                        source_id=source,
                        consumer_id=consumer,
                        relation="image"
                        if row.metadata.get("bound_to_regulatory_chunk_id")
                        else "aggregate",
                        path=next_path,
                        quote=by_id[source].text
                        if by_id[source].text in row.text
                        else "",
                    )
                )
                pending.append((consumer, next_path))
        # Short headings/common tokens do not establish text ownership.
        text = by_id[source].text
        if len(text.strip()) >= 60:
            found.extend(
                AmendmentConsumer(
                    source_id=source,
                    consumer_id=row.id,
                    relation="text_occurrence",
                    path=[source, row.id],
                    quote=text,
                )
                for row in rows
                if row.id not in visited and text in row.text
            )
    return found


def merge_windows(windows: list[Window]) -> list[Window]:
    result: list[Window] = []
    for start, end in sorted(set(windows)):
        if start >= end:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def changed_windows(
    before: list[AnnexCanonicalSnapshot], after: list[AnnexCanonicalSnapshot]
) -> dict[str, list[Window]]:
    old, new = {r.id: r for r in before}, {r.id: r for r in after}
    result: dict[str, list[Window]] = {}
    for identifier in old.keys() | new.keys():
        left, right = old.get(identifier), new.get(identifier)
        boundaries = sorted(
            {
                date.min,
                date.max,
                *(
                    d
                    for row in (left, right)
                    if row
                    for d in (row.validity_start_date, row.validity_end_date)
                    if d is not None
                ),
            }
        )

        def representation(row: AnnexCanonicalSnapshot | None, when: date) -> object:
            if row is None or not (
                (row.validity_start_date or date.min)
                <= when
                < (row.validity_end_date or date.max)
            ):
                return None
            return (
                row.text,
                row.heading_path,
                row.metadata,
                row.position,
                row.chunk_type,
            )

        windows = merge_windows(
            [
                (start, end)
                for start, end in zip(boundaries, boundaries[1:])
                if representation(left, start) != representation(right, start)
            ]
        )
        if windows:
            result[identifier] = windows
    return result


def structural_windows(
    before: list[AnnexCanonicalSnapshot], after: list[AnnexCanonicalSnapshot]
) -> dict[str, list[Window]]:
    impacts = changed_windows(before, after)
    consumers: dict[str, dict[str, list[Window]]] = defaultdict(dict)
    for row in [*before, *after]:
        for source in source_ids(row):
            if source != row.id:
                windows = consumers[source].setdefault(row.id, [])
                windows.append(
                    (
                        row.validity_start_date or date.min,
                        row.validity_end_date or date.max,
                    )
                )
    pending = list(impacts)
    while pending:
        source = pending.pop()
        for consumer, lifetimes in consumers[source].items():
            overlaps = [
                (max(a, c), min(b, d))
                for a, b in impacts[source]
                for c, d in lifetimes
                if max(a, c) < min(b, d)
            ]
            if not overlaps:
                continue
            combined = merge_windows(impacts.get(consumer, []) + overlaps)
            if combined != impacts.get(consumer):
                impacts[consumer] = combined
                pending.append(consumer)
    return impacts


class ContextImpactDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: str
    affected: bool
    quote: str
    reason: str
    uncertain: bool = False
    source_id: str | None = None
    source_quote: str = ""
    source_side: Literal["before", "after"] | None = None


class ContextImpactResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decisions: list[ContextImpactDecision]


ContextAuditResolver = Callable[
    [str, Callable[[], ContextImpactResult]], ContextImpactResult
]


def validate_context_decisions(
    contexts: dict[str, str],
    decisions: list[ContextImpactDecision],
    *,
    changes: dict[str, dict[str, dict[str, str]]] | None = None,
) -> set[str]:
    if len(decisions) != len(contexts) or {d.key for d in decisions} != set(contexts):
        raise ValueError(
            "context impact audit must cover every supplied context exactly once"
        )
    for decision in decisions:
        if decision.uncertain:
            raise ValueError("unresolved context impact: " + decision.reason)
        if not decision.reason.strip():
            raise ValueError("context impact audit requires an explanation")
        if decision.affected and (
            not decision.quote.strip() or decision.quote not in contexts[decision.key]
        ):
            raise ValueError(
                "context impact must quote an actual affected context statement"
            )
        if decision.affected and changes is not None:
            source = (
                changes[decision.key]
                .get(decision.source_side or "", {})
                .get(decision.source_id or "")
            )
            if (
                source is None
                or not decision.source_quote.strip()
                or decision.source_quote not in source
            ):
                raise ValueError("context impact must quote an actual changed source")
    return {d.key for d in decisions if d.affected}


def include_context_consumers(
    impacts: dict[str, list[Window]],
    *,
    before: list[AnnexCanonicalSnapshot],
    after: list[AnnexCanonicalSnapshot],
    bindings: list[AnnexTemporalProjection],
    llm: LLM | None,
    evidence: list[AmendmentContextEvidence] | None = None,
    audit_cache: ContextAuditResolver | None = None,
) -> dict[str, list[Window]]:
    """Audit frozen output, not mere membership in a document-wide LLM input.

    A shared source document is not proof that every generated summary changed.
    Deduplicate actual context outputs; do not regenerate or embed candidates.
    Incomplete audits fail explicitly rather than escalating to file reindexing.
    """
    result = {key: list(windows) for key, windows in impacts.items()}
    boundaries = sorted(
        {d for ranges in impacts.values() for window in ranges for d in window}
        | {
            d
            for row in before + after
            for d in (row.validity_start_date, row.validity_end_date)
            if d is not None
        }
    )
    windows = [
        (start, end)
        for start, end in zip(boundaries, boundaries[1:])
        if any(
            left <= start and end <= right
            for ranges in impacts.values()
            for left, right in ranges
        )
    ]
    groups: dict[tuple[str, date, date], set[str]] = defaultdict(set)
    for binding in bindings:
        source = json.loads(binding.projection.source_json)
        identifier = source["regulatory_chunk_id"]
        context = "\n".join(
            value
            for value in (source.get("doc_summary"), source.get("chunk_context"))
            if value
        )
        if not context.strip():
            continue
        for start, end in windows:
            start = max(start, binding.effective_start or date.min)
            end = min(end, binding.effective_end or date.max)
            if start >= end or any(
                left <= start and end <= right
                for left, right in impacts.get(identifier, [])
            ):
                continue
            groups[(context, start, end)].add(identifier)
    if not groups:
        return result
    if llm is None:
        raise ValueError(
            "existing contextual embeddings require a context impact audit"
        )
    direct = changed_windows(before, after)

    def effective_changes(
        rows: list[AnnexCanonicalSnapshot], when: date
    ) -> list[dict[str, object]]:
        return [
            {
                "id": row.id,
                "text": row.text,
                "heading_path": row.heading_path,
                "metadata": row.metadata,
            }
            for row in rows
            if row.id in direct
            and any(a <= when < b for a, b in direct[row.id])
            and (row.validity_start_date or date.min)
            <= when
            < (row.validity_end_date or date.max)
        ]

    entries = {str(i): entry for i, entry in enumerate(groups)}
    contexts = {key: entry[0] for key, entry in entries.items()}
    changes = {
        f"{start}:{end}": {
            "before": effective_changes(before, start),
            "after": effective_changes(after, start),
        }
        for _, start, end in entries.values()
    }
    cases = {
        key: {
            "text": context,
            "effective_start": start.isoformat(),
            "effective_end": end.isoformat(),
            "change_key": f"{start}:{end}",
        }
        for key, (context, start, end) in entries.items()
    }
    batches: list[dict[str, str]] = []
    current: dict[str, str] = {}
    characters = 0
    for key, context in contexts.items():
        if current and (len(current) >= 8 or characters + len(context) > 24000):
            batches.append(current)
            current = {}
            characters = 0
        current[key] = context
        characters += len(context)
    if current:
        batches.append(current)
    affected: set[str] = set()

    def source_proofs(batch: dict[str, str]) -> dict[str, dict[str, dict[str, str]]]:
        return {
            key: {
                side: {str(r["id"]): str(r["text"]) for r in rows}
                for side, rows in changes[str(cases[key]["change_key"])].items()
            }
            for key in batch
        }

    def audit_batch(batch: dict[str, str]) -> ContextImpactResult:
        # Bound each unit of work without timing out later groups before they start.
        deadline = monotonic() + 180
        proofs = source_proofs(batch)

        def generate_audit() -> ContextImpactResult:
            feedback: str | None = None
            for attempt in range(2):
                response = generate_structured(
                    llm,
                    flow=LLMFlow.REGULATORY_CONTEXTUAL_BATCH,
                    system_prompt=CONTEXT_IMPACT_AUDIT,
                    user_prompt=json.dumps(
                        {
                            "contexts": {key: cases[key] for key in batch},
                            "validation_feedback": feedback,
                            "changes": {
                                str(cases[key]["change_key"]): changes[
                                    str(cases[key]["change_key"])
                                ]
                                for key in batch
                            },
                        },
                        ensure_ascii=False,
                    ),
                    response_model=ContextImpactResult,
                    timeout_override=60,
                    deadline=deadline,
                    max_tokens=12000,
                )
                try:
                    validate_context_decisions(
                        batch, response.decisions, changes=proofs
                    )
                    return response
                except ValueError as error:
                    if attempt == 1:
                        raise
                    feedback = str(error)
            raise AssertionError("context audit attempts exhausted")

        return (
            audit_cache(
                context_hash(
                    [
                        {key: cases[key] for key in batch},
                        {
                            str(cases[key]["change_key"]): changes[
                                str(cases[key]["change_key"])
                            ]
                            for key in batch
                        },
                        contextual_model_fingerprint(llm),
                        CONTEXT_IMPACT_AUDIT,
                    ]
                ),
                generate_audit,
            )
            if audit_cache is not None
            else generate_audit()
        )

    # The shared helper preserves tenant/tracing context and joins every call on failure.
    responses = cast(
        list[ContextImpactResult],
        run_functions_tuples_in_parallel(
            [(audit_batch, (batch,)) for batch in batches], max_workers=4
        ),
    )
    for batch, response in zip(batches, responses):
        proofs = source_proofs(batch)
        if evidence is not None:
            for decision in response.decisions:
                if decision.key not in batch:
                    continue
                entry = entries[decision.key]
                evidence.append(
                    AmendmentContextEvidence(
                        consumer_ids=sorted(groups[entry]),
                        effective_start=entry[1],
                        effective_end=entry[2],
                        context_sha256=context_hash(entry[0]),
                        audit_input_sha256=context_hash(
                            [
                                cases[decision.key],
                                changes[str(cases[decision.key]["change_key"])],
                                contextual_model_fingerprint(llm),
                                CONTEXT_IMPACT_AUDIT,
                            ]
                        ),
                        outcome="unresolved"
                        if decision.uncertain
                        else "affected"
                        if decision.affected
                        else "unchanged",
                        quote=decision.quote,
                        reason=decision.reason,
                        source_id=decision.source_id,
                        source_side=decision.source_side,
                        source_quote=decision.source_quote,
                    )
                )
        affected.update(
            validate_context_decisions(batch, response.decisions, changes=proofs)
        )
    for key in affected:
        _, start, end = entries[key]
        for identifier in groups[entries[key]]:
            result[identifier] = merge_windows(
                result.get(identifier, []) + [(start, end)]
            )
    logger.info(
        "amendment_context_impact unique_contexts=%d affected_contexts=%d affected_chunks=%d audit_groups=%d",
        len(contexts),
        len(affected),
        len(result),
        len(batches),
    )
    return result


def validate_published_source_membership(
    before: list[AnnexCanonicalSnapshot],
    bindings: list[AnnexTemporalProjection],
    changed: dict[str, list[Window]],
) -> None:
    """Never silently retain a dated representation with different source lineage.

    Annex publications can advance derived sources independently of canonical
    metadata. Such transitions need reconciliation before the normal writer can
    safely construct canonical successors.
    """
    rows = {r.id: r for r in before}
    recovered = recover_source_membership(before)
    affected = dict(changed)
    while True:
        previous = dict(affected)
        for binding in bindings:
            if binding.derived_role not in {
                "hierarchical_aggregate",
                "image_companion",
            }:
                continue
            identifier = json.loads(binding.projection.source_json)[
                "regulatory_chunk_id"
            ]
            lower, upper = (
                binding.effective_start or date.min,
                binding.effective_end or date.max,
            )
            windows = merge_windows(
                [
                    (max(lower, start), min(upper, end))
                    for source in [identifier, *binding.dependency_ids]
                    for start, end in affected.get(source, [])
                    if max(lower, start) < min(upper, end)
                ]
            )
            if not windows:
                continue
            row = rows.get(identifier)
            members = (recovered.get(identifier, source_ids(row))) if row else []
            if (
                row is None
                or members != binding.dependency_ids
                or source_ids(
                    row.model_copy(update={"metadata": binding.representation_metadata})
                )
                != members
            ):
                raise ValueError(
                    f"published source membership requires reconciliation: {identifier}"
                )
            affected[identifier] = merge_windows(affected.get(identifier, []) + windows)
        if affected == previous:
            return


def analyze_amendment_impact(
    *,
    before: list[AnnexCanonicalSnapshot],
    after: list[AnnexCanonicalSnapshot],
    bindings: list[AnnexTemporalProjection],
    llm: LLM | None,
    audit_cache: ContextAuditResolver | None = None,
) -> AmendmentImpactReport:
    """The approval writer and inspection tooling share one evidence-based selector."""
    evidence: list[AmendmentContextEvidence] = []
    direct = changed_windows(before, after)
    dependencies = find_source_consumers(before, set(direct) & {r.id for r in before})
    dependencies.extend(
        find_source_consumers(after, set(direct) & {r.id for r in after})
    )
    selected = structural_windows(before, after)
    unresolved: list[str] = []
    try:
        validate_published_source_membership(before, bindings, selected)
        selected = include_context_consumers(
            selected,
            before=before,
            after=after,
            bindings=bindings,
            llm=llm,
            evidence=evidence,
            audit_cache=audit_cache,
        )
    except ValueError as error:
        unresolved.append(str(error))
    return AmendmentImpactReport(
        canonical_before_sha256=context_hash(
            [r.model_dump(mode="json") for r in before]
        ),
        canonical_after_sha256=context_hash([r.model_dump(mode="json") for r in after]),
        affected_windows=selected,
        dependencies=list({d.model_dump_json(): d for d in dependencies}.values()),
        context_evidence=evidence,
        unchanged_ids=[]
        if unresolved
        else sorted({r.id for r in after} - selected.keys()),
        unresolved=unresolved,
    )
