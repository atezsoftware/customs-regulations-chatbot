"""One bounded repair cycle for an already-reviewed Guardrails v3 answer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, JsonValue

from onyx.asv3.candidate_audit import CandidateAudit
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.guardrails_v3 import GuardrailsV3ReviewOutcome
from onyx.asv3.models import RunContext
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.llm.interfaces import LLM
from onyx.llm.models import ReasoningEffort, UserMessage

_REPAIR_MODEL = "gemini-3.8-flash"
_MAX_REPAIR_EVIDENCE_CHARS = 64_000


class GuardrailsV3Finalization(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    answer: str
    action: str
    repair_applied: bool = False
    recheck_completed: bool = False
    failure_reason: str | None = None


def finalize_guardrails_v3(
    *,
    question: str = "",
    candidate_answer: str,
    initial_review: GuardrailsV3ReviewOutcome,
    ledger: EvidenceLedger,
    context: RunContext,
    repair_llm: LLM | None,
    recheck: Callable[..., GuardrailsV3ReviewOutcome],
    candidate_audit: CandidateAudit | None = None,
    broker: CorpusBroker | None = None,
) -> GuardrailsV3Finalization:
    """Use delivered evidence first and accept only one clean, changed recheck."""
    if not initial_review.review_completed or not initial_review.repair_requested:
        return GuardrailsV3Finalization(answer=candidate_answer, action="none")
    remaining_seconds = max(0.0, context.deadline - time.monotonic())
    if (
        repair_llm is None
        or remaining_seconds < 10
        or repair_llm.config.model_provider not in {"vertex_ai", "gemini"}
        or repair_llm.config.model_name != _REPAIR_MODEL
    ):
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="disclose_gap",
            failure_reason="repair_unavailable",
        )
    citations = list(ledger.citation_numbers())
    delivered = [
        number for number in citations if _has_complete_delivery(ledger, number)
    ]
    action = "patch_delivered_evidence"
    focused_target = _focused_search_target(initial_review)
    if focused_target is not None and broker is not None:
        focused = _search_review_gap(
            question=question,
            target=focused_target,
            broker=broker,
            ledger=ledger,
            context=context,
        )
        if focused:
            delivered = sorted(set(delivered).union(focused))
            action = "focused_search"
        else:
            return GuardrailsV3Finalization(
                answer=candidate_answer,
                action="disclose_gap",
                failure_reason="focused_search_evidence_unavailable",
            )
    elif not delivered and candidate_audit is not None and broker is not None:
        recovered = _recover_audited_candidate(candidate_audit, broker, ledger, context)
        if recovered:
            delivered = recovered
            action = "recover_audited_candidate"
    if not delivered:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="disclose_gap",
            failure_reason="repair_evidence_undelivered",
        )
    remaining_seconds = max(0.0, context.deadline - time.monotonic())
    if remaining_seconds < 10:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="disclose_gap",
            failure_reason="repair_unavailable",
        )
    repair_evidence = _prepare_repair_evidence(ledger, candidate_answer, delivered)
    if not repair_evidence:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="disclose_gap",
            failure_reason="repair_evidence_undelivered",
        )
    ledger.record_delivery(
        "guardrails-v3-repair",
        "asv3_guardrails_v3_repair",
        [
            {"citation": record["citation"], "text": record["text"]}
            for record in repair_evidence
        ],
    )
    allowed_citations = ledger.completely_delivered("guardrails-v3-repair")
    if not allowed_citations:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="disclose_gap",
            failure_reason="repair_evidence_undelivered",
        )
    try:
        response = repair_llm.invoke(
            [
                UserMessage(
                    content=json.dumps(
                        {
                            "candidate_answer": candidate_answer,
                            "findings": [
                                item.model_dump(mode="json")
                                for item in initial_review.findings
                            ],
                            "evidence": repair_evidence,
                            "allowed_citations": sorted(allowed_citations),
                        },
                        ensure_ascii=False,
                    )
                )
            ],
            max_tokens=8192,
            reasoning_effort=ReasoningEffort.LOW,
            use_streaming=False,
            provider_compatibility_attempts=1,
            timeout_override=max(1, int(remaining_seconds)),
        )
        repaired = (response.choice.message.content or "").strip()
    except Exception:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action=action,
            failure_reason="repair_failed",
        )
    if not repaired or repaired == candidate_answer:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action=action,
            failure_reason="repair_unchanged",
        )
    repaired_citations = set(extract_citation_numbers(repaired))
    if repaired_citations - allowed_citations or (
        extract_citation_numbers(candidate_answer) and not repaired_citations
    ):
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action=action,
            failure_reason="invalid_repair_citations",
        )
    review = recheck(repaired, initial_review.findings)
    if not review.review_completed or review.findings:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action=action,
            failure_reason="recheck_failed",
        )
    return GuardrailsV3Finalization(
        answer=repaired,
        action=action,
        repair_applied=True,
        recheck_completed=True,
    )


def _has_complete_delivery(ledger: EvidenceLedger, citation: int) -> bool:
    deliveries = ledger.inspect(citation)["deliveries"]
    if not isinstance(deliveries, list):
        return False
    return any(
        isinstance(delivery, dict)
        and isinstance(records := delivery.get("records"), list)
        and any(
            isinstance(record, dict)
            and record.get("citation") == citation
            and record.get("complete") is True
            for record in records
        )
        for delivery in deliveries
    )


def _prepare_repair_evidence(
    ledger: EvidenceLedger,
    candidate_answer: str,
    delivered: list[int],
) -> list[dict[str, JsonValue]]:
    required = set(extract_citation_numbers(candidate_answer))
    if not required.issubset(delivered):
        return []
    ordered = list(
        dict.fromkeys([*extract_citation_numbers(candidate_answer), *delivered])
    )
    evidence: list[dict[str, JsonValue]] = []
    used_chars = 2
    for citation in ordered:
        item = ledger.get(citation)
        if item is None:
            if citation in required:
                return []
            continue
        record: dict[str, JsonValue] = {
            "citation": citation,
            "source_id": item.source_id,
            "chunk_id": item.chunk_id,
            "text": item.text,
        }
        cost = len(json.dumps(record, ensure_ascii=False)) + (1 if evidence else 0)
        if used_chars + cost > _MAX_REPAIR_EVIDENCE_CHARS:
            if citation in required:
                return []
            continue
        used_chars += cost
        evidence.append(record)
    return evidence


def _focused_search_target(
    review: GuardrailsV3ReviewOutcome,
) -> str | None:
    """Select a checklist-owned research target without naming any authority."""
    for finding in review.findings:
        for dimension in finding.dimensions:
            if len(dimension) == 2 and dimension[0] == "d" and dimension[1].isdigit():
                return f"m2.{dimension}"
    return None


def _search_review_gap(
    *,
    question: str,
    target: str,
    broker: CorpusBroker,
    ledger: EvidenceLedger,
    context: RunContext,
) -> list[int]:
    search = broker.search_adapter
    if search is None or not question.strip():
        return []
    try:
        outcome = search(
            {
                "query": question,
                "mode": "hybrid",
                "coverage_item": target,
                "evidence_target": f"review_gap:{target}",
                "expand_query": True,
            },
            context,
        )
        if not outcome.evidence:
            return []
        numbers = ledger.add(outcome.evidence, context)
        delivery: list[dict[str, JsonValue]] = [
            {"citation": number, "text": item.text}
            for number, item in zip(numbers, outcome.evidence, strict=True)
        ]
        ledger.record_delivery(
            "guardrails-v3-focused-search",
            "asv3_guardrails_v3_repair",
            delivery,
        )
        delivered = ledger.completely_delivered("guardrails-v3-focused-search")
        return numbers if set(numbers).issubset(delivered) else []
    except Exception:
        return []


def _recover_audited_candidate(
    audit: CandidateAudit,
    broker: CorpusBroker,
    ledger: EvidenceLedger,
    context: RunContext,
) -> list[int]:
    for record in audit.records():
        locator = record.hydration_locator
        if record.status != "excluded" or locator is None:
            continue
        try:
            doc = SearchDoc(
                document_id=locator.document_id,
                chunk_ind=locator.chunk_ind,
                semantic_identifier=locator.semantic_identifier,
                blurb=locator.blurb,
                source_type=DocumentSource(locator.source_type),
                boost=0,
                hidden=False,
                metadata={"regulatory_chunk_id": locator.regulatory_chunk_id},
                match_highlights=[],
            )
            evidence = broker.hydrate_search_evidence(doc, context)
            if not evidence:
                continue
            numbers = ledger.add(evidence, context)
            ledger.record_delivery(
                "guardrails-v3-recovery",
                "asv3_guardrails_v3_repair",
                [
                    {"citation": number, "text": item.text}
                    for number, item in zip(numbers, evidence, strict=True)
                ],
            )
            delivered = ledger.completely_delivered("guardrails-v3-recovery")
            if set(numbers).issubset(delivered):
                return numbers
        except Exception:
            continue
    return []
