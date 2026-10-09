"""One bounded repair cycle for an already-reviewed Guardrails v3 answer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

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


class GuardrailsV3Finalization(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    answer: str
    action: str
    repair_applied: bool = False
    recheck_completed: bool = False
    failure_reason: str | None = None


def finalize_guardrails_v3(
    *,
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
    if repair_llm is None or remaining_seconds < 10:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="disclose_gap",
            failure_reason="repair_unavailable",
        )
    citations = list(ledger.citation_numbers())
    delivered = [number for number in citations if ledger.inspect(number)["deliveries"]]
    action = "patch_delivered_evidence"
    if not delivered and candidate_audit is not None and broker is not None:
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
                            "allowed_citations": delivered,
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
    if set(extract_citation_numbers(repaired)) - set(delivered):
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
