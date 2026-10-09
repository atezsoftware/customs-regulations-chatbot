"""One bounded repair cycle for an already-reviewed Guardrails v3 answer."""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.guardrails_v3 import GuardrailsV3ReviewOutcome
from onyx.asv3.models import RunContext
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
            action="patch_delivered_evidence",
            failure_reason="repair_failed",
        )
    if not repaired or repaired == candidate_answer:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="patch_delivered_evidence",
            failure_reason="repair_unchanged",
        )
    if set(extract_citation_numbers(repaired)) - set(delivered):
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="patch_delivered_evidence",
            failure_reason="invalid_repair_citations",
        )
    review = recheck(repaired, initial_review.findings)
    if not review.review_completed or review.findings:
        return GuardrailsV3Finalization(
            answer=candidate_answer,
            action="patch_delivered_evidence",
            failure_reason="recheck_failed",
        )
    return GuardrailsV3Finalization(
        answer=repaired,
        action="patch_delivered_evidence",
        repair_applied=True,
        recheck_completed=True,
    )
