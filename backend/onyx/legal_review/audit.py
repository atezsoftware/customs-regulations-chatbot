"""Trace retrieval candidates without changing selection or evidence capacity."""

from collections.abc import Mapping, Sequence
from threading import RLock

from pydantic import JsonValue

from onyx.asv3.candidate_audit import CandidateAudit, CandidateAuditRecord
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from onyx.legal_review.models import DraftAnswer, RequirementRecord


class LegalReviewCandidateAudit(CandidateAudit):
    def __init__(self, context: RunContext, *, request: str) -> None:
        super().__init__(context, request=request)
        self._audit_lock = RLock()

    def record(self, record: CandidateAuditRecord) -> None:
        self._context.check_active()
        checked = CandidateAuditRecord.model_validate(record)
        # The workflow deadline bounds this metadata ledger. Audit capacity must
        # not silently drop later searches or alter legal evidence acquisition.
        with self._audit_lock:
            self._records[checked.identity] = checked

    def records(self) -> tuple[CandidateAuditRecord, ...]:
        with self._audit_lock:
            return tuple(self._records.values())


def evidence_journey(
    ledger: EvidenceLedger,
    receipts: Sequence[dict[str, JsonValue]],
    assessments: Sequence[dict[str, JsonValue]],
    requirements: Mapping[str, RequirementRecord],
    draft: DraftAnswer | None,
) -> list[dict[str, JsonValue]]:
    """Join canonical identities to acquisition, reading, findings and literal draft use."""
    result: list[dict[str, JsonValue]] = []
    for citation in ledger.citation_numbers():
        item = ledger.get(citation)
        assert item is not None
        source_assessments = [
            row for row in assessments if row.get("source_id") == item.source_id
        ]
        result.append(
            {
                "citation": citation,
                "source_id": item.source_id,
                "chunk_id": item.chunk_id,
                "text_hash": item.text_hash,
                "receipt_ids": [
                    row.get("call_id")
                    for row in receipts
                    if isinstance(ids := row.get("evidence_ids"), list)
                    and citation in ids
                ],
                "reading_dispositions": [
                    row.get("disposition") for row in source_assessments
                ]
                or ["open"],
                "requirement_ids": [
                    identity
                    for identity, row in requirements.items()
                    if any(support.citation == citation for support in row.supports)
                ],
                "draft_claim_ids": [
                    claim.claim_id
                    for claim in (draft.claims if draft else [])
                    if any(support.citation == citation for support in claim.supports)
                ],
            }
        )
    return result
