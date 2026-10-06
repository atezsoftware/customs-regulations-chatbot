"""Host receipts for exact accepted child answers, not semantic approvals."""

from __future__ import annotations

import hashlib
import json
import threading
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome

BodyValidator = Callable[[], ToolOutcome | None]


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class _OriginalBinding(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    citation: int = Field(strict=True, ge=1)
    source_id: str = Field(min_length=1)
    chunk_id: str | None
    text_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    metadata_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    start_char: Literal[0] = 0
    end_char: int = Field(strict=True, ge=1)
    passage_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    complete: Literal[True] = True


class _Receipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    receipt_id: str = Field(pattern=r"^parallel_[a-f0-9]{64}$")
    task_id: str = Field(min_length=1)
    assignment_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    answer: str = Field(min_length=1)
    answer_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    status: Literal[OutcomeStatus.FOUND, OutcomeStatus.PARTIAL]
    model_call_id: str = Field(min_length=1)
    delivery_flow: str | None
    originals: list[_OriginalBinding]
    source_state_hash: str = Field(pattern=r"^[a-f0-9]{64}$")


class _Checkpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    run_id: str
    user_id: str | None
    scope_hash: str
    request_hash: str
    receipts: list[_Receipt]
    integrity: str


class ParallelAnswerReceipts:
    """Only host publication guards may create or validate these records.

    The checksum detects corrupted checkpoints; ownership and the host-only checkpoint
    store are the trust boundary. A receipt proves exact text and delivery, not entailment.
    Source-state projections supplied by the caller must belong to the target task only.
    """

    def __init__(
        self, context: RunContext, request: str, *, user_id: str | None
    ) -> None:
        self.run_id = context.run_id
        self.user_id = user_id
        self.scope_hash = _digest(context.scope)
        self.request_hash = _digest(request)
        self._receipts: dict[str, _Receipt] = {}
        self._lock = threading.RLock()

    def _fence(self, context: RunContext) -> None:
        if (context.run_id, _digest(context.scope)) != (
            self.run_id,
            self.scope_hash,
        ):
            raise ValueError("Parallel answer run or scope changed")

    @staticmethod
    def _check_guard(validate_body: BodyValidator) -> None:
        if validate_body() is not None:
            raise ValueError("Parallel answer publication guard rejected the body")

    @staticmethod
    def _originals(
        answer: str, model_call_id: str, ledger: EvidenceLedger
    ) -> list[_OriginalBinding]:
        citations = extract_citation_numbers(answer)
        delivered = ledger.completely_delivered(model_call_id)
        citable = ledger.citation_mapping()
        bindings: list[_OriginalBinding] = []
        for number in citations:
            item = ledger.get(number)
            if item is None or number not in citable or number not in delivered:
                raise ValueError(
                    "Parallel answer needs its actual complete original delivery"
                )
            deliveries = ledger.inspect(number).get("deliveries")
            if not isinstance(deliveries, list):
                raise ValueError("Parallel answer delivery record is missing")
            matched = False
            for delivery in deliveries:
                if (
                    not isinstance(delivery, dict)
                    or delivery.get("call_id") != model_call_id
                ):
                    continue
                records = delivery.get("records")
                if not isinstance(records, list):
                    continue
                matched = any(
                    isinstance(row, dict)
                    and row.get("citation") == number
                    and row.get("source_id") == item.source_id
                    and row.get("chunk_id") == item.chunk_id
                    and row.get("text_hash") == item.text_hash
                    and type(row.get("start_char")) is int
                    and row.get("start_char") == 0
                    and type(row.get("end_char")) is int
                    and row.get("end_char") == len(item.text)
                    and row.get("passage_hash") == item.text_hash
                    and row.get("complete") is True
                    for row in records
                )
                if matched:
                    break
            if not matched:
                raise ValueError("Parallel answer original delivery identity changed")
            bindings.append(
                _OriginalBinding(
                    citation=number,
                    source_id=item.source_id,
                    chunk_id=item.chunk_id,
                    text_hash=item.text_hash,
                    metadata_hash=_digest(item.metadata),
                    end_char=len(item.text),
                    passage_hash=item.text_hash,
                )
            )
        return bindings

    @staticmethod
    def _receipt_identity(receipt: _Receipt) -> str:
        return "parallel_" + _digest(
            receipt.model_dump(mode="json", exclude={"receipt_id"})
        )

    def seal(
        self,
        child_context: RunContext,
        *,
        assignment: dict[str, JsonValue],
        answer: str,
        status: OutcomeStatus,
        model_call_id: str,
        ledger: EvidenceLedger,
        validate_body: BodyValidator,
        source_state: dict[str, JsonValue] | None = None,
    ) -> str:
        self._fence(child_context)
        task_id = child_context.services.get("task_id")
        if child_context.depth < 1 or not isinstance(task_id, str) or not task_id:
            raise ValueError("Parallel answer needs its child task owner")
        if (
            ("task_id" in assignment and assignment["task_id"] != task_id)
            or (
                child_context.services.get("assignment_id") is not None
                and child_context.services["assignment_id"]
                != assignment.get("question_id")
            )
            or (
                "outcome_ids" in assignment
                and child_context.services.get("task_outcome_ids") is not None
                and child_context.services["task_outcome_ids"]
                != assignment["outcome_ids"]
            )
        ):
            raise ValueError(
                "Parallel answer assignment does not belong to its child task"
            )
        if child_context.services.get("last_model_call_id") != model_call_id:
            raise ValueError("Parallel answer needs its actual last child model call")
        if status not in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}:
            raise ValueError("Parallel answer status is not publishable")
        if not answer.strip() or not model_call_id:
            raise ValueError("Parallel answer and accepted model call must be nonempty")
        self._check_guard(validate_body)
        originals = self._originals(answer, model_call_id, ledger)
        receipt = _Receipt(
            receipt_id="parallel_" + "0" * 64,
            task_id=task_id,
            assignment_hash=_digest(assignment),
            answer=answer,
            answer_hash=hashlib.sha256(answer.encode("utf-8")).hexdigest(),
            status=(
                OutcomeStatus.FOUND
                if status == OutcomeStatus.FOUND
                else OutcomeStatus.PARTIAL
            ),
            model_call_id=model_call_id,
            delivery_flow=ledger.delivery_flow(model_call_id),
            originals=originals,
            source_state_hash=_digest(source_state or {}),
        )
        receipt = receipt.model_copy(
            update={"receipt_id": self._receipt_identity(receipt)}
        )
        if originals:
            ledger.pin_delivery(model_call_id)
        with self._lock:
            self._receipts[receipt.receipt_id] = receipt
        return receipt.receipt_id

    def verify(
        self,
        context: RunContext,
        *,
        receipt_id: str,
        task_id: str,
        assignment: dict[str, JsonValue],
        answer: str,
        status: OutcomeStatus,
        ledger: EvidenceLedger,
        validate_body: BodyValidator,
        source_state: dict[str, JsonValue] | None = None,
    ) -> None:
        self._fence(context)
        with self._lock:
            receipt = self._receipts.get(receipt_id)
        if receipt is None:
            raise ValueError("Parallel answer receipt is unknown")
        if (
            receipt.task_id != task_id
            or receipt.assignment_hash != _digest(assignment)
            or receipt.answer != answer
            or receipt.status != status
            or receipt.source_state_hash != _digest(source_state or {})
        ):
            raise ValueError(
                "Parallel answer body, task, assignment, status or source state changed"
            )
        self._validate_record(receipt, ledger)
        self._check_guard(validate_body)

    def _validate_record(self, receipt: _Receipt, ledger: EvidenceLedger) -> None:
        if (
            receipt.receipt_id != self._receipt_identity(receipt)
            or receipt.answer_hash
            != hashlib.sha256(receipt.answer.encode("utf-8")).hexdigest()
            or receipt.originals
            != self._originals(receipt.answer, receipt.model_call_id, ledger)
            or receipt.delivery_flow != ledger.delivery_flow(receipt.model_call_id)
        ):
            raise ValueError("Parallel answer receipt or canonical original changed")

    def model_call_id(self, receipt_id: str) -> str:
        """Let the host recheck the exact target owner's accepted call."""
        with self._lock:
            receipt = self._receipts.get(receipt_id)
            if receipt is None:
                raise ValueError("Parallel answer receipt is unknown")
            return receipt.model_call_id

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            content: dict[str, JsonValue] = {
                "version": 1,
                "run_id": self.run_id,
                "user_id": self.user_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "receipts": [
                    row.model_dump(mode="json") for row in self._receipts.values()
                ],
            }
        return {**content, "integrity": _digest(content)}

    def restore(
        self,
        snapshot: dict[str, JsonValue],
        context: RunContext,
        request: str,
        ledger: EvidenceLedger,
    ) -> None:
        self._fence(context)
        checkpoint = _Checkpoint.model_validate(snapshot)
        if (
            checkpoint.run_id != self.run_id
            or checkpoint.user_id != self.user_id
            or checkpoint.scope_hash != self.scope_hash
            or checkpoint.request_hash != self.request_hash
            or _digest(request) != self.request_hash
            or checkpoint.integrity
            != _digest(checkpoint.model_dump(mode="json", exclude={"integrity"}))
        ):
            raise ValueError(
                "Parallel answer checkpoint ownership or integrity changed"
            )
        records: dict[str, _Receipt] = {}
        for receipt in checkpoint.receipts:
            if receipt.receipt_id in records:
                raise ValueError("Duplicate parallel answer receipt")
            self._validate_record(receipt, ledger)
            records[receipt.receipt_id] = receipt
        with self._lock:
            if self._receipts:
                raise ValueError("Cannot restore into an active parallel answer store")
            for receipt in records.values():
                if receipt.originals:
                    ledger.pin_delivery(receipt.model_call_id)
            self._receipts = records
