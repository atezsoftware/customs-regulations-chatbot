"""Run-local, source-bound material review for the isolated Fast Tek publisher."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import OrderedDict
from collections.abc import Callable
from threading import RLock
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome

FAST_MATERIAL_REVIEW_STOP_SERVICE = "asv3_fast_material_review_stop"


class FastMaterialFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    kind: Literal[
        "condition",
        "exception",
        "deadline",
        "proof",
        "scope",
        "remedy",
        "contradiction",
        "omission",
    ]
    detail: str = Field(min_length=1)
    applicability: str = Field(min_length=1)
    citation: int = Field(ge=1)
    source_quote: str = Field(min_length=1)
    answer_quote: str


class FastMaterialReviewResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["complete", "uncertain"]
    candidate_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_set_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    examined_citations: list[int]
    findings: list[FastMaterialFinding] = Field(max_length=64)


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


class FastMaterialReview:
    """Cache semantic results only; access and publication fences stay with the host."""

    def __init__(
        self,
        context: RunContext,
        ledger: EvidenceLedger,
        assess: Callable[[dict[str, JsonValue]], FastMaterialReviewResult],
    ) -> None:
        self.context, self.ledger, self.assess = context, ledger, assess
        self._identity = (context.run_id, _digest(context.scope))
        self._cache: OrderedDict[str, ToolOutcome | Exception | None] = OrderedDict()
        self._reviews = 0
        self._lock = RLock()

    @staticmethod
    def _gap(
        status: str,
        candidate_hash: str,
        source_hash: str,
        findings: list[FastMaterialFinding] | None = None,
    ) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary=(
                "Correct the source-bound material conditions before publication."
                if status == "material_findings"
                else "The final material-condition review is unverified; do not publish as complete."
            ),
            data={
                "asv3_fast_material_review": {
                    "status": status,
                    "candidate_sha256": candidate_hash,
                    "source_set_sha256": source_hash,
                    "findings": [row.model_dump(mode="json") for row in findings or []],
                }
            },
        )

    def publication_gap(
        self, answer: str, writer_call_id: str | None, question: str
    ) -> ToolOutcome | None:
        self.context.check_active()
        if self._identity != (self.context.run_id, _digest(self.context.scope)):
            raise PermissionError("Material review run or scope changed")
        candidate_hash = hashlib.sha256(answer.encode()).hexdigest()
        numbers = sorted(self.ledger.completely_delivered(writer_call_id or ""))
        originals: list[JsonValue] = []
        source_text: dict[int, str] = {}
        for number in numbers:
            item = self.ledger.get(number)
            if (
                item is None
                or item.search_doc is None
                or not item.text.strip()
                or hashlib.sha256(item.text.encode()).hexdigest() != item.text_hash
            ):
                return self._gap("invalid_original", candidate_hash, "")
            source_text[number] = item.text
            originals.append(
                {
                    "citation": number,
                    "source_id": item.source_id,
                    "chunk_id": item.chunk_id,
                    "text_hash": item.text_hash,
                    "text": item.text,
                    "start_char": 0,
                    "end_char": len(item.text),
                    "metadata": copy.deepcopy(item.metadata),
                    "search_metadata": copy.deepcopy(item.search_doc.metadata),
                }
            )
        source_hash = _digest(originals)
        if not question.strip() or not answer.strip() or not numbers:
            return self._gap("missing_review_input", candidate_hash, source_hash)
        key = _digest([*self._identity, question, candidate_hash, source_hash])
        with self._lock:
            self.context.check_active()
            if key in self._cache:
                cached = self._cache[key]
                self._cache.move_to_end(key)
                if isinstance(cached, Exception):
                    raise cached
                return cached.model_copy(deep=True) if cached is not None else None
            if self._reviews >= 2:
                return self._gap("review_limit", candidate_hash, source_hash)
            self._reviews += 1
            payload: dict[str, JsonValue] = {
                "question": question,
                "candidate": answer,
                "candidate_sha256": candidate_hash,
                "source_set_sha256": source_hash,
                "original_evidence": originals,
                "required_evidence_numbers": numbers,
            }
            try:
                result = self.assess(payload)
                self.context.check_active()
                valid = (
                    result.candidate_sha256 == candidate_hash
                    and result.source_set_sha256 == source_hash
                    and len(result.examined_citations) == len(numbers)
                    and set(result.examined_citations) == set(numbers)
                    and all(
                        row.citation in source_text
                        and bool(row.source_quote.strip())
                        and row.source_quote in source_text[row.citation]
                        and bool(row.detail.strip())
                        and bool(row.applicability.strip())
                        and (
                            bool(row.answer_quote.strip())
                            and row.answer_quote in answer
                            or row.kind == "omission"
                            and row.answer_quote == ""
                        )
                        for row in result.findings
                    )
                )
                outcome = (
                    self._gap("invalid_review_binding", candidate_hash, source_hash)
                    if not valid
                    else self._gap("uncertain", candidate_hash, source_hash)
                    if result.status != "complete"
                    else self._gap(
                        "material_findings",
                        candidate_hash,
                        source_hash,
                        result.findings,
                    )
                    if result.findings
                    else None
                )
            except Exception as error:
                self._cache[key] = error
                raise
            self._cache[key] = outcome
            while len(self._cache) > 8:
                self._cache.popitem(last=False)
            return outcome.model_copy(deep=True) if outcome is not None else None
