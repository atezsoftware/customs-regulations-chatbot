"""Typed, vector-free state needed to resume amendment matching."""

import json
from hashlib import sha256
from typing import Any

from pydantic import BaseModel

from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.amendments.structural_target import (
    normalize_appendix_label,
    parse_amendment_structural_target,
)

MATCH_CONTRACT_VERSION = 2


class MatchEvidence(BaseModel):
    scope_sha256: str
    candidates: dict[str, str]


class MatchedInstruction(BaseModel):
    evidence: MatchEvidence | None = None
    instruction_index: int
    instruction: AmendmentInstruction
    candidates: list[CandidateChunk]
    match: MatchResult

    @property
    def group_key(self) -> tuple[str, str | int]:
        old_chunk_id = self.match.old_chunk_id
        target = parse_amendment_structural_target(self.instruction)
        if old_chunk_id is not None and target and target.appendix_label:
            file_id = next(
                (
                    candidate.user_file_id
                    for candidate in self.candidates
                    if candidate.chunk_id == old_chunk_id
                ),
                old_chunk_id,
            )
            return (
                "appendix",
                f"{file_id}:{normalize_appendix_label(target.appendix_label)}",
            )
        return (
            ("existing", old_chunk_id)
            if old_chunk_id
            else ("new", self.instruction_index)
        )


def match_input_sha256(**inputs: Any) -> str:
    payload = {"contract_version": MATCH_CONTRACT_VERSION, **inputs}
    return sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
