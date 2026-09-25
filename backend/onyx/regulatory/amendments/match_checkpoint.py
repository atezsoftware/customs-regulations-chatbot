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

MATCH_CONTRACT_VERSION = 3


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

    def draft_group_key(self, heading_articles: set[str]) -> tuple[str, str | int]:
        """Join heading dependencies only inside a verified physical source."""
        from onyx.regulatory.amendments.target_scope import validated_addition_anchor

        target = parse_amendment_structural_target(self.instruction)
        if (
            target is None
            or target.appendix_label
            or target.article_no not in heading_articles
        ):
            return self.group_key
        candidate = next(
            (
                item
                for item in self.candidates
                if item.chunk_id == self.match.old_chunk_id
            ),
            None,
        )
        if self.match.old_chunk_id is None:
            candidate = validated_addition_anchor(self.instruction, self.candidates)
        if candidate is None:
            return self.group_key
        return "article_heading", f"{candidate.user_file_id}:{target.article_no}"

    def draft_lane_key(self) -> tuple[str, str]:
        """Keep overlapping article/annex drafts in one serial lane."""
        from onyx.regulatory.amendments.target_scope import validated_addition_anchor

        candidate = next(
            (
                item
                for item in self.candidates
                if item.chunk_id == self.match.old_chunk_id
            ),
            None,
        )
        if self.match.old_chunk_id is None:
            candidate = validated_addition_anchor(self.instruction, self.candidates)
        if candidate is None:
            return "unverified", "source"
        target = parse_amendment_structural_target(self.instruction)
        if target is None:
            return candidate.user_file_id, "source"
        if target.appendix_label:
            return (
                candidate.user_file_id,
                f"appendix:{normalize_appendix_label(target.appendix_label)}",
            )
        return (
            candidate.user_file_id,
            f"article:{target.article_no}" if target.article_no else "source",
        )


def match_input_sha256(**inputs: Any) -> str:
    payload = {"contract_version": MATCH_CONTRACT_VERSION, **inputs}
    return sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode()
    ).hexdigest()
