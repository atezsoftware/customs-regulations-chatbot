"""Amendment (Resmi Gazete update) analysis pipeline for indexed chunks.

Given raw pasted amendment text, produces reviewable {old_chunk, new_chunk}
proposals without writing anything to storage — persistence and approval
happen at the REST layer (`fs_explorer_api.server`), which is what actually
calls into `PostgresStorage`'s amendment batch/proposal methods.
"""

from .models import (
    AnalyzeAmendmentRequest,
    AnalyzeAmendmentResponse,
    ApproveProposalsRequest,
    ApproveProposalsResponse,
    DecideProposalRequest,
    ProposalFailure,
    ProposalRecord,
)
from .pipeline import analyze_amendment, flag_duplicate_targets_in_records

__all__ = [
    "AnalyzeAmendmentRequest",
    "AnalyzeAmendmentResponse",
    "ApproveProposalsRequest",
    "ApproveProposalsResponse",
    "DecideProposalRequest",
    "ProposalFailure",
    "ProposalRecord",
    "analyze_amendment",
    "flag_duplicate_targets_in_records",
]
