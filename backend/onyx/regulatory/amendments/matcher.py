"""LLM confirmation of which hybrid-search candidate an amendment instruction targets."""

import json

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at matching Turkish regulatory amendment instructions to the existing text they amend.

You will be given one amendment instruction and a list of candidate existing text chunks. The candidates were found via fuzzy text/heading matching and exact article-number matching — heading_path is only a best-effort reconstruction from document formatting, so it can be unreliable. Read each candidate's actual TEXT carefully; do not rely on the heading or the scores alone.

Your task: decide which candidate (if any) this instruction amends.

- If exactly one candidate is clearly the existing article/provision this instruction changes, set `old_chunk_id` to that candidate's id.
- CRITICAL — only set `old_chunk_id` to null if the instruction EXPLICITLY adds a brand-new article/provision (e.g. "... eklenmiştir", "yeni madde"). If the instruction is amending, replacing, clarifying, or repealing something and a matching candidate exists, you MUST select it — never treat an ordinary amendment as a new addition just because the wording differs from the candidate.
- Set `confidence` to a 0.0-1.0 score.
- Set `rationale` to a brief explanation.

Only ever use an id from the given candidates. Never invent an id."""
# ruff: noqa: E501 end


def _format_candidates(candidates: list[CandidateChunk]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            json.dumps(
                {
                    "id": candidate.chunk_id,
                    "source_name": candidate.source_name,
                    "text": candidate.text,
                    "metadata": candidate.metadata,
                    "scores": {
                        "text_similarity": round(candidate.text_trgm_score, 3),
                        "heading_similarity": round(candidate.heading_trgm_score, 3),
                        "structured_match": candidate.structured_match,
                    },
                },
                ensure_ascii=False,
            )
        )
    return "\n\n".join(blocks)


def confirm_match(
    llm: LLM,
    *,
    instruction: AmendmentInstruction,
    candidates: list[CandidateChunk],
) -> MatchResult:
    prompt = (
        f"Amendment instruction:\n{instruction.instruction_text}\n\n"
        f"Article reference: {instruction.article_reference or '(not stated)'}\n\n"
        f"Target source: {instruction.target_source or '(not stated)'}\n\n"
        f"Candidate chunks:\n{_format_candidates(candidates)}"
    )
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_MATCH_CONFIRMATION,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=MatchResult,
    )
