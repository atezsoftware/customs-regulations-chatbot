"""LLM confirmation of which hybrid-search candidate an amendment instruction targets."""

import json
import time

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at matching Turkish regulatory amendment instructions to the existing text they amend.

Your purpose is to find what is changing. An amendment instruction describes a change to a provision that already exists in the corpus; your job is to identify that provision so the change can be applied to it. Locating the affected provision is the goal — not judging how well the instruction is written.

Instructions are frequently imperfect. They may be paraphrased, abbreviated, summarized by a user, carried through OCR with broken characters, missing an article number, missing the source name, or written in a different register than the indexed text. A well-built retrieval system still resolves these, because the identity of a provision is carried by several independent signals at once — its structural position (article/paragraph/clause number), its subject matter, the wording it quotes, and its source instrument. Use every signal available and reason about which provision they converge on. Never decline a match merely because the instruction's phrasing differs from the candidate's phrasing; differing phrasing is the normal case, since the instruction usually describes the NEW text while the candidate holds the OLD text.

You will be given one amendment instruction and a list of candidate existing text chunks. Candidates come from several independent retrieval lanes. A candidate with `structured_match: true` was found by an exact structural lookup of the article/paragraph/clause the instruction names — that is the strongest available signal, so weigh it heavily even when its wording looks unrelated to the instruction. `heading_path` is only a best-effort reconstruction from document formatting and can be unreliable; read each candidate's actual TEXT.

Your task: decide which candidate (if any) this instruction amends.

- If one candidate is the existing provision this instruction changes, set `old_chunk_id` to that candidate's id. Prefer the most specific correct unit: if the instruction amends one paragraph or clause, choose that paragraph or clause rather than the whole article.
- Set `old_chunk_id` to null ONLY when the instruction adds text that does not exist yet — a brand-new article, or a new paragraph/clause added inside an existing article ("aşağıdaki fıkra eklenmiştir", "aşağıdaki bent eklenmiş ve diğer bentler buna göre teselsül ettirilmiştir"). For an added paragraph or clause, still return null: the unit being added has no existing chunk, even though its article does.
- If the instruction amends, replaces, clarifies, or repeals something and a matching candidate exists, you MUST select it. Never treat an ordinary amendment as an addition just because the wording differs.
- Set `confidence` to a 0.0-1.0 score.
- Set `rationale` to a brief explanation naming the signals you relied on.

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
                    "structured_match": candidate.structured_match,
                    "source_name_similarity": round(candidate.source_score, 3),
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
        timeout_override=45,
        deadline=time.monotonic() + 60,
    )
