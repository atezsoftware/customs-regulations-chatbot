"""LLM confirmation of which hybrid-search candidate an amendment instruction targets."""

import json
import re
import time

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.amendment_context import AmendmentContext
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.ranker import CandidateChunk
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at matching Turkish regulatory amendment instructions to the existing text they amend.

Your purpose is to find what is changing: identify the existing provision an instruction affects so the change can be applied to it.

Instructions are routinely imperfect — paraphrased, abbreviated, summarized by a user, damaged by OCR, missing an article number or the source name. Resolve them anyway: a provision's identity is carried by several independent signals at once (structural position, subject matter, quoted wording, source instrument). Differing phrasing is the normal case, never a reason to decline: the instruction describes the NEW text while the candidate holds the OLD text.

You will be given one amendment instruction and candidate existing chunks from several retrieval lanes. A candidate with `structured_match: true` was found by an exact structural lookup of the article/paragraph/clause the instruction names — the strongest signal available, so weigh it heavily even when its wording looks unrelated. `resolved_article_no` and `scope_evidence`, when supplied, identify an article from its actual opening heading and bounded canonical continuation, separately from legacy metadata. Use that evidence to interpret a table split from its heading. A source match alone is not an article match. `source_ambiguous: true` means other verified source versions may exist beyond this bounded candidate list; structural metadata alone cannot distinguish them. Require explicit version/date or source-text evidence, otherwise return not_found. `structure_conflict`, when present, means the complete source contradicts the scalar metadata or contains several pieces claiming this unit. Do not resolve that conflict from labels or list order; use the existing body and bounded source evidence, otherwise return not_found. `heading_path` is a best-effort reconstruction from document formatting and can be unreliable; read each candidate's actual TEXT.

Your task: decide which candidate (if any) this instruction amends.

- If a candidate is the existing provision this instruction changes, set `old_chunk_id` to its id. Prefer the most specific correct unit: for an amendment to one paragraph or clause, choose that paragraph or clause rather than the whole article.
- Set `outcome` to `matched` for a verified existing target. Set `outcome` to `not_found` and `old_chunk_id` to null when the source, target unit, or old wording cannot be supported. Explain which evidence is missing; never select a citing law instead of the named law.
- Set `outcome` to `new_provision` and `old_chunk_id` to null when the instruction adds text that does not exist yet: a brand-new article, or a new paragraph/clause added inside an existing article ("aşağıdaki fıkra eklenmiştir", "aşağıdaki bent eklenmiş ve diğer bentler buna göre teselsül ettirilmiştir"). If the instruction amends, replaces, clarifies, or repeals something and a matching candidate exists, you MUST select it.
- Set `confidence` to a 0.0-1.0 score and `rationale` to a brief explanation naming the signals you used.

You may also be given the full text of the amendment the instruction was taken from. Read it to understand the instruction — which source "Aynı Tebliğ" names, what a term defined in another article means, which article a cross-reference points at. It is context only: decide the target of the ONE instruction you were given, never of another article you read there.

Only ever use an id from the given candidates. Never invent an id."""
# ruff: noqa: E501 end


# An annex table chunk can run to tens of thousands of characters. Identity is
# decided from the opening of a candidate plus its structural metadata, so a
# generous bound keeps one oversized candidate from crowding out every other one.
_MAX_CANDIDATE_TEXT_CHARS = 6000


def _bounded_candidate_text(text: str, instruction_text: str = "") -> str:
    if len(text) <= _MAX_CANDIDATE_TEXT_CHARS:
        return text
    # Keep the title and evidence around quoted old wording even in long tables.
    windows: list[tuple[int, int]] = [(0, 2000)]
    for phrase in re.findall(r'[“"]([^”"]+)[”"]', instruction_text):
        if len(phrase) < 4:
            continue
        offset = text.find(phrase)
        if offset < 0 or any(start <= offset < end for start, end in windows):
            continue
        start = max(0, offset - 500)
        windows.append((start, min(len(text), max(offset + len(phrase), start + 1800))))
        if len(windows) == 3:
            break
    if len(windows) == 1:
        windows = [(0, _MAX_CANDIDATE_TEXT_CHARS)]
    remaining = _MAX_CANDIDATE_TEXT_CHARS
    excerpts = []
    for start, end in sorted(windows):
        end = min(end, start + remaining)
        excerpts.append(f"[canonical text offset {start}:{end}]\n{text[start:end]}")
        remaining -= end - start
    return "\n".join(excerpts) + "\n[other text omitted for matching]"


def _format_candidates(
    candidates: list[CandidateChunk], instruction_text: str = ""
) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            json.dumps(
                {
                    "id": candidate.chunk_id,
                    "source_name": candidate.source_name,
                    "text": _bounded_candidate_text(candidate.text, instruction_text),
                    "metadata": candidate.metadata,
                    "structured_match": candidate.structured_match,
                    "source_verified": candidate.source_verified,
                    "source_ambiguous": candidate.source_ambiguous,
                    "structure_conflict": candidate.structure_conflict,
                    "resolved_article_no": candidate.resolved_article_no,
                    "scope_evidence": candidate.scope_evidence,
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
    amendment_context: AmendmentContext | None = None,
) -> MatchResult:
    prompt = (
        f"Amendment instruction:\n{instruction.instruction_text}\n\n"
        f"Article reference: {instruction.article_reference or '(not stated)'}\n\n"
        f"Target source: {instruction.target_source or '(not stated)'}\n\n"
        f"Candidate chunks:\n{_format_candidates(candidates, instruction.instruction_text)}"
    )
    if amendment_context is not None:
        prompt += f"\n\n{amendment_context.prompt_section()}"
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_MATCH_CONFIRMATION,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=MatchResult,
        timeout_override=60,
        deadline=time.monotonic() + 90,
    )
