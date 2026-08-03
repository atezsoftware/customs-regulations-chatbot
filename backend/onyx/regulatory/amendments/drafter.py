"""LLM drafting of the amended chunk content and its effective dates."""

import json
from typing import Any

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    DraftResult,
)
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at drafting amended Turkish regulatory text.

You will be given an amendment instruction, (if any) the existing chunk it amends, and the amendment's reference/publication date. Your task is to produce the new chunk content and its effective dates.

For `new_chunk`:
- `text`: the FULL amended chunk text per the instruction (not just the changed part — the whole chunk, with the amendment applied).
- `chunk_type`: usually stays the same as the old chunk; only change it if the nature of the amendment requires it.
- `heading_path`: leave null to carry over the old chunk's heading_path unchanged. ONLY set this if there is NO old chunk (a brand-new article) — in that case you will be given a `sibling_reference` (an example chunk from the same document); base heading_path on sibling_reference's heading_path (same BÖLÜM/KISIM/top-level heading), updating only the MADDE number/title for the new article.
- `metadata_changes`: ONLY the metadata fields that actually change (e.g. article_no if the article was renumbered). Do NOT repeat fields that stay the same — they carry over automatically from the old chunk. Leave empty ({}) if nothing in metadata changes.

  IF there is NO old chunk (this is a brand-new article/provision): there is no old metadata to merge onto, so `metadata_changes` is effectively the metadata in full. Derive document_type, document_number, etc. from `sibling_reference` as appropriate. Fill article_no from the instruction if you can (e.g. "Madde 7 eklenmiştir") — citations look at article_no first, heading_path only as a fallback.

For `dates`:
- `effective_start_date`: the date (YYYY-MM-DD) this new text takes effect. If the instruction states a concrete date (e.g. "1 Ocak 2027'den itibaren"), use it directly. If it states a relative phrase (e.g. "yayımı tarihinden itibaren") AND you were given a reference/publication date, resolve it against that date. If there is NEITHER a concrete date NOR a usable reference date, DO NOT GUESS — leave this null; the system will then use the approval date as a safe default (matching the Turkish regulatory default of "yürürlüğe giriş, aksi belirtilmedikçe yayım tarihinde" and the best information actually available). NOTE: this date (or the approval-date fallback) is also used as the date the OLD chunk's validity ENDS.
- `effective_end_date`: ONLY set this if the instruction ITSELF explicitly states this new provision is also temporary/time-limited (e.g. "31.12.2027 tarihine kadar geçerlidir"). Otherwise null — never invent a default end date; null means "valid indefinitely", which is the correct default.
- `rationale`: briefly explain how you derived these dates (or why you left them null).

Use ONLY information explicitly present in the given texts. Never invent or assume anything not stated."""
# ruff: noqa: E501 end


def _chunk_to_review_dict(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk.get("id"),
        "text": chunk.get("text"),
        "chunk_type": chunk.get("chunk_type"),
        "heading_path": chunk.get("heading_path"),
        "metadata": chunk.get("chunk_metadata") or chunk.get("metadata"),
    }


def draft_new_chunk(
    llm: LLM,
    *,
    instruction: AmendmentInstruction,
    old_chunk: dict[str, Any] | None,
    sibling_reference: dict[str, Any] | None,
    reference_date: str | None,
) -> DraftResult:
    old_chunk_json = (
        json.dumps(_chunk_to_review_dict(old_chunk), ensure_ascii=False, indent=2)
        if old_chunk is not None
        else "(none — this adds a new article/provision)"
    )
    sibling_json = (
        json.dumps(
            _chunk_to_review_dict(sibling_reference), ensure_ascii=False, indent=2
        )
        if sibling_reference is not None
        else "(none)"
    )
    prompt = (
        f"Amendment instruction:\n{instruction.instruction_text}\n\n"
        f"Reference/publication date: {reference_date or '(not stated)'}\n\n"
        f"Natural-language date phrase: {instruction.raw_date_phrase or '(none)'}\n\n"
        f"Old chunk:\n{old_chunk_json}\n\n"
        f"Sibling chunk from the same document (only used when there is no old "
        f"chunk, for heading_path/metadata convention):\n{sibling_json}"
    )
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_DRAFTING,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=DraftResult,
    )
