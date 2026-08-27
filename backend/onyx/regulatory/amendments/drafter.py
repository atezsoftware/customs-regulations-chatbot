"""LLM drafting of the amended chunk content and its effective dates."""

import json
import unicodedata
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

You will be given one or more amendment instructions, (if any) the existing chunk they amend, and the amendment's reference/publication date. Your task is to apply every listed instruction and produce ONE full replacement chunk containing all listed changes, plus its single effective-date window. Never return separate or partial chunk texts.

For `new_chunk`:
- `text`: the FULL amended chunk text after applying EVERY listed instruction (not just the changed parts — one complete replacement chunk containing all changes).
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


def _normalized_date_phrase(phrase: str) -> str:
    normalized_case = (
        unicodedata.normalize("NFKC", phrase)
        .translate(str.maketrans({"İ": "i", "I": "ı"}))
        .casefold()
    )
    return " ".join(normalized_case.split())


def _group_date_phrase(instructions: list[AmendmentInstruction]) -> str | None:
    phrases_by_normalized_value: dict[str, str] = {}
    for instruction in instructions:
        if instruction.raw_date_phrase is None:
            continue
        display_phrase = " ".join(instruction.raw_date_phrase.split())
        phrases_by_normalized_value.setdefault(
            _normalized_date_phrase(display_phrase), display_phrase
        )
    if len(phrases_by_normalized_value) > 1:
        phrases = ", ".join(
            repr(value) for value in phrases_by_normalized_value.values()
        )
        raise RuntimeError(
            "Same-target amendment instructions contain incompatible explicit "
            f"effective-date phrases: {phrases}"
        )
    return next(iter(phrases_by_normalized_value.values()), None)


def draft_combined_chunk(
    llm: LLM,
    *,
    instructions: list[AmendmentInstruction],
    old_chunk: dict[str, Any] | None,
    sibling_reference: dict[str, Any] | None,
    reference_date: str | None,
) -> DraftResult:
    if not instructions:
        raise ValueError(
            "Combined amendment drafting requires at least one instruction"
        )
    group_date_phrase = _group_date_phrase(instructions)
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
    instruction_sections = []
    for display_index, instruction in enumerate(instructions, start=1):
        own_date_phrase = (
            instruction.raw_date_phrase.strip()
            if instruction.raw_date_phrase is not None
            else (
                f"(none; inherits group phrase: {group_date_phrase})"
                if group_date_phrase is not None
                else "(none)"
            )
        )
        instruction_sections.append(
            f"[Instruction {display_index}]\n"
            f"Text: {instruction.instruction_text}\n"
            f"Natural-language date phrase: {own_date_phrase}"
        )
    prompt = (
        "Amendment instructions, in application order:\n\n"
        f"{'\n\n'.join(instruction_sections)}\n\n"
        f"Reference/publication date: {reference_date or '(not stated)'}\n\n"
        f"Shared effective-date phrase for this chunk version: "
        f"{group_date_phrase or '(none)'}\n\n"
        f"Old chunk:\n{old_chunk_json}\n\n"
        f"Sibling chunk from the same document (only used when there is no old "
        f"chunk, for heading_path/metadata convention):\n{sibling_json}\n\n"
        "Return one full replacement chunk containing every listed change."
    )
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_DRAFTING,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=DraftResult,
    )


def draft_new_chunk(
    llm: LLM,
    *,
    instruction: AmendmentInstruction,
    old_chunk: dict[str, Any] | None,
    sibling_reference: dict[str, Any] | None,
    reference_date: str | None,
) -> DraftResult:
    return draft_combined_chunk(
        llm,
        instructions=[instruction],
        old_chunk=old_chunk,
        sibling_reference=sibling_reference,
        reference_date=reference_date,
    )
