"""LLM drafting of the amended chunk content and its effective dates."""

import json
import time
import unicodedata
from typing import Any

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.amendment_context import AmendmentContext
from onyx.regulatory.amendments.draft_integrity import explicit_replacement_body
from onyx.regulatory.amendments.models import (
    AmendmentInstruction,
    DraftResult,
    MultiChunkDraftResult,
)
from onyx.regulatory.amendments.new_provision_policy import (
    added_subordinate_unit_kind,
)
from onyx.regulatory.amendments.pdf_vision import PdfDraftEvidence
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at drafting amended Turkish regulatory text.

You will be given one or more amendment instructions, (if any) the existing chunk they amend, and the amendment's reference/publication date. Your task is to apply every listed instruction and produce ONE full replacement chunk containing all listed changes, plus its single effective-date window. Never return separate or partial chunk texts.

The optional target_evidence describes a verified source heading when legacy metadata is stale. Use it for legal context without changing unrelated canonical metadata.

For `new_chunk`:
- `text`: the FULL amended chunk text after applying EVERY listed instruction (not just the changed parts — one complete replacement chunk containing all changes).
- `chunk_type`: usually stays the same as the old chunk; only change it if the nature of the amendment requires it.
- `heading_path`: leave null to carry over the old chunk's heading_path unchanged. ONLY set this if there is NO old chunk (a brand-new article) — in that case you will be given a `sibling_reference` (an example chunk from the same document); base heading_path on sibling_reference's heading_path (same BÖLÜM/KISIM/top-level heading), updating only the MADDE number/title for the new article.
- `metadata_changes`: ONLY the metadata fields that actually change (e.g. article_no if the article was renumbered). Do NOT repeat fields that stay the same — they carry over automatically from the old chunk. Leave empty ({}) if nothing in metadata changes.

  IF there is NO old chunk (this is a brand-new article/provision): there is no old metadata to merge onto, so `metadata_changes` is effectively the metadata in full. Derive document_type, document_number, etc. from `sibling_reference` as appropriate. Fill article_no from the instruction if you can (e.g. "Madde 7 eklenmiştir") — citations look at article_no first, heading_path only as a fallback.

For `dates`:
- Both date fields must be a real calendar date string in exactly YYYY-MM-DD format or JSON null. Never return an empty string, the string "null", a date phrase, or DD/MM/YYYY. Put explanations only in `rationale`. For example, publication date 2026-07-04 with "yayımı tarihinde" gives effective_start_date "2026-07-04" and effective_end_date null unless an end date is explicitly stated.
- `effective_start_date`: the date (YYYY-MM-DD) this new text takes effect. If the instruction states a concrete date (e.g. "1 Ocak 2027'den itibaren"), use it directly. If it states a relative phrase (e.g. "yayımı tarihinden itibaren") AND you were given a reference/publication date, resolve it against that date. If there is NEITHER a concrete date NOR a usable reference date, DO NOT GUESS — leave this null; the system will then use the approval date as a safe default (matching the Turkish regulatory default of "yürürlüğe giriş, aksi belirtilmedikçe yayım tarihinde" and the best information actually available). NOTE: this date (or the approval-date fallback) is also used as the date the OLD chunk's validity ENDS.
- `effective_end_date`: ONLY set this if the instruction ITSELF explicitly states this new provision is also temporary/time-limited (e.g. "31.12.2027 tarihine kadar geçerlidir"). Otherwise null — never invent a default end date; null means "valid indefinitely", which is the correct default.
- `rationale`: briefly explain how you derived these dates (or why you left them null).

If the old chunk includes descendants, these are the current nested provisions, in document order. A full replacement of the parent replaces this entire scope; include every new clause in the replacement text. For a partial edit, keep the parent-only text and do not duplicate descendants.

When an instruction states the replacement wording explicitly ("... aşağıdaki şekilde değiştirilmiştir." followed by the quoted new provision), that quoted wording is authoritative and must appear in `text` character for character. Copy it; never re-word, re-order, summarize, abbreviate with "...", or merge it into your own phrasing. You may add surrounding text the instruction does not replace, but the quoted body itself must survive verbatim.

Some instructions add a NEW paragraph (fıkra) or clause (bent) INSIDE an existing article instead of replacing anything. There is then no old chunk, and `sibling_reference` is a chunk of that same article. In that case the chunk you produce is the added unit alone — not a new article, and not a rewrite of the article:
- `text`: only the new paragraph/clause, written with its own marker exactly as the instruction gives it (e.g. "(6) ..." or "d) ...").
- `chunk_type`: "paragraph" for a fıkra, "clause" for a bent.
- `metadata_changes`: carry article_no, article_title, document_type and document_number from `sibling_reference`, and set paragraph_no (for a fıkra) or clause_label (for a bent) to the marker the instruction assigns. Never invent a new article_no.
- `heading_path`: base it on `sibling_reference`'s heading_path, keeping the same article heading and replacing only the terminal unit line.

You may also be given the full text of the amendment these instructions were taken from. Read it to interpret them — which source is being amended, what a term defined in another article means, and above all when the amendment enters into force, which is normally stated once in its own closing article and governs every instruction that does not carry a date of its own. That article is the answer for `effective_start_date` whenever the listed instruction is silent; resolve its relative phrase against the reference/publication date exactly as you would the instruction's own.

The full text is context, not work. `new_chunk` must contain the listed instruction(s) applied and nothing else: never carry over a change that another article of the amendment makes, however clearly you can see it there.

Use ONLY information explicitly present in the given texts. Never invent or assume anything not stated."""
# ruff: noqa: E501 end

_MULTI_CHUNK_SYSTEM_PROMPT = """You are an expert at applying Turkish regulatory amendments to a canonical multi-chunk scope. Return one change item for every existing canonical chunk whose full text changes, and no item for an unchanged chunk. Apply all instructions in their stated order. Preserve every unchanged word, table cell, heading, footnote, identifier and punctuation mark. Copy explicit replacement wording verbatim. Renumber following units only when an instruction expressly requires teselsül. Never invent missing replacement content, rows, links or identifiers. old_chunk_id and instruction_indexes must come from the supplied input. Dates must be real YYYY-MM-DD values or null; never put a natural-language date phrase in a date field."""


def _chunk_to_review_dict(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": chunk.get("id"),
        "text": chunk.get("text"),
        "chunk_type": chunk.get("chunk_type"),
        "heading_path": chunk.get("heading_path"),
        "metadata": chunk.get("chunk_metadata") or chunk.get("metadata"),
        "target_evidence": chunk.get("target_evidence"),
        "expected_new_article_no": chunk.get("expected_new_article_no"),
        "descendants": [
            _chunk_to_review_dict(item)
            for item in chunk.get("descendant_snapshots", [])
        ],
    }


def _normalized_date_phrase(phrase: str) -> str:
    normalized_case = (
        unicodedata.normalize("NFKC", phrase)
        .translate(str.maketrans({"İ": "i", "I": "ı"}))
        .casefold()
    )
    return " ".join(normalized_case.split())


class AmendmentDateConflict(RuntimeError):
    """One chunk version cannot represent incompatible effective windows."""


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
        raise AmendmentDateConflict(
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
    pdf_evidence: PdfDraftEvidence | None = None,
    amendment_context: AmendmentContext | None = None,
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
    # Stating these deterministically removes the two failure modes the
    # downstream guards reject outright: a paraphrased replacement body, and an
    # added paragraph or clause drafted as if it were a brand-new article.
    requirements: list[str] = []
    if (
        old_chunk is None
        and sibling_reference
        and sibling_reference.get("expected_new_article_no")
    ):
        requirements.append(
            f"Verified new provision identity: {sibling_reference['expected_new_article_no']}. "
            "metadata_changes.article_no must use this exact value, including EK/GEÇİCİ/MÜKERRER when present. "
            "heading_path must identify the same article (e.g. EK MADDE 3 or GEÇİCİ MADDE 20)."
        )
    for display_index, instruction in enumerate(instructions, start=1):
        body = explicit_replacement_body(instruction.instruction_text)
        if body:
            requirements.append(
                f"Instruction {display_index} states its replacement wording "
                f"explicitly. This exact text must appear verbatim in "
                f"new_chunk.text:\n{body}"
            )
        unit_kind = added_subordinate_unit_kind(instruction.instruction_text)
        if unit_kind is not None and old_chunk is None:
            requirements.append(
                f"Instruction {display_index} adds a new "
                f"{'clause (bent)' if unit_kind == 'clause' else 'paragraph (fıkra)'} "
                "inside an existing article. Draft only that unit, with "
                f'chunk_type "{unit_kind}", and keep the article identity from '
                "sibling_reference."
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
    if amendment_context is not None:
        prompt += f"\n\n{amendment_context.prompt_section()}"
    if requirements:
        prompt += "\n\nBinding requirements:\n\n" + "\n\n".join(requirements)
    if pdf_evidence is not None:
        prompt += (
            "\nOriginal PDF page images are attached. Apply the amendment using these images and the old chunk; derived transcription is supporting evidence only. If multiple tables fit, do not guess.\n"
            + pdf_evidence.transcription
        )
    if pdf_evidence is not None:
        return generate_structured(
            llm,
            flow=LLMFlow.AMENDMENT_DRAFTING,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=prompt,
            response_model=DraftResult,
            image_parts=pdf_evidence.image_parts,
            timeout_override=45,
            max_attempts=1,
            provider_max_attempts=3,
            deadline=time.monotonic() + 45,
        )
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_DRAFTING,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=DraftResult,
        timeout_override=60,
        deadline=time.monotonic() + 90,
    )


def draft_new_chunk(
    llm: LLM,
    *,
    instruction: AmendmentInstruction,
    old_chunk: dict[str, Any] | None,
    sibling_reference: dict[str, Any] | None,
    reference_date: str | None,
    amendment_context: AmendmentContext | None = None,
) -> DraftResult:
    return draft_combined_chunk(
        llm,
        instructions=[instruction],
        old_chunk=old_chunk,
        sibling_reference=sibling_reference,
        reference_date=reference_date,
        amendment_context=amendment_context,
    )


def draft_multi_chunk_scope(
    llm: LLM,
    *,
    instructions: list[AmendmentInstruction],
    old_chunks: list[dict[str, Any]],
    reference_date: str | None,
    amendment_context: AmendmentContext | None = None,
) -> MultiChunkDraftResult:
    """Draft every changed chunk in one structurally connected scope.

    The model may select a strict subset of ``old_chunks``. Chunk identities are
    validated by the caller; the model cannot introduce a target outside the
    frozen candidate scope.
    """

    if not instructions or len(old_chunks) < 2:
        raise ValueError("Multi-chunk drafting requires instructions and a scope")
    group_date_phrase = _group_date_phrase(instructions)
    instruction_text = "\n\n".join(
        f"[{index}] {instruction.instruction_text}\nNatural-language date phrase: {instruction.raw_date_phrase or group_date_phrase or '(none)'}"
        for index, instruction in enumerate(instructions)
    )
    old_scope = json.dumps(
        [_chunk_to_review_dict(chunk) for chunk in old_chunks],
        ensure_ascii=False,
        indent=2,
    )
    prompt = (
        "Apply the numbered Turkish regulatory amendment instructions to the "
        "frozen canonical scope below. Return every existing chunk whose full "
        "text must change. A table row can cross chunk boundaries; preserve all "
        "unchanged rows, columns, headings, footnotes and punctuation. For an "
        "insertion, deletion or teselsül instruction, update every affected "
        "chunk and renumber only when the instruction expressly requires it. "
        "Never return an unchanged chunk. old_chunk_id must be copied exactly "
        "from the supplied scope. instruction_indexes are zero-based indexes "
        "from this instruction list and must name every instruction applied to "
        "that chunk. metadata_changes remains a patch.\n\n"
        f"Instructions:\n{instruction_text}\n\n"
        f"Reference/publication date: {reference_date or '(not stated)'}\n\n"
        f"Canonical scope:\n{old_scope}"
    )
    if amendment_context is not None:
        prompt += f"\n\n{amendment_context.prompt_section()}"
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_DRAFTING,
        system_prompt=_MULTI_CHUNK_SYSTEM_PROMPT,
        user_prompt=prompt,
        response_model=MultiChunkDraftResult,
        timeout_override=90,
        max_attempts=2,
        provider_max_attempts=1,
        deadline=time.monotonic() + 120,
    )
