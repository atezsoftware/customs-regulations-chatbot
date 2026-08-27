"""LLM segmentation of a pasted amendment text into atomic instructions."""

import re

from pydantic import Field

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import AmendmentInstruction, SegmentationResult
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at analyzing Turkish regulatory amendment texts (Resmi Gazete değişiklik/protokol metinleri).

You will be given a pasted amendment/update text. Your job:

1. Split the text into atomic amendment instructions, each affecting exactly ONE independently retrievable legal rule or provision.
   Common patterns in Turkish regulatory amendments:
   - "MADDE N ... aşağıdaki şekilde değiştirilmiştir." (article N is changed to read as follows)
   - "... aşağıdaki fıkra/bent eklenmiştir." (the following paragraph/clause is added)
   - "... yürürlükten kaldırılmıştır." (repealed)
   Put the exact text needed to fully understand each change (the instruction sentence plus any new article text) into `instruction_text`.
   The input may instead be a concise natural-language summary in any language. Such a summary is still an amendment request. Split coordinated changes into separate instructions whenever they concern different operative rules, even when no article number is given. Do not leave multiple independently searchable changes in one instruction.
   CRITICAL: Never keep a changed numerical threshold/cap and a distinct exception, fallback, or alternative procedure that applies beyond that threshold in the same instruction. Emit one instruction for the threshold and another for the beyond-threshold procedure. Each instruction must be answerable by one focused search question and target one existing chunk.

2. CRITICAL — only treat an instruction as adding a brand-new article/provision if the text EXPLICITLY says so (e.g. "... eklenmiştir", "yeni bir madde olarak", "MADDE N eklenmiştir"). If the text is merely amending, replacing, or clarifying an existing article, it is NOT a new article — leave `article_reference` pointing at the existing article being changed. Never infer a new-article addition from ambiguous or silent phrasing.

3. For each instruction, fill `article_reference` with the relevant article/paragraph reference (e.g. "Madde 3", "Madde 5 fıkra 2") if stated; otherwise null.

4. For each instruction, fill `target_source` with the complete name/identity of the EXISTING regulation being changed (for example, "Gümrük Genel Tebliği (Transit Rejimi) (Seri No: 4)"). Resolve anaphoric phrases such as "Aynı Tebliğin" or "Aynı Yönetmeliğin" from the concrete source named earlier in the full pasted text; do not return "Aynı Tebliğ" as the source. This source identity is a positive retrieval anchor, not permission to invent a match.

5. For each instruction, fill `raw_date_phrase` with the natural-language effective-date phrase verbatim (e.g. "yayımı tarihinden itibaren yürürlüğe girer", "1 Ocak 2027 tarihinde yürürlüğe girer") if any; otherwise null.

6. For each instruction, write `search_query` as ONE concise question that asks for the currently indexed rule being changed. Use the likely language of the existing source (Turkish for Turkish regulations), retain the legal mechanism and subject nouns, and do not combine independent issues. The query should search for the old rule, not merely repeat phrases such as "updated" or "amended".

7. For each instruction, write `recovery_query` as ONE short alternate lexical query using the source/mechanism anchors and the most discriminative nouns. It must target the same rule as `search_query`, not broaden to another issue.

8. Fill `reference_date` with the text's own official publication/signing date (YYYY-MM-DD), if stated — this will be used as the anchor for resolving relative date phrases (e.g. "yayımı tarihinden itibaren"). Leave null if not stated.

Use ONLY information explicitly present in the given text. Never invent or assume anything not stated."""
# ruff: noqa: E501 end

_NAMED_SOURCE_RE = re.compile(
    r"(?:yayımlanan|yayimlanan)\s+"
    r"(?P<source>[A-ZÇĞİÖŞÜ][^.\n]{2,180}?"
    r"(?:Kanunu|Yönetmeliği|Yonetmeligi|Tebliği|Tebligi|Kararı|Karari)"
    r"(?:\s*\([^\n)]{1,80}\)){0,3})"
    r"[’']?(?:nin|nın|nun|nün)",
    re.IGNORECASE,
)
_SAME_SOURCE_RE = re.compile(
    r"\b(?:aynı|ayni)\s+(?:tebliğ|teblig|yönetmelik|yonetmelik|kanun|karar)",
    re.IGNORECASE,
)


class _RetrievalPlannedInstruction(AmendmentInstruction):
    """Require retrieval lanes for newly generated segmentation checkpoints."""

    search_query: str = Field(min_length=1, max_length=500)
    recovery_query: str = Field(min_length=1, max_length=500)


class _RetrievalPlannedSegmentationResult(SegmentationResult):
    instructions: list[_RetrievalPlannedInstruction]


def propagate_target_sources(
    instructions: list[AmendmentInstruction],
) -> list[AmendmentInstruction]:
    """Resolve explicit and ``Aynı ...`` source anchors across one amendment."""

    current_source: str | None = None
    resolved: list[AmendmentInstruction] = []
    for instruction in instructions:
        target_source = (instruction.target_source or "").strip() or None
        is_same_source = bool(
            _SAME_SOURCE_RE.search(target_source or "")
            or _SAME_SOURCE_RE.search(instruction.instruction_text)
        )
        if target_source and not _SAME_SOURCE_RE.search(target_source):
            current_source = target_source
        elif is_same_source and current_source:
            target_source = current_source
        elif target_source is None:
            source_match = _NAMED_SOURCE_RE.search(instruction.instruction_text)
            if source_match:
                target_source = " ".join(source_match.group("source").split())
                current_source = target_source
        resolved.append(instruction.model_copy(update={"target_source": target_source}))
    return resolved


def segment_amendment_text(llm: LLM, raw_text: str) -> SegmentationResult:
    result = generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_SEGMENTATION,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=raw_text,
        response_model=_RetrievalPlannedSegmentationResult,
    )
    return SegmentationResult(
        reference_date=result.reference_date,
        instructions=propagate_target_sources(
            [
                AmendmentInstruction.model_validate(item.model_dump())
                for item in result.instructions
            ]
        ),
    )
