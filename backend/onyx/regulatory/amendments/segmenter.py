"""LLM segmentation of a pasted amendment text into atomic instructions."""

from onyx.llm.interfaces import LLM
from onyx.regulatory.amendments.models import SegmentationResult
from onyx.regulatory.structured_llm import generate_structured
from onyx.tracing.flows import LLMFlow

# ruff: noqa: E501 start
_SYSTEM_PROMPT = """You are an expert at analyzing Turkish regulatory amendment texts (Resmi Gazete değişiklik/protokol metinleri).

You will be given a pasted amendment/update text. Your job:

1. Split the text into atomic amendment instructions, each affecting exactly ONE article/provision.
   Common patterns in Turkish regulatory amendments:
   - "MADDE N ... aşağıdaki şekilde değiştirilmiştir." (article N is changed to read as follows)
   - "... aşağıdaki fıkra/bent eklenmiştir." (the following paragraph/clause is added)
   - "... yürürlükten kaldırılmıştır." (repealed)
   Put the exact text needed to fully understand each change (the instruction sentence plus any new article text) into `instruction_text`.

2. CRITICAL — only treat an instruction as adding a brand-new article/provision if the text EXPLICITLY says so (e.g. "... eklenmiştir", "yeni bir madde olarak", "MADDE N eklenmiştir"). If the text is merely amending, replacing, or clarifying an existing article, it is NOT a new article — leave `article_reference` pointing at the existing article being changed. Never infer a new-article addition from ambiguous or silent phrasing.

3. For each instruction, fill `article_reference` with the relevant article/paragraph reference (e.g. "Madde 3", "Madde 5 fıkra 2") if stated; otherwise null.

4. For each instruction, fill `raw_date_phrase` with the natural-language effective-date phrase verbatim (e.g. "yayımı tarihinden itibaren yürürlüğe girer", "1 Ocak 2027 tarihinde yürürlüğe girer") if any; otherwise null.

5. Fill `reference_date` with the text's own official publication/signing date (YYYY-MM-DD), if stated — this will be used as the anchor for resolving relative date phrases (e.g. "yayımı tarihinden itibaren"). Leave null if not stated.

Use ONLY information explicitly present in the given text. Never invent or assume anything not stated."""
# ruff: noqa: E501 end


def segment_amendment_text(llm: LLM, raw_text: str) -> SegmentationResult:
    return generate_structured(
        llm,
        flow=LLMFlow.AMENDMENT_SEGMENTATION,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=raw_text,
        response_model=SegmentationResult,
    )
