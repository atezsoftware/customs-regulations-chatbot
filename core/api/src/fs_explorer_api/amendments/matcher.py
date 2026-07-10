"""LLM confirmation of which hybrid-search candidate an amendment instruction targets."""

from __future__ import annotations

import json

from ..llm import ChatTurn, LLMClient, LLMUsage
from .models import AmendmentInstruction, MatchResult
from .ranker import CandidateChunk

_SYSTEM_PROMPT = """Sen bir Türk mevzuatı değişiklik eşleştirme uzmanısın.

Sana bir değişiklik talimatı ve bu talimatın hangi mevcut metin parçasını (chunk)
etkilediğine dair bir aday listesi verilecek. Adaylar semantik arama, bulanık
(trigram) metin/başlık eşleştirmesi ve yapısal madde/doküman numarası eşleştirmesiyle
bulundu — başlık yolu (heading_path) belge biçimlendirmesinden tahmin edildiği için
güvenilmez olabilir, bu yüzden adayların METNİNE de dikkatlice bak, sadece başlığa
veya skorlara güvenme.

Görevin: bu talimatın adaylardan HANGİSİNİ değiştirdiğine karar vermek.

- Eğer adaylardan biri açıkça bu talimatın değiştirdiği mevcut madde/hüküm ise,
  `old_chunk_id` alanına o adayın chunk_id'sini yaz.
- Eğer talimat YENİ bir madde/hüküm ekliyorsa (mevcut hiçbir adayı değiştirmiyorsa),
  `old_chunk_id`'yi null bırak.
- `confidence` alanına 0.0-1.0 arası bir güven skoru yaz.
- `rationale` alanına kısa bir açıklama yaz.

Sadece verilen adaylardan birinin chunk_id'sini kullan, asla uydurma bir id yazma."""


def _format_candidates(candidates: list[CandidateChunk]) -> str:
    blocks = []
    for candidate in candidates:
        blocks.append(
            json.dumps(
                {
                    "chunk_id": candidate.chunk_id,
                    "relative_path": candidate.relative_path,
                    "text": candidate.text,
                    "metadata": candidate.metadata,
                    "scores": {
                        "semantic": round(candidate.semantic_score, 3),
                        "text_similarity": round(candidate.text_trgm_score, 3),
                        "heading_similarity": round(candidate.heading_trgm_score, 3),
                        "structured_match": candidate.structured_match,
                    },
                },
                ensure_ascii=False,
            )
        )
    return "\n\n".join(blocks)


async def confirm_match(
    llm: LLMClient,
    *,
    instruction: AmendmentInstruction,
    candidates: list[CandidateChunk],
) -> tuple[MatchResult, LLMUsage]:
    prompt = (
        f"Değişiklik talimatı:\n{instruction.instruction_text}\n\n"
        f"Madde referansı: {instruction.article_reference or '(belirtilmemiş)'}\n\n"
        f"Aday chunk'lar:\n{_format_candidates(candidates)}"
    )
    history = [ChatTurn(role="user", text=prompt)]
    return await llm.generate_structured(history, _SYSTEM_PROMPT, MatchResult)
