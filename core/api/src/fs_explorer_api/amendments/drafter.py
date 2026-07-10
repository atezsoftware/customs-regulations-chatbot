"""LLM drafting of the amended chunk content and its effective dates."""

from __future__ import annotations

import json
from typing import Any

from fs_explorer_shared.storage import chunk_to_review_dict

from ..llm import ChatTurn, LLMClient, LLMUsage
from .models import AmendmentInstruction, DraftResult

_SYSTEM_PROMPT = """Sen bir Türk mevzuatı değişiklik metni yazım uzmanısın.

Sana bir değişiklik talimatı, (varsa) değiştirdiği mevcut metin parçası (chunk) ve
metnin referans/yayım tarihi verilecek. Görevin bu talimata göre YENİ chunk içeriğini
ve yürürlük tarihlerini üretmek.

new_chunk alanı için:
- `text`: değişiklik talimatına göre güncellenmiş TAM chunk metni (sadece değişen
  kısmı değil, chunk'ın tamamını, değişiklik uygulanmış haliyle yaz).
- `chunk_type`: genellikle eski chunk ile aynı kalır, değişikliğin doğası
  gerektirmedikçe değiştirme.
- `metadata`: eski chunk'ın metadata alanlarını (varsa) koru, sadece talimatın
  gerektirdiği alanları güncelle (örn. madde numarası değişiyorsa article_no). Eski
  chunk yoksa (yeni madde ekleniyorsa), talimat metninden çıkarabildiğin alanları
  doldur. metadata içindeki HERHANGİ bir alanı değiştirme yetkin var.

dates alanı için:
- `effective_start_date`: bu yeni metnin YÜRÜRLÜĞE GİRDİĞİ tarih (YYYY-MM-DD).
  Talimatta doğal dil ifadesi varsa (örn. "yayımı tarihinden itibaren", "1 ay sonra")
  bunu verilen referans tarihine göre kesin bir tarihe çevir. Hiçbir tarih bilgisi
  yoksa null bırak. NOT: bu tarih aynı zamanda eski chunk'ın geçerliliğinin SONA
  ERDİĞİ tarih olarak kullanılacak (yeni metin başladığında eski metin biter).
- `effective_end_date`: SADECE talimatın kendisi bu yeni hükmün de geçici/süreli
  olduğunu açıkça belirtiyorsa doldur (örn. "31.12.2027 tarihine kadar geçerlidir").
  Aksi halde null.
- `rationale`: bu tarihlere nasıl ulaştığını kısaca açıkla.

Sadece verilen metinlerde yer alan bilgiyi kullan, hiçbir şey uydurma."""


async def draft_new_chunk(
    llm: LLMClient,
    *,
    instruction: AmendmentInstruction,
    old_chunk: dict[str, Any] | None,
    reference_date: str | None,
) -> tuple[DraftResult, LLMUsage]:
    old_chunk_json = (
        json.dumps(chunk_to_review_dict(old_chunk), ensure_ascii=False, indent=2)
        if old_chunk is not None
        else "(yok — bu yeni bir madde/hüküm ekliyor)"
    )
    prompt = (
        f"Değişiklik talimatı:\n{instruction.instruction_text}\n\n"
        f"Referans/yayım tarihi: {reference_date or '(belirtilmemiş)'}\n\n"
        f"Doğal dil tarih ifadesi: {instruction.raw_date_phrase or '(yok)'}\n\n"
        f"Eski chunk:\n{old_chunk_json}"
    )
    history = [ChatTurn(role="user", text=prompt)]
    return await llm.generate_structured(history, _SYSTEM_PROMPT, DraftResult)
