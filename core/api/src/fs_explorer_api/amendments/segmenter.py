"""LLM segmentation of a pasted gazette amendment text into atomic instructions."""

from __future__ import annotations

from ..llm import ChatTurn, LLMClient, LLMUsage
from .models import SegmentationResult

_SYSTEM_PROMPT = """Sen bir Türk mevzuatı (Resmi Gazete) değişiklik metni analiz uzmanısın.

Sana yapıştırılan bir Resmi Gazete değişiklik/protokol metni verilecek. Görevin:

1. Metni, her biri TEK bir maddeyi/hükmü etkileyen atomik değişiklik talimatlarına ayır.
   Türkçe mevzuat değişikliklerinde sık görülen kalıplar:
   - "MADDE N ... aşağıdaki şekilde değiştirilmiştir."
   - "... aşağıdaki fıkra/bent eklenmiştir."
   - "... yürürlükten kaldırılmıştır."
   Her talimatın `instruction_text` alanına, o değişikliği tam olarak anlamak için
   gereken ilgili metni (talimat cümlesi + varsa yeni madde metni) koy.

2. Her talimat için varsa `article_reference` alanına ilgili madde/fıkra referansını
   (örn. "Madde 3", "Madde 5 fıkra 2") koy; belirtilmemişse null bırak.

3. Her talimat için varsa `raw_date_phrase` alanına yürürlük/geçerlilik tarihiyle ilgili
   doğal dil ifadesini (örn. "yayımı tarihinden itibaren yürürlüğe girer",
   "1 Ocak 2027 tarihinde yürürlüğe girer") olduğu gibi koy; belirtilmemişse null bırak.

4. `reference_date` alanına, metnin kendisinde belirtilen resmi yayım/imza tarihini
   (varsa) YYYY-MM-DD formatında koy — bu, göreli tarih ifadelerini (örn. "yayımı
   tarihinden itibaren") çözmek için referans noktası olarak kullanılacak. Belirtilmemişse
   null bırak.

Sadece verilen metinde açıkça yer alan bilgiyi kullan, hiçbir şey uydurma."""


async def segment_amendment_text(
    llm: LLMClient, raw_text: str
) -> tuple[SegmentationResult, LLMUsage]:
    history = [ChatTurn(role="user", text=raw_text)]
    return await llm.generate_structured(history, _SYSTEM_PROMPT, SegmentationResult)
