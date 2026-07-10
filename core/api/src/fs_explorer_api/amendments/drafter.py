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
- `metadata_changes`: SADECE değişen metadata alanlarını buraya yaz (örn. madde
  numarası değişiyorsa sadece article_no). heading_path, document_date gibi
  DEĞİŞMEYEN alanları BURAYA TEKRAR YAZMA — bunlar otomatik olarak eski chunk'tan
  korunacak, sen sadece farkı belirtiyorsun. Hiçbir metadata alanı değişmiyorsa
  boş obje ({}) döndür. metadata içindeki HERHANGİ bir alanı değiştirme yetkin
  var.

  EĞER eski chunk YOKSA (bu tamamen yeni bir madde/hüküm ekliyor): merge
  edilecek eski bir metadata olmadığı için `metadata_changes` pratikte TAM
  metadata'nın kendisi olur — bu durumda sana "Aynı dokümandan örnek bir chunk"
  (`sibling_reference`) verilecek. heading_path'i BOŞ BIRAKMA — sibling_reference'ın
  heading_path'ini TEMEL AL (aynı BÖLÜM/KISIM/ana başlık seviyesinde kal, sadece
  MADDE numarasını/başlığını yeni maddeye göre güncelle) ki arama ve alıntılama
  (citation) bu yeni chunk için de doğru çalışsın. document_type, document_number
  gibi diğer alanları da sibling_reference'tan uygun şekilde türet. article_no'yu
  talimattan çıkarabiliyorsan (örn. "Madde 7 eklenmiştir") mutlaka doldur —
  citation'lar öncelikle article_no'ya bakar, heading_path'e ancak o yoksa
  bakar.

dates alanı için:
- `effective_start_date`: bu yeni metnin YÜRÜRLÜĞE GİRDİĞİ tarih (YYYY-MM-DD).
  Talimatta somut bir tarih varsa (örn. "1 Ocak 2027'den itibaren") direkt onu
  kullan. "Yayımı tarihinden itibaren" gibi göreli bir ifade varsa VE sana
  verilen referans/yayım tarihi doluysa, bunu o tarihe göre kesin bir tarihe
  çevir. NE somut bir tarih NE de kullanabileceğin bir referans/yayım tarihi
  yoksa (örn. "yayımı tarihinden itibaren geçerlidir" deniyor ama referans
  tarihi "(belirtilmemiş)" ise) TAHMİN ETME, UYDURMA — null bırak. Bu durumda
  sistem otomatik olarak onay tarihini (bu değişikliğin fiilen veritabanına
  işlendiği günü) kullanacak; bu hem Türk mevzuatındaki "aksi belirtilmedikçe
  yayım tarihinde yürürlüğe girer" varsayılan kuralına hem de pratik olarak
  elimizdeki en iyi bilgiye uyar. NOT: bu tarih (veya null ise sistemin
  dolduracağı onay tarihi) aynı zamanda eski chunk'ın geçerliliğinin SONA
  ERDİĞİ tarih olarak kullanılacak (yeni metin başladığında eski metin biter).
- `effective_end_date`: SADECE talimatın kendisi bu yeni hükmün de geçici/süreli
  olduğunu açıkça belirtiyorsa doldur (örn. "31.12.2027 tarihine kadar geçerlidir").
  Aksi halde null — burada ASLA bir varsayılan tarih uydurma, boş bırakmak
  "süresiz geçerli" anlamına gelir ve bu doğru varsayılandır.
- `rationale`: bu tarihlere nasıl ulaştığını (veya neden null bıraktığını)
  kısaca açıkla.

Sadece verilen metinlerde yer alan bilgiyi kullan, hiçbir şey uydurma."""


async def draft_new_chunk(
    llm: LLMClient,
    *,
    instruction: AmendmentInstruction,
    old_chunk: dict[str, Any] | None,
    sibling_reference: dict[str, Any] | None = None,
    reference_date: str | None,
) -> tuple[DraftResult, LLMUsage]:
    old_chunk_json = (
        json.dumps(chunk_to_review_dict(old_chunk), ensure_ascii=False, indent=2)
        if old_chunk is not None
        else "(yok — bu yeni bir madde/hüküm ekliyor)"
    )
    sibling_json = (
        json.dumps(sibling_reference, ensure_ascii=False, indent=2)
        if sibling_reference is not None
        else "(yok)"
    )
    prompt = (
        f"Değişiklik talimatı:\n{instruction.instruction_text}\n\n"
        f"Referans/yayım tarihi: {reference_date or '(belirtilmemiş)'}\n\n"
        f"Doğal dil tarih ifadesi: {instruction.raw_date_phrase or '(yok)'}\n\n"
        f"Eski chunk:\n{old_chunk_json}\n\n"
        f"Aynı dokümandan örnek bir chunk (sibling_reference, sadece eski chunk "
        f"yoksa heading_path/metadata konvansiyonu için kullan):\n{sibling_json}"
    )
    history = [ChatTurn(role="user", text=prompt)]
    return await llm.generate_structured(history, _SYSTEM_PROMPT, DraftResult)
