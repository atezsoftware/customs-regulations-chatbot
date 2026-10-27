# Karma belge ZIP'i

Updates → New document → **File** → **Choose document or ZIP** yolundan yükleyin. Her üye kendi gerçek belge biçimine göre hazırlanır. PDF, HTML/HTM, DOCX, XLSX, desteklenen görseller, düz metin ve UTF-8 Markdown kullanılabilir. Markdown dönüşümü atlar; metadata, etiketleme, chunk incelemesi ve toplu onay devam eder.

Ana belge ve her ek ayrı mevzuat kaydıdır. Bağımsız belgeler aynı arşivde bulunabilir. Dosya adından, arşiv sırasından veya aynı klasörde bulunmalarından üst belge ilişkisi üretilmez. Başlık, metadata çıkarımında belgenin kendi kaynak metninden alınır; orijinal dosya yolu kaynak kanıtında saklanır.

```text
belgeler/
  ana.html
  ekler/
    ek-1.pdf
    ek-2.md
  bagimsiz.docx
```

`ana.html` içindeki `<a href="ekler/ek-1.pdf">EK-1</a>` ve `<a href="ekler/ek-2.md">EK-2</a>` bağlantıları iki eki ana belgeye bağlar. Markdown içindeki `[EK-1](ekler/ek-1.pdf)` aynı biçimde çözülür. Dosya yolu büyük/küçük harf ve Türkçe karakterler dahil birebir eşleşmelidir. `bagimsiz.docx`, üst belge bağlantısı olmadığından bağımsız kalır.

Manifest zorunlu değildir. Orijinal web ek adresini arşivdeki dosyaya bağlamak veya çoklu üst belge adaylarını açıkça seçmek için arşiv köküne isteğe bağlı `manifest.json` eklenebilir:

```json
{
  "documents": [
    {"path": "ana.html", "source_url": "https://example.gov/ana.html"},
    {
      "path": "ekler/ek-1.pdf",
      "parent": "ana.html",
      "source_url": "https://example.gov/ek-1.pdf"
    },
    {"path": "ekler/ek-2.md", "parent": "ana.html"},
    {"path": "bagimsiz.docx"}
  ]
}
```

Manifest varsa bütün gerçek üyeler birer kez listelenir; yollar manifestin bulunduğu klasöre göredir. `parent` gerçek kaynak bağlantısıyla desteklenmelidir. `source_url` tam HTTP/HTTPS adresidir; yalnızca dosya adının benzemesi eşleştirme sayılmaz. Eşleşmeyen gerçek internet ek bağlantıları mevcut güvenli kaynak indirme akışından alınır. Eksik yerel ek, belirsiz üst belge, desteklenmeyen biçim veya alınamayan internet eki görünür hata üretir; eksik paket onaya açılmaz.

Sınırlar: yüklenen ZIP en fazla 25 MiB, açılmış kaynakların toplamı 100 MiB, her kaynak en fazla 25 MiB ve toplam en fazla 21 gerçek belge. İnternetten veya gömülü dosyalardan alınan ekler de aynı paket sınırına dahildir. İç içe ZIP, şifreli dosya, özel dosya veya güvenli olmayan yol kabul edilmez. Birebir aynı içeriğin arşivde birden fazla kopyasını koymayın.

**Markdown package** seçiminin `legislation-markdown-v1` manifesti ve tek ana belge kuralı aynen devam eder. Bu belge **File** yolundaki karma ZIP içindir.

Örnek: [HTML ana belge, PDF/Markdown ekler ve bağımsız Markdown belgesi](examples/mixed-document-zip.zip). Bu paket gerçek mevzuat değildir; yalnızca yazılım denemesi için hazırlanmıştır. Beklenen sonuç iki bağımsız ana belge ve bunlardan birine bağlı iki ek, toplam dört ayrı kayıttır.
