# Markdown mevzuat paketi

Bu biçim, **yeni belge ekleme** işleminde metni hazır bir ana mevzuatı ve eklerini birlikte yüklemek içindir. Ana belge ve her ek ayrı mevzuat kaydı olur. Ana belge–ek ilişkisi korunur; belgelerin metinleri birleştirilmez.

Metin zaten Markdown olarak hazırlanmış olmalıdır. Bu yol PDF, tarama veya görsel dönüştürmez; bağlantılardan içerik indirmez. Tablolar, dipnotlar, madde numaraları ve tüm hükümler Markdown dosyalarında tam metin olarak yer almalıdır. Mevcut mevzuatı değiştirme akışı bu biçimin kapsamı dışındadır.

## Hazır örnek

[Ana belge ve iki ek içeren örnek ZIP'i indir](examples/legislation-markdown-v1.zip). Dosyalar örnektir; gerçek bir düzenleme veya hukuki kaynak değildir.

Tek belge için UTF-8 kodlamalı bir `.md` dosyası yeterlidir. Dosyanın ilk birinci düzey başlığı (`# Başlık`) belge adı olarak kullanılır; böyle bir başlık yoksa dosya adı kullanılır. Ayrı yerel eklere başvuruyorsa tek dosya yerine ZIP yükleyin.

## ZIP yapısı

ZIP içinde `manifest.json` zorunludur. Dosyalar ZIP kökünde veya tek bir dış klasörde bulunabilir. Bütün yollar manifest'in bulunduğu klasöre göre yazılır.

```text
manifest.json
ana.md
ekler/
  ek-1.md
  ek-2.md
```

```json
{
  "format": "legislation-markdown-v1",
  "main": "ana.md",
  "documents": [
    {
      "path": "ana.md",
      "title": "Ana mevzuatın tam adı",
      "source_url": "https://example.com/mevzuat"
    },
    {
      "path": "ekler/ek-1.md",
      "title": "EK-1 — Başvuru bilgileri",
      "parent": "ana.md"
    },
    {
      "path": "ekler/ek-2.md",
      "title": "EK-2 — Süreler",
      "parent": "ana.md"
    }
  ]
}
```

- `format` tam olarak `legislation-markdown-v1` olmalıdır.
- `main`, ana Markdown dosyasının yoludur. Ana belgenin `parent` alanı olamaz.
- `documents`, ZIP'teki **bütün ve yalnızca** Markdown dosyalarını birer kez listeler. `path` dosya yoluyla büyük/küçük harf ve Unicode yazımı dahil birebir eşleşmelidir.
- `title`, arayüzde görünen belge adıdır. ZIP yüklemesinde Markdown başlığı yerine bu ad kullanılır.
- Ana belge dışındaki her belgenin bir `parent` yolu olmalıdır. Üst belge de manifest'te bulunmalıdır. Bu ilişkiler ana belgede köklenen, döngüsüz bir ağaç oluşturmalıdır. Bir ekin kendi alt eki olabilir.
- `source_url` isteğe bağlı HTTP/HTTPS kaynak bilgisidir. Yalnızca kaynak bilgisi olarak saklanır; adres ziyaret edilmez.
- Listelenmemiş `.md`, PDF, HTML, görsel, başka ZIP veya yardımcı dosya bulunamaz. `manifest.json` dışında başka metadata dosyası gerekmez.
- Finder'ın eklediği `.DS_Store` ve `__MACOSX` altındaki `._dosya` kaynak çatalları, ZIP güvenlik ve boyut kontrollerinden sonra yok sayılır. Diğer yardımcı dosyalar kabul edilmez.

## Gerçek Markdown bağlantıları

Manifest'te bir ekin üst belgesi olarak gösterilen dosya, o eke gerçek bir yerel Markdown bağlantısı içermelidir. Örneğin `ana.md`:

```md
# Ana mevzuatın tam adı

MADDE 1 - Başvuruya ilişkin bilgiler eklerde yer alır.

[EK-1: Başvuru bilgileri](ekler/ek-1.md)
[EK-2: Süreler](ekler/ek-2.md)
```

Referans biçimi ve HTML bağlantısı da tanınır:

```md
[EK-1][basvuru]

[basvuru]: ekler/ek-1.md

<a href="ekler/ek-2.md">EK-2</a>
```

Alt klasördeki bir ekten ana belgeye bağlantı `../ana.md` biçiminde yazılabilir. Bağlantı çözüldüğünde paket sınırları içinde kalmalıdır. Mevcut dosyalara çapraz bağlantılar ve `ek-1.md#tablo` gibi bölüm işaretleri saklanır. Bölüm işaretinin sayfada varlığı denetlenmez; dosya yolunun varlığı denetlenir.

Kod bloğu veya satır içi kod içindeki bağlantı örnekleri belge ilişkisi sayılmaz. Yalnızca `Ekler için tıklayınız` yazmak yeterli değildir; yerel Markdown dosyasına gerçek bağlantı gerekir. Üst belgenin bağlantısı eksikse paket kabul edilmez.

`[Resmî kaynak](https://example.com/kaynak.pdf)` gibi dış HTTP/HTTPS atıfları metinde kalır; takip edilmez ve ek dosyası oluşturmaz. İçe alınacak ekin tam metni pakette yerel `.md` dosyası olarak bulunmalıdır.

Markdown'ı orijinal PDF/HTML'den oluştururken **içe alınacak eklere ait bağlantıları yerel `.md` yollarına çevirin**. Örneğin ekin web adresini `ekler/ek-1.md` yapın; asıl adresi manifest'teki o ekin `source_url` alanında koruyabilirsiniz. Dosya uzantısı veya benzer dosya adından otomatik eşleştirme yapılmaz.

## Metin ve tablolar

Dosyalar UTF-8 olmalıdır; baştaki UTF-8 BOM kabul edilir. Kaynak baytları ve karması aynen saklanır. Ham Markdown okunurken yalnızca baştaki BOM kaldırılır. Chunk hazırlığında doğrulanmış ek bağlantısının `Ekler için tıklayınız — EK-1` gibi navigasyon etiketi `EK-1` yapılır; bağlantı hedefi ve ayrı ek kaydı korunur. Hükümler, tablolar ve dipnotlar özetlenmez. Asıl dosya ve kaynak ilişkileri ilk halleriyle tutulur.

Madde başlığı varsa numara ve başlığı aynı Markdown başlık satırında tutun; hükmü sonraki paragrafta yazın. Başlıksız maddelerde `MADDE 1 - Hükmün tam metni` biçimi de kabul edilir. Örneğin:

```md
## MADDE 1 — Başvuru belgeleri

(1) Başvuruda sunulacak belgeler EK-1'de gösterilir.
```

```md
| Alan | Açıklama |
| --- | --- |
| Başvuran | Adı ve unvanı |
| Belge | Başvuru numarası |
```

Uzun paragraflar metinleri korunarak token sınırına göre bölünür; madde başlığı ilk hüküm parçasıyla birlikte tutulur. Bir tablonun satırları bölünmez. Tek bir satır veya başlık sınırı aşıyorsa içerik kesilmez; hazırlama hatası açıkça gösterilir.

Görsel Markdown (`![...](...)`), `<img>`, gömülü PDF/görsel nesneleri ve görsel dosyasına bağlantılar kabul edilmez. Görsel veya taramadaki bilgi önce eksiksiz metne/tabloya aktarılmalıdır; yalnızca bir görsele bağlantı vermek yeterli değildir.

## Onaydan önce inceleme ve belge ilişkileri

Yeni belge ekranı, Update ile aynı `Before / After` alan tablosunu kullanır. Chunk'lar doğrudan görünür; yeni kayıt için `Before` boştur. Etiketleme işlemi bittikten sonra hazır veya kısmi incelemede `After` tarafında metni, başlık yolunu, chunk türünü, hukuki metadata alanlarını ve geçerlilik tarihlerini tek tek düzenleyebilirsiniz. Etiketleme çalışırken düzenleme kapalıdır. Kayıt kimlikleri, kaynak karmaları ve teknik ilişki/audit alanları salt okunurdur.

Chunk oluşturulduktan sonra Gemini 3.5 Flash yalnızca yeni dosyaların hukuki metadata ve yürürlük tarihlerini çıkarır. Belgenin kapanıştaki yürürlük maddesi ve maddeye özel istisnalar birlikte okunur; tarihlerin tam metin dayanağı gösterilir. Başka mevzuata ait atıf tarihleri yeni belgenin tarihi sayılmaz. Belirtilmeyen veya çözülemeyen tarihler boş kalır. Her alanın chunk oluşturma, LLM veya manuel incelemeden geldiği gösterilir; bu adım Markdown dönüştürme değildir.

Kaydedilmemiş değişiklikler varken toplu onay kapalıdır. Kaydetme, yalnızca değişen dosyanın etiketlerini Gemini 3.5 Flash ile yeniden üretir; paketteki diğer dosyaların doğrulanmış etiketleri yeniden kullanılır. Yeni etiketler tamamlandıktan sonra güncel incelemeyi onaylayın. Önceki ve düzeltilmiş chunk revizyonları saklanır. Metin, başlık yolu veya chunk türü değişirse o dosyanın eski LLM kanıtları güncel değil olarak işaretlenir ve LLM'den türetilmiş tarihler temizlenir; gerçekten elle düzeltilmiş değerler korunur. Bu uyarıdaki alanları güncel metne göre inceleyin. Etiketleme başarısızsa ilgili chunk kartında nedeni gösterilir ve onay kapalı kalır.

Her dosya için PC Külliyatı'ndaki tamamlanmış belgelerin adlarından önceki mevzuat adayları aranır. Bu kontrol dosya düzeyindedir; madde/chunk karşılaştırması yapmaz. Otomatik öneride sayılar ve dosya uzantıları göz ardı edilir; elle aramada belge numarası da kullanılabilir. Öneri tek başına ilişki kurmaz. Önceki dosyayı seçip ilişkiyi kaydetmeniz gerekir.

`successor` önceki/sonraki mevzuat bağlantısı kurar. `repeals` ayrıca önceki belgeye dosya düzeyinde mülga bilgisi ekler; bunun için yürürlük tarihi, açık dayanak ve arayüzde ayrı onay gerekir. Yeni bir numara veya benzer ad mülga kararı sayılmaz. İlişki, bütün yeni dosyalar başarıyla indekslendikten sonra etkinleşir. Gelecekteki tarih için yürürlükten kalkma planlanmış olarak gösterilir. Her iki dosyanın ayrıntılarından karşı belgeye geçilebilir.

Bu işlem eski dosyanın metnini, chunk'larını, etiketlerini veya arama projeksiyonlarını yeniden yazmaz. Onaylanmış mülga kaydı, yürürlükten kalkma tarihinde ve sonrasındaki güncel kelime/vektör/hibrit aramalarda eski dosyayı dışlar. Önceki tarihli sorgular, kaydedilmiş kaynak önizlemeleri ve eski belgenin doğrudan okunması korunur. Aynı kontrol madde, komşu chunk ve başlık genişletmelerinde de uygulanır.

Eski belgenin mülga tarihi, yeni belgenin yürürlük başlangıcını kendiliğinden belirlemez. Yeni belgenin başlangıcı bilinmiyorsa tarih uydurulmaz; bilinen tarihler ayrıca girilmelidir. Onaydan sonra önceki dosya setten kaldırılmışsa ilişki açıkça engellenmiş görünür; dosyayı sete geri bağlayıp ilişkiyi yeniden deneyin. Bu yeniden deneme chunk veya etiket üretmez.

## Sınırlar ve hata durumları

| Sınır | Değer |
| --- | --- |
| Yükleme boyutu | 16 MiB |
| Markdown dosyası | 2 MiB ve en fazla 2.000.000 karakter |
| ZIP'in toplam açılmış boyutu | 32 MiB |
| Markdown belge sayısı | 21 (1 ana belge ve en fazla 20 ek) |
| ZIP girdisi sayısı, klasörler dahil | 128 |
| Girdi başına sıkıştırma oranı | En fazla 100:1 |
| Gerçek bağlantı sayısı, dış atıflar dahil | 500 |
| Manifest boyutu | 256 KiB |

Sınır aşıldığında içerik kesilmez; paket açık bir hatayla reddedilir. Bağımsız ana belgeleri ayrı paketlerde yükleyin. Tek mevzuatın bağlı ekleri 20 ek sınırını aşıyorsa yöneticinize başvurun; bağlantılarını korumak için bağlı ekleri ayrı paketlere bölmeyin. Eksik dosyalar, bağlantısız ekler, döngüler, aynı içeriğe sahip iki Markdown dosyası, harf/Unicode farkıyla çakışan yollar, yinelenen manifest alanları ve referans tanımları kabul edilmez. Güvensiz ZIP yolları, sembolik bağlantılar, şifreli dosyalar ve özel dosya girdileri de reddedilir.

Hata iletisindeki kod ve paket yolu sorunun konumunu gösterir. Manifest'i ve gerçek bağlantıları düzeltip paketi yeniden yükleyin. Bu doğrulama metnin hukuken eksiksiz olduğunu doğrulamaz; etiketler ve parçalar yeni belge inceleme ekranında ayrıca gözden geçirilmelidir.
