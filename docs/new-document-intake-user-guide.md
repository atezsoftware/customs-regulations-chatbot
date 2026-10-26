# Yeni belge ve ekleri ekleme

PC Külliyatı için kullanım kılavuzu

Bu kılavuz, yeni bir mevzuatı ve eklerini **Updates** ekranından inceleyip PC Külliyatı'na eklemek içindir. Ana belge ve her ek, ayrı bir mevzuat kaydı olarak hazırlanır. Aralarındaki bağlantılar korunur.

## 1. Ekranı açın ve kaynağı seçin

Yönetici hesabınızla **Updates** ekranını açın ve **New document** işaretini etkinleştirin. Doğrudan ekran yolu: `/admin/amendments?new_document=true`.

DEV test ekranı: [Yeni belge ekleme](https://dev-customs-regulations.singlewindow.io/admin/amendments?new_document=true).

Hedef set **PC Külliyatı** olarak sabittir; başka bir document set seçimi yapılmaz. Bu işlemdeki yeni dosyalar üzerinde çalışılır. Setteki diğer belgeler yeniden etiketlenmez. Bir önceki mevzuatın mülga olarak işaretlenmesi, ayrıca seçilip onaylanan bir belge ilişkisidir.

| Elinizdeki kaynak | Ekrandaki seçim | Yapılacak işlem |
| --- | --- | --- |
| Orijinal PDF veya başka desteklenen dosya | File | Choose document ile dosyayı seçin. |
| Belgenin internet adresi | URL | New document source URL alanına doğrudan kaynak adresini girin. |
| Hazır tek Markdown dosyası | Markdown package | Choose Markdown or ZIP ile `.md` dosyasını seçin. |
| Ana belge ve eklerinin hazır Markdown dosyaları | Markdown package | Manifest içeren ZIP paketini seçin. |

**File**, PDF yanında DOCX, HTML/HTM, XLSX ve desteklenen görsel dosyalarını da alır. Hazır `.md` ve mevzuat ZIP'i için **Markdown package** seçimini kullanın.

Dosya yüklerken **Original document URL** alanı, orijinal belgenin web adresidir. Belgedeki `/ekler/ek1.pdf` gibi göreli bağlantıların çözülebilmesi için bu adresi girin. Yerel bilgisayarınızdaki dosya yolu bu alanın yerine geçmez. URL yolunda kaynak adresi zaten verilmiştir.

**Prepare source and attachments** düğmesine basın. Kaynak paketindeki dosya adlarını ve belge sayısını kontrol edin. Paket **ready** olunca **Prepare new documents** düğmesine basın.

Etiketleme ve hukuki metadata için model **Gemini 3.5 Flash** olarak sabittir. Model veya GCS adresi seçmeniz gerekmez. Bağlantı yapılandırma hatası görürseniz yöneticinizin Language Models bağlantısını kontrol etmesini sağlayın; bu hata çözülmeden belge hazırlama başlatılamaz.

## 2. PDF ve URL içindeki ekleri kontrol edin

Kaynakta “Ekler için tıklayınız” gibi bir bağlantı varsa, sistem bağlantıyı belge bağlamıyla birlikte değerlendirir. Alınan ana belge ile PDF, HTML veya diğer desteklenen ekler ayrı mevzuat kayıtları olarak gösterilir. Bir ekin alt ekleri de ilişkili olabilir.

Her dosya kartında **Main document** veya **Attachment to ...** açıklaması bulunur. **Open ...** ve **Back to ...** bağlantılarıyla ana belge ile ekleri arasında geçin. İnceleme metnindeki doğrulanmış ek bağlantıları da ilgili dosyaya götürür.

Bir ek alınamazsa paket eksiksizmiş gibi ilerlemez. Hata bölümünde ekin etiketi/adresi ve nedeni gösterilir. Örneğin kaynak adresi eksikliği, bulunamayan dosya veya indirilemeyen bağlantı görülebilir. Önce eksikliği giderin; uygunsa **Retry source preparation** ile yeniden deneyin. Kaynak paketinin hazır olması, mevzuatın hukuki içeriğinin doğruluğuna ilişkin kullanıcı incelemesinin yerine geçmez.

## 3. Markdown ve ZIP standardı

Hazır Markdown verildiğinde Markdown'a dönüştürme adımı atlanır. Chunk hazırlama, hukuki metadata, yürürlük tarihi çıkarımı ve etiketleme yine yapılır.

Tek belge için UTF-8 kodlamalı bir `.md` dosyası yeterlidir. İlk `# Başlık` belge adı olarak kullanılır; yoksa dosya adı kullanılır. Ana belge ile yerel ekler birlikteyse aşağıdaki ZIP yapısını kullanın:

Bir pakette bir ana belge ve en fazla 20 ek, toplam 21 Markdown belgesi bulunabilir. Alt ekler de bu sayıya dahildir. Bağımsız ana belgeleri ayrı paketlerde yükleyin. Tek mevzuatın bağlı ekleri bu sınırı aşıyorsa yöneticinize başvurun; bağlantılarını korumak için bağlı ekleri ayrı paketlere bölmeyin.

```text
manifest.json
ana.md
ekler/
  ek-1.md
```

`manifest.json` örneği:

```json
{
  "format": "legislation-markdown-v1",
  "main": "ana.md",
  "documents": [
    {
      "path": "ana.md",
      "title": "Ana mevzuatın tam adı"
    },
    {
      "path": "ekler/ek-1.md",
      "title": "EK-1 - Başvuru belgeleri",
      "parent": "ana.md"
    }
  ]
}
```

`ana.md` içinde eke gerçek bağlantı bulunmalıdır:

```md
# Ana mevzuatın tam adı

## MADDE 1 - Başvuru belgeleri

(1) Başvuruda sunulacak belgeler EK-1'de gösterilir.

[Ekler için tıklayınız - EK-1](ekler/ek-1.md)
```

**Eşleştirme için zorunlu kurallar:**

1. `documents`, ZIP'teki bütün Markdown dosyalarını birer kez listelesin. `path` değerleri dosya adlarıyla büyük/küçük harf ve Türkçe karakterler dahil birebir eşleşsin.
2. Ana belge dışındaki her belgenin `parent` alanı bulunsun. Üst belge, o eke gerçek bir yerel Markdown bağlantısı versin. Yalnızca “Ekler için tıklayınız” yazısı yeterli değildir.
3. PDF/HTML'deki ek adresini paketteki yerel `.md` yoluna çevirin. Orijinal web adresini isterseniz ilgili manifest kaydına `source_url` olarak ekleyin.
4. Bütün hükümleri, tabloları ve dipnotları tam metin aktarın. Madde numarası ve başlığı aynı başlık satırında, hüküm de hemen altında olsun. Görseldeki bilgiyi metin veya tabloya aktarın.
5. ZIP'e yalnızca `manifest.json` ve listelenen `.md` dosyalarını koyun. Orijinal PDF'leri, görselleri, başka ZIP'leri veya yardımcı metadata dosyalarını bu pakete eklemeyin. Tek bir dış klasör kullanılabilir.

Markdown paketindeki dış HTTP/HTTPS atıfları indirilmez. İçe alınacak ekin tam metni yerel `.md` olarak pakette bulunmalıdır. `source_url`, bu yolda yalnızca kaynak bilgisidir.

**Hazır örnek:** [Ana belge ve iki ek içeren örnek ZIP](examples/legislation-markdown-v1.zip). Ekrandaki **Download Markdown package template** bağlantısı da şablonu indirir. Örnek dosyalar gerçek mevzuat değildir. Ayrıntılı ve bağlayıcı paket kuralları: [Markdown mevzuat paketi](markdown-legislation-package.md).

## 4. Chunk'ları, etiketleri ve tarihleri inceleyin

Her dosyada **New legislation**, her yeni chunk'ta **Added** işareti bulunur. Chunk'lar doğrudan görünür; gerektiğinde **Show added chunks**, **Previous chunks** ve **Next chunks** ile inceleyin.

Update ekranındaki **Before / After** alan tablosu kullanılır. Yeni belge için **Before** boştur. Eski belgeyle chunk karşılaştırması yapılmaz. Yeni metin ve düzenlenebilir alanlar **After** tarafındadır.

Özellikle şu alanları kontrol edin:

- **Metin:** Hükümler tam mı; başlık, tablo ve dipnotlar doğru mu? Uzun bir madde birden fazla chunk'a bölünebilir; devam parçalarını da okuyun.
- **Başlık yolu ve chunk türü:** Madde doğru başlığın altında mı? Başlık yolunda her satır ayrı bir başlıktır.
- **Hukuki metadata:** Belge türü/numarası, madde numarası/başlığı, fıkra-bent, ek ve tablo bilgileri kaynağa uyuyor mu?
- **Geçerlilik başlangıcı ve sonu:** Tarihlerin kaynak dayanağı doğru mu? `YYYY-MM-DD` kullanın; bilinmeyen tarihi boş bırakın. Bitiş, başlangıçtan sonra olmalıdır.
- **Etiketler:** Dosya ve chunk etiketlerini inceleyin. Başarısız bir etiketlemenin nedeni ilgili chunk kartında gösterilir.

Alanların yanında bilginin chunk hazırlığından, LLM'den veya manuel incelemeden geldiği gösterilir. LLM'nin tarih çıkarımında kullandığı alıntıyı ve kaynak chunk'ı kontrol edin. Belgenin genel yürürlük maddesi ile maddeye özel istisna farklı tarihler doğurabilir. Tarayıcının yazdırma tarihi, mevzuatın yayım veya yürürlük tarihi değildir. Başka bir mevzuata yapılan atıftaki tarih de yeni belgenin yürürlük tarihi olarak kabul edilmemelidir. Dayanağı bulunmayan veya çözülemeyen tarih boş kalır.

**Düzeltme:** Hazır veya kısmi incelemede `After` alanlarını değiştirin, ardından **Save chunk changes** ile kaydedin. **Discard chunk changes**, kaydedilmemiş değişiklikleri geri alır. Etiketleme çalışırken düzenleme kapalıdır. Teknik kimlikler, kaynak bilgileri ve ilişki/audit alanları salt okunurdur.

Kaydetme, yalnızca değişen dosyanın etiketlerini yeniden üretir; paketteki diğer dosyaların doğrulanmış etiketleri korunur. Yeni etiketleme bitince sonucu tekrar inceleyin. Etiketleri bu tablodan doğrudan değiştirmezsiniz.

Metin, başlık yolu veya chunk türü değişirse o dosyaya ait önceki LLM kanıtları güncel değil olarak işaretlenir ve LLM'den türetilmiş tarihler temizlenir. Bu alanları güncel bağlama göre yeniden kontrol edin. Gerçekten elle düzelttiğiniz değerler korunur. Önceki ve düzeltilmiş chunk revizyonları saklanır.

## 5. Önceki mevzuat ve mülga ilişkisini belirleyin

Önceki mevzuat incelemesini dosya bazında açın. Adaylar belge adlarından önerilir; gerekirse **Search previous legislation** ile ad veya numara arayın. Doğru kaydı **Select ...** ile seçin. Bu, belge düzeyindeki kısa bir kontroldür; chunk karşılaştırması değildir.

Örneğin 4000 sayılı bir düzenleme varken 5000 sayılı yeni düzenlemenin gelmesi, tek başına eskiyi mülga yapmaz. Yalnızca önceki/sonraki belge bağı kuracaksanız önceki belgeyi seçip **Save previous legislation relationship** ile kaydedin.

Yeni belge eskiyi açıkça yürürlükten kaldırıyorsa:

1. **This document explicitly repeals the previous legislation** kutusunu işaretleyin.
2. **Repeal effective date** alanına yürürlükten kalkma tarihini girin.
3. **Repeal provision** alanına açık kaldırma hükmünü/dayanağını yazın.
4. **Save previous legislation relationship** ile ilişkiyi kaydedin.

**No previous legislation** seçimiyle önceki belge seçimini kaldırabilirsiniz. **Discard relationship changes**, henüz kaydedilmemiş ilişki değişikliklerini geri alır.

Karşılıklı ilişki, toplu onaydan sonra bütün yeni dosyalar başarıyla indekslenince etkinleşir. Mülga tarihinden itibaren eski belge güncel aramalardan çıkar; eski tarihli sorgular ve doğrudan tarihsel kaynak okuması korunur. Gelecekteki tarih için mülga durumu planlanmış görünür. Eski belgenin metni, chunk'ları ve etiketleri yeniden yazılmaz. Eski belgenin mülga tarihi, yeni belgenin başlangıç tarihini kendiliğinden doldurmaz.

## 6. Toplu onay ve tamamlanma

Ana belgeyi ve bütün ekleri, chunk'ları, etiketleri, tarihleri ve varsa önceki mevzuat ilişkisini kontrol edin. Kaydedilmemiş değişiklik bırakmayın.

Paket **ready** olduğunda **Approve all new documents** düğmesi açılır. Bu düğme, bu içe aktarımdaki bütün yeni dosyaları onaylar; PC Külliyatı'ndaki diğer dosyaları topluca değiştirmez. Kısmi veya başarısız etiketleme varken onay kapalıdır.

Onaydan sonra **approving / indexing**, işlemin sırada veya sürmekte olduğunu gösterir. **completed**, bütün yeni dosyaların indekslenmesi tamamlandığında görünür. Yalnızca onaya basılmış olmasını tamamlanma saymayın.

Geçmiş içe aktarımlar ekrandaki belge adı ve durum düğmelerinde listelenir. Devam eden veya hatalı bir işlemi buradan tekrar açın. Yeni bağımsız bir yükleme için **Start another import** kullanın.

## 7. Hata olduğunda ne yapmalı?

| Ekrandaki durum | Kullanıcı işlemi |
| --- | --- |
| Eksik/indirilemeyen ek veya kaynak paketi hatası | Adresi, Original document URL alanını veya kaynak dosyayı kontrol edin. Sunuluyorsa Retry source preparation kullanın. |
| ZIP yolu, manifest veya bağlantı hatası | Bildirilen paket yolunu düzeltin; dosya adlarını, parent değerlerini ve gerçek yerel bağlantıları eşleştirip paketi yeniden yükleyin. |
| Metadata ya da etiketleme hatası; partial / failed | İlgili dosya ve chunk hata açıklamasını okuyun. Sunuluyorsa Retry new documents kullanın. Hazır görünmeyen işlemi onaylamayın. |
| Kaydetme sırasında inceleme değişti uyarısı | Gerekirse düzenleme metninizi kopyalayın; güncel durumu yeniden açın, güncel chunk ve tarihleri kontrol edip tekrar kaydedin. |
| Onaydan sonra indeksleme hatası | Aynı içe aktarımı tekrar açıp Retry new documents kullanın. Aynı dosyaları yeniden yükleyerek ikinci bir içe aktarım oluşturmayın. |
| Belge ilişkisi engellendi | Önceki dosyanın PC Külliyatı üyeliğini geri sağlayın; Retry document relationship kullanın. Bu işlem yayınlanmış chunk ve etiketleri yeniden üretmez. |

Sorun devam ederse yöneticinize belge adını, ekrandaki durumu ve hata iletisini iletin. API anahtarını paylaşmayın.

## 8. Gerçek PDF ile hızlı kontrol listesi

Kurumunuzun DEV ortamında, sonucu karşılaştırabileceğiniz gerçek bir PDF ve varsa erişilebilir ek adresleriyle aşağıdaki adımları uygulayın. Bu bölüm bir kontrol listesidir; belirli bir PDF'nin testten geçtiği iddiası değildir.

- PDF'yi **File** yolundan seçin; varsa göreli ekler için orijinal web adresini girin.
- Kaynak paketindeki ana belge ve ek sayısını, dosya adlarını ve bağlantıları kaynakla karşılaştırın.
- Beklenen bir ek alınamıyorsa hatanın açıkça gösterildiğini ve ilerlemenin engellendiğini kontrol edin.
- Bütün dosyaların ayrı **New legislation** kartı olduğunu ve ana belge-ek bağlantılarının çalıştığını kontrol edin.
- İlk madde başlığı, uzun bir maddenin devamı, tablo ve yürürlük maddesini kaynak PDF ile karşılaştırın.
- Tarih alıntısını ve varsa maddeye özel yürürlük istisnasını kontrol edin. Eski mevzuat atfındaki tarihin yeni belgeye aktarılmadığını doğrulayın.
- Bir chunk alanını değiştirip **Discard chunk changes** deneyin. Sonra gerekli bir düzeltmeyi kaydedin; yalnızca o dosyanın etiketlerinin yenilendiğini kontrol edin.
- Varsa önceki mevzuatı seçin; mülga bilgisi için açık hüküm ve tarih kullanın.
- DEV'de yayımlanması istenen dosyalar için toplu onay verin ve **completed** durumunu bekleyin. Sonra yeni belgeleri PC Külliyatı'nda arayın ve ek bağlantılarını yeniden kontrol edin.

Markdown yolunu ayrıca denemek için hazır örnek ZIP'i kullanın. Bu örnekte beklenen yapı bir ana belge ve iki ayrı ektir.

## 9. Ek bağlantısı olan gerçek PDF örneği

Örnek kaynak, 5 Ocak 2025 tarihli Resmî Gazete'de yayımlanan **“Ev ve Büro Tipi Elektrik-Elektronik Cihazların Hazırda Bekleme, Kapalı ve Ağ Bağlantılı Hazırda Bekleme Konumlarının Elektrik Enerjisi Tüketimi ile İlgili Çevreye Duyarlı Tasarım Gerekliliklerine Dair Tebliğ (2023/826/AB) (SGM-2024/5)”** belgesidir.

Üç sayfalık `resmigazete.gov.tr_eskiler_2025_01_20250105-3.htm.pdf` dosyasını **File** yoluyla seçin. **Original document URL** için şu kaynak adresini kullanın:

https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-3.htm

PDF'nin üçüncü sayfasındaki **“Ekleri için tıklayınız.”** bağlantısı şu ek PDF'ye gider:

https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-3-1.pdf

İncelemede ana tebliğin ve alınabilen ekin ayrı mevzuat kartları olarak görünmesini, aralarında bağlantı kurulmasını ve kaynaktaki yürürlük hükmünün doğru değerlendirilmesini kontrol edin. Ek adresi erişilemiyorsa eksikliğin açıkça gösterilmesini bekleyin; “ekler mevcut” varsayımıyla onay vermeyin. Bu örnek için tamamlanmış test sonucu, yayın kontrolü kaydında ayrıca belirtilmelidir.
