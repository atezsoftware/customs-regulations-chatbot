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

**Düzeltme:** Hazır veya kısmi incelemede `After` alanlarını değiştirin. Aynı incelemede başka chunk'a, **Previous chunks / Next chunks** ile başka sayfaya veya ana belge/ek kartına geçebilirsiniz. Metin, başlık yolu, tür, metadata ve tarih taslakları korunur; dosya kartını gizleyip açmak da taslağı silmez. Taslaklar kaydedilene kadar bu ekrandadır.

Birden fazla chunk'ı düzenlediğinizde üstteki **Save all chunk changes** ile hepsini birlikte kaydedin. Bu işlem tek bir etiketleme çalışması başlatır; yalnızca değişen dosyalar yeniden işlenir, diğer dosyaların doğrulanmış etiketleri korunur. Tek chunk düzenlediyseniz karttaki **Save chunk changes** de kullanılabilir. Yeni etiketleme bitince sonucu tekrar inceleyin. Etiketleri bu tablodan doğrudan değiştirmezsiniz.

Kaydetmeden önce düzenlenen metinler ve bunlardan yeniden oluşacak toplu chunk'lar, hazırlamadaki metin boyutu sınırına göre kontrol edilir. Bir parça sınırı aşarsa hata, ilgili chunk'ı, ölçülen boyutu ve izin verilen sınırı gösterir; hiçbir değişiklik kaydedilmez ve etiketleme başlamaz. Taslaklarınız korunur. Sığdırmak için hukuki içeriği çıkarmayın; kaynağı koruyan biçim düzeltmesi yeterli değilse yöneticiden yeniden hazırlama desteği isteyin.

**Discard chunk changes** tek chunk'ın, **Discard all chunk changes** bütün sayfa ve dosyalardaki kaydedilmemiş değişiklikleri geri alır. Kaydetme başarısız olursa taslaklar korunur. İnceleme sürümü değişmişse uyarıyı okuyup güncel kaynakla yeniden değerlendirin. Kaydedilmemiş değişiklik varken başka içe aktarıma geçiş, yeniden deneme, yeniden hazırlama ve toplu onay kapalıdır. Etiketleme çalışırken düzenleme kapalıdır. Teknik kimlikler, kaynak bilgileri ve ilişki/audit alanları salt okunurdur.

Etiketleme başarısızlıklarının sayısı hem dosya kartında hem içe aktarımın üstünde gösterilir. İlgili chunk'ta **Labeling: failed** ve hata nedeni bulunur. Bunu yalnızca etiket atanmamış olmasıyla karıştırmayın; başarısız veya kanıtı yetersiz bir chunk varken toplu onay açılmaz.

Metin, başlık yolu veya chunk türü değişirse o dosyaya ait önceki LLM kanıtları güncel değil olarak işaretlenir ve LLM'den türetilmiş tarihler temizlenir. Bu alanları güncel bağlama göre yeniden kontrol edin. Gerçekten elle düzelttiğiniz değerler korunur. Önceki ve düzeltilmiş chunk revizyonları saklanır. Dosya bağlamı uyarısı metadata bölümünde, eski LLM kanıtı uyarısı ilgili alanda gösterilir; bu, her boş veya yapısal alanın değiştiği anlamına gelmez. **Retry new documents**, manuel metin düzenlemesinden sonra metadata'yı LLM'den yeniden çıkarmaz; etiketlemeyi yeniden dener. Metadata değerlerini ve dayanaklarını onaydan önce güncel kaynakla karşılaştırın.

Türetilmiş toplu chunk'ların okumada kullanılacağı tarih aralığı, bağlı oldukları bütün kaynak parçalarının geçerlilik aralıklarının kesişimiyle sınırlıdır. Bu sınırlandırma, kayıtlı kaynak metnini ve elle onayladığınız tarih değerlerini değiştirmez.

**Chunk sınırı hatası varsa önce yeniden hazırlamayı değerlendirin.** Kaynakta devamı olduğu halde yalnız “2.” veya tek başına bir tablo başlığı oluşmuşsa, incelemenin altındaki **Reprepare chunks from frozen source** bölümüne bakın. Bu işlem aynı yeni dosyaları, dondurulmuş dönüştürülmüş metni ve ana belge-ek ilişkilerini kullanır; güncel hazırlama kurallarıyla chunk'ları ve etiketleri yeniden üretir. PC Külliyatı'ndaki diğer kayıtları değiştirmez.

Yeniden hazırlama, yalnızca henüz onaylanmamış, aktif çalışması kalmamış ve chunk’larda kaydedilmiş manuel değişiklik içermeyen uygun bir içe aktarımda açılır. Ekran, uygun değilse sunucunun engelleme nedenini gösterir. **Bu nedenle sınırları, manuel düzeltmeleri kaydetmeden önce kontrol edin.** Kaydedilmemiş taslakları önce geri almanız gerekir; kaydedilmiş manuel değişiklik varsa yeniden hazırlama bunların üzerine yazmaz. Bu işlem dondurulmuş kaynak metnindeki okuma/yazım hatasını düzeltmez. Kaynak metni düzeltmek için orijinal belgeye dayanarak `After` alanlarını kullanın. Bir etiketi zorla alabilmek için kaynakta olmayan hüküm eklemeyin.

## 5. Önceki mevzuat ve mülga ilişkisini belirleyin

Önceki mevzuat incelemesini dosya bazında açın. Adaylar belgenin kendi hukuki başlığı ve numarasıyla değerlendirilir. Ham URL veya `20250105-2.htm.pdf` gibi teknik dosya adı tek başına mevzuat kimliği değildir; metindeki başka düzenlemelere ait atıf numaraları da belgenin kendi numarası sayılmaz. Gerekirse **Search previous legislation** ile ad veya numara arayın ve kaynağa uyan kaydı **Select ...** ile seçin. Bu, belge düzeyindeki kısa bir kontroldür; chunk karşılaştırması değildir.

Açık kaldırma hükmünde adı geçen eski düzenleme sette yoksa başka bir kaydı onun yerine seçmeyin. Aday bulunmaması, benzer adlı dosyanın eski düzenleme sayılmasını gerektirmez; doğru kayıt bulunana kadar ilişkiyi boş bırakın.

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

Onaydan sonra **approving / indexing**, işlemin sırada veya sürmekte olduğunu gösterir. **completed**, bütün yeni dosyaların indekslenmesi tamamlandığında görünür. Yalnızca onaya basılmış olmasını tamamlanma saymayın. Updates ekranında tamamlanma göründükten sonra PC Külliyatı dosya listesi eski durumu gösteriyorsa sayfayı yenileyin; yeniden **Index** düğmesine basmanız gerekmez.

Tamamlanan belgeyi PC Külliyatı dosya listesindeki **PDF** veya dosya ayrıntısındaki **Document PDF** ile açabilirsiniz. Kaynak PDF ise yüklenen orijinal PDF aynen gösterilir. Office, HTML veya görsel gibi diğer içe aktarılan biçimlerde, kaynak dosyanın ve hazırlanmış metnin bütünlüğü doğrulanır; hazırlanmış metin PDF olarak gösterilir ve başlıkta **(extracted text)** yazar. Markdown dosyaları metinlerinden PDF olarak görüntülenir.

Geçmiş içe aktarımlar ekrandaki belge adı ve durum düğmelerinde listelenir. Devam eden veya hatalı bir işlemi buradan tekrar açın. Yeni bağımsız bir yükleme için **Start another import** kullanın.

## 7. Hata olduğunda ne yapmalı?

| Ekrandaki durum | Kullanıcı işlemi |
| --- | --- |
| Eksik/indirilemeyen ek veya kaynak paketi hatası | Adresi, Original document URL alanını veya kaynak dosyayı kontrol edin. Sunuluyorsa Retry source preparation kullanın. |
| ZIP yolu, manifest veya bağlantı hatası | Bildirilen paket yolunu düzeltin; dosya adlarını, parent değerlerini ve gerçek yerel bağlantıları eşleştirip paketi yeniden yükleyin. |
| Metadata ya da etiketleme hatası; partial / failed | İlgili dosya ve chunk hata açıklamasını okuyun. Sunuluyorsa Retry new documents kullanın. Hazır görünmeyen işlemi onaylamayın. |
| Etiket alıntısı chunk metniyle eşleşmiyor | Chunk'ı orijinal kaynakla karşılaştırın. Metin doğruysa Retry new documents kullanın: geçerli başarılı etiketler korunur, yalnız başarısız veya güncelliğini yitirmiş sonuçlar yeniden işlenir. Metni yalnız orijinal kaynağa göre düzeltin. |
| Kaydetme sırasında inceleme değişti uyarısı | Taslaklar korunur. Gerekirse düzenleme metninizi kopyalayın; güncel durumu yeniden açın, güncel chunk ve tarihleri kontrol edip tekrar kaydedin. |
| Tek başına kalan başlık veya yanlış chunk sınırı | Manuel düzeltmeleri kaydetmeden önce Reprepare chunks from frozen source uygunluğunu kontrol edin. Engelleniyorsa gösterilen nedeni değerlendirin; Retry etiketleme, mevcut chunk sınırını değiştirmez. |
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
- Chunk sınırı hatası varsa, manuel düzeltme kaydetmeden önce **Reprepare chunks from frozen source** uygunluğunu kontrol edin.
- Ana belge ve ekte farklı sayfalardaki birkaç chunk alanını değiştirin; sayfalar arasında dönünce taslakların korunduğunu kontrol edin. Önce **Discard all chunk changes**, sonra gerekli düzeltmeler için **Save all chunk changes** deneyin; tek etiketleme çalışmasıyla yalnız değişen dosyaların işlendiğini kontrol edin.
- Başarısız etiket sayısının ve ilgili chunk'ın hata nedeninin açık olduğunu, bu sırada onayın kapalı kaldığını kontrol edin.
- Varsa önceki mevzuatı seçin; mülga bilgisi için açık hüküm ve tarih kullanın.
- DEV'de yayımlanması istenen dosyalar için toplu onay verin ve **completed** durumunu bekleyin. Sonra yeni belgeleri PC Külliyatı'nda arayın ve ek bağlantılarını yeniden kontrol edin.

Markdown yolunu ayrıca denemek için hazır örnek ZIP'i kullanın. Bu örnekte beklenen yapı bir ana belge ve iki ayrı ektir.

## 9. Ek bağlantısı olan gerçek PDF örneği

Örnek kaynak, 5 Ocak 2025 tarihli Resmî Gazete'de yayımlanan **“Ev ve Büro Tipi Elektrik-Elektronik Cihazların Hazırda Bekleme, Kapalı ve Ağ Bağlantılı Hazırda Bekleme Konumlarının Elektrik Enerjisi Tüketimi ile İlgili Çevreye Duyarlı Tasarım Gerekliliklerine Dair Tebliğ (2023/826/AB) (SGM-2024/5)”** belgesidir.

Üç sayfalık `resmigazete.gov.tr_eskiler_2025_01_20250105-3.htm.pdf` dosyasını **File** yoluyla seçin. **Original document URL** için şu kaynak adresini kullanın:

https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-3.htm

PDF'nin üçüncü sayfasındaki **“Ekleri için tıklayınız.”** bağlantısı şu ek PDF'ye gider:

https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-3-1.pdf

İncelemede ana tebliğin ve alınabilen ekin ayrı mevzuat kartları olarak görünmesini, aralarında bağlantı kurulmasını ve kaynaktaki yürürlük hükmünün doğru değerlendirilmesini kontrol edin. Ek adresi erişilemiyorsa eksikliğin açıkça gösterilmesini bekleyin; “ekler mevcut” varsayımıyla onay vermeyin. Bu örneğin yerel doğrulama sonucu aşağıdadır.

Kimlik ve yayım bilgilerini ana PDF'nin ilk sayfasından, yürürlük istisnalarını ana PDF'nin 3. sayfasından, ek sürelerini ek PDF'nin fiziksel 3. sayfasından karşılaştırın:

| Kaynak / alan | Beklenen değer |
| --- | --- |
| Ana tebliğin kendi numarası | `SGM-2024/5` |
| Ana tebliğin yayımı | `2025-01-05`; Resmî Gazete sayısı `32773` |
| Madde 1, 11, 12 ve madde 6'nın (1) numaralı fıkrası | Yürürlük başlangıcı `2025-01-05` |
| Diğer hükümler; madde 6'nın (2) ve (3) numaralı fıkraları dahil | Yürürlük başlangıcı `2025-05-09` |
| Ek II, 1(a) | `2025-07-01` tarihine kadar geçerli hüküm; geçerlilik sonu `2025-07-01` |
| Ek II, 1(b) | `2025-07-01` tarihinden itibaren geçerli hüküm; başlangıç `2025-07-01` |
| Ekin kendi belge numarası | Bilinmiyor; boş bırakılır. `SGM-2024/3`, atıf yapılan kapsam dışı düzenlemedir. |

Madde 11(b), diğer hükümlerin 9 Mayıs'ta başlayacağını anlatır; madde 11'in kendi yürürlük başlangıcı yine `2025-01-05` olur. Boş veya yanlış LLM alanlarını **After** tarafında kaynakla karşılaştırarak düzeltin ve onaydan önce **Save chunk changes** ile kaydedin; yenilenen etiketlemeyi de inceleyin.

## 10. Gerçek örneğin yerel doğrulama sonucu

26 Ekim 2026 tarihli yerel testte yukarıdaki ana PDF ve bağlı ek PDF, aynı içe aktarımda ayrı mevzuat kayıtları olarak işlendi. İki dosya da **completed** durumuna ulaştı ve indeksleme istekleri başarıyla tamamlandı. Test kapsamı yerel geliştirme ortamıdır; örnek DEV ortamına yüklenmedi.

| Kontrol | Yerel test sonucu |
| --- | --- |
| Ana belge ve ek | Ayrı kayıtlar; ana belge-ek bağlantıları ve aralarında geçiş doğrulandı. |
| Chunk'lar | Ana belgede 41, ekte 136; toplam 177 chunk. Bunların 171'i atomik metin parçası, 6'sı bu parçalardan türetilmiş toplu chunk'tır. |
| Etiketleme | 171 atomik parçanın tamamının etiketlemesi tamamlandı. Yalnızca bu içe aktarımın yeni dosyaları işlendi. |
| İnsan incelemesi | Kaynakla karşılaştırılan iki tarih alanı elle düzeltildi ve kaydedildi. Bu değerler otomatik çıkarım başarısı olarak sayılmadı; manuel kaynak bilgisi korundu. |
| İnceleme ve onay | Kaydedilmemiş değişiklik varken onayın kapalı olması, Discard ile geri alma ve Approve all new documents akışı doğrulandı. |

Bu örnek, otomatik sonuçların onaydan önce kaynakla karşılaştırılması gerektiğini gösterir. Özellikle maddeye özel yürürlük tarihlerini ve eklerin ayrı kimliklerini kontrol edin.

## 11. İkinci gerçek PDF'nin yerel kullanıcı ekranı testi

27 Ekim 2026 tarihli yerel testte, `SGM-2024/4` ana belgesinin üç sayfalık PDF'si ve bağlı 21 sayfalık taranmış ek PDF, aynı içe aktarımda kullanıcı ekranından incelendi. **Approve all new documents** ile verilen tek toplu onayın ardından iki dosya da **completed** durumuna ulaştı. Bu test yerel geliştirme ortamındadır; dosyalar DEV ortamına yüklenmedi. Test, önceden 16 test dosyası bulunan yalıtılmış yerel veritabanında yapıldı; iki yeni dosyayla toplam 18 oldu. Bu sayı gerçek DEV külliyatının boyutunu göstermez.

- Ana kaynak: https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-2.htm
- Ek PDF: https://www.resmigazete.gov.tr/eskiler/2025/01/20250105-2-1.pdf

| Kontrol | Yerel kullanıcı ekranı sonucu |
| --- | --- |
| Ana belge ve ek | Ayrı mevzuat kayıtları; ana belgede 43, ekte 177, toplam 220 chunk. Bunların 212'si atomik, 8'i türetilmiş toplu chunk. |
| Kaynak ve chunk incelemesi | Aynı dondurulmuş kaynaklardan yeniden hazırlama kullanıldı. Yedi OCR/metin düzeltmesi, ana belgedeki iki 5 Ocak 2025 tarihi ve bir Yürütme başlığı alanı elle düzeltildi: toplam 10 manuel alan. Bunlar otomatik çıkarım başarısı sayılmadı. |
| Toplu kayıt ve taslak koruması | Yedi metin ve iki tarih düzeltmesi **Save all chunk changes** ile birlikte kaydedildi. İlk denemede 313 token'lık parça, hazırlamada sabitlenmiş 312 token'lık içerik sınırını aştığı için kayıt durdu; taslaklar korundu. Kaynağı koruyan biçim düzeltmesiyle 311 token'a indirilen kayıt birlikte uygulandı. Başlık metadata'sı tek chunk kaydıyla düzeltildi. |
| Etiketleme | İki başarısız sonuç yeniden işlenirken 210 geçerli sonuç korundu; sonunda 212 atomik parçanın tamamı etiketlendi. Alıntı doğrulaması gevşetilmedi; etiketi geçirmek için kaynakta olmayan hüküm eklenmedi. |
| Önceki mevzuat | Açık kaldırma hükmünde atıf yapılan `SGM-2013/6` için sette aday bulunmadı. Başka bir kayıt onun yerine seçilmedi; yanlış mülga ilişkisi kurulmadı. |
| Onay ve indeksleme | Aynı içe aktarımın iki dosyası da toplu onaydan sonra **completed** oldu. Geçmişten yeniden açıldığında durum korundu; PC Külliyatı listesi yenilenince iki dosya da Completed, chunk sayıları 43 ve 177 olarak göründü. |
| Mevcut kayıtların korunması | Önceden var olan 16 test dosyası, bunların 204 kanonik chunk'ı, 758 etiket kaydı, 652 blobun içeriği (toplam 94.484.817 bayt), 580 arama indeksi belgesi ve indeks kimlikleri birebir korundu. Uygulamanın genel yayınlama sayacı bu işlem sırasında ilerler; karşılaştırma bütün veritabanının bayt eşitliği iddiası değildir. |
| Kaynak bütünlüğü | Ana belge-ek kaynak bağlantıları ve orijinal PDF baytları aynı kaldı. Orijinal 21 sayfalık ek PDF, tamamlanmadan sonra kullanıcı ekranından açıldı. |

**Görsel kaynak sınırı:** Orijinal PDF'ler korundu. Taranmış ekteki grafik ve şekiller metinde bütünüyle yeniden üretilmedi; bu bilgileri orijinal PDF'den kontrol edin. Tamamlanan indeksleme, otomatik metin ve metadata'nın insan incelemesinden bağımsız olarak eksiksiz veya hukuken doğru olduğu anlamına gelmez.
