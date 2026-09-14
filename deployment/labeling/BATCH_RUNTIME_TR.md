# Chunk labeling Batch çalışma biçimi

Labeling, mevcut kanonik chunk kayıtlarının dondurulmuş metin ve bağlamını Gemini 3.8 Flash'a gönderir. Etiket tanımları iş başlarken veritabanından alınır. Etiketleme yeni kaynak chunk veya embedding oluşturmaz; mevcut arama indeksine yazmaz. Türetilmiş chunk'lar, kanonik kaynaklarının etiketlerini projeksiyon aşamasında alır.

## Hazırlık ve Google gönderimi

Yerel hazırlık 128 chunk'lık sayfalarla ilerler. Aynı sayfanın istekleri toplu SQL ile kaydedilir; her sayfada bütün işin sayaçları yeniden taranmaz. Hazırlık boyunca aynı anda işlenen veri miktarı sabit kalır. Bütün istekler hazırlandıktan sonra gönderim başlar.

Yerel hazırlık paketleri ayrı Google işleri değildir. Gönderim sırasında, aynı işin henüz gönderilmemiş paketleri birleştirilir. Tek Google Batch işi için üst sınır **200.000 istek veya 1.000.000.000 byte JSONL giriş** boyutudur; önce dolan sınır belirleyicidir. Büyük metin ve etiket açıklamaları nedeniyle byte sınırı genellikle önce dolar. Son paket daha küçük olabilir.

Bu sınırlar [Google'ın Batch dokümantasyonuna](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/capabilities/batch-inference) dayanır. Google, kendi kapasitesini, kuyruğunu ve iş içindeki paralelliği yönetir. Uygulama aynı anda en fazla dört uzak işi takip eder.

Birleştirme; daha önce denenmiş, sonucu belirsiz veya uzak iş kimliği bulunan paketlere dokunmaz. Dondurulmuş istekler tekrar hesaplanıp hash'leri doğrulanır. Üyelik değişimi tek veritabanı işlemiyle yapılır. Bu işlem bittikten sonra iptal durumu yeniden kontrol edilir.

## Bellek ve disk sınırları

| İşlem | Sınır |
| --- | --- |
| Yerel hazırlık / istek okuma / sonuç uygulama | 128 chunk |
| Google giriş dosyası | 1.000.000.000 byte, en fazla 200.000 istek |
| GCS upload / download tamponu | 8 MiB / 4 MiB |
| İstek korelasyon manifesti | 64 MiB, en fazla 200.000 kayıt |
| Ham ve normalleştirilmiş sonuç akışı | 8 GiB |
| Geçici sıkıştırılmış sonuç veritabanı | 2 GiB disk, yaklaşık 4 MiB SQLite cache |

Giriş dosyası yerel geçici dosyaya akışla yazılır. Sonuçlar da akışla okunur ve sıkıştırılarak geçici SQLite dosyasında tutulur; bütün sonuçlar bir Python sözlüğüne yüklenmez. Geçici dosyalar işlem sonunda ve hata durumunda temizlenir. Disk veya çıktı sınırı aşılırsa eksik veri üzerinden etiket uygulanmaz; GCS sonuçları mevcut saklama politikası boyunca inceleme için kalır.

## İptal, hata ve devam etme

Uzun aktarım sırasında worker sahipliği 30 saniyede bir kontrol edilip 300 saniyelik lease yenilenir. Süresi dolmuş veya başka worker'a geçmiş lease ile sonuç kaydedilemez.

Google işinin oluşturulmasından önce gelen iptal doğrudan yerelde uygulanır. Google'ın işi kabul etmiş olabileceği durumlarda aynı gönderim anahtarıyla uzak iş aranır; hemen ikinci bir ücretli iş açılmaz. Görünürlük için ayrılan 10 dakikalık süre, uzun dosya hazırlığı tarafından tüketilmez ve her sorguda yeniden uzatılmaz.

Tüm çıktıdaki üyelik, tekrar ve JSON sözleşmesi doğrulanmadan başarılı etiketler kaydedilmeye başlanmaz. Daha sonra sonuçlar 128'lik işlemlerle kaydedilir ve ilerleme sayaçları güncellenir. Worker bu aşamada kapanırsa aynı uzak çıktı yeniden okunur; daha önce tamamlanan chunk'lar tekrar uygulanmaz. Her chunk'ın bağlamı kendi dondurulmuş kaynak aralığıyla doğrulanır.

Birleştirme işlemi sırasında iptal isteği veritabanı kilidini bekleyebilir. Google aktarımı ve sonuç uygulaması boyunca ise iptal, periyodik sahiplik kontrolünde işlenir.

## Doğrulama kapsamı

Kontroller; gerçek, izole PostgreSQL üzerinde paket birleştirme, byte/istek sınırları, lease süresi, iptal, belirsiz gönderim, yarıda kesilen sonuç uygulaması, yanlış stale üretmeme ve toplu hazırlık yazımlarını kapsar. Provider taşıma ve geçici sonuç dosyası testleri; akış, üyelik, tekrar, boyut sınırı ve kaynak temizliğini doğrular.

128 istekli yerel hazırlık kaydı ölçümünde SQL sayısı 134'ten 5'e düştü. 50.000 satırlık sentetik işte yalnız kayıt adımı yaklaşık 1,06 saniyeden 0,25 saniyeye indi. Bu ölçüm canlı sistemin toplam etiketleme hızını veya Google işlem süresini temsil etmez.
