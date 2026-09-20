# Kaynak hazırlığı

`Prepare source`, verilen sayfayı ve içeriğinde keşfedilen ek bağlantılarını indirir. Resmî Gazete örneğindeki `20260704-17.htm` sayfası, `20260704-17-1.pdf` dosyasına bağlanır. Bu dosya 12 sayfalık görüntü içerir; güvenilir metin ve tablo yapısı için görsel modelle okunur.

URL, doğrudan dosya yükleme ve yapıştırılan metin aynı paket hazırlama akışını kullanır. Görsel okuma bütçesi hem PDF sayfalarına hem de yüklenen görsellerin karelerine uygulanır. Metin, HTML, DOCX ve XLSX kaynakları kendi ayrıştırıcılarıyla okunur.

İndirme ve kaynak keşfi 180 saniyeyle sınırlıdır. Paket hazırlığının toplam süre sınırı, yeniden görsel okuma gerektiren PDF'lerin toplam sayfa sayısına göre hesaplanır:

`min(7200, 180 + 60 + 90 × sayfa sayısı)` saniye.

Bu bir azami çalışma süresidir; işlem erken biterse bekletilmez. 12 sayfa için üst sınır 22 dakika, 50 sayfa için 79 dakikadır. Daha önce doğrulanıp kaydedilmiş PDF çıktıları tekrar modele gönderilmez.

Sayfa sayısı, kaynak dosyasından izole parser ile elde edilir. En çok üç sayfalık PDF tek çağrıda işlenir; daha uzun PDF'ler gerçek, ardışık ve en çok üç sayfalık alt PDF'lere bölünerek gruplar halinde modele gönderilir. Sayfalar sırayla okunur, orijinal sayfa numaraları ve görsel kanıt kontrolleri korunur. Kaynak PDF çağrıları nonstreaming çalışır; her çağrı ve toplam hazırlık için süre kontrolleri uygulanır. Diğer LLM akışlarının varsayılan davranışı değişmez.

Normal tablo hücreleri satır düzeninde aktarılır. Birleşik veya uzun form hücrelerinde satır ilişkisi belirsizse, hücreler görsel konumlarına göre ayrı ayrı aktarılır. Gerçek iki boyutlu hücre çakışması doğrulama hatasıdır ve mevcut model düzeltme denemesini tetikler; yalnız yatay aralığın aynı olması çakışma sayılmaz.

Yeni PDF transkriptleri v2 kuralını kaydeder; sürüm bilgisi olmayan eski kayıtlar özgün v1 kurallarıyla doğrulanır. Böylece önceki kaynakların metinleri ve hashleri değişmez. Görsellerden yalnız gerçekten görünen metin çıkarılır; fotoğraf betimlemesi veya tahmini mevzuat oluşturulmaz. Analiz, kullanıcı bağlamını da değerlendirir: “bu tablonun yeni hali” gibi güncelleme niyeti geçerlidir ve resmî değişiklik ifadeleri aranmaz. Güncelleme bağlamı olmayan ilgisiz içerikten talimat veya taslak üretilmez; eksik hedef ve değerler uydurulmaz.

İndirme sonrasında worker'ın iş sahipliği süresi, hesaplanan kalan süreye beş dakikalık kayıt payı eklenerek uzatılır. Uzatma için aynı ortam, token, aktif durum ve henüz dolmamış sahiplik gerekir. Eski worker veya tekrar tıklama aynı paketi eşzamanlı çalıştıramaz.

Paket işlenirken aynı kaynağın `Prepare source` düğmesi pasif kalır. Süre aşımı anlaşılır bir mesajla gösterilir; `Retry` aynı paketi yeniden ele alır. Eksik veya doğrulanmamış içerik hazır sayılmaz.

Bu değişiklikler kaynak hazırlığıyla sınırlıdır. Kanonik chunk, embedding ve labeling kayıtlarını değiştirmez. DEV deploy'u ortak background worker'larını yeniden başlattığı için çalışan labeling geçici olarak duraklayabilir; mevcut kalıcı iş kaydı ve kurtarma görevi üzerinden devam eder. Deploy sonrasında aynı labeling iş kimliği ve hazırlık sayacının ilerlemesi kontrol edilmelidir.
