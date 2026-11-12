# ASv3 Tek / Exp Cropped session dosyaları

PDF, DOCX, XLSX, CSV, TXT ve MD dosyaları özel sohbet eki olarak yüklenir.
Document set, mevzuat kataloğu, Elasticsearch veya embedding indeksine eklenmez.
PDF orijinali saklanır; yükleme sırasında yalnızca dosya boyutu, şifre ve sayfa
sınırları doğrulanır. PDF için metin katmanı, OCR veya ön okuma gerekmez.
Diğer formatlardan yerel parser metin çıkarır. Yükleme LLM çağrısı veya
importer/Celery işi başlatmaz.

Soru sorulduğunda orijinal PDF, soru ve araştırma talimatlarıyla birlikte ilk
normal ASv3 karar isteğine native dosya olarak eklenir. PDF içeren bir sorunun
ilk karar modeli `gemini-3.5-flash-lite` olur; ayrı bir belge okuma/özetleme
çağrısı yapılmaz. Sonraki kararlar mevcut model politikasını kullanır ve
orijinallere aynı istek içinden erişir. Dosya içeriği talimat kabul edilmez.

Web'de varsayılan sohbet ekranındaki **Dosya ekle** düğmesi doğrudan cihazın
dosya seçicisini açar; Recent Files listesi gösterilmez. Dosya seçildiğinde ASv3
Tek etkinleşir. ASv3 Tek veya Exp Cropped zaten seçiliyse bu seçim korunur.
Yüklenen dosyalar mesaj kutusunda, gönderildikten sonra ilgili session'da görünür.

Bu doküman bu değişikliğin API sözleşmesidir; kurulu bir ortamda kullanılabilmesi
için bu sürümün backend ve web bileşenlerinin yayımlanması gerekir.

## Kimlik ve session sınırı

Mevcut browser oturumu, API key veya PAT kullanılır. Örnekler frontend proxy
adresindeki `/api` yollarını kullanır. Dış uygulamanın backend'i API key'i tutmalı;
key tarayıcıya verilmemelidir.

Session'ı oluşturan kimlikle yükleme, mesaj gönderme ve dosya işlemleri yapılır.
API key mevcut service account kimliğine çözülür. `user` alanındaki dış kullanıcı
adı gösterim/attribution bilgisidir; dosya erişim yetkisi sağlamaz. Bir key'i
paylaşan uygulama kullanıcılarının ayrımını çağıran uygulama uygulamalıdır.
Yükleme/kaldırma `WRITE_CHAT`, listeleme/indirme `READ_CHAT` gerektirir.
Admin yetkisi başka bir sahibin dosyasını açmaz; public session paylaşımı dosya
erişimi sağlamaz. Default assistant (`persona_id: 0`) ve proje dışı session gerekir.

## Önerilen çağrı sırası

### 1. Session oluştur

```http
POST /api/chat/create-chat-session
Authorization: Bearer <API_KEY>
Content-Type: application/json

{"persona_id": 0, "user": "Dış Kullanıcı Adı"}
```

```json
{"chat_session_id": "01234567-89ab-cdef-0123-456789abcdef"}
```

### 2. Dosyaları aynı session'a yükle

```bash
curl --fail-with-body "$BASE_URL/api/chat/files/upload" \
  -H "Authorization: Bearer $API_KEY" \
  -F "chat_session_id=$SESSION_ID" \
  -F "files=@/absolute/path/beyanname.pdf" \
  -F "files=@/absolute/path/kalemler.xlsx"
```

`Content-Type` başlığını elle vermeyin; multipart boundary'yi HTTP istemcisi üretir.
Birden fazla dosya için aynı `files` alanı tekrarlanır.

```json
{
  "files": [
    {
      "id": "d7d95615-8755-47f2-9746-493d8ef53892",
      "type": "document",
      "name": "beyanname.pdf",
      "token_count": 1200,
      "size": 48123,
      "is_session_attachment": true
    }
  ],
  "rejected_files": [
    {"name": "sifreli.pdf", "reason": "Password-protected PDF files are not supported."}
  ]
}
```

`type` için sunucunun döndürdüğü değeri kullanın. HTTP 200 bütün dosyaların kabul
edildiği anlamına gelmez: `rejected_files` mutlaka kontrol edilmelidir. Reddedilen
dosya modelin bağlamına girmez. Aynı dosyayı tekrar yüklemek yeni bir ek oluşturur.

`chat_session_id` verilmezse dosya sahibine bağlı taslak yüklenir. İlk mesajdaki
`file_descriptors` ile bir session'a bağlanır. Bağlandıktan sonra başka session'a
taşınamaz. Web arayüzü ilk sorudan önce dosya eklemek için bu yolu kullanır.

### 3. ASv3 Tek veya Exp Cropped sorusu gönder

```http
POST /api/chat/send-chat-message
Authorization: Bearer <API_KEY>
Content-Type: application/json

{
  "chat_session_id": "01234567-89ab-cdef-0123-456789abcdef",
  "message": "Yüklediğim beyannamenin ve kalemlerin gümrük yükümlülüklerini incele.",
  "atez_search_v3": true,
  "asv3_research_profile": "normal",
  "asv3_exp_cropped": false,
  "file_descriptors": [
    {
      "id": "d7d95615-8755-47f2-9746-493d8ef53892",
      "type": "document",
      "name": "beyanname.pdf",
      "is_session_attachment": true
    }
  ],
  "stream": true
}
```

ASv3 Tek için `asv3_exp_cropped: false`, Exp Cropped için
`asv3_exp_cropped: true` kullanılır. İki durumda da `atez_search_v3: true` ve
`asv3_research_profile: "normal"` gerekir. Mevcut model seçim alanları ve
mesaj response/stream sözleşmesi değişmez. Tam JSON yanıt için `stream: false`
gönderilebilir; uzun araştırmalar için istemci timeout'u buna uygun olmalıdır.

Session'a önceden bağlı bütün aktif ekler otomatik kullanılır. Sonraki sorularda
aynı `chat_session_id` yeterlidir; dosyaları yeniden yüklemek veya descriptor'ları
yeniden göndermek gerekmez. `file_descriptors`, yeni taslakları bağlamak ve
mesajda dosya kartını göstermek için kullanılır. Dosya adı/tipi sunucudaki kayıttan
yeniden doğrulanır; payload dosya kimliği veya sahipliği değiştiremez.

## Listeleme, indirme ve kaldırma

| İşlem | Endpoint | Sonuç |
| --- | --- | --- |
| Session eklerini listele | `GET /api/chat/sessions/{session_id}/files` | Aktif dosya snapshot listesi |
| Sahibe ait bağlanmamış taslakları listele | `GET /api/chat/files` | Taslak snapshot listesi; kullanılmayan taslaklar silinebilir |
| Orijinal dosyayı indir | `GET /api/chat/files/{file_id}` | Sahibine özel dosya indirme |
| Session'dan kaldır | `DELETE /api/chat/sessions/{session_id}/files/{file_id}` | 204; sonraki sorular bu eki kullanmaz |
| Bağlanmamış taslağı sil | `DELETE /api/chat/files/{file_id}` | 204; taslak dosya silinir |

Session'dan kaldırma önceki mesajdaki orijinal eki korur. Önceden üretilmiş
yanıtları veya başlamış araştırmayı geriye dönük değiştirmez. Korunan orijinaller
session'ın toplam 20 MiB depolama sınırına dahil kalır; kaldırma bu kapasiteyi
boşaltmaz ve dosyayı sessizce silmez. Kapasite dolduğunda yeni session açılmalıdır.
Bir araştırmayı
`asv3_resume_message_id` ile tekrar açarken ek kümesi/içeriği değişmişse istek
reddedilir; güncel dosyalarla yeni soru gönderilmelidir.

## Formatlar, sınırlar ve hata davranışı

- Bir session'da en fazla 10 aktif dosya ve toplam 200.000 aktif çıkarılmış metin
  karakteri bulunabilir. Session'da korunan bütün orijinallerin toplamı, kaldırılan
  ekler dahil, 20 MiB'ı aşamaz. Sahibin bağlanmamış taslakları da en fazla 10 dosya,
  toplam 20 MiB orijinal içerik ve 200.000 çıkarılmış metin karakteri içerebilir.
- PDF en fazla 500 sayfa; Office ZIP genişlemesi en fazla 64 MiB olabilir.
- XLSX/CSV en fazla 20.000 satır ve 200.000 hücre; XLSX en fazla 100 sayfa içerir.
- Bir yükleme isteği en fazla 10 dosya içerir. Session/taslak kapasitesi ayrıca denetlenir.
- Model bağlamı için bütün session eklerinin toplamı seçilen modelin giriş
  kapasitesinin yarısını ve 64.000 token'ı aşamaz. Aşımda 413 döner; metin kesilmez.
  PDF sayfası başına 4.096 token bağlam rezervi ayrılır; base64 taşıma verisi
  metin token'ı sayılmaz. PDF `token_count` değeri bu rezervdir, faturalandırılmış
  veya sağlayıcıdan ölçülmüş tüketim değildir.
- PDF'ler metin katmanı bulunmasa da kabul edilir; taranmış sayfalar, tablolar ve
  görseller orijinal PDF içinden model tarafından okunur. Şifreli, bozuk, boş
  veya sayfa sınırını aşan PDF reddedilir. DOCX'ten yerel metin çıkarılır;
  DOCX içindeki gömülü görseller bu native PDF akışına dahil değildir.
- XLSX formülleri formül olarak korunur; çalıştırılmaz veya hesaplanmaz.
  CSV/TXT/MD UTF-8 ya da BOM içeren UTF-16 olmalıdır. XLS, DOC ve arşiv yükleme
  desteklenmez.
- Dosya içeriği kullanıcı vaka verisi kabul edilir. Hukuki iddialar yetkili
  mevzuat kaynaklarından doğrulanır; ekler mevzuat citation numarası almaz.

API hataları standart `{"error_code": "...", "detail": "..."}` biçimindedir.
Yetkisiz/başka session'a bağlı/kaldırılmış descriptor'lar kabul edilmez.
