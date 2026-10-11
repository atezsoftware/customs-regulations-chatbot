# Session feedback API

Her session için tek güncel değerlendirme tutulur. Mevcut mesaj beğeni/beğenmeme
kayıtları ayrı çalışmaya devam eder.

| `rating` | Label |
| --- | --- |
| 1 | Yanlış |
| 2 | Eksik |
| 3 | Kısmi doğru |
| 4 | Doğru |

## Kaydetme ve güncelleme

`PUT /api/chat/chat-session/{session_id}/feedback`

`session_id`, chat API'sinden alınan mevcut session UUID'sidir. Aynı endpoint ilk
kaydı oluşturur ve sonraki değerlendirmelerde onu günceller. Label sunucuda üretilir.

```http
PUT /api/chat/chat-session/01234567-89ab-cdef-0123-456789abcdef/feedback
Authorization: Bearer <MEVCUT_API_KEY>
Content-Type: application/json

{
  "rating": 3,
  "feedback_text": "İlgili istisna açıklanmamış.",
  "expected_version": 0
}
```

| Body alanı | Kural |
| --- | --- |
| `rating` | Zorunlu tam sayı: 1, 2, 3 veya 4. Boolean, float ve string kabul edilmez. |
| `feedback_text` | Opsiyonel string; dış boşluklar temizlendikten sonra en fazla 5.000 karakter. Boş metin `null` olur. Null karakteri kabul edilmez. |
| `expected_version` | Opsiyonel, sıfır veya pozitif tam sayı. İlk kayıt için 0; güncelleme için önceki response'un `version` değeri. |

`{"rating": 4}` geçerlidir. PUT tam değiştirme yapar: `feedback_text` gönderilmezse
veya `null` gönderilirse önceki açıklama temizlenir.

Başarılı çağrı 200 döndürür:

```json
{
  "chat_session_id": "01234567-89ab-cdef-0123-456789abcdef",
  "rating": 3,
  "label": "Kısmi doğru",
  "feedback_text": "İlgili istisna açıklanmamış.",
  "version": 1,
  "time_created": "2026-10-11T18:00:00Z",
  "time_updated": "2026-10-11T18:00:00Z",
  "evaluated_through_message_id": null
}
```

Aynı puan, metin ve mesaj kapsamında tekrar kaydetmek sürümü veya zamanı
değiştirmez. `expected_version` güncel sürümle uyuşmazsa 409 döner ve mevcut kayıt
korunur. Alan gönderilmezse son başarılı yazma geçerlidir. Web formu sürüm kontrolü
kullanır; çakışmada güncel kaydı yükler, kullanıcının taslağını korur ve tekrar
kaydetmesini bekler.

## Kimlik ve kapsam

Mevcut browser oturumu, API key veya PAT kullanılır; `WRITE_CHAT` yetkisi gerekir.
Session sahibi authenticated kullanıcıyla tam eşleşmelidir. API key kendi service
account'ına çözülür: session'ı oluşturan aynı key kullanılabilir. Admin olmak,
public paylaşım veya başka bir kullanıcının key'i sahiplik sağlamaz. Bu endpoint
sahipsiz session'a sahip atamaz ve payload'dan kullanıcı kimliği almaz.

`evaluated_through_message_id`, kaydetme anında aktif daldaki son kaydedilmiş,
görüntülenebilir assistant yanıtıdır. Normal sohbetlerde durdurulmuş kısmi yanıt da
kapsama girebilir; bu, eksik yanıtın değerlendirilmesini sağlar. Placeholder,
hatalı yanıt ve yayımlanmamış regulatory taslak kapsam dışında kalır. Uygun mesaj
yoksa değer `null` olur. Sohbet devam ettiğinde eski değerlendirme kendiliğinden
yeni yanıtlara taşınmaz; tekrar kaydetme mesaj kapsamını ve sürümü günceller.

Feedback yazmak session'ın sidebar sırasını değiştirmez. Soft delete kaydı korur
ama yeni yazmayı engeller; hard delete feedback'i cascade ile siler. Klonlanan
session değerlendirmeyi miras almaz.

## Okuma ve admin

Mevcut `GET /api/chat/get-chat-session/{session_id}` yanıtındaki `session_feedback`
alanı kaydı döndürür; kayıt yoksa `null` gelir. `is_shared=true` görünümünde sahibine
de `null` döner; açıklama public paylaşımda gösterilmez.

Admin Query History listesi/detayı puanı, açıklamayı ve zamanı gösterir. Mevcut
liste API'sinde bağımsız `session_rating=1..4` filtresi vardır. CSV'nin Q/A
satırlarında `session_feedback_rating`, `session_feedback_label`,
`session_feedback_text`, `session_feedback_updated_at` kolonları bulunur. Mevcut
mesaj feedback filtreleri ve kolonları korunur; yeni metin de CSV formula
sanitization işleminden geçer.

## Hatalar ve migration

| HTTP | Durum |
| --- | --- |
| 403 | Mevcut auth zincirinde eksik/geçersiz kimlik veya yetersiz permission/PAT scope. |
| 404 | Session bulunamadı, silinmiş, sahipsiz veya başka kullanıcıya ait. |
| 409 | Feedback sürümü değişmiş veya aktif mesaj dalı tutarsız. |
| 422 | Geçersiz UUID veya body doğrulaması. |
| 429 | Mevcut kullanıcı bazlı feedback rate limit aşıldı. |

Auth hataları mevcut handler biçimini kullanır; domain hataları `OnyxError`
(`error_code`, `detail`), validation hataları mevcut global 422 handler'ından
(`status_code`, `message`, `data`) döner.

Deploy edilen backend için Alembic head'e upgrade gerekir. Additive revision
`e5fc62d469ab`, `4ecf8032a916` sonrasına `chat_session_feedback` tablosunu ekler;
mevcut session ve mesaj feedback verilerini dönüştürmez.
