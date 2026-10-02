# ASv3 araştırma workflow’u

ASv3, mevcut ATEZ Search ve ATEZ Search v2 seçeneklerinden ayrı çalışan, kaynak araştırmasını araçlarla yöneten bir harness’tır. Giriş noktası `backend/onyx/asv3/runtime.py::run_asv3_loop` fonksiyonudur. Bu doküman uygulanan sözleşmeleri açıklar; belirli bir ortamın servislerinin, sağlayıcılarının veya bütün araçlarının canlı doğrulandığı anlamına gelmez.

## Seçim ve model

Frontend’de ASv3 seçimi `atez_search_v3: true` gönderir. ATEZ Search, v2 ve Deep Research aynı istekte birlikte seçilemez. Backend bu çakışmaları da doğrular. Mevcut kapsam, varsayılan asistanla ve proje dışında çalışmadır; özel asistan ve proje kapsamı desteklenmez. Birden fazla yanıt modeli seçilmesi de desteklenmez.

Koordinatör kullanıcı tarafından seçilen modelle çalışır. Araştırmacılar, iddia doğrulayıcısı, dil profili ve final üretimi aynı seçilmiş LLM nesnesini kullanır; ASv3 içinde sabit bir sağlayıcı/model seçimi bulunmaz. Paralel araştırmacı oluşturmak, farklı sağlayıcılara otomatik geçiş anlamına gelmez. Modelin araç çağrısı, bağlam ve gerektiğinde görsel giriş yetenekleri ayrıca uygun olmalıdır.

Frontend normal gönderme, düzenleme, yeniden üretme ve devam isteklerinde mevcut model seçimi akışını kullanır. ASv3 bayrağı yanıtın ilk boş yer tutucusuna da yazılır: ilk yerelleştirilmiş paket gelene kadar yalnızca nötr `ASv3` ve bekleme göstergesi görünür.

## Araştırmanın yürütülmesi

Koordinatör soruları, belirleyici olguları, olumsuz koşulları ve alternatif senaryoları kaydeder; ihtiyaca uygun araçları seçer. Bilinen kaynak ve madde doğrudan okunabilir. Kaynak kimliği belirsizse önce çözülür; gerektiğinde kaynak içi metin araması, mevcut indeks araması, atıf takibi ve asıl dosya incelemesi kullanılır. Tek bir başarısız arama, hükmün külliyatta bulunmadığını kanıtlamaz.

Bağımsız ihtiyaçlar araştırmacılara dağıtılabilir veya bağımlılıkları açık bir araç programıyla paralel yürütülebilir. Araştırmacılar aynı erişim kapsamını, kanıt defterini ve ortak bütçeyi paylaşır. Sonuçlar kaynak metniyle doğrulanır; araştırmacının özeti tek başına birincil hukuki kanıt sayılmaz. Finalden önce eksikler, kaynak koşulları, süreler ve atıflar ayrıca denetlenir. Bütçe veya servis sınırında ulaşılamayan hususlar kesin sonuç gibi sunulmamalıdır.

## Sabit araç kataloğu: 29 araç

Araç şemalarının gerçek kaynağı `registry.py`, `corpus_tools.py`, `source_tools.py`, `supplemental_tools.py`, `sandbox.py` ve `workers.py` içindeki fabrikalardır. Aşağıdaki tablo ana girdileri özetler; bütün isteğe bağlı alanlar ve sınırlar için bu şemalar veya `discover_tools` sonucu esas alınır.

| Araç | Ana girdi ve sözleşme |
| --- | --- |
| `discover_tools` | İsteğe bağlı `names`; çalışma için yetkili araçların şemalarını verir. Servisin sağlıklı olduğunu kanıtlamaz. |
| `read_evidence` | Global `citation`, isteğe bağlı karakter aralığı; kaydedilmiş özgün kanıt metnini okur. |
| `inspect_evidence_path` | `citation`; gözlenen kayıt ve final dahil edilme bilgilerini verir, görülmeyen aşamaları uydurmaz. |
| `read_research_state` | Girdi gerekmez; senaryo, araştırmacılar ve ortak bütçenin durumunu verir. Özgün metinler her durumda tekrar taşınmaz. |
| `resolve_source` | `query`, isteğe bağlı sayfalama; yetkili kanonik kaynak adaylarını bulur, birden fazla adayı belirsiz bırakır. |
| `read_source_range` | `source_id`, isteğe bağlı `start`, `limit`; sıralı kanonik parçaları okur ve devam konumunu bildirir. |
| `read_provision` | `source_id`, `article`; isteğe bağlı `paragraph`, `clause`; hükmü devamı ve gerekli bağlamıyla okur. Geçici ve mükerrer madde kimlikleri ayrıdır. |
| `search_source_text` | `source_id`, `pattern`, isteğe bağlı `mode: literal/regex`; kaynak içinde özgün konumlarıyla arar. Regex süreyle sınırlıdır. |
| `query_corpus` | `operation: inventory/headings` ve kapsam/sayfalama alanları; envanter veya başlık okur, serbest SQL çalıştırmaz. |
| `follow_reference` | Görülebilir başlangıç `source_id`, `chunk_id`; isteğe bağlı hedef kaynak/madde ve derinlik; açık mevzuat atfını takip eder. |
| `diagnose_source` | `source_id`; gözlenen kaynak, DB ve yayın metadata’sını inceler. Servis hatasını metin yokluğu saymaz. |
| `compare_versions` | `source_id`, `old_date`, `new_date`; açık tarihli kaynak görünümlerini karşılaştırır. Bilinmeyen geçerlilik sınırları bilinmiyor olarak kalır. |
| `search_corpus` | `query`, isteğe bağlı `mode: hybrid/keyword/full_text`; mevcut kapsamlı SearchTool indeks adaptörünü kullanır. |
| `open_source_file` | `source_id`, isteğe bağlı metin aralığı; erişimi ve sürümü uygun asıl dosyanın baytlarını doğrular, metin veya manifest verir. |
| `inspect_source_page` | `source_id`, birden başlayan `page`, isteğe bağlı `vision`; doğrulanmış PDF/görsel sayfasını ve mevcut native metni inceler. Görsel yorumlama model yeteneğine bağlıdır. |
| `extract_table` | `source_id`, gerektiğinde `page`, `sheet`, `start_row`, `limit`; CSV/XLSX/HTML veya PDF sayfasından konum bilgili tablo çıkarır, çıkarım belirsizliğini korur. |
| `record_scenario` | `questions`, `facts`; kullanıcı sorularını ve olgularını tutar. Kaydedilen olgu, kaynaklı hukuk kuralı değildir. |
| `report_progress` | `title`, `message`; sorunun dilinde doğal, senaryoya özgü durum bildirir. Araç adları, kod, yollar ve gizli muhakeme kullanıcıya aktarılmaz. |
| `verify_claim` | `claim`, global `citations`, isteğe bağlı `facts`; iddiayı kaydedilmiş özgün kanıt, koşul, tarih ve olgularla karşılaştıran odaklı LLM kontrolü yapar. Bağımsız doğruluk garantisi değildir. |
| `load_skill` | İzinli `name`; sürümlü araştırma yönergelerini yükler. Yönergeler kaynak yetkisini genişletmez ve hukuki kanıt değildir. |
| `calculate` | `operation` ve tutar/tarih girdileri; Decimal hesapları, dağıtım ve süre hesabı yapar. İş günü hesabı açık tatil takvimi ister; hukuki süre kuralını kendisi üretmez. |
| `compose_tool_calls` | `steps`, isteğe bağlı `max_parallel`; bağımlılıklar, `$ref` yolları ve koşullarla en fazla 20 adımlı program yürütür. Her iç çağrı aynı kayıt ve kapsam kontrolünden geçer. |
| `run_research_code` | `code`, isteğe bağlı `language: python/bash`, `source_ids`, `timeout_ms`; yapılandırılmış izole serviste kod çalıştırır. En fazla beş yetkili kanonik kaynak manifesti aktarılır; uygulama host’unda komut çalıştırmaz. |
| `spawn_researcher` | `task`; bağımsız bilgi ihtiyacını araştırmacıya devreder. Araştırmacı kendi araçlarını seçer. |
| `send_update` | `task_id`, `message`; yeni görev açmadan araştırmacıya ek bilgi iletir. |
| `followup_researcher` | `task_id`, `message`; çalışan araştırmacıyı yönlendirir veya bitmiş görev için bağlı devam başlatır. |
| `list_researchers` | Girdi gerekmez; görev durumlarını ve tamamlanmış sonuçları verir. |
| `wait_researcher` | İsteğe bağlı `task_id`, `timeout_seconds`; en fazla 30 saniye bekler. Çalışıyor durumu tamamlanmış sonuç değildir. |
| `cancel_researcher` | `task_id`; araştırmacıyı iptal eder, geç gelen sonuçların kabulünü engeller. |

`load_skill` için mevcut adlar `legal_conditions`, `source_recovery`, `temporal_calculation` ve `claim_verification` şeklindedir. Bunlar kullanıcıya otomatik yeni izin veren harici skill sistemi değildir.

Araç sonuçları `ToolOutcome` sözleşmesinde `status`, `summary`, `data`, `evidence` ve `artifacts` taşır. Durumlar arasında `found`, `partial`, `ambiguous`, `not_found`, `unavailable`, `denied`, `truncated`, `version_unknown`, `cancelled`, `invalid` ve `error` bulunur. Özellikle `unavailable`, `not_found` ile aynı anlama gelmez. `ToolReceipt`, gerçek çağrı, sonuç, geçen süre ve global kanıt numaralarını tutar.

## Külliyat ve dış kaynak sınırı

Varsayılan çalışma yalnızca yetkili külliyat içindedir. Dış okuma için hem istekte `asv3_allow_external: true` hem de kullanıcı niyetini değerlendiren dil profilinde `external_requested: true` gerekir. Frontend’de mevcut WebSearchTool’un açıkça seçilmesi ve devre dışı olmaması izin bayrağını oluşturabilir; araç listede bulunduğu veya genel olarak etkin olduğu için izin verilmez. Açık külliyatla sınırlama dış okumaya genişletilmemelidir.

Dış araçlar sabit 29 araca dahil değildir. `external_tools.py`, isteğe gerçekten sağlanan WebSearchTool/OpenURLTool ile izin listesindeki mevcut MCPTool/CustomTool nesnelerini `external_<özgün_function_adı>` olarak sarar. Parametre şeması sağlanan araçtan gelir. Yeni endpoint, kimlik bilgisi veya tarayıcı oturumu yaratılmaz.

`ASV3_EXTERNAL_READ_TOOLS`, mevcut MCP/Custom araçlarının `tool.name` değerlerini virgülle belirleyen isteğe bağlı okuma izin listesidir. Varsayılanı boştur. Bu ayar bilerek deployment şablonuna eklenmemiştir; ortam sahibi yalnızca gerçekten salt okunur olduğunu denetlediği araçları ayrıca yapılandırmalıdır. Listeye ad yazmak aracı oluşturmaz, asistanın yetkisini genişletmez ve kullanıcı izninin yerine geçmez.

Dış çıktı güvenilmeyen kaynak verisi olarak işaretlenir; içindeki talimatlar uygulanacak sistem talimatı değildir. Her API cevabı atıf yapılabilir kanıta çevrilmez: mevcut SearchDocsResponse biçiminde kaynak taşıyan çıktı bu dönüşüme uygundur. Kayıtlı dış kanıtın yeniden kullanımı taze yetkili erişim gerektirir. Bu erişimi yeniden yapan devam yolu mevcut olmadığı için atıf yapılabilir dış kanıt içeren kesilmiş çalışmada devam düğmesi sunulmaz.

## Kanıt, atıf ve devam

Kanıt defteri kaynak kimliği, parça kimliği ve metin SHA-256 değeri üzerinden kaydı birleştirir; paralel araştırmacılar aynı global atıf numaralarını kullanır. Finalde kullanılan kaynaklar erişim, tarih/yayın ve kanıt bütünlüğü yönünden tekrar denetlenir. Başlık eşleşmesi veya üretilmiş hesap çıktısı, özgün hükmün yerine geçmez.

Kanonik atıflar mevcut parça kimliğini ve preview akışını korur. Asıl dosyadan türetilen kanıt, sahte kanonik parça kimliği almaz: negatif `chunk_ind`, kaynak SHA-256 ve sayfa/satır locator’ı ile ayrı işaretlenir. `preview_url`, frontend’de `/api/asv3/citation/{assistant_message_id}/{citation_num}` yoluna gider; backend kullanıcıya ait mesajı, güncel erişim/yayını, özgün dosya hash’ini ve türetilmiş metin bütünlüğünü denetleyerek ilgili alıntıyı verir. Preview başarısızsa tüm dosyaya sessiz dönüş yapılmaz. Gerçek dış web araması kaynağı ise sahipli preview bulunmadığında güvenli HTTP(S) özgün bağlantıyla açılır.

Checkpoint, mevcut ToolCall logunda `-3001` ayırıcı kimliğiyle, tenant ve mesaj sahibi kontrolü altında saklanır. Özel checkpoint içeriği gelecekteki LLM sohbet geçmişine veya genel araç gösterimine eklenmez. Devam aynı yetkili kapsamı ve soruyu gerektirir; kaydedilmiş doğrulanmış dil profilini kullanır ve kaynakları tekrar doğrular. Tamamlanmış/başarısız/iptal terminal bildirimi ya da kaydedilmiş durdurma cevabı tekrar devam düğmesine dönüştürülmez.

## Bellek ve izolasyon

Ortak bütçe varsayılan olarak 64 araç çağrısı, 32 model kararı, 2.000.000 bayt kanıt ve 4.000.000 bayt artifact ile sınırlıdır; eşzamanlı araç ve model slotları ayrı ayrı dört adettir. Orkestrasyon araçları slot tutmaz; iç araştırma çağrıları kendi kontrollerinden geçer. Final için üç karar ayrılır. Runtime çalışma süresi 900 saniye, araştırmadan sonra final için ayrılan süre 180 saniyedir. Bunlar mevcut kod varsayılanlarıdır, her isteğin bu sınırları tüketmesi beklenmez.

LLM’ye kanıt özetleri sınırlı taşınır; ihtiyaç duyulan özgün metin `read_evidence` ile aralıklı okunabilir. Checkpoint için serileştirilmiş bütçe 2.500.000, açılmış içerik bütçesi 8.000.000 bayttır. Asıl dosya okuma 25 MiB ile sınırlıdır. Frontend ilerleme paketlerini 256 paketten sonra daraltır; ilk çalışma kimliği, son durumlar ve yakın geçmiş tutulurken kaynak, atıf, cevap ve kontrol paketleri korunur. Bunlar bütün uygulamanın bellek kullanımının sabit olduğu iddiası değildir.

Araştırma kodu uygulama host’una, üretim DB erişimine veya S3 kimlik bilgilerine erişim verilmeden ayrı servise gönderilir. Bash için kısa ömürlü, ağ erişimi kapalı servis oturumu ve servis desteği gerekir. Manifestler kanonik kaynaklardan gelir; hesap veya kod ürünü hukuki otorite sayılmaz. Sonlandırma ve iptal kontrolleri ortak kapsamda çalışır; iptalden sonraki geç sonuçlar yayınlanmaz.

## Servis bağımlılıkları ve kullanılabilirlik

| Yetenek | Gerekli bağımlılık ve sınır |
| --- | --- |
| Kaynak kimliği, kanonik okuma, checkpoint | PostgreSQL, tenant/kullanıcı kapsamı ve yayın/geçerlilik kontrolleri. |
| `search_corpus` | Mevcut SearchTool, indeks ve seçilen arama modunun bağımlılıkları; hibrit aramada embedding hizmeti de gerekebilir. |
| Asıl dosya ve sayfa | Erişilebilir doğrulanmış file store baytları, uygun dosya biçimi ve render/extraction bağımlılıkları. Her dosya/sürüm kabul edilmez. |
| Görsel inceleme ve PDF tablo kurma | Seçilmiş LLM’nin ilgili görsel giriş yeteneği ve geçerli kaynak sayfası; native metin mevcut olması görsel yorumlamanın çalıştığını göstermez. |
| `run_research_code` | `CODE_INTERPRETER_BASE_URL`, DB’de etkin Code Interpreter sunucusu politikası ve sağlıklı servis. Bash ayrıca session/execute-bash desteği ister. |
| Dış okuma | İsteğe sağlanan mevcut yetkili araç, gerekiyorsa izin listesi, kullanıcı izni ve sağlayıcı/connector erişimi. |

Kayıtlı araç şemasının varlığı bu bağımlılıkların DEV veya başka bir ortamda sağlıklı olduğunu kanıtlamaz. Eksik bağımlılıkta sonucu `unavailable`/`denied` olarak değerlendirmek ve gerçekten mevcut alternatif yöntemi denemek gerekir; desteklenmeyen bir sağlayıcıya sessiz geçiş yapılmaz.

## Bildirimler ve gözlemlenebilirlik

`asv3_progress` paketleri çalışma/olay kimliği, sıra, dil, aşama/durum, doğal başlık/mesaj ve gerektiğinde görev ilişkisini taşır. Frontend kimlik ve sıra denetimiyle eski veya başka çalışmaya ait olayları eler; görev sonlandırmalarını korur. Kullanıcıya teknik aşama veya araç adları yerine sorusuyla ilgili doğal bildirim gösterilir. Dil profilinin ilk yüklenişinden önce nötr bekleme görünümü kullanılır.

Model trace ayrımları `asv3_language`, `asv3_coordinator`, `asv3_researcher`, `asv3_verification`, `asv3_source_vision` ve `asv3_final` şeklindedir. Bunlar geliştirici gözlemlenebilirliği içindir. Süre/kalite karşılaştırması için aynı soru, kaynak kapsamı, model ve test koşullarına ait gerçek çalıştırma kanıtı gerekir; bu mimari tek başına hız veya kalite üstünlüğünü kanıtlamaz.

## İlgili uygulama noktaları

- Backend: `backend/onyx/asv3/`, `backend/onyx/db/asv3_corpus.py`, `backend/onyx/db/asv3_runs.py`, `backend/onyx/server/asv3_citations.py`.
- Giriş ve replay: `backend/onyx/chat/process_message.py`, `backend/onyx/server/query_and_chat/models.py`, `session_loading.py`, `streaming_models.py`.
- Frontend: `web/src/lib/asv3/`, `web/src/sections/asv3/`, `web/src/hooks/useChatController.ts`, mevcut citation/PreviewModal bileşenleri.
- Testler: `backend/tests/unit/onyx/asv3/`, session-loading testleri, `web/src/lib/asv3/` testleri ve `web/tests/e2e/chat/asv3_workflow.spec.ts`. Browser testinin deterministik cevap akışı canlı LLM/külliyat kalitesi kanıtı olarak yorumlanmamalıdır.
