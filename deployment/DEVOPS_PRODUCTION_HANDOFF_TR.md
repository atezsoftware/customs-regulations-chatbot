# Customs Regulations üretim geçişi — DevOps teslim notu

## Karar

Üretimde yerel embedding, rerank veya LLM modeli çalıştırılmayacak. CUDA, Docling ve
`model_server` bağımlılıkları üretim runtime imajında ve Kubernetes topolojisinde bulunmayacak.
Uygulama boş veritabanıyla model-server olmadan açılacak; embedding ve LLM seçimi daha sonra Admin
arayüzünden yapılacak.

Uygulama iş yükleri dört servisten üç servise iner:

| Mevcut iş yükü | Üretim kararı | İmaj |
| --- | --- | --- |
| `customs-regulations-web` | Kalır | Web runtime |
| `customs-regulations-api` | Kalır | Backend runtime-lite |
| `customs-regulations-background` | Kalır | Aynı backend runtime-lite |
| `customs-regulations-model-server` | Kaldırılır | Üretilmez ve prod registry'ye alınmaz |

Bu üç servisli özel runtime-lite topolojisi esas alınmalıdır. Repo içindeki standart Onyx Helm
chart'ına yalnız `values-cloud-models.yaml` eklemek yeterli değildir: o dosya model podlarını kapatır
ama standart chart'ın primary/docfetching/docprocessing/generic user-file-processing/Beat
release'lerini tek başına kapatmaz. Bu genel worker release'leri prod'a eklenmemelidir. Buna karşılık
`customs-regulations-background` içinde `supervisord-lite.conf` ile çalışan özel
`celery_worker_regulatory_indexing` ve `celery_beat_regulatory_indexing` zorunludur; bunlar ayrı bir
Helm release değildir.

PostgreSQL, Redis, Elasticsearch ve MinIO uygulama altyapısıdır; model-server ile birlikte
kaldırılmayacaklardır. Redis Celery koordinasyonu/cache için, Elasticsearch arama indeksi için, MinIO
dosya deposu için gereklidir.

## Helm değerlerinde istenen değişiklikler

`customs-regulations-api` ve `customs-regulations-background` için:

```yaml
environment:
  enabled: true
  parameters:
    - name: "DOCUMENT_IMPORT_ENABLED"
      value: "false"
    - name: "DISABLE_MODEL_SERVER"
      value: "true"
    - name: "MARKDOWN_IMPORT_ENABLED"
      value: "true"
    - name: "REGULATORY_BATCH_INDEXING_ENABLED"
      value: "false"
```

Yalnız background için `REGULATORY_INDEXING_GCS_URI` ve lease/retry/poll/embedding batch tuning
değişkenleri eklenir. Arşiv limitleri yalnız API'de bulunur. Değişkenlerin tam kapsamı ve varsayılanları
canonical `REGULATORY_PRODUCTION_RUNBOOK.md` tablosuyla birebir eşleşmelidir; secret değerler bu
teslim notuna yazılmaz.

İki servisten de aşağıdaki değişkenler kaldırılmalıdır:

```text
MODEL_SERVER_HOST
MODEL_SERVER_PORT
INDEXING_MODEL_SERVER_HOST
INDEXING_MODEL_SERVER_PORT
```

`customs-regulations-model-server-values.yaml` üretim release listesinden çıkarılmalı; Deployment,
Service, HPA, probe ve model-server imajı oluşturulmamalıdır. Ağ politikasında API ve background
podlarından `openrouter.ai:443` çıkışına izin verilmelidir.

`REGULATORY_BATCH_INDEXING_ENABLED` varsayılan olarak `false` kalır. Önce singleton
`alembic upgrade head`, GCS yetki testi, Vertex contextual model testi ve aktif OpenRouter embedding
ayarının doğrulaması tamamlanır. Sonra bayrak API ve background için birlikte `true` yapılır ve iki
iş yükü de kontrollü olarak yeniden başlatılır. Bayrağın yalnız bir pod türünde değiştirilmesi veya
migration'dan önce açılması kabul edilmez. GCS URI secret değildir; servis hesabı/Workload Identity
credential'ı Vault/Kubernetes kimlik katmanında kalır.

Background podunun supervisor sözleşmesi tam olarak beş worker, bir özel Beat ve bir log yönlendirici
olmak üzere yedi process'tir:

| Process | Queue/görev |
| --- | --- |
| `celery_worker_regulatory_benchmark` | `regulatory_benchmark` |
| `celery_worker_regulatory_indexing` | `user_file_processing,regulatory_indexing` |
| `celery_worker_user_file_maintenance` | `user_file_project_sync,user_file_delete` |
| `celery_worker_light` | Onaylı hafif bakım queue'ları |
| `celery_worker_monitoring` | `monitoring` |
| `celery_beat_regulatory_indexing` | Yalnız stale recovery ve queue monitoring |
| `log-redirect-handler` | Altı Celery logunu stdout'a aktarır |

Beat readiness ve liveness dosyaları sırasıyla
`/tmp/onyx_k8s_regulatoryindexingbeat_readiness.txt` ve
`/tmp/onyx_k8s_regulatoryindexingbeat_liveness.txt` olmalıdır. Generic Beat schedule, primary,
docfetching, docprocessing, generic indexing ve Elasticsearch migration queue'ları bu topolojide
yasaktır. `user_file_processing` ise yalnız özel regulatory indexing worker üzerinde zorunludur.

Her background podu kendi Beat process'ini çalıştırabilir. Shelf pod-local
`/tmp/regulatory-indexing-beat-schedule` altında kalır; replica'lar aynı dosyayı paylaşmaz. Yayın
öncesinde tenant + schedule entry + UTC time-slot için bounded TTL'li Redis `SET NX EX` claim alınır;
aynı slotu yalnız bir replica yayınlar ve follower healthy kalır. Claim sahibi `SET NX EX` sonrasında
broker yayını öncesinde durursa aynı slot yeniden yayınlanmaz; deterministik failover en geç sonraki
UTC slotunda olur: monitoring için 10 saniye, stale recovery için 60 saniye. İki interval'lik TTL
yalnız stale key retention ve clock anomaly etkisini sınırlar; aynı-slot takeover garantisi değildir.
External Helm replica sayısının bir olduğu varsayılmaz.

Beat başlangıcında eski readiness/liveness dosyaları Redis ve PostgreSQL beklenmeden önce silinir.
Yeni dosyalar supervisor'daki güncel PID ve instance UUID'sini taşır. Compose ve CodeBuild PID/instance
eşleşmesini ve liveness mtime değerinin en fazla 150 saniye olmasını çalıştırarak doğrular; yalnız
dosya varlığı kabul edilmez. Başarısız schedule refresh liveness'ı yenilemez, kontrollü shutdown iki
probe'u siler. Follower replica'nın dispatch claim sahibi olması readiness şartı değildir.
CodeBuild yeni image tag'ini çalıştıran, terminating olmayan tüm Ready background replica'ları
enumerate eder ve her birinde supervisor PID/instance/freshness doğrulamasını çalıştırır. Sıfır eşleşme
veya replica'lardan herhangi birinin probe hatası readiness'i başarısız yapar; yalnız ilk podun
kontrolü yeterli değildir.

Mevcut dört servisli kurulumdan geçiliyorsa model-server podu önce drain/stop edilmeli ve eski
Deployment/Service kaldırılmalıdır. Ardından cloud preflight ve üç servisli rollout çalıştırılır;
preflight çalışan eski veya başka projeye ait model-server container/podunu bilinçli olarak blocker
kabul eder.

`image.tag: latest` ve `image.policy: Always` yerine aynı Git commit'inden üretilmiş, onaylı ve
değişmez digest referansları kullanılmalıdır. API ile background aynı backend runtime-lite digest'ini
kullanmalıdır. Kullanılan kurumsal chart yalnız `name` + `tag` kabul ediyorsa DevOps önce chart'a
`repository@sha256:...`/`digest` desteği eklemeli veya `name` alanında digest kullanımını doğrulamalıdır;
`latest` ile prod geçişi yapılmamalıdır.

Gönderilen Vault şablonundaki MinIO adları backend'in okuduğu adlarla eşleşmiyor. Secret değerleri
değişmeden aşağıdaki environment adlarına map edilmelidir:

| Mevcut Vault export'u | Uygulamanın beklediği ad |
| --- | --- |
| `MINIO_ENDPOINT_URL` | `S3_ENDPOINT_URL` |
| `MINIO_ACCESS_KEY` | `S3_AWS_ACCESS_KEY_ID` |
| `MINIO_SECRET_KEY` | `S3_AWS_SECRET_ACCESS_KEY` |
| `MINIO_BUCKET_NAME` | `S3_FILE_STORE_BUCKET_NAME` |
| `MINIO_REGION_NAME` | `AWS_REGION_NAME` |

Vault template karşılığı:

```sh
export S3_ENDPOINT_URL="{{ .Data.minio_endpoint_url }}"
export S3_AWS_ACCESS_KEY_ID="{{ .Data.minio_access_key }}"
export S3_AWS_SECRET_ACCESS_KEY="{{ .Data.minio_secret_key }}"
export S3_FILE_STORE_BUCKET_NAME="{{ .Data.minio_bucket_name }}"
export AWS_REGION_NAME="{{ .Data.minio_region_name }}"
```

Bu mapping hem API hem background podunda bulunmalıdır. Aksi halde `FILE_STORE_BACKEND=s3` seçilmiş
olsa bile uygulama MinIO credential/bucket ayarlarını kullanmaz.

## Admin akışı

Embedding endpoint'i veya vektör boyutu Helm/Vault değişkeni değildir ve kullanıcı tarafından
yazılmayacaktır.

1. Admin, **Search Settings > OpenRouter** seçeneğini açar.
2. OpenRouter API anahtarını girer ve **Modelleri getir** işlemini çalıştırır.
3. Embedding modelini listeden seçer.
4. Uygulama sabit OpenRouter embedding adresine bir test çağrısı yapar ve vektör boyutunu yanıttan
   otomatik belirler.
5. Admin ayarı uygular. Boş kurulumda yeni cloud Search Settings doğrudan aktif edilir.
6. Chat LLM'i ayrıca mevcut **Language Models > OpenRouter** ekranından seçilir.

OpenRouter anahtarı uygulamanın credential kaydı olarak PostgreSQL'de şifreli tutulur. Bu nedenle
`ENCRYPTION_KEY_SECRET` bütün podlarda aynı ve sürümler arasında kalıcı olmalıdır. Admin akışı
kullanılıyorsa OpenRouter embedding adresi veya anahtarı Helm değerlerine eklenmez.

Mevcut Vault şablonundaki `GEN_AI_API_KEY` prod'dan kaldırılmalı/unset bırakılmalıdır. Bu değişken
yalnız geliştirme akışı içindir ve boş veritabanında otomatik bir varsayılan **OpenAI** LLM sağlayıcısı
oluşturur; Admin'den seçilecek OpenRouter sağlayıcısıyla çelişir.

## Geçiş sırası

1. Backend runtime-lite ve web imajlarını aynı commit'ten üretip digest ile yayınlayın. Prod için
   model-server imajı üretmeyin.
2. PostgreSQL, Redis, Elasticsearch ve MinIO bağlantılarını; yukarıdaki S3 environment mapping'ini,
   secret enjeksiyonunu ve kalıcı volume'leri doğrulayın.
3. Üç uygulama iş yükünü `DISABLE_MODEL_SERVER=true` ve `DOCUMENT_IMPORT_ENABLED=false` ile açın.
4. Model-server Deployment/Service olmadığını ve podlarda CUDA/Docling/Torch bağımlılıklarının
   bulunmadığını doğrulayın.
5. Trafiği açmadan önce Admin üzerinden OpenRouter embedding modelini seçip test edin ve aktif edin.
6. Admin üzerinden OpenRouter chat LLM'ini seçip test edin.
7. Singleton migration'ı tamamlayın; `REGULATORY_INDEXING_GCS_URI` ve GCS/Vertex yetkilerini
   doğruladıktan sonra durable indexing
   bayrağını API ve background'da birlikte açıp ikisini yeniden başlatın.
8. Yedi supervisor process'ini, worker/Beat readiness-liveness dosyalarını, queue setini, bir küçük
   Markdown canary indeksini ve stale recovery'yi doğrulayın.
9. Sağlık, oturum açma, boş arama, chat ve bir küçük benchmark smoke testi çalıştırın; ardından kullanıcı
   trafiğini açın.

Bu ilk aktivasyon yalnızca PostgreSQL'de belge, gerçek connector veya tamamlanmış user file yokken
indexer olmadan yapılabilir. Veri bulunan bir ortamda embedding modeli değişikliği yeniden indeksleme
gerektirir; bu işlem prod runtime içinde değil, ayrı ve yetkili importer/indexing ortamında
tamamlanmalıdır.

## Rerank kapsamı

Mevcut aktif arama akışında bağımsız bir rerank sağlayıcı çağrısı yoktur. Bu üretim geçişi bu nedenle
bir rerank endpoint'i veya rerank secret'ı istemez. OpenRouter rerank daha sonra ürüne eklenmek
istenirse veri çıkışı, timeout/fail-open davranışı ve sıralama testleri olan ayrı bir uygulama
değişikliği olarak ele alınmalıdır.

## Kabul kriterleri

- Kubernetes'te `customs-regulations-model-server` Deployment, Service veya pod yoktur.
- API ve background ortamında `DISABLE_MODEL_SERVER=true` görünür; model-server host değişkeni yoktur.
- Backend runtime-lite imajında `torch`, `triton`, NVIDIA/CUDA ve Docling paketleri yoktur.
- Background supervisor'da tam yedi process çalışır; özel worker iki queue'yu tüketir ve özel Beat
  yalnız stale recovery ile queue monitoring yayınlar.
- Migration ve GCS/Vertex/OpenRouter kontrollerinden önce durable indexing bayrağı kapalıdır; açıldıktan
  sonra API ve background birlikte restart edilmiş ve Markdown canary tamamlanmıştır.
- Uygulama model-server DNS'i bulunmadan sağlıklı başlar ve Admin ekranı açılır.
- Admin OpenRouter embedding modellerini listeler; endpoint/dimension alanı göstermez.
- Seçilen embedding modeli test edilir, aktif Search Settings'e kaydedilir ve boş arama model-server'a
  bağlanmadan çalışır.
- PostgreSQL, Redis, Elasticsearch ve MinIO dışarıya açık değildir ve uygulama podlarından erişilebilirdir.
- Uygulama imajları `latest` yerine onaylı digest ile sabitlenmiştir.

Ek worker'ın ölçülen cold-import maksimum RSS değeri uygulama ortamında yaklaşık `216568 KiB` olmuştur.
Repo harici Helm resource değerleri burada sahiplenilmediği için sabit limit uydurulmaz. DevOps rollout
öncesi ve sonrası cgroup `memory.current`, `memory.peak`, `memory.max`, `memory.events` ile process RSS
kanıtını arşivlemeli; OOM veya yetersiz headroom kabul kriterini başarısız saymalıdır.
