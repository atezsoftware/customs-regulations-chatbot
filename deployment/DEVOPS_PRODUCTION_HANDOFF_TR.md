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
ama standart chart'ın primary/docfetching/docprocessing/user-file-processing/Beat worker'larını tek
başına kapatmaz. Bu worker release'leri prod'a eklenmemelidir.

PostgreSQL, Redis, OpenSearch ve MinIO uygulama altyapısıdır; model-server ile birlikte
kaldırılmayacaklardır. Redis Celery koordinasyonu/cache için, OpenSearch arama indeksi için, MinIO
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
```

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
2. PostgreSQL, Redis, OpenSearch ve MinIO bağlantılarını; yukarıdaki S3 environment mapping'ini,
   secret enjeksiyonunu ve kalıcı volume'leri doğrulayın.
3. Üç uygulama iş yükünü `DISABLE_MODEL_SERVER=true` ve `DOCUMENT_IMPORT_ENABLED=false` ile açın.
4. Model-server Deployment/Service olmadığını ve podlarda CUDA/Docling/Torch bağımlılıklarının
   bulunmadığını doğrulayın.
5. Trafiği açmadan önce Admin üzerinden OpenRouter embedding modelini seçip test edin ve aktif edin.
6. Admin üzerinden OpenRouter chat LLM'ini seçip test edin.
7. Sağlık, oturum açma, boş arama, chat ve bir küçük benchmark smoke testi çalıştırın; ardından kullanıcı
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
- Uygulama model-server DNS'i bulunmadan sağlıklı başlar ve Admin ekranı açılır.
- Admin OpenRouter embedding modellerini listeler; endpoint/dimension alanı göstermez.
- Seçilen embedding modeli test edilir, aktif Search Settings'e kaydedilir ve boş arama model-server'a
  bağlanmadan çalışır.
- PostgreSQL, Redis, OpenSearch ve MinIO dışarıya açık değildir ve uygulama podlarından erişilebilirdir.
- Uygulama imajları `latest` yerine onaylı digest ile sabitlenmiştir.
