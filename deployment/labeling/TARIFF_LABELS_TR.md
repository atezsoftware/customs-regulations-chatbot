# TARIFF Regulatory Intelligence etiket sözlüğü

Kaynak: `TARIFF_REGULATORY_INTELLIGENCE_Full_Scope_Data_Model.pdf`, sürüm 2.1, 30 Ağustos 2026; etiket tabloları s.13-23. Belgenin 51 sayfası tarandı; örnekler ve eklerde yeni etiket bulunmadı.

Bu dosya yalnızca etiket kodlarını, adlarını ve açıklamalarını çıkarır. PDF’deki mimari, chunk oluşturma, saklama, cardinality ve retrieval kararları uygulama gereksinimi olarak alınmamıştır.

Bu 255 etiket ve açıklaması başlangıçta migration ile DB'ye kaydedilir. Document set ekranında **Label Settings** ile ekleme ve düzenleme yapılabilir. **Start Labeling**, DB'deki güncel tanımları LLM promptuna aktarır. Bu katalog ve [JSON kopyası](tariff-regulatory-intelligence-v2.1.json) ilk v2.1 tanımlarının referansıdır; sonradan DB'de yapılan düzenlemeleri yansıtmaz. `backend/onyx/regulatory/labeling/data/` altındaki kopya geçmiş migration için sabit tutulur; iki başlangıç kopyasının eşitliği test edilir.

## Sayım

| Grup | Sayı | Kaynak sayfalar |
| --- | ---: | --- |
| Hukuk alanı | 26 | 13-14 |
| Konu (üst gruplar dahil) | 90 | 14-18 |
| Hüküm ve etki türü | 33 | 18-19 |
| Ek türü | 30 | 19-20 |
| Sektör | 12 | 20-21 |
| Relevance alanı | 24 | 22-23 |
| Uygulanabilirlik: ticaret akışı | 7 | 23 |
| Uygulanabilirlik: gümrük rejimi | 11 | 23 |
| Uygulanabilirlik: aktör | 14 | 23 |
| Uygulanabilirlik: sistem | 8 | 23 |
| **Toplam** | **255** | |

## Aktarım notları

- Ana 215 etikette PDF’deki kodlar ve Türkçe adlar korundu. Açıklama sütunları kaynak dilinde alındı; PDF satır sonlarında bölünmüş kod ve kelimeler birleştirildi.
- Uygulanabilirlikte PDF yalnızca 40 kod değeri verir. Bu değerlerin Türkçe görünen adları ve kısa boyut açıklamaları aktarım için eklendi; kaynakta ayrı tanım metinleri bulunmaz. Sistem kısaltmaları genişletilmedi.
- `export` gibi farklı boyutlarda tekrarlanan değerlerin çakışmaması için yalnızca uygulanabilirlik etiketlerinin ID’lerine boyut eklendi: `trade_flow.export`, `customs_regime.export`. Kaynak değer değişmedi.
- `geography` için PDF kapalı bir liste vermiyor; yalnızca ISO ülke/yönetilen bölge kodunu işaret ediyor. Ülke etiketi üretilmedi.
- `core`, `substantial`, `partial`, `limited` (s.21,47) relevance düzeyleridir, içerik etiketleri değildir. Mevcut job çıktı şemasına düzey alanı eklenmedi.
- Kaynak kimliği/türü/durumu, ilişki türleri, retrieval profilleri ve örnek sayısal ID’ler içerik etiketi olarak sözlüğe eklenmedi.

## Hukuk alanı (26)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `customs_law` | Gümrük hukuku | Customs procedures, declarations, tariff, origin, valuation, debt, control and customs regimes. | - | 13 |
| `international_trade_law` | Uluslararası ticaret hukuku | Import, export, trade policy, preferential trade and cross-border restrictions. | - | 13 |
| `tax_law` | Vergi hukuku | VAT, excise, duties, tax procedure and fiscal liabilities. | - | 13 |
| `administrative_law` | İdare hukuku | Administrative powers, procedures, organisation and public administration. | - | 13 |
| `administrative_enforcement_law` | İdari yaptırım hukuku | Administrative fines, regulatory violations, licence sanctions and non-criminal enforcement. | - | 13 |
| `criminal_law` | Ceza hukuku | Criminal offences, smuggling, seizure, confiscation and criminal penalties. | - | 13 |
| `civil_and_obligations_law` | Medeni hukuk ve borçlar hukuku | Persons, property, contracts, delivery, liability and private-law obligations. | - | 13 |
| `commercial_and_company_law` | Ticaret ve şirketler hukuku | Commercial enterprise, companies, negotiable instruments and commercial records. | - | 13 |
| `transport_law` | Taşıma hukuku | Road, maritime, aviation and rail transport, including carrier obligations. | - | 13 |
| `environmental_law` | Çevre hukuku | Waste, emissions, hazardous substances, environmental permits and nature protection. | - | 13 |
| `product_compliance_law` | Ürün uygunluğu ve teknik düzenlemeler hukuku | Product safety, standards, conformity assessment and market surveillance. | - | 13 |
| `strategic_trade_control_law` | Stratejik ticaret kontrolü hukuku | Weapons, defence goods, dual-use goods and controlled technology. | - | 13 |
| `sanctions_law` | Yaptırımlar hukuku | Embargoes, asset freezes and country-, person- or entity-based restrictive measures. | - | 13 |
| `labour_and_occupational_safety_law` | İş ve iş sağlığı güvenliği hukuku | Employment relationships, collective rights and occupational health and safety. | - | 13 |
| `financial_and_currency_law` | Finans ve kambiyo hukuku | Foreign exchange, currency controls, banking, interest, funds and precious metals. | - | 13 |
| `public_finance_and_receivables_law` | Kamu maliyesi ve kamu alacakları hukuku | Public accounting, collection of public debts, restructuring, interest and public guarantees. | - | 13 |
| `public_procurement_law` | Kamu ihale hukuku | Public purchasing, tenders, direct procurement, contracting and contractor processes. | - | 13 |
| `state_aid_and_investment_incentives_law` | Devlet yardımları ve yatırım teşvikleri hukuku | Investment certificates, project-based aid, support funds and export incentives. | - | 13 |
| `public_employment_law` | Kamu personeli hukuku | Civil servants, appointments, leave, discipline and public-sector employment. | - | 13 |
| `judicial_procedure_and_enforcement_law` | Yargılama, icra ve takip hukuku | Litigation, administrative courts, enforcement, bankruptcy, service and judicial assistance. | - | 13 |
| `information_and_digital_law` | Bilgi ve dijital hukuk | Access to information, electronic signatures, electronic notification, data and communications. | - | 14 |
| `public_property_and_real_estate_law` | Kamu malları ve taşınmaz hukuku | Treasury property, public movables, easements, occupation permits and public assets. | - | 14 |
| `constitutional_and_human_rights_law` | Anayasa ve insan hakları hukuku | Constitutional review, equality, fundamental rights and rights-based public-law rules. | - | 14 |
| `public_international_and_treaty_law` | Uluslararası kamu hukuku ve antlaşmalar hukuku | Treaties, diplomatic status, international organisations and international obligations. | - | 14 |
| `associations_and_foundations_law` | Dernekler ve vakıflar hukuku | Membership, donations, records, governance and nonprofit legal administration. | - | 14 |
| `regulated_professions_law` | Düzenlenmiş meslekler hukuku | Licensing, competence, fees, discipline and responsibility of regulated professionals. | - | 14 |

## Konu (90)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `SUB.CUS` | Gümrük işlemleri | Gümrük beyanı, tarife, menşe, kıymet, borç, kontrol ve idari işlemler üst grubu. | - | 14 |
| `SUB.CUS.DECL` | Gümrük beyanı | Beyanname, özet beyan, sözlü/eksik/tamamlayıcı beyan ve düzeltme/iptal. | SUB.CUS | 14 |
| `SUB.CUS.ENTRY` | Giriş, varış ve çıkış | Taşıt/eşyanın gümrük bölgesine gelişi, varış bildirimi, boşaltma ve çıkış. | SUB.CUS | 14 |
| `SUB.CUS.RELEASE` | Teslim ve serbest bırakma | Eşyanın teslimi, serbest bırakılması ve kontrol sonrası çekilmesi. | SUB.CUS | 14 |
| `SUB.CUS.REP` | Temsil ve gümrük müşavirliği | Doğrudan/dolaylı temsil, gümrük müşaviri ve YGM yetki/sorumluluğu. | SUB.CUS | 14 |
| `SUB.CUS.SIMPL` | Basitleştirmeler ve yetkilendirilmiş yükümlü | YYS/AEO, OKS, izinli gönderici/alıcı ve basitleştirilmiş usuller. | SUB.CUS | 14 |
| `SUB.CUS.TARIFF` | Tarife ve sınıflandırma | GTİP/TGTC, bağlayıcı tarife bilgisi ve sınıflandırma kararları. | SUB.CUS | 14 |
| `SUB.CUS.ORIGIN` | Menşe ve menşe ispatı | Tercihli/tercihsiz menşe, kümülasyon ve ispat belgeleri. | SUB.CUS | 14 |
| `SUB.CUS.VALUE` | Gümrük kıymeti | Satış bedeli ve alternatif yöntemler, ilaveler, royalti, navlun ve kıymet araştırması. | SUB.CUS | 14 |
| `SUB.CUS.DUTY` | Gümrük vergileri ve mali yükler | Gümrük vergisi, İGV, EMY, fon ve ithalde alınan mali yükler. | SUB.CUS | 14 |
| `SUB.CUS.DEBT` | Gümrük borcu ve tahakkuk | Gümrük yükümlülüğünün doğması, borçlu, tahakkuk ve sona erme. | SUB.CUS | 14 |
| `SUB.CUS.GUAR` | Teminat | Bireysel/toplu/kapsamlı teminat, muafiyet ve çözüm işlemleri. | SUB.CUS | 14 |
| `SUB.CUS.REFUND` | Geri verme ve kaldırma | Gümrük vergilerinin geri verilmesi/kaldırılması ve başvurusu. | SUB.CUS | 15 |
| `SUB.CUS.CONTROL` | Risk, kontrol, muayene ve laboratuvar | Risk analizi, belge kontrolü, fiziki muayene, numune, tahlil ve laboratuvar. | SUB.CUS | 15 |
| `SUB.CUS.POSTCONTROL` | Sonradan kontrol ve denetim | Sonradan kontrol, ikincil kontrol ve gümrük denetimi. | SUB.CUS | 15 |
| `SUB.CUS.TEMPSTORE` | Geçici depolama | Geçici depolama statüsü, yerleri, süre ve kayıtları. | SUB.CUS | 15 |
| `SUB.CUS.FACILITY` | Antrepo ve depolama tesisi | Antrepo/geçici depolama yerinin açılması, işletilmesi, donanımı ve stok kontrolü. | SUB.CUS | 15 |
| `SUB.CUS.OFFICE` | Gümrük idaresi, sınır kapısı ve güzergâh | Yetkili gümrükler, ihtisas gümrükleri, sınır kapıları ve güzergâhlar. | SUB.CUS | 15 |
| `SUB.CUS.DIGITAL` | Dijital gümrük sistemleri | BİLGE, NCTS, Tek Pencere, TAREKS entegrasyonu ve elektronik gümrük süreçleri. | SUB.CUS | 15 |
| `SUB.CUS.IPR` | Fikri mülkiyetin gümrükte korunması | Sahte/taklit eşya için işlemlerin durdurulması, başvuru ve imha. | SUB.CUS | 15 |
| `SUB.CUS.DISPOSAL` | Tasfiye, imha ve gümrüğe terk | Tasfiyelik eşya, satış, imha ve gümrüğe terk süreçleri. | SUB.CUS | 15 |
| `SUB.REG` | Gümrük rejimleri ve özel kullanımlar | Gümrük rejimleri, statüler ve özel kullanımlar üst grubu. | - | 15 |
| `SUB.REG.FREE_CIRC` | Serbest dolaşıma giriş | Serbest dolaşımda olmayan eşyanın ithalat yükleri uygulanarak serbest dolaşıma girişi. | SUB.REG | 15 |
| `SUB.REG.EXPORT` | İhracat rejimi | Serbest dolaşımdaki eşyanın kesin ihracı ve çıkış rejimi. | SUB.REG | 15 |
| `SUB.REG.TRANSIT` | Transit - karma veya alt tür belirsiz | Transit merkezi konu olmakla birlikte ulusal/ortak/TIR ayrımı yapılamıyorsa. | SUB.REG | 15 |
| `SUB.REG.NAT_TRANSIT` | Ulusal transit | Türkiye içindeki ulusal transit işlemleri. | SUB.REG.TRANSIT | 15 |
| `SUB.REG.COMMON_TRANSIT` | Ortak transit ve NCTS | Ortak Transit Sözleşmesi ve NCTS kapsamındaki transit. | SUB.REG.TRANSIT | 15 |
| `SUB.REG.TIR` | TIR transit sistemi | TIR Sözleşmesi, TIR Karnesi, volet ve kefalet zinciri. | SUB.REG.TRANSIT | 15 |
| `SUB.REG.WAREHOUSE` | Gümrük antrepo rejimi | Antrepo rejimine giriş, stok, devir, elleçleme ve çıkış. | SUB.REG | 15 |
| `SUB.REG.INWARD` | Dahilde işleme | DİİB/Dİİ, şartlı muafiyet, geri ödeme ve taahhüt hesabı. | SUB.REG | 15 |
| `SUB.REG.OUTWARD` | Hariçte işleme | HİİB/Hİİ, tamir, standart değişim ve işlenmiş ürünün geri ithali. | SUB.REG | 15 |
| `SUB.REG.TEMP_ADM` | Geçici ithalat | Geçici ithalat izni, ATA Karnesi, süre, ayniyet ve yeniden ihracat. | SUB.REG | 15 |
| `SUB.REG.END_USE` | Nihai kullanım | Belirli kullanıma bağlı indirimli/sıfır vergi ve izleme. | SUB.REG | 15 |
| `SUB.REG.FREE_ZONE` | Serbest bölge işlemleri | Serbest bölgeye giriş/çıkış, kullanıcı, faaliyet ve özel hesap. | SUB.REG | 15 |
| `SUB.REG.REEXPORT` | Yeniden ihracat | Serbest dolaşımda olmayan veya geçici ithal eşyanın yeniden ihracı. | SUB.REG | 15 |
| `SUB.REG.RETURNED` | Geri gelen eşya | İhraç edilen eşyanın şartlarla yeniden ithali ve muafiyeti. | SUB.REG | 15 |
| `SUB.REG.SHIP_SUPPLY` | Kumanya ve ihrakiye | Gemi/uçak kumanyası, yakıt ve donatım teslimleri. | SUB.REG | 15 |
| `SUB.TRD` | Dış ticaret ve ticaret politikası | İthalat/ihracat kontrolleri, önlemler ve anlaşmalar üst grubu. | - | 16 |
| `SUB.TRD.IMPORT_CONTROL` | İthalat kontrolü ve izin | İthalat izni, ön izin, kayıt belgesi, kullanılmış eşya ve ihtisas gümrüğü. | SUB.TRD | 16 |
| `SUB.TRD.EXPORT_CONTROL` | İhracat kontrolü ve izin | İhracat izni, kayda bağlı ihracat, yasak ve kısıtlar. | SUB.TRD | 16 |
| `SUB.TRD.SURVEILLANCE` | İthalatta gözetim | Gözetim belgesi, kıymet/miktar eşiği ve gözetim uygulaması. | SUB.TRD | 16 |
| `SUB.TRD.DEFENCE` | Damping ve sübvansiyona karşı önlemler | Damping/sübvansiyon soruşturması, kesin/geçici önlem ve vergi. | SUB.TRD | 16 |
| `SUB.TRD.SAFEGUARD` | Korunma önlemleri | Korunma önlemi soruşturması, kota ve ek mali yük. | SUB.TRD | 16 |
| `SUB.TRD.QUOTA` | Kota, tarife kontenjanı ve tahsis | Kota/kontenjan açılması, tahsis yöntemi ve kullanım şartları. | SUB.TRD | 16 |
| `SUB.TRD.PREFERENTIAL` | Tercihli ticaret ve gümrük birliği | STA/TTA, GTS, Gümrük Birliği ve tercih düzenlemeleri. | SUB.TRD | 16 |
| `SUB.TRD.PRODUCT_COMPLIANCE` | Ürün güvenliği ve teknik uygunluk | TAREKS, CE/TSE, teknik düzenleme, uygunluk değerlendirmesi ve denetim. | SUB.TRD | 16 |
| `SUB.TRD.STRATEGIC` | Stratejik ve çift kullanımlı eşya kontrolü | Silah, mühimmat, çift kullanımlı, nükleer ve kontrollü teknoloji. | SUB.TRD | 16 |
| `SUB.TRD.SANCTIONS` | Yaptırım, ambargo ve kısıtlayıcı tedbir | Ülke/kişi/kuruluş temelli ambargo, malvarlığı dondurma ve ticaret yasağı. | SUB.TRD | 16 |
| `SUB.TAX` | Vergi, mali ve muhasebe | Vergi, tahsilat, kambiyo, ödeme ve kayıt üst grubu. | - | 16 |
| `SUB.TAX.VAT` | Katma değer vergisi | KDV'nin konusu, matrahı, oranı, ithalatı, istisnası, indirimi ve iadesi. | SUB.TAX | 16 |
| `SUB.TAX.EXCISE` | Özel tüketim vergisi | ÖTV kapsamı, listeler, oranlar, istisna ve bandrol bağlantısı. | SUB.TAX | 16 |
| `SUB.TAX.OTHER` | Diğer vergi, harç ve fonlar | Damga, gelir, kurumlar vergisi, harç, KKDF ve diğer fonlar. | SUB.TAX | 16 |
| `SUB.TAX.PROCEDURE` | Vergi usulü ve beyan | Vergi beyanı, tarh, tebligat, defter-belge ve usul hükümleri. | SUB.TAX | 16 |
| `SUB.TAX.COLLECTION` | Kamu alacağı, tahsilat ve yapılandırma | Tahsil, takip, tecil, faiz, zam ve alacakların yapılandırılması. | SUB.TAX | 16 |
| `SUB.TAX.FX` | Kambiyo, döviz ve ihracat bedeli | Türk parasını koruma, döviz işlemleri ve ihracat bedellerinin yurda getirilmesi. | SUB.TAX | 16 |
| `SUB.TAX.PAYMENT` | Ödeme, transfer ve finansman | Ödeme, banka transferi, kredi, finansman ve nakit hareketleri. | SUB.TAX | 16 |
| `SUB.TAX.ACCOUNTING` | Muhasebe, defter ve mali raporlama | Muhasebe kayıtları, ticari defter, mali tablo ve raporlama. | SUB.TAX | 16 |
| `SUB.LOG` | Taşıma, lojistik ve yolcu | Taşıma türleri, lojistik düğümler ve yolcu/posta üst grubu. | - | 16 |
| `SUB.LOG.MULTI` | Taşıma ve taşıyıcı - karma | Taşıma modu karma veya belirtilmemiş; taşıyıcı sorumluluğu. | SUB.LOG | 16 |
| `SUB.LOG.ROAD` | Karayolu taşımacılığı | Karayolu taşıma yetkisi, geçiş belgesi, yabancı plaka ve ücretler. | SUB.LOG | 16 |
| `SUB.LOG.SEA` | Deniz taşımacılığı ve liman | Gemi, liman, kabotaj, Ro-Ro ve deniz taşıma belgeleri. | SUB.LOG | 16 |
| `SUB.LOG.AIR` | Havayolu taşımacılığı | Uçak, hava kargo, havalimanı ve hava taşıma işlemleri. | SUB.LOG | 16 |
| `SUB.LOG.RAIL` | Demiryolu taşımacılığı | Demiryolu, CIM/COTIF ve trenle taşıma. | SUB.LOG | 17 |
| `SUB.LOG.CONTAINER` | Konteyner, liman ve lojistik tesis | Konteyner kayıt/takip, liman operasyonu ve aktarma. | SUB.LOG | 17 |
| `SUB.LOG.PASSENGER` | Yolcu, posta ve hızlı kargo | Yolcu eşyası, pasaport sınır işlemleri, posta ve hızlı kargo. | SUB.LOG | 17 |
| `SUB.ENF` | Yaptırım ve uyuşmazlık | Ceza, kaçakçılık, el koyma ve başvuru yolları üst grubu. | - | 17 |
| `SUB.ENF.ADMIN_PENALTY` | İdari para cezası ve usulsüzlük | İdari para cezası, usulsüzlük ve vergi kaybı yaptırımı. | SUB.ENF | 17 |
| `SUB.ENF.SMUGGLING` | Kaçakçılık ve ceza soruşturması | Kaçakçılık suçu, adli takip ve etkin pişmanlık. | SUB.ENF | 17 |
| `SUB.ENF.SEIZURE` | El koyma, müsadere ve yakalama | El koyma, müsadere, yakalama ve adli emanet. | SUB.ENF | 17 |
| `SUB.ENF.OBJECTION` | İtiraz ve idari başvuru | İdari itiraz, şikayet ve yeniden inceleme yolları. | SUB.ENF | 17 |
| `SUB.ENF.SETTLEMENT` | Uzlaşma | Gümrük/vergi alacakları ve cezalarında uzlaşma. | SUB.ENF | 17 |
| `SUB.ENF.LITIGATION` | Dava, yargı ve içtihat | Mahkeme kararı, dava, temyiz ve yargılama usulü. | SUB.ENF | 17 |
| `SUB.ADM` | Kamu yönetimi ve kurumsal süreç | Teşkilat, personel, iç denetim ve kamu süreçleri üst grubu. | - | 17 |
| `SUB.ADM.ORG` | Teşkilat, görev ve yetki | Kamu kurumunun teşkilatı, görev dağılımı ve birim yetkisi. | SUB.ADM | 17 |
| `SUB.ADM.PERSONNEL` | Kamu personeli ve disiplin | Atama, özlük, izin, emeklilik ve disiplin. | SUB.ADM | 17 |
| `SUB.ADM.AUDIT` | İç denetim ve kurumsal kontrol | İç denetim, teftiş, kalite ve kurumsal risk yönetimi. | SUB.ADM | 17 |
| `SUB.ADM.PROCUREMENT` | Kamu ihalesi ve varlık yönetimi | Kamu alımı, ihale, taşınır/taşınmaz ve ihtiyaç fazlası mal. | SUB.ADM | 17 |
| `SUB.ADM.DATA` | Bilgi, belge, elektronik işlem ve veri | Bilgi edinme, e-imza, e-tebligat, veri ve elektronik idari süreç. | SUB.ADM | 17 |
| `SUB.EMP.RIGHTS` | İşçi hakları ve çalışma | İş ilişkisi, çalışma süresi, ücretli izin, toplu iş sözleşmesi ve işçi hakları. | - | 17 |
| `SUB.EMP.SAFETY` | İş sağlığı ve güvenliği | İş sağlığı, iş güvenliği, iş kazası, risk değerlendirmesi ve işyeri hekimi. | - | 17 |
| `SUB.ENV.CHEMICAL` | Kimyasallar ve tehlikeli maddeler | Kimyasal ve tehlikeli maddeler, güvenlik bilgi formu, REACH ve biyosidal ürünler. | - | 17 |
| `SUB.ENV.COMPLIANCE` | Çevresel uygunluk | Çevre izinleri, emisyon, sera gazı, çevre kirliliği ve çevresel uyum yükümlülükleri. | - | 17 |
| `SUB.ENV.WASTE` | Atık ve geri kazanım | Atık yönetimi, tehlikeli atık, ambalaj atığı ve geri kazanım. | - | 17 |
| `SUB.HEALTH.VET` | Sağlık, veteriner ve bitki sağlığı | Veteriner, bitki sağlığı, gıda güvenliği, hayvan sağlığı ve karantina. | - | 17 |
| `SUB.GEN` | Genel hukuk ve diğer düzenleme | Karma/diğer hukuk alanları üst grubu. | - | 17 |
| `SUB.GEN.TREATY` | Uluslararası antlaşma ve işbirliği | Antlaşma, protokol ve karşılıklı idari yardımın genel çerçevesi. | SUB.GEN | 17 |
| `SUB.GEN.COMPANY` | Şirketler ve ticaret hukuku | Şirket, ticari işletme, sicil ve ticari kayıt düzenlemeleri. | SUB.GEN | 17 |
| `SUB.GEN.CONTRACT` | Sözleşme ve borç ilişkileri | Sözleşme, teslim, sorumluluk ve özel hukuk borçları. | SUB.GEN | 18 |
| `SUB.GEN.PROFESSION` | Düzenlenmiş mesleki hizmet | Meslek ruhsatı, ücret, disiplin ve sorumluluk kuralları. | SUB.GEN | 18 |
| `SUB.GEN.OTHER` | Diğer düzenleyici konu | Kontrollü konu listesine güvenle eşlenemeyen fakat düzenleyici içerik taşıyan kayıt. | SUB.GEN | 18 |

## Hüküm ve etki türü (33)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `EFF.PURPOSE` | Amaç | Düzenlemenin hedefini açıklar. | - | 18 |
| `EFF.SCOPE` | Kapsam ve uygulanabilirlik | Kişi, eşya, işlem, yer veya zaman kapsamını belirler. | - | 18 |
| `EFF.BASIS` | Hukuki dayanak | Üst norma veya yetki kaynağına dayanır. | - | 18 |
| `EFF.DEFINITION` | Tanım | Terim veya kavram tanımlar. | - | 18 |
| `EFF.COMPETENCE` | Yetki ve görev | Yetkili makamı, görevi veya sorumluluğu belirler. | - | 18 |
| `EFF.GENERAL_RULE` | Genel kural | Diğer özel etkilere girmeyen temel normatif kural. | - | 18 |
| `EFF.OBLIGATION` | Yükümlülük | Bir aktöre yapılması gereken davranış yükler. | - | 18 |
| `EFF.PROHIBITION` | Yasak | Bir davranışı veya işlemi yasaklar. | - | 18 |
| `EFF.PERMISSION` | İzin verilen işlem | Bir davranışa izin verir veya seçim tanır. | - | 18 |
| `EFF.AUTHORISATION` | Başvuru, izin ve yetkilendirme | Başvuru, ruhsat, sertifika veya yetki şartı getirir. | - | 18 |
| `EFF.ELIGIBILITY` | Uygunluk ve hak kazanma şartı | Hak, statü veya kolaylığa uygunluk ölçütü belirler. | - | 18 |
| `EFF.PROCEDURE` | Usul ve işlem adımı | Sıralı işlem, yöntem veya uygulama adımı tanımlar. | - | 18 |
| `EFF.CONDITION` | Koşul | Bir sonucun gerçekleşmesini koşula bağlar. | - | 18 |
| `EFF.EXCEPTION` | İstisna / ayrık durum | Genel kuraldan ayrılan durumu tanımlar. | - | 18 |
| `EFF.EXEMPTION` | Muafiyet | Vergi, izin veya yükümlülükten muafiyet tanır. | - | 18 |
| `EFF.DECLARATION` | Beyan yükümlülüğü | Beyan verilmesini veya içeriğini zorunlu kılar. | - | 18 |
| `EFF.DOCUMENT` | Belge ibrazı / düzenlenmesi | Belge, sertifika, fatura veya form ister/düzenletir. | - | 18 |
| `EFF.RECORDKEEP` | Kayıt ve saklama | Kayıt tutma, arşivleme veya saklama süresi belirler. | - | 18 |
| `EFF.REPORTING` | Bildirim ve raporlama | Bildirim, rapor veya bilgi verme yükümlülüğü getirir. | - | 18 |
| `EFF.DEADLINE` | Süre ve son tarih | İşlem süresi, uzatma, zamanaşımı veya son tarih belirler. | - | 18 |
| `EFF.CALCULATION` | Hesaplama | Tutar, oran, miktar veya formül hesaplar. | - | 18 |
| `EFF.VALUATION` | Değerleme | Kıymet/değer tespit yöntemi belirler. | - | 18 |
| `EFF.PAYMENT` | Ödeme ve tahsil | Ödeme, tahsil, faiz veya transfer yükümlülüğü. | - | 19 |
| `EFF.REFUND` | İade / geri verme | İade, geri verme veya kaldırma hakkı/usulü. | - | 19 |
| `EFF.INSPECTION` | Kontrol ve denetim | Muayene, kontrol, numune, denetim veya doğrulama öngörür. | - | 19 |
| `EFF.PENALTY` | Ceza ve yaptırım | İdari/cezai sonuç veya müeyyide getirir. | - | 19 |
| `EFF.REMEDY` | İtiraz ve dava yolu | İtiraz, uzlaşma, dava veya temyiz yolu tanır. | - | 19 |
| `EFF.TRANSITIONAL` | Geçiş hükmü | Eski-yeni düzenleme arasında geçiş veya kazanılmış hak kuralı. | - | 19 |
| `EFF.AMENDMENT` | Değişiklik hükmü | Başka metni değiştirir, ekler veya yeniden yazar. | - | 19 |
| `EFF.REPEAL` | Yürürlükten kaldırma | Başka hükmü tamamen/kısmen yürürlükten kaldırır. | - | 19 |
| `EFF.COMMENCEMENT` | Yürürlük / başlangıç | Yürürlük veya uygulama başlangıcını belirler. | - | 19 |
| `EFF.EXECUTION` | Yürütme | Düzenlemeyi yürütecek makamı belirtir. | - | 19 |
| `EFF.CROSSREF` | Atıf ve ilişki | Başka act, madde veya eke normatif atıf kurar. | - | 19 |

## Ek türü (30)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `ANX.GOODS_LIST` | Eşya / ürün listesi | Kod içermeyen veya karma ürün/eşya listesi. | - | 19 |
| `ANX.COMMODITY_CODE_LIST` | GTİP / tarife kodu listesi | GTİP, tarife pozisyonu veya fasıl satırları. | - | 19 |
| `ANX.PROHIBITED_LIST` | Yasaklı eşya listesi | İthal/ihracı yasak eşya. | - | 19 |
| `ANX.RESTRICTED_LIST` | Kontrollü / kısıtlı eşya listesi | İzin, kontrol veya özel şart kapsamındaki eşya. | - | 19 |
| `ANX.COUNTRY_LIST` | Ülke / bölge listesi | Ülke, ülke grubu veya coğrafi kapsam. | - | 19 |
| `ANX.AUTHORITY_LIST` | Yetkili makam listesi | Yetkili kurum, kuruluş veya birim. | - | 19 |
| `ANX.OFFICE_ROUTE_LIST` | Gümrük idaresi / kapı / güzergâh listesi | İdare, sınır kapısı, rota veya liman listesi. | - | 19 |
| `ANX.PERSON_ENTITY_LIST` | Kişi / firma / kuruluş listesi | Listelenmiş gerçek/tüzel kişi, firma veya tesis. | - | 19 |
| `ANX.DOC_CODE_LIST` | Belge kodu listesi | Belge türü ve kod eşleştirmesi. | - | 19 |
| `ANX.CONDITION_CODE_LIST` | Koşul kodu listesi | Koşul, muafiyet veya işlem kodları. | - | 19 |
| `ANX.ADDITIONAL_CODE_LIST` | Ek kod listesi | Ek tarife, vergi veya sistem kodları. | - | 19 |
| `ANX.DUTY_RATE_TABLE` | Vergi / oran tablosu | Vergi, mali yük, oran veya fiyat tablosu. | - | 19 |
| `ANX.QUOTA_THRESHOLD_TABLE` | Kota / eşik / miktar tablosu | Kota, kontenjan, limit, ağırlık veya parasal eşik. | - | 20 |
| `ANX.FEE_PENALTY_TABLE` | Ücret / ceza tablosu | Ücret, harç veya ceza miktarı. | - | 20 |
| `ANX.CALCULATION` | Hesaplama formülü / örneği | Formül, hesap tablosu veya sayısal örnek. | - | 20 |
| `ANX.APPLICATION_FORM` | Başvuru formu | İzin, belge, statü veya karar başvurusu. | - | 20 |
| `ANX.DECLARATION_FORM` | Beyan formu | Gümrük, vergi veya ticari beyan şablonu. | - | 20 |
| `ANX.CERTIFICATE` | Sertifika / şahadetname örneği | Menşe, dolaşım, uygunluk veya başka sertifika. | - | 20 |
| `ANX.LICENCE_PERMIT` | İzin / lisans örneği | Ruhsat, izin, lisans veya yetki belgesi. | - | 20 |
| `ANX.REPORT_TEMPLATE` | Rapor / kayıt / muhasebe şablonu | Rapor, cetvel, kayıt veya mali şablon. | - | 20 |
| `ANX.REQUIRED_DOCS` | Gerekli belgeler listesi | Başvuru/beyanla sunulacak belgeler. | - | 20 |
| `ANX.TECH_SPEC` | Teknik şartname | Teknik özellik, performans veya tolerans. | - | 20 |
| `ANX.TEST_METHOD` | Test / analiz yöntemi | Numune, laboratuvar veya test prosedürü. | - | 20 |
| `ANX.PROCEDURE_GUIDE` | Uygulama talimatı / iş akışı | Adımlar, ekranlar, kılavuz veya iş akışı. | - | 20 |
| `ANX.EXPLANATORY_NOTES` | Açıklama notları / izahname | Tarife veya düzenleme açıklama notları. | - | 20 |
| `ANX.AGREEMENT_PROTOCOL` | Antlaşma eki / protokol | Uluslararası anlaşma protokolü veya eki. | - | 20 |
| `ANX.CORRELATION_TABLE` | Korelasyon / karşılaştırma tablosu | Eski-yeni kod, madde veya sistem eşleştirmesi. | - | 20 |
| `ANX.EMBEDDED_INSTRUMENT` | Ekli bağımsız düzenleme / rehber | Üst yazıya eklenen ayrı yönerge, genelge, rehber veya karar. | - | 20 |
| `ANX.CORRESPONDENCE_ATTACHMENT` | Yazışma eki | 'Ek: 1 adet yazı' gibi idari yazışma eki; hukuki annex değildir. | - | 20 |
| `ANX.OTHER` | Diğer annex | Türü metin/tablo yapısından belirlenemeyen annex; inceleme gerekir. | - | 20 |

## Sektör (12)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `SEC.AGRI_FOOD` | Tarım ve gıda | Tarım ürünleri, gıda, yem ve balıkçılık. | - | 20 |
| `SEC.PLANT_ANIMAL` | Bitki ve hayvan sağlığı | Veteriner, karantina, bitki sağlığı ve canlı hayvan. | - | 21 |
| `SEC.HEALTH` | Sağlık, ilaç ve tıbbi cihaz | İlaç, tıbbi cihaz ve sağlık ürünü. | - | 21 |
| `SEC.CHEM_ENV` | Kimyasal, çevre, atık ve tehlikeli madde | Atık, kimyasal, ozon, radyoaktif ve tehlikeli madde. | - | 21 |
| `SEC.ENERGY` | Enerji, petrol ve akaryakıt | Elektrik, gaz, petrol, LPG, yakıt ve maden. | - | 21 |
| `SEC.TOBACCO_ALCOHOL` | Tütün ve alkol | Tütün, sigara, alkollü içki ve bandrol. | - | 21 |
| `SEC.TEXTILE` | Tekstil ve hazır giyim | İplik, kumaş, tekstil ve hazır giyim. | - | 21 |
| `SEC.VEHICLE_MACHINE` | Taşıt, makina ve aksam | Kara/hava/deniz taşıtı, makina ve parçalar. | - | 21 |
| `SEC.PRECIOUS` | Kıymetli maden, taş ve mücevher | Altın, gümüş, kıymetli taş ve rafineri. | - | 21 |
| `SEC.CULTURAL` | Kültür varlığı | Sanat, arkeoloji ve kültür varlığı. | - | 21 |
| `SEC.DEFENCE` | Savunma, silah ve çift kullanım | Silah, mühimmat, askeri malzeme ve çift kullanım. | - | 21 |
| `SEC.TELECOM` | Elektronik ve telekom cihazları | Telsiz, haberleşme, elektronik cihaz ve bileşen. | - | 21 |

## Relevance alanı (24)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `cross_border_regulatory` | Sınır-ötesi düzenleyici | Customs, import/export, trade policy, origin, transport, border control and cross-border enforcement. | - | 22 |
| `accounting` | Muhasebe | Books, records, tax accounting, costing, reporting and documentary accounting obligations. | - | 22 |
| `finance` | Finans | Payments, collection, foreign exchange, funds, credit, guarantees, interest and financial effects. | - | 22 |
| `tax_and_public_revenue` | Vergi ve kamu gelirleri | Taxes, fees, duties, VAT/excise, assessment, collection and public receivables. | - | 22 |
| `supply_chain_and_logistics` | Tedarik zinciri ve lojistik | Transit, transport, carriers, ports, warehousing, storage, containers and border operations. | - | 22 |
| `environmental_compliance` | Çevresel uyum | Waste, emissions, ozone, permits, hazardous materials and nature-protection compliance. | - | 22 |
| `product_compliance` | Ürün uygunluğu | Product safety, technical rules, CE/TSE, TAREKS, conformity assessment and surveillance. | - | 22 |
| `strategic_trade_control` | Stratejik ticaret kontrolü | Defence goods, weapons, ammunition, dual-use goods and controlled technology. | - | 22 |
| `sanctions_and_restrictive_measures` | Yaptırımlar ve kısıtlayıcı tedbirler | Embargoes, UN sanctions, asset freezes, travel bans and listed-party measures. | - | 22 |
| `employment_compliance` | İstihdam ve işyeri uyumu | Employment rights, contracts, collective rights and occupational health and safety. | - | 22 |
| `public_sector_governance` | Kamu yönetimi ve personel | Public organisation, powers, personnel, appointments, discipline, audit and internal processes. | - | 22 |
| `public_procurement` | Kamu ihalesi ve sözleşmeleri | Public purchasing, tenders, direct procurement, public contracts and contractor processes. | - | 22 |
| `investment_and_incentives` | Yatırım ve teşvikler | Investment certificates, project aid, state support, funds and export support. | - | 22 |
| `disputes_and_enforcement` | Uyuşmazlık ve yaptırım süreçleri | Appeals, litigation, settlement, enforcement, bankruptcy, fines, seizure and confiscation. | - | 22 |
| `digital_and_data` | Dijital, veri ve elektronik işlemler | Electronic signatures, e-notification, e-documents, single-window systems, data and cyber processes. | - | 22 |
| `public_assets_and_property` | Kamu varlıkları ve taşınmazlar | Treasury land, public movables, easements, allocation and public-asset management. | - | 22 |
| `agri_food_veterinary` | Tarım, gıda ve veterinerlik | Agriculture, plant/animal health, food safety, veterinary controls, fisheries, tobacco and alcohol products. | - | 22 |
| `health_and_life_sciences` | Sağlık ve yaşam bilimleri | Medicines, medical devices, pharmacy, healthcare and biological or chemical health controls. | - | 22 |
| `energy_and_natural_resources` | Enerji ve doğal kaynaklar | Electricity, gas, petroleum, fuels, LPG, mining, coal and energy markets. | - | 22 |
| `nonprofit_and_associations` | Dernek, vakıf ve kâr amacı gütmeyenler | Associations, foundations, donations, membership and nonprofit records or operations. | - | 22 |
| `market_conduct_and_consumer` | Piyasa davranışı ve tüketici | Consumer protection, competition, advertising, prices, labels and unfair commercial practices. | - | 22 |
| `intellectual_property_and_brand_protection` | Fikri mülkiyet ve marka koruması | Trade marks, patents, copyright, geographical indications and counterfeit-goods protection. | - | 22 |
| `regulated_professional_services` | Düzenlenmiş mesleki hizmetler | Customs brokerage, YGM, legal, accounting and other licensed professional services. | - | 23 |
| `international_and_treaty_compliance` | Uluslararası anlaşma ve yükümlülükler | Treaties, bilateral arrangements, NATO/diplomatic status and international obligations. | - | 23 |

## Ticaret akışı (7)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `trade_flow.import` | İthalat | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: import. Bu boyuttaki İthalat kapsamını ifade eder. | - | 23 |
| `trade_flow.export` | İhracat | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: export. Bu boyuttaki İhracat kapsamını ifade eder. | - | 23 |
| `trade_flow.transit` | Transit | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: transit. Bu boyuttaki Transit kapsamını ifade eder. | - | 23 |
| `trade_flow.entry` | Giriş | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: entry. Bu boyuttaki Giriş kapsamını ifade eder. | - | 23 |
| `trade_flow.exit` | Çıkış | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: exit. Bu boyuttaki Çıkış kapsamını ifade eder. | - | 23 |
| `trade_flow.domestic` | Yurt içi | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: domestic. Bu boyuttaki Yurt içi kapsamını ifade eder. | - | 23 |
| `trade_flow.re_export` | Yeniden ihracat | Uygulanabilirlik boyutu: trade_flow (Ticaret akışı). Kaynak değer: re_export. Bu boyuttaki Yeniden ihracat kapsamını ifade eder. | - | 23 |

## Gümrük rejimi (11)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `customs_regime.free_circulation` | Serbest dolaşıma giriş | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: free_circulation. Bu boyuttaki Serbest dolaşıma giriş kapsamını ifade eder. | - | 23 |
| `customs_regime.export` | İhracat | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: export. Bu boyuttaki İhracat kapsamını ifade eder. | - | 23 |
| `customs_regime.national_transit` | Ulusal transit | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: national_transit. Bu boyuttaki Ulusal transit kapsamını ifade eder. | - | 23 |
| `customs_regime.common_transit` | Ortak transit | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: common_transit. Bu boyuttaki Ortak transit kapsamını ifade eder. | - | 23 |
| `customs_regime.tir` | TIR | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: tir. Bu boyuttaki TIR kapsamını ifade eder. | - | 23 |
| `customs_regime.warehouse` | Antrepo | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: warehouse. Bu boyuttaki Antrepo kapsamını ifade eder. | - | 23 |
| `customs_regime.inward_processing` | Dahilde işleme | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: inward_processing. Bu boyuttaki Dahilde işleme kapsamını ifade eder. | - | 23 |
| `customs_regime.outward_processing` | Hariçte işleme | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: outward_processing. Bu boyuttaki Hariçte işleme kapsamını ifade eder. | - | 23 |
| `customs_regime.temporary_admission` | Geçici ithalat | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: temporary_admission. Bu boyuttaki Geçici ithalat kapsamını ifade eder. | - | 23 |
| `customs_regime.end_use` | Nihai kullanım | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: end_use. Bu boyuttaki Nihai kullanım kapsamını ifade eder. | - | 23 |
| `customs_regime.free_zone` | Serbest bölge | Uygulanabilirlik boyutu: customs_regime (Gümrük rejimi). Kaynak değer: free_zone. Bu boyuttaki Serbest bölge kapsamını ifade eder. | - | 23 |

## Aktör (14)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `actor.importer` | İthalatçı | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: importer. Bu boyuttaki İthalatçı kapsamını ifade eder. | - | 23 |
| `actor.exporter` | İhracatçı | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: exporter. Bu boyuttaki İhracatçı kapsamını ifade eder. | - | 23 |
| `actor.declarant` | Beyan sahibi | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: declarant. Bu boyuttaki Beyan sahibi kapsamını ifade eder. | - | 23 |
| `actor.representative` | Temsilci | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: representative. Bu boyuttaki Temsilci kapsamını ifade eder. | - | 23 |
| `actor.customs_broker` | Gümrük müşaviri | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: customs_broker. Bu boyuttaki Gümrük müşaviri kapsamını ifade eder. | - | 23 |
| `actor.YGM` | YGM | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: YGM. Bu boyuttaki YGM kapsamını ifade eder. | - | 23 |
| `actor.carrier` | Taşıyıcı | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: carrier. Bu boyuttaki Taşıyıcı kapsamını ifade eder. | - | 23 |
| `actor.warehouse_operator` | Antrepo işletmecisi | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: warehouse_operator. Bu boyuttaki Antrepo işletmecisi kapsamını ifade eder. | - | 23 |
| `actor.AEO` | AEO | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: AEO. Bu boyuttaki AEO kapsamını ifade eder. | - | 23 |
| `actor.manufacturer` | Üretici | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: manufacturer. Bu boyuttaki Üretici kapsamını ifade eder. | - | 23 |
| `actor.passenger` | Yolcu | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: passenger. Bu boyuttaki Yolcu kapsamını ifade eder. | - | 23 |
| `actor.postal_operator` | Posta işletmecisi | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: postal_operator. Bu boyuttaki Posta işletmecisi kapsamını ifade eder. | - | 23 |
| `actor.bank` | Banka | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: bank. Bu boyuttaki Banka kapsamını ifade eder. | - | 23 |
| `actor.authority` | Yetkili makam | Uygulanabilirlik boyutu: actor (Aktör). Kaynak değer: authority. Bu boyuttaki Yetkili makam kapsamını ifade eder. | - | 23 |

## Sistem (8)

| Kod | Türkçe ad | Açıklama | Üst konu | Sayfa |
| --- | --- | --- | --- | ---: |
| `system.BILGE` | BILGE | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: BILGE. Bu boyuttaki BILGE kapsamını ifade eder. | - | 23 |
| `system.NCTS` | NCTS | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: NCTS. Bu boyuttaki NCTS kapsamını ifade eder. | - | 23 |
| `system.TPS` | TPS | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: TPS. Bu boyuttaki TPS kapsamını ifade eder. | - | 23 |
| `system.TAREKS` | TAREKS | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: TAREKS. Bu boyuttaki TAREKS kapsamını ifade eder. | - | 23 |
| `system.SEBIS` | SEBIS | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: SEBIS. Bu boyuttaki SEBIS kapsamını ifade eder. | - | 23 |
| `system.ETGB` | ETGB | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: ETGB. Bu boyuttaki ETGB kapsamını ifade eder. | - | 23 |
| `system.KTS` | KTS | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: KTS. Bu boyuttaki KTS kapsamını ifade eder. | - | 23 |
| `system.TIR_TRACKING` | TIR_TRACKING | Uygulanabilirlik boyutu: system (Sistem). Kaynak değer: TIR_TRACKING. Bu boyuttaki TIR_TRACKING kapsamını ifade eder. | - | 23 |
