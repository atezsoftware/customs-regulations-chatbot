"""Local operative deadlines must retain their exact article and paragraph scope."""

import hashlib
import json

import pytest

from onyx.regulatory.document_intake import metadata

# Exact referenced source chunks and full cached ACK bodies from the Resmi Gazete
# regression. Unreferenced document chunks are unnecessary for this replay.
_CACHED_SOURCES_JSON = r"""[
{"chunk_id":"rc_6fbfb0a8f5ef35e9b6eca376413c977426fdad49","chunk_type":"free_text","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK"],"structural_metadata":{"appendix_label":null,"article_no":null,"article_title":null,"clause_label":null,"document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":null,"subclause_label":null,"table_index":null,"table_row_index":null},"text":"20 Aralık 2025 CUMARTESİ\nResmî Gazete\nSayı : 33113\nYÖNETMELİK\nTicaret Bakanlığından:\nPAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL\nKONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI\nTERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK\nBaşlangıç Hükümleri"},
{"chunk_id":"rc_f5a356ca5327dded19505770ed685c9d676ea49c","chunk_type":"clause","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","ON İKİNCİ BÖLÜM","GEÇİCİ MADDE 1 - Menşe ispat belgelerine ibare eklenmesi","(1) 31/12/2025 tarihine kadar;","a) \"Gözlemler\""],"structural_metadata":{"appendix_label":null,"article_no":"GEÇİCİ 1","article_title":"Menşe ispat belgelerine ibare eklenmesi","clause_label":"a","document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":"1","subclause_label":null,"table_index":null,"table_row_index":null},"text":"GEÇİCİ MADDE 1-\n(1) 31/12/2025 tarihine kadar;\na) Bu Yönetmeliğe uygun olarak düzenlenen EUR.1 dolaşım belgelerinin (7) numaralı “Gözlemler” kutusuna İngilizce “REVISED RULES” ibaresi kaydedilir."},
{"chunk_id":"rc_d0f9fbe915d0b1781ac31759e05ab0717ab198c3","chunk_type":"paragraph","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","ON İKİNCİ BÖLÜM","MADDE 60 - Yürürlük","(1) Bu Yönetmelik 1/1/2025 tarihinden geçerli olmak üzere yayımı tarihinde yürürlüğe girer."],"structural_metadata":{"appendix_label":null,"article_no":"60","article_title":"Yürürlük","clause_label":null,"document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":"1","subclause_label":null,"table_index":null,"table_row_index":null},"text":"MADDE 60-\n(1) Bu Yönetmelik 1/1/2025 tarihinden geçerli olmak üzere yayımı tarihinde yürürlüğe girer."},
{"chunk_id":"rc_3d832da309c3955162cd29dd3dcfdc867f9372d7","chunk_type":"clause","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","ON İKİNCİ BÖLÜM","GEÇİCİ MADDE 1 - Menşe ispat belgelerine ibare eklenmesi","(1) 31/12/2025 tarihine kadar;","b) \"REVISED RULES\""],"structural_metadata":{"appendix_label":null,"article_no":"GEÇİCİ 1","article_title":"Menşe ispat belgelerine ibare eklenmesi","clause_label":"b","document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":"1","subclause_label":null,"table_index":null,"table_row_index":null},"text":"b) Bu Yönetmeliğe uygun olarak hazırlanan menşe beyanlarına, metnin sonuna yazılmak üzere İngilizce “REVISED RULES” ibaresi eklenir.\n2012 kurallarına ilişkin mevzuata göre düzenlenmiş olan menşe ispat belgelerinin sonradan kontrol işlemleri"},
{"chunk_id":"rc_d87d9b5a39d24a08dc80092da398f35efd70a7be","chunk_type":"clause","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","ON İKİNCİ BÖLÜM","GEÇİCİ MADDE 2","(1) 31/12/2029 tarihine kadar;","a) 56 ncı madde hükümlerinin uygulanmasıyla kabul edilen ve 2012 kurallarına ilişkin mevzu..."],"structural_metadata":{"appendix_label":null,"article_no":"GEÇİCİ 2","article_title":null,"clause_label":"a","document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":"1","subclause_label":null,"table_index":null,"table_row_index":null},"text":"GEÇİCİ MADDE 2-\n(1) 31/12/2029 tarihine kadar;\na) 56 ncı madde hükümlerinin uygulanmasıyla kabul edilen ve 2012 kurallarına ilişkin mevzuata göre düzenlenmiş olan menşe ispat belgelerine ilişkin olarak on ikinci bölüm hükümleri çerçevesinde sonradan kontrol işlemleri gerçekleştirilebilir."},
{"chunk_id":"rc_991b6e42e4143e397b5dfb1cc09312aacadf30b9","chunk_type":"clause","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","ON İKİNCİ BÖLÜM","GEÇİCİ MADDE 2","(1) 31/12/2029 tarihine kadar;","b) 2012 kurallarına ilişkin mevzuata uygun olarak 56 ncı maddede belirtilen tarihlerden ön..."],"structural_metadata":{"appendix_label":null,"article_no":"GEÇİCİ 2","article_title":null,"clause_label":"b","document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":"1","subclause_label":null,"table_index":null,"table_row_index":null},"text":"b) 2012 kurallarına ilişkin mevzuata uygun olarak 56 ncı maddede belirtilen tarihlerden önce Türkiye’de düzenlenen ve tercihli tarife talebiyle Taraf ülke gümrük idarelerine ibraz edilen menşe ispat belgelerine ilişkin olarak ithalatın gerçekleştirildiği Taraf ülkeden alınan sonradan kontrol talepleri on ikinci bölüm hükümlerine göre karşılanır."},
{"chunk_id":"rc_9a2937af197ecb89bcc2c9fd3e9306f667c16768","chunk_type":"paragraph","heading_path":["PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","ON İKİNCİ BÖLÜM","MADDE 61 - Yürütme","(1) Bu Yönetmelik hükümlerini Ticaret Bakanı yürütür."],"structural_metadata":{"appendix_label":null,"article_no":"61","article_title":"Yürütme","clause_label":null,"document_date":null,"document_number":null,"document_type":"yonetmelik","paragraph_no":"1","subclause_label":null,"table_index":null,"table_row_index":null},"text":"MADDE 61-\n(1) Bu Yönetmelik hükümlerini Ticaret Bakanı yürütür.\n[20251220-3-1](20251220-3-1.pdf)"}
]"""

_CACHED_ACKS: dict[str, tuple[str, str]] = {
    "8a1dc3a0bc95eaf1a0105d10e8adfdf2f1b2a5aa595f9667494e34b975f7b594": (
        "331daf8c476ff0061139d4e1570348d96c5793ccf712527704b6506ff63e0a61",
        '{"chunks":[{"chunk_id":"rc_f5a356ca5327dded19505770ed685c9d676ea49c","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"7482fc1d95fb6200ccfe0a5a96136793d6628bfe246be1a0113c2484c5a93d54","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"GEÇİCİ MADDE 1-","evidence_id":"2fa8543a977f088e8a132a1277b037d84c30242dae86e3354af73d2c8096ae00","basis":"explicit","reference_evidence_id":""},{"field":"article_title","value":"GEÇİCİ MADDE 1 - Menşe ispat belgelerine ibare eklenmesi","evidence_id":"d8f96c7699b81a7ada2270fb047dd2544df67c73cf19e2fad6f1385d27e1b784","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""},{"field":"validity_end_date","value":"2025-12-31","evidence_id":"6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8","basis":"explicit","reference_evidence_id":""}]}],"warnings":[]}',
    ),
    "6b870bc840566fab37232a3ce734ce08ccf3446b9d2b7f9ef5a0ba8f95a7dcd7": (
        "b30a6ff51571b65791507c0b904c717e653d414e72ed7a57ffd2fe3b43766a96",
        '{"chunks":[{"chunk_id":"rc_f5a356ca5327dded19505770ed685c9d676ea49c","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"publication_number","value":"33113","evidence_id":"e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"GEÇİCİ MADDE 1","evidence_id":"2fa8543a977f088e8a132a1277b037d84c30242dae86e3354af73d2c8096ae00","basis":"explicit","reference_evidence_id":""},{"field":"article_title","value":"Menşe ispat belgelerine ibare eklenmesi","evidence_id":"d8f96c7699b81a7ada2270fb047dd2544df67c73cf19e2fad6f1385d27e1b784","basis":"explicit","reference_evidence_id":""},{"field":"paragraph_no","value":"1","evidence_id":"6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8","basis":"explicit","reference_evidence_id":""},{"field":"clause_label","value":"a","evidence_id":"c4a441f3a2ad7b63986184fabd04941596cb806f1e4d6a53613927204ae18d3b","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""},{"field":"validity_end_date","value":"2025-12-31","evidence_id":"6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8","basis":"explicit","reference_evidence_id":""}]},{"chunk_id":"rc_3d832da309c3955162cd29dd3dcfdc867f9372d7","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"publication_number","value":"33113","evidence_id":"e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"GEÇİCİ MADDE 1","evidence_id":"2fa8543a977f088e8a132a1277b037d84c30242dae86e3354af73d2c8096ae00","basis":"explicit","reference_evidence_id":""},{"field":"article_title","value":"Menşe ispat belgelerine ibare eklenmesi","evidence_id":"d8f96c7699b81a7ada2270fb047dd2544df67c73cf19e2fad6f1385d27e1b784","basis":"explicit","reference_evidence_id":""},{"field":"paragraph_no","value":"1","evidence_id":"6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8","basis":"explicit","reference_evidence_id":""},{"field":"clause_label","value":"b","evidence_id":"79d08a930525ad2ad32a1ce1dd5747322a3e7992d39e2706ebb7921c1365bb58","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""},{"field":"validity_end_date","value":"2025-12-31","evidence_id":"6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8","basis":"explicit","reference_evidence_id":""}]},{"chunk_id":"rc_d87d9b5a39d24a08dc80092da398f35efd70a7be","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"publication_number","value":"33113","evidence_id":"e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"GEÇİCİ MADDE 2","evidence_id":"bdb5f5f1c71471a49ae44fdc828653e710e5c0af8c6d2c356657ae784835a93e","basis":"explicit","reference_evidence_id":""},{"field":"paragraph_no","value":"1","evidence_id":"a2ac93f6a26bbf341f6b2dab7d1cf9a7e6fab15a8de81002b93c5c6247e9b06a","basis":"explicit","reference_evidence_id":""},{"field":"clause_label","value":"a","evidence_id":"6207b70cfaeba299dcc43051b863f7a089bbce4c5d4832896e99ae121565cc7d","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""},{"field":"validity_end_date","value":"2029-12-31","evidence_id":"a2ac93f6a26bbf341f6b2dab7d1cf9a7e6fab15a8de81002b93c5c6247e9b06a","basis":"explicit","reference_evidence_id":""}]},{"chunk_id":"rc_991b6e42e4143e397b5dfb1cc09312aacadf30b9","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"publication_number","value":"33113","evidence_id":"e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"GEÇİCİ MADDE 2","evidence_id":"bdb5f5f1c71471a49ae44fdc828653e710e5c0af8c6d2c356657ae784835a93e","basis":"explicit","reference_evidence_id":""},{"field":"paragraph_no","value":"1","evidence_id":"a2ac93f6a26bbf341f6b2dab7d1cf9a7e6fab15a8de81002b93c5c6247e9b06a","basis":"explicit","reference_evidence_id":""},{"field":"clause_label","value":"b","evidence_id":"dfceb32532cc736df0d1bececdf5f5c0055a657c47d84c74629d6e47a1ddcd9c","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""},{"field":"validity_end_date","value":"2029-12-31","evidence_id":"a2ac93f6a26bbf341f6b2dab7d1cf9a7e6fab15a8de81002b93c5c6247e9b06a","basis":"explicit","reference_evidence_id":""}]},{"chunk_id":"rc_d0f9fbe915d0b1781ac31759e05ab0717ab198c3","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"publication_number","value":"33113","evidence_id":"e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"60","evidence_id":"0c4f472dba1e63df27d48bccd282b547e9bc55eb3137efff5f0f607ee99852bf","basis":"explicit","reference_evidence_id":""},{"field":"article_title","value":"Yürürlük","evidence_id":"0f75ccf3930c0f0bd574b420e1929155e227e35deb2890330b39a053c54dce44","basis":"explicit","reference_evidence_id":""},{"field":"paragraph_no","value":"1","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""}]},{"chunk_id":"rc_9a2937af197ecb89bcc2c9fd3e9306f667c16768","fields":[{"field":"document_title","value":"PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK","evidence_id":"a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b","basis":"explicit","reference_evidence_id":""},{"field":"document_type","value":"YÖNETMELİK","evidence_id":"39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b","basis":"explicit","reference_evidence_id":""},{"field":"publication_date","value":"2025-12-20","evidence_id":"766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3","basis":"explicit","reference_evidence_id":""},{"field":"publication_number","value":"33113","evidence_id":"e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53","basis":"explicit","reference_evidence_id":""},{"field":"article_no","value":"61","evidence_id":"7176499b705c30b3e4241f6c12dd3e11a72cf97a5f50d719da834ce34f54d3c5","basis":"explicit","reference_evidence_id":""},{"field":"article_title","value":"Yürütme","evidence_id":"737ebf537de861b749590d87071b67bf142da11d873459ed0e6f5847e07643b7","basis":"explicit","reference_evidence_id":""},{"field":"paragraph_no","value":"1","evidence_id":"e4a325e081f9fc97834dfa68607498c71ee4ab4baf7a6aaf1d9aef6b58009c40","basis":"explicit","reference_evidence_id":""},{"field":"validity_start_date","value":"2025-01-01","evidence_id":"f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49","basis":"explicit","reference_evidence_id":""}]}],"warnings":["No document_date explicitly specified in the text of the regulation."]}',
    ),
    "78436097795073f046d3d06e4abd7c13ccf0e03dbafb6e1503992816e8aed765": (
        "676352113495b47c03f763d962c06bf4c8f5b985f0f46d671e1fe46d8f05dba9",
        '{\n  "chunks": [\n    {\n      "chunk_id": "rc_f5a356ca5327dded19505770ed685c9d676ea49c",\n      "fields": [\n        {\n          "field": "article_no",\n          "value": "GEÇİCİ MADDE 1",\n          "evidence_id": "2fa8543a977f088e8a132a1277b037d84c30242dae86e3354af73d2c8096ae00",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "article_title",\n          "value": "Menşe ispat belgelerine ibare eklenmesi",\n          "evidence_id": "d8f96c7699b81a7ada2270fb047dd2544df67c73cf19e2fad6f1385d27e1b784",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "clause_label",\n          "value": "a",\n          "evidence_id": "c4a441f3a2ad7b63986184fabd04941596cb806f1e4d6a53613927204ae18d3b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "document_title",\n          "value": "PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK",\n          "evidence_id": "a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "document_type",\n          "value": "YÖNETMELİK",\n          "evidence_id": "39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "publication_date",\n          "value": "2025-12-20",\n          "evidence_id": "766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "publication_number",\n          "value": "33113",\n          "evidence_id": "e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "validity_start_date",\n          "value": "2025-12-20",\n          "evidence_id": "f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49",\n          "basis": "relative_publication",\n          "reference_evidence_id": "766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3"\n        },\n        {\n          "field": "validity_end_date",\n          "value": "2025-12-31",\n          "evidence_id": "6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        }\n      ]\n    },\n    {\n      "chunk_id": "rc_3d832da309c3955162cd29dd3dcfdc867f9372d7",\n      "fields": [\n        {\n          "field": "article_no",\n          "value": "GEÇİCİ MADDE 1",\n          "evidence_id": "2fa8543a977f088e8a132a1277b037d84c30242dae86e3354af73d2c8096ae00",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "article_title",\n          "value": "Menşe ispat belgelerine ibare eklenmesi",\n          "evidence_id": "d8f96c7699b81a7ada2270fb047dd2544df67c73cf19e2fad6f1385d27e1b784",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "clause_label",\n          "value": "b",\n          "evidence_id": "79d08a930525ad2ad32a1ce1dd5747322a3e7992d39e2706ebb7921c1365bb58",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "document_title",\n          "value": "PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK",\n          "evidence_id": "a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "document_type",\n          "value": "YÖNETMELİK",\n          "evidence_id": "39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "publication_date",\n          "value": "2025-12-20",\n          "evidence_id": "766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "publication_number",\n          "value": "33113",\n          "evidence_id": "e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "validity_start_date",\n          "value": "2025-12-20",\n          "evidence_id": "f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49",\n          "basis": "relative_publication",\n          "reference_evidence_id": "766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3"\n        },\n        {\n          "field": "validity_end_date",\n          "value": "2025-12-31",\n          "evidence_id": "6aaac8bb66a711013eca4f210e661eb163787d63a7be98a17232985fc9b6e5c8",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        }\n      ]\n    },\n    {\n      "chunk_id": "rc_d87d9b5a39d24a08dc80092da398f35efd70a7be",\n      "fields": [\n        {\n          "field": "article_no",\n          "value": "GEÇİCİ MADDE 2",\n          "evidence_id": "bdb5f5f1c71471a49ae44fdc828653e710e5c0af8c6d2c356657ae784835a93e",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "clause_label",\n          "value": "a",\n          "evidence_id": "6207b70cfaeba299dcc43051b863f7a089bbce4c5d4832896e99ae121565cc7d",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "document_title",\n          "value": "PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK",\n          "evidence_id": "a2f4c4c1a777d7c9c0bcfcae00213cc1a3bdd0e4a74b08c8ec26db228597055b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "document_type",\n          "value": "YÖNETMELİK",\n          "evidence_id": "39a509c2156b115ccee968c69da715d9a7da332d87bffc9bd0125a749a66318b",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "publication_date",\n          "value": "2025-12-20",\n          "evidence_id": "766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "publication_number",\n          "value": "33113",\n          "evidence_id": "e5bec667c48a908e34cb7692f585d11f7cb7d671d6588e7d4833f15341884a53",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        },\n        {\n          "field": "validity_start_date",\n          "value": "2025-12-20",\n          "evidence_id": "f9204318dd492eea9562bdfc5c7aea23ce597c20bb8f745e86ddb221fce24c49",\n          "basis": "relative_publication",\n          "reference_evidence_id": "766ac080b147956611e7665889a86a3d094cef9ac09834587d49cfaf9fe543b3"\n        },\n        {\n          "field": "validity_end_date",\n          "value": "2029-12-31",\n          "evidence_id": "a2ac93f6a26bbf341f6b2dab7d1cf9a7e6fab15a8de81002b93c5c6247e9b06a",\n          "basis": "explicit",\n          "reference_evidence_id": ""\n        }\n      ]\n    }\n  ],\n  "warnings": []\n}',
    ),
}


def _source(
    identifier: str = "source",
    *,
    article: str = "GEÇİCİ MADDE 1",
    paragraph: str = "1",
    quote: str = "(1) 31/12/2025 tarihine kadar;",
    text: str | None = None,
    headings: list[str] | None = None,
) -> metadata.MetadataChunk:
    return metadata.MetadataChunk(
        chunk_id=identifier,
        text=text
        if text is not None
        else f"{article}-\n{quote}\na) Geçici kural uygulanır.",
        heading_path=headings
        if headings is not None
        else [article, quote, "a) Geçici kural"],
        chunk_type="clause",
        structural_metadata={"paragraph_no": paragraph},
    )


def _response(
    target: metadata.MetadataChunk, source: metadata.MetadataChunk, quote: str
) -> str:
    return metadata.LegalMetadataResult(
        chunks=[
            metadata.ChunkLegalMetadata(
                chunk_id=target.chunk_id,
                fields=[
                    metadata.LegalMetadataFact(
                        field="validity_end_date",
                        value="2025-12-31",
                        source_chunk_id=source.chunk_id,
                        evidence_quote=quote,
                        basis="explicit",
                    )
                ],
            )
        ]
    ).model_dump_json()


def _validate(
    target: metadata.MetadataChunk, source: metadata.MetadataChunk, quote: str
) -> metadata.LegalMetadataResult:
    chunks = [source] if target.chunk_id == source.chunk_id else [source, target]
    return metadata.validate_metadata_response(
        _response(target, source, quote), chunks=chunks, target_ids=[target.chunk_id]
    )


@pytest.mark.parametrize("request_hash", list(_CACHED_ACKS))
def test_exact_cached_single_and_parent_window_acks_keep_values_and_provenance(
    request_hash: str,
) -> None:
    expected_sha, response = _CACHED_ACKS[request_hash]
    assert hashlib.sha256(response.encode()).hexdigest() == expected_sha
    chunks = [
        metadata.MetadataChunk.model_validate(raw)
        for raw in json.loads(_CACHED_SOURCES_JSON)
    ]
    targets = [target["chunk_id"] for target in json.loads(response)["chunks"]]
    assert len(targets) in {1, 3, 6}
    input_hash = metadata.metadata_input_hash(chunks)
    request = metadata.build_metadata_request(chunks, targets)
    request_before = request.model_dump_json()
    result = metadata.validate_metadata_transport_response(
        response, chunks=chunks, target_ids=targets
    )
    catalog = {
        entry.evidence_id: entry
        for entry in metadata.build_metadata_evidence_catalog(chunks)
    }
    validated = {
        target.chunk_id: {fact.field: fact for fact in target.fields}
        for target in result.chunks
    }
    for target in json.loads(response)["chunks"]:
        for original in target["fields"]:
            fact = validated[target["chunk_id"]][original["field"]]
            assert fact.value == original["value"]
            assert (
                fact.source_chunk_id == catalog[original["evidence_id"]].source_chunk_id
            )
            assert fact.evidence_quote == catalog[original["evidence_id"]].text
    assert metadata.metadata_input_hash(chunks) == input_hash
    assert request.model_dump_json() == request_before
    assert hashlib.sha256(response.encode()).hexdigest() == expected_sha


@pytest.mark.parametrize("article", ["MADDE 1", "GEÇİCİ MADDE 1", "EK MADDE 1"])
def test_bare_operative_body_introduction_can_end_its_own_provision(
    article: str,
) -> None:
    source = _source(article=article)
    fact = _validate(source, source, source.text.splitlines()[1]).chunks[0].fields[0]
    assert fact.value == "2025-12-31"
    assert fact.source_chunk_id == source.chunk_id


def test_matching_sibling_requires_the_same_explicit_parent_introduction() -> None:
    source = _source()
    target = _source(
        "target",
        text="b) Diğer geçici kural uygulanır.",
        headings=source.heading_path[:-1] + ["b) Diğer kural"],
    )
    assert (
        _validate(target, source, source.text.splitlines()[1]).chunks[0].fields[0].value
        == "2025-12-31"
    )


@pytest.mark.parametrize("target_article", ["MADDE 1", "GEÇİCİ MADDE 2", "EK MADDE 1"])
def test_deadline_cannot_cross_article_numbers_or_namespaces(
    target_article: str,
) -> None:
    source = _source()
    target = _source("target", article=target_article)
    with pytest.raises(ValueError, match="another law"):
        _validate(target, source, source.text.splitlines()[1])


@pytest.mark.parametrize(
    "change",
    [
        {"structural_metadata": {"paragraph_no": "2"}},
        {"text": "(2) b) Başka fıkranın kuralı uygulanır."},
        {"text": "MADDE 1-\n(1) Başka madde kuralı uygulanır."},
        {
            "heading_path": [
                "GEÇİCİ MADDE 1",
                "(2) 31/12/2025 tarihine kadar;",
                "b) Kural",
            ]
        },
        {"heading_path": ["GEÇİCİ MADDE 1", "b) Kural"]},
    ],
)
def test_wrong_paragraph_or_missing_common_parent_proof_is_rejected(
    change: dict[str, object],
) -> None:
    source = _source()
    target = _source(
        "target",
        text="b) Diğer geçici kural uygulanır.",
        headings=source.heading_path[:-1] + ["b) Diğer kural"],
    ).model_copy(update=change)
    with pytest.raises(ValueError, match="another law"):
        _validate(target, source, source.text.splitlines()[1])


@pytest.mark.parametrize(
    "quote",
    [
        "(1) Başvurular 31/12/2025 tarihine kadar yapılır.",
        "(1) Yükümlülükler 31/12/2025 tarihine kadar yerine getirilir.",
        "4000 sayılı Kanun 31/12/2025 tarihinde yürürlükten kaldırılmıştır.",
        "(1) 31/12/2025 tarihine kadar başvuru yapılır;",
        "(1) 31/12/2025 tarihine kadar; önceki kanun uygulanır.",
    ],
)
def test_application_compliance_and_other_law_repeal_quotes_are_not_local_expiry(
    quote: str,
) -> None:
    source = _source(quote=quote)
    with pytest.raises(ValueError, match="another law"):
        _validate(source, source, quote)


def test_quoted_deadline_after_another_operative_paragraph_is_not_an_introduction() -> (
    None
):
    quote = "(2) 31/12/2025 tarihine kadar;"
    source = _source(
        paragraph="2",
        quote=quote,
        text=f"GEÇİCİ MADDE 1-\n(1) Kural uygulanır.\n{quote}\na) İşlem yapılır.",
    )
    with pytest.raises(ValueError, match="another law"):
        _validate(source, source, quote)


def test_unscoped_bare_deadline_does_not_supply_its_own_article_identity() -> None:
    quote = "(1) 31/12/2025 tarihine kadar;"
    source = _source(text=quote + "\na) Kural uygulanır.", headings=[])
    with pytest.raises(ValueError, match="another law"):
        _validate(source, source, quote)


def test_heading_only_deadline_remains_unusable_for_validity_dates() -> None:
    source = _source(text="a) Kural uygulanır.")
    entry = next(
        entry
        for entry in metadata.build_metadata_evidence_catalog([source])
        if entry.scope == "heading" and entry.text == source.heading_path[1]
    )
    response = json.dumps(
        {
            "chunks": [
                {
                    "chunk_id": source.chunk_id,
                    "fields": [
                        {
                            "field": "validity_end_date",
                            "value": "2025-12-31",
                            "evidence_id": entry.evidence_id,
                            "basis": "explicit",
                            "reference_evidence_id": "",
                        }
                    ],
                }
            ]
        }
    )
    with pytest.raises(ValueError, match="requires body evidence"):
        metadata.validate_metadata_transport_response(
            response, chunks=[source], target_ids=[source.chunk_id]
        )
