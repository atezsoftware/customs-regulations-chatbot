"""Cropped authoring keeps source ownership without weakening original-source guards."""

import hashlib
import json

import pytest

from onyx.asv3.authority import cited_lower_statute_gap
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.cropped_repair_model import integrate_repair
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.prompts.asv3.research import DIRECT_ANSWER_OPENING
from onyx.prompts.asv3_cropped_writing import CROPPED_SOURCE_OWNERSHIP
from tests.unit.onyx.asv3.test_cropped_repair_model import checks, current, fixture
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_native_model_adapter import native_response

# Exact captured GY579/2 body: run20261013T161849Z, source601c6f1f...,
# chunk rc_13f903efa4c09f34c2fb9ac5b0380c57cafb4fe8, citation383.
_REGULATION = (
    "(2) Soruşturma veya kovuşturma sonucunda Kanunda tayin edilen cezalardan daha ağır "
    "bir para cezasına karar verildiği veya fiilin suç teşkil ettiğine ve suçtan dolayı "
    "yaptırım uygulanmasına karar verildiği takdirde, kararın kesinleşmesi ve hükmolunan "
    "paranın tahsili şartıyla Kanuna göre verilmiş olan ceza kaldırılır ve alınmış olan "
    "para geri verilir."
)
# Exact captured noncited GK235/5 record provides the canonical bare-law alias.
_GK_ORIGINAL = '{"source_id":"12328001-2f21-407e-b230-7d50a8dbf28f","text":"**5.** Türkiye Gümrük Bölgesine getirilen ve varış gümrük idaresine karayolu ile sevk edilmek üzere transit rejimine konu edilen serbest dolaşımda olmayan eşyanın, yapılan kontrol veya muayene sonucunda, beyan edilenden belirgin bir şekilde farklı cinste olduğunun tahlil, teknik inceleme ve araştırmaya gerek olmaksızın tespiti hâlinde fiilin niteliğine göre bu fıkranın (a) ve (b) bentlerine, transit rejimine konu edilen serbest dolaşımda olmayan eşyanın, hareket gümrük idaresinden sevk edilmesinden sonra varış gümrük idaresine varışından önce ya da varış gümrük idaresinde yapılan kontrolü veya muayenesi sonucunda beyana göre eksiklik veya fazlalık tespit edilmesi hâlinde fiilin niteliğine göre bu fıkranın (c) veya (d) bentleri uyarınca işlem yapılır.\xa0Bu fıkranın (c) ve (d) bentleri kapsamında değerlendirilemeyecek eksiklik ve fazlalık durumları ile bu durumların tespitine ilişkin usul ve esaslar Bakanlıkça belirlenir.","chunk_id":"rc_791eeb6fcac4fd50ae86f823254bd085cb414a40","search_doc":{"document_id":"12328001-2f21-407e-b230-7d50a8dbf28f","chunk_ind":1070,"semantic_identifier":"İhracat Mevzuatı/Kanun, Kararname, Yönetmelik, Tebliğ, Genelge/gumruk_kanunu_.md","link":null,"blurb":"**5.** Türkiye Gümrük Bölgesine getirilen ve varış gümrük idaresine karayolu ile sevk edilmek üzere transit rejimine konu edilen serbest dolaşımda olmayan eşyanın, yapılan kontrol veya muayene sonucunda, beyan edilenden belirgin bir şekilde","source_type":"file","boost":0,"hidden":false,"metadata":{"regulatory_chunk_id":"rc_791eeb6fcac4fd50ae86f823254bd085cb414a40","regulatory_heading_path":["4458 SAYILI GÜMRÜK KANUNU","Cezalar","İKİNCİ BÖLÜM","MADDE 235","5. Türkiye Gümrük Bölgesine getirilen ve varış gümrük idaresine karayolu ile sevk edilmek..."],"regulatory_validity_start_date":"","regulatory_validity_end_date":""},"score":null,"is_relevant":null,"relevance_explanation":null,"match_highlights":[],"updated_at":null,"primary_owners":null,"secondary_owners":null,"is_internet":false,"file_id":"f3bf6768-cbe0-4cd2-926d-8958578530e6"},"question_ids":[],"metadata":{"canonical_metadata":{"document_id":"12328001-2f21-407e-b230-7d50a8dbf28f","chunk_index":1070,"title":"İhracat Mevzuatı/Kanun, Kararname, Yönetmelik, Tebliğ, Genelge/gumruk_kanunu_.md","source_type":"user_file","semantic_identifier":"İhracat Mevzuatı/Kanun, Kararname, Yönetmelik, Tebliğ, Genelge/gumruk_kanunu_.md — 4458 SAYILI GÜMRÜK KANUNU > Cezalar > İKİNCİ BÖLÜM > MADDE 235 > 5. Türkiye Gümrük Bölgesine getirilen ve varış gümrük idaresine karayolu ile sevk edilmek...","source_links":"{\\"0\\": \\"\\"}","document_sets":["PC Külliyatı"],"regulatory_chunk_id":"rc_791eeb6fcac4fd50ae86f823254bd085cb414a40","heading_path":["4458 SAYILI GÜMRÜK KANUNU","Cezalar","İKİNCİ BÖLÜM","MADDE 235","5. Türkiye Gümrük Bölgesine getirilen ve varış gümrük idaresine karayolu ile sevk edilmek..."],"provision_identifiers":["madde 235"],"legal_dates":["1999-10-27","1999-11-04"],"article_no":"235","clause_label":null,"paragraph_no":"5","article_title":null,"chunk_variant":"atomic","document_date":"1999-10-27","document_type":"kanun","source_regulatory_chunk_ids":[],"read_as_of_date":"2026-10-13"},"position":1070},"text_hash":"c6a7be4a3a8e43c9bbd3f73722a8fdaa764d31f57458d9a4d5e601ba87c7d7e8"}'

_OPTIONS = {
    "strict_reference_boundaries": True,
    "syntactic_reference_binding": True,
    "strict_numeric_paragraphs": True,
}


@pytest.mark.parametrize("writing", [False, True])
@pytest.mark.parametrize(
    "cropped,cheap", [(True, False), (False, False), (False, True)]
)
def test_actual_native_fit_scopes_source_ownership_to_cropped(
    writing: bool, cropped: bool, cheap: bool
) -> None:
    adapter, ledger, selected = fixture()
    adapter.context.services.update(asv3_exp_cropped=cropped, asv3_exp_cheap=cheap)
    adapter._writing_answer = writing
    adapter.answer_llm = selected
    prompt, _, _ = adapter._fit_native_decision(current(ledger))
    system = prompt[0].content
    assert isinstance(system, str)
    assert system.count(CROPPED_SOURCE_OWNERSHIP) == int(cropped and not cheap)
    assert system.count(DIRECT_ANSWER_OPENING) == int(cropped)
    selected.invoke.assert_not_called()


def test_fresh_retained_integration_receives_instruction_and_actual_originals() -> None:
    adapter, ledger, selected = fixture()
    selected.invoke.return_value = native_response("submit_retained_answer", "{}")
    answer = current(ledger).draft_to_repair
    result = integrate_repair(adapter, current(ledger), checks())
    assert result.calls[0].arguments["answer"] == answer
    assert selected.invoke.call_count == 1
    system = selected.invoke.call_args.kwargs["prompt"][0].content
    assert isinstance(system, str) and system.count(CROPPED_SOURCE_OWNERSHIP) == 1
    assert adapter.last_call_id is not None
    assert ledger.completely_delivered(adapter.last_call_id) == {1, 2, 3}


def test_exact_owner_quotation_avoids_false_bare_law_obligation() -> None:
    assert hashlib.sha256(_REGULATION.encode()).hexdigest() == (
        "be390164922b671964b9ce23e4a027940054f9f0378e6789a661516cf3400512"
    )
    ledger = EvidenceLedger()
    ledger.add(
        [
            original("Gümrük Yönetmeliği", "579", kind="yonetmelik", text=_REGULATION),
            EvidenceItem.model_validate(json.loads(_GK_ORIGINAL)),
        ],
        RunContext(),
    )
    blocked = (
        "Yönetmelik m.579/2, kararın kesinleşmesi ve paranın tahsili şartıyla "
        "Gümrük Kanunu cezasının kaldırılmasını düzenler [1]."
    )
    assert canonical_named_authority_gap(blocked, ledger, **_OPTIONS)
    quoted = f'Gümrük Yönetmeliği m.579/2 şöyledir: "{_REGULATION}" [1].'
    assert canonical_named_authority_gap(quoted, ledger, **_OPTIONS) is None
    assert cited_lower_statute_gap(quoted, ledger, {1}) is None
    # The quote does not bless an additional statutory claim in the same unit.
    mixed = quoted + " 6455 sayılı Kanun m.11 uyarınca izin iptal edilir [1]."
    assert canonical_named_authority_gap(mixed, ledger, **_OPTIONS)


@pytest.mark.parametrize("article", ["11", "10/A"])
def test_quoted_lower_statute_still_requires_its_own_original(article: str) -> None:
    number, name = (
        ("6455", "İşlem Kanunu")
        if article == "11"
        else (
            "4458",
            "Gümrük Kanunu",
        )
    )
    lower = f"{number} sayılı {name} m.{article} uyarınca izin gerekir."
    ledger = EvidenceLedger()
    ledger.add(
        [original("Uygulama Tebliği", "3", kind="teblig", text=lower)], RunContext()
    )
    claim = f'Bu hüküm dosyaya uygulanır: "{lower}" [1].'
    gap = cited_lower_statute_gap(claim, ledger, {1})
    assert gap is not None and gap["unread_cited_statute_references"]


def test_reading_article10_never_supplies_article10a() -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(
                "4458 sayılı Gümrük Kanunu", "10", text="**MADDE 10.-** İzin gerekir."
            )
        ],
        RunContext(),
    )
    assert canonical_named_authority_gap(
        "4458 sayılı Gümrük Kanunu m.10/A uyarınca izin gerekir [1].",
        ledger,
        **_OPTIONS,
    )
