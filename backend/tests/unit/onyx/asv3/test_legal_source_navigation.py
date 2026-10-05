"""Title relationships remain scoped navigation, never additional legal evidence."""

from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import cast
from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs, evidence_for_chunk
from onyx.asv3.legal_source_navigation import (
    ProvisionNavigationAnchor,
    derive_provision_navigation_anchor,
    match_related_source_name,
)
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.models import User

AYM_SOURCE_NAME = (
    "Kararname ve Mahkeme Kararları/"
    "anayasa_mahkemesinin_2026-72_gumruk_kanunu_usulsuzluk_cezasi_241_"
    "maddesinin_1_fikrasinda_yetki_ifadelerinin_iptal_edildigi.md"
)


def original(
    title: str = "Gümrük Kanunu", article: str = "241", kind: str | None = "kanun"
) -> EvidenceItem:
    identifier = uuid4()
    source = CorpusSource(identifier, f"Sources/{title}.md", str(uuid4()))
    chunk = CorpusChunk(
        "canonical-original",
        identifier,
        "Read operative original.",
        1,
        1,
        (title, article if "MADDE" in article else f"MADDE {article}"),
        {"document_type": kind, "title": title},
        None,
        None,
        "active",
    )
    return evidence_for_chunk(source, chunk)


def anchor(
    item: EvidenceItem, article: str = "241", qualifier: str | None = None
) -> ProvisionNavigationAnchor:
    result = derive_provision_navigation_anchor(
        item.source_id, [item], article, qualifier
    )
    assert result is not None
    return result


def broker() -> CorpusBroker:
    return CorpusBroker(cast(User, object()), IndexFilters(access_control_list=[]))


def source(name: str) -> CorpusSource:
    identifier = uuid4()
    return CorpusSource(identifier, name, str(uuid4()))


def test_actual_catalog_title_has_explicit_relationship_without_statute_number() -> (
    None
):
    item = original()
    actual = anchor(item)
    assert actual.instrument_name == "Gümrük Kanunu"
    assert actual.instrument_number is None
    assert match_related_source_name(actual, AYM_SOURCE_NAME) == "judicial_candidate"
    assert (
        match_related_source_name(
            actual, "Tasarruflu Yazılar/01.07.2026_gk_241-1_hakkinda.md"
        )
        is None
    )


@pytest.mark.parametrize(
    ("law", "article", "name", "role"),
    [
        (
            "Katma Değer Vergisi Kanunu",
            "17",
            "danistayin_katma_deger_vergisi_kanununun_17_maddesindeki_istisna_karari.md",
            "judicial_candidate",
        ),
        (
            "Özel Tüketim Vergisi Kanunu",
            "8",
            "ozel_tuketim_vergisi_kanunu_8_maddesine_iliskin_cumhurbaskani_karari.md",
            "executive_candidate",
        ),
        (
            "Denizcilik Vergisi Kanunu",
            "6",
            "denizcilik_vergisi_kanunu_madde_6_degisiklik_kanunu.md",
            "amendment_candidate",
        ),
        (
            "Customs Act",
            "19",
            "court_customs_act_article_19_referral.pdf",
            "referral_candidate",
        ),
    ],
)
def test_other_instruments_and_source_roles(
    law: str, article: str, name: str, role: str
) -> None:
    assert (
        match_related_source_name(anchor(original(law, article), article), name) == role
    )


@pytest.mark.parametrize(
    "name",
    [
        "mahkeme_diger_kanun_241_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_1241_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_2412_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_2024-241_hakkinda.md",
        "gumruk_kanunu_mahkeme_241_maddesi/mahkeme_diger_konu.md",
        "gumruk_kanunu_241_maddesinin_uygulamasi.md",
        "mahkeme_gumruk_kanunu_mukerrer_241_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_gecici_241_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_241a_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_241.1_maddesinin_iptali.md",
        "mahkeme_gumruk_kanunu_234_maddesi_medeni_kanunun_241_maddesinin_iptali.md",
    ],
)
def test_unrelated_numbers_qualifiers_paths_and_ordinary_guidance_are_not_candidates(
    name: str,
) -> None:
    assert match_related_source_name(anchor(original()), name) is None


@pytest.mark.parametrize("qualifier", ["gecici", "mukerrer", "ek"])
def test_qualified_articles_are_distinct(qualifier: str) -> None:
    item = original("Denizcilik Vergisi Kanunu", f"{qualifier} MADDE 7")
    actual = anchor(item, "7", qualifier)
    assert (
        match_related_source_name(
            actual,
            f"mahkeme_denizcilik_vergisi_kanunu_{qualifier}_7_maddesinin_iptali.md",
        )
        == "judicial_candidate"
    )
    assert (
        match_related_source_name(
            actual, "mahkeme_denizcilik_vergisi_kanunu_7_maddesinin_iptali.md"
        )
        is None
    )


def test_slash_article_is_not_conflated_with_letter_or_paragraph_identity() -> None:
    actual = anchor(original("Denizcilik Vergisi Kanunu", "12/A"), "12/A")
    for name in (
        "mahkeme_denizcilik_vergisi_kanunu_12a_maddesinin_iptali.md",
        "mahkeme_denizcilik_vergisi_kanunu_12_maddesi_a_bendi.md",
    ):
        assert match_related_source_name(actual, name) is None


@pytest.mark.parametrize(
    "defect",
    [
        "lower_source",
        "derived",
        "external",
        "wrong_chunk",
        "no_own_heading",
        "wrong_article",
        "conflicting_title",
        "unbound_chunk",
    ],
)
def test_anchor_needs_a_genuine_own_governing_original(defect: str) -> None:
    item = original()
    if defect == "lower_source":
        item.metadata["canonical_metadata"] = {
            "document_type": "tebliğ",
            "title": "Gümrük Kanunu",
        }
    elif defect in {"derived", "external"}:
        item.metadata[defect] = True
    elif defect == "wrong_chunk":
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_chunk_id"] = "another-chunk"
    elif defect == "unbound_chunk":
        item.chunk_id = None
        assert item.search_doc is not None
        item.search_doc.metadata.pop("regulatory_chunk_id")
    elif defect == "no_own_heading":
        item.metadata["heading_path"] = [
            "A court decision discussing Gümrük Kanunu 241"
        ]
    elif defect == "wrong_article":
        item.metadata["heading_path"] = ["Gümrük Kanunu", "MADDE 234"]
    else:
        item.metadata["canonical_metadata"] = {
            "document_type": "kanun",
            "title": "Katma Değer Vergisi Kanunu",
        }
    assert (
        derive_provision_navigation_anchor(item.source_id, [item], "241", None) is None
    )


def test_lookup_is_shared_paged_and_does_not_hydrate_candidate_originals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = broker()
    item = original()
    candidate = source(AYM_SOURCE_NAME)
    lookup = Mock(
        return_value=([candidate, source("mahkeme_diger_kanun_241_maddesi.md")], True)
    )
    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    context = RunContext()
    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(
            executor.map(
                lambda _: current.related_sources_for_evidence(item, context), range(8)
            )
        )
    assert lookup.call_count == 1
    record = records[0]
    assert record is not None
    assert record["navigation_only"] is True and record["absence_proven"] is False
    assert record["has_more"] is True and record["next_offset"] == 50
    assert record["candidates"] == [
        {
            "source_id": str(candidate.id),
            "name": AYM_SOURCE_NAME,
            "candidate_role": "judicial_candidate",
        }
    ]
    assert "text" not in str(record["candidates"]) and "citation" not in str(
        record["candidates"]
    )
    assert current.related_source_navigation() == [record]
    record["candidates"] = []
    assert current.related_source_navigation()[0]["candidates"]


@pytest.mark.parametrize(
    "error",
    [
        PermissionError("denied"),
        CorpusScopeUnavailable("unavailable"),
        SQLAlchemyError("database unavailable"),
    ],
)
def test_catalog_access_failure_does_not_erase_the_read_original(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = broker()
    monkeypatch.setattr(current, "related_catalog_sources", Mock(side_effect=error))
    item = original()
    result = current.related_sources_for_evidence(item, RunContext())
    assert result is not None and result["candidates"] == []
    assert result["status"] in {"denied", "unavailable"}
    assert result["absence_proven"] is False
    assert item.text == "Read operative original."


@pytest.mark.parametrize(
    ("field", "value"),
    [("tenant_id", "other-tenant"), ("as_of_date", date(2025, 1, 1))],
)
def test_scope_or_date_change_never_reuses_old_navigation(
    field: str,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = broker()
    lookup = Mock(return_value=([source(AYM_SOURCE_NAME)], False))
    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    item = original()
    current.related_sources_for_evidence(item, RunContext())
    assert current.related_source_navigation()
    current.filters = current.filters.model_copy(update={field: value})
    assert current.related_source_navigation() == []
    lookup.return_value = ([], False)
    result = current.related_sources_for_evidence(item, RunContext())
    assert result is not None and result["candidates"] == []
    assert result["absence_proven"] is False
    assert lookup.call_count == 2


def test_nonlegal_or_source_identity_unknown_reads_do_not_query_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = broker()
    lookup = Mock(side_effect=AssertionError("No catalogue lookup expected"))
    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    assert (
        current.related_sources_for_evidence(
            original("Welcome", "1", None), RunContext()
        )
        is None
    )
    lookup.assert_not_called()


def test_numeric_official_heading_preserves_name_and_number_without_filename_guessing() -> (
    None
):
    actual = anchor(original("4458 SAYILI GÜMRÜK KANUNU"))
    assert actual.instrument_name == "GÜMRÜK KANUNU"
    assert actual.instrument_number == "4458"
    assert match_related_source_name(actual, AYM_SOURCE_NAME) == "judicial_candidate"
    assert (
        match_related_source_name(
            actual, "mahkeme_4459_sayili_gumruk_kanunu_241_maddesinin_iptali.md"
        )
        is None
    )
    assert (
        match_related_source_name(
            actual, "mahkeme_4458_sayili_kanunun_241_maddesinin_iptali.md"
        )
        == "judicial_candidate"
    )


def test_read_provision_retains_evidence_when_optional_catalog_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = broker()
    loaded = source("Sources/gumruk_kanunu.md")
    chunk = CorpusChunk(
        "operative",
        loaded.id,
        "Operative original.",
        0,
        0,
        ("Gümrük Kanunu", "MADDE 241"),
        {"document_type": "kanun"},
        None,
        None,
        "active",
    )
    monkeypatch.setattr(current, "source", Mock(return_value=loaded))
    monkeypatch.setattr(current, "provision_start", Mock(return_value=0))
    monkeypatch.setattr(current, "page", Mock(return_value=(loaded, [chunk], False)))
    monkeypatch.setattr(
        current,
        "related_catalog_sources",
        Mock(side_effect=SQLAlchemyError("unavailable")),
    )
    result = CapabilityRegistry(build_corpus_specs(current)).dispatch(
        CapabilityCall(
            name="read_provision",
            arguments={"source_id": str(loaded.id), "article": "241"},
        ),
        RunContext(),
    )
    assert result.status == OutcomeStatus.FOUND
    assert [item.text for item in result.evidence] == ["Operative original."]
    navigation = result.data["related_source_candidates"]
    assert isinstance(navigation, dict) and navigation["status"] == "unavailable"
    assert navigation["absence_proven"] is False


def test_catalog_lookup_uses_existing_scoped_access_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import corpus_tools

    current = broker()
    current.filters = current.filters.model_copy(
        update={"tenant_id": "tenant-A", "asv3_document_set_id": 73}
    )
    session = Mock()

    class SessionContext:
        def __enter__(self) -> Mock:
            return session

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(corpus_tools, "get_session_with_current_tenant", SessionContext)
    lookup = Mock(return_value=([source(AYM_SOURCE_NAME)], False))
    monkeypatch.setattr(corpus_tools, "find_related_sources", lookup)
    result = current.related_sources_for_evidence(original(), RunContext())
    assert result is not None and result["candidates"]
    assert lookup.call_args.args == (session,)
    assert lookup.call_args.kwargs["user"] is current.user
    assert lookup.call_args.kwargs["filters"] is current.filters
    assert lookup.call_args.kwargs["query_variants"] == ("Gümrük Kanunu 241",)


def test_cancellation_is_not_downgraded_to_an_optional_navigation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = broker()
    monkeypatch.setattr(
        current, "related_catalog_sources", Mock(side_effect=RunStopped("cancelled"))
    )
    with pytest.raises(RunStopped):
        current.related_sources_for_evidence(original(), RunContext())


def test_name_and_number_only_titles_share_one_catalog_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = broker()
    numbered = source("mahkeme_4458_sayili_kanunun_241_maddesinin_iptali.md")
    named = source(AYM_SOURCE_NAME)
    lookup = Mock(return_value=([named, numbered], False))
    monkeypatch.setattr(current, "related_catalog_sources", lookup)
    result = current.related_sources_for_evidence(
        original("4458 SAYILI GÜMRÜK KANUNU"), RunContext()
    )
    assert result is not None
    assert result["query_variants"] == ["GÜMRÜK KANUNU 241", "4458 sayılı Kanun 241"]
    assert lookup.call_count == 1
    variants = result["query_variants"]
    assert isinstance(variants, list)
    assert lookup.call_args.args[0] == tuple(variants)
    candidates = result["candidates"]
    assert isinstance(candidates, list)
    assert {
        candidate["source_id"]
        for candidate in candidates
        if isinstance(candidate, dict)
    } == {str(named.id), str(numbered.id)}
