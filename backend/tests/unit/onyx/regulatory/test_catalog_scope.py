from uuid import uuid4

import pytest

from onyx.regulatory.catalog_scope import (
    CatalogSourceTitle,
    resolve_catalog_source_titles,
    select_catalog_heading_prefixes,
)


def test_explicit_inflected_title_matches_own_instrument_not_citing_file() -> None:
    law = CatalogSourceTitle(
        file_id=uuid4(), name="Kanunlar/gumruk_kanunu_.md", root_heading="GÜMRÜK KANUNU"
    )
    citing = CatalogSourceTitle(
        file_id=uuid4(),
        name="Kararlar/gumruk_kanununun_bazi_maddeleri_hakkinda_karar.md",
        root_heading="Uygulama Kararı",
    )
    assert resolve_catalog_source_titles(
        "gümrük kanununun cezalar ile ilgili hükümleri", [law, citing]
    ) == [law]
    assert resolve_catalog_source_titles("gümrük cezaları", [law, citing]) == []
    assert (
        resolve_catalog_source_titles(
            "Gümrük Kanunu ile Kabahatler Kanununu karşılaştır", [law, citing]
        )
        == []
    )


def test_hierarchy_selects_parent_and_descendants_but_not_unrelated_topic() -> None:
    paths = [
        ["GÜMRÜK KANUNU", "ONBİRİNCİ KISIM Cezalar", "BİRİNCİ BÖLÜM", "Madde 231"],
        ["GÜMRÜK KANUNU", "ONBİRİNCİ KISIM Cezalar", "İKİNCİ BÖLÜM", "Madde 234"],
        ["GÜMRÜK KANUNU", "İtirazlar", "Madde 242"],
    ]
    assert select_catalog_heading_prefixes(
        "Gümrük Kanununun cezalar ile ilgili hükümleri", "Gümrük Kanununun", paths
    ) == [("GÜMRÜK KANUNU", "ONBİRİNCİ KISIM Cezalar")]
    assert (
        select_catalog_heading_prefixes(
            "Gümrük Kanunu cezalar ve itirazlar", "Gümrük Kanunu", paths
        )
        == []
    )
    assert (
        select_catalog_heading_prefixes("Gümrük Kanunu teminat", "Gümrük Kanunu", paths)
        == []
    )


def test_source_number_and_accented_uppercase_are_not_fuzzy_topic_matches() -> None:
    law = CatalogSourceTitle(
        file_id=uuid4(),
        name="gumruk_kanunu.md",
        root_heading="4458 SAYILI GÜMRÜK KANUNU",
    )
    other = CatalogSourceTitle(
        file_id=uuid4(),
        name="kabahatler_kanunu.md",
        root_heading="5326 SAYILI KABAHATLER KANUNU",
    )
    assert resolve_catalog_source_titles(
        "4458 sayılı kanundaki cezalar", [law, other]
    ) == [law]
    assert resolve_catalog_source_titles("GÜMRÜK KANUNU cezalar", [law, other]) == [law]


def test_topic_matching_does_not_turn_article_clause_text_into_section_scope() -> None:
    paths = [
        ["Gümrük Kanunu", "Cezalar", "MADDE 231"],
        [
            "Gümrük Kanunu",
            "Diğer Hükümler",
            "MADDE 227",
            "e) Ceza veya disiplin soruşturması sonucunda çıkarılmış olmamak",
        ],
    ]
    assert select_catalog_heading_prefixes(
        "Gümrük Kanunu cezalar", "Gümrük Kanunu", paths
    ) == [("Gümrük Kanunu", "Cezalar")]


def test_numeric_source_does_not_accept_derivative_instrument() -> None:
    law = CatalogSourceTitle(
        file_id=uuid4(),
        name="gumruk_kanunu.md",
        root_heading="4458 SAYILI GÜMRÜK KANUNU",
    )
    derivative = CatalogSourceTitle(
        file_id=uuid4(),
        name="4458_sayili_gumruk_kanununun_bazi_maddelerinin_uygulanmasi_hakkinda_karar.md",
        root_heading="4458 Sayılı Gümrük Kanununun Bazı Maddelerinin Uygulanması Hakkında Karar",
    )
    assert resolve_catalog_source_titles(
        "4458 sayılı kanundaki cezalar", [law, derivative]
    ) == [law]


@pytest.mark.parametrize(
    "query", ["bana gümrük kanunundaki cezaları göster", "cezalar için gümrük kanunu"]
)
def test_source_title_is_resolved_inside_natural_request(query: str) -> None:
    law = CatalogSourceTitle(file_id=uuid4(), name="gumruk_kanunu.md")
    assert resolve_catalog_source_titles(query, [law]) == [law]


def test_long_lowercase_title_and_numeric_decision_identity() -> None:
    law = CatalogSourceTitle(
        file_id=uuid4(), name="ithalatta_haksiz_rekabetin_onlenmesi_hakkinda_kanun.md"
    )
    decision = CatalogSourceTitle(
        file_id=uuid4(),
        name="bkk_2009-15481_gumruk_kanununun_bazi_maddelerinin_uygulanmasi_hakkinda_karar.md",
    )
    assert resolve_catalog_source_titles(
        "ithalatta haksız rekabetin önlenmesi hakkında kanundaki hükümler", [law]
    ) == [law]
    assert resolve_catalog_source_titles(
        "2009/15481 sayılı karar hükümleri", [decision]
    ) == [decision]
