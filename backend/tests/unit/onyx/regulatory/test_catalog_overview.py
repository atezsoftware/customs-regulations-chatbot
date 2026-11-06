import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from onyx.regulatory.catalog_overview import (
    catalog_overview_payload,
    validate_catalog_overview,
)
from onyx.regulatory.catalog_summary import CatalogSummaryUnavailable
from onyx.server.features.regulatory.catalog_models import (
    CatalogFile,
    CatalogSearchHit,
    CatalogSearchPage,
)
from onyx.server.features.regulatory.models import RegulatoryChunkSnapshot


def _page(count: int = 50) -> CatalogSearchPage:
    now = datetime.now(timezone.utc)
    f = CatalogFile(
        id=uuid4(),
        name="Gümrük Kanunu",
        folder_path="",
        user_id=None,
        file_id="f",
        status="COMPLETED",
        file_type="text",
        created_at=now,
        chunk_count=count,
        document_sets=[],
        pending_review=False,
    )
    return CatalogSearchPage(
        items=[
            CatalogSearchHit(
                file=f,
                score=1.0,
                chunk=RegulatoryChunkSnapshot(
                    id=str(i),
                    user_file_id=str(f.id),
                    text=("Bir hüküm ve istisnası birlikte değerlendirilir.\n\n" * 200),
                    position=i,
                    chunk_type="article",
                    heading_path=["Gümrük Kanunu", "Cezalar", f"Madde {i + 1}"],
                    chunk_metadata={},
                    validity_start_date=None,
                    validity_end_date=None,
                    status="active",
                    source="indexed",
                    supersedes_chunk_id=None,
                    superseded_by_chunk_id=None,
                    created_at=now,
                    updated_at=now,
                ),
            )
            for i in range(count)
        ],
        total=count,
        offset=0,
        limit=count,
        has_more=False,
    )


def test_overview_packet_is_bounded_and_citations_retain_global_reference() -> None:
    page = _page()
    packet = catalog_overview_payload("cezalar", page)
    assert len(json.dumps(packet, ensure_ascii=False)) < 18000
    assert packet["total_chunks"] == 50
    assert 1 <= len(packet["evidence"]) <= 20
    refs = [e["reference"] for e in packet["evidence"]]
    assert len(set(refs)) == len(refs)
    assert all(
        page.items[e["reference"] - 1].chunk.id == e["chunk_id"]
        for e in packet["evidence"]
    )
    assert validate_catalog_overview(
        "Cezalar başlığı altında ilgili hükümler bulundu. [1]", packet
    )


@pytest.mark.parametrize(
    "text",
    [
        "Uydurulan sonuç [99]",
        "Madde 999 bulunur. [1]",
        "Atıfsız özet.",
        "Cezalar 1–50 maddelerindedir. [1]",
    ],
)
def test_rejects_unknown_citations_numbers_and_unverified_ranges(text: str) -> None:
    with pytest.raises(CatalogSummaryUnavailable):
        validate_catalog_overview(text, catalog_overview_payload("cezalar", _page()))


def test_each_sentence_uses_its_own_cited_source_for_numbers() -> None:
    page = _page(2)
    page.items[0].chunk.text = "Ödeme süresi 10 gündür."
    page.items[1].chunk.text = "Ödeme süresi 30 gündür."
    packet = catalog_overview_payload("ödeme", page)
    with pytest.raises(CatalogSummaryUnavailable):
        validate_catalog_overview(
            "Ödeme süresi 30 gündür [1]. Diğer dosyada ödeme süresi 10 gündür [2].",
            packet,
        )
    assert validate_catalog_overview(
        "Ödeme süresi 10 gündür [1]. Diğer dosyada ödeme süresi 30 gündür [2].", packet
    )


def test_named_section_overview_uses_outline_without_detached_clause_prose() -> None:
    from onyx.regulatory.catalog_scope import CatalogSearchScope

    page = _page(2)
    page.items[0].chunk.heading_path = [
        "Gümrük Kanunu",
        "Cezalar",
        "Madde 231",
        "1. Oran yüzde 999",
    ]
    page.items[1].chunk.heading_path = [
        "Gümrük Kanunu",
        "Cezalar",
        "Madde 232",
        "1. Koşul devamı",
    ]
    page.scope = CatalogSearchScope(
        file_ids=[page.items[0].file.id],
        heading_prefixes=[["Gümrük Kanunu", "Cezalar"]],
        heading_paths=[item.chunk.heading_path for item in page.items],
    )
    packet = catalog_overview_payload("cezalar", page)
    assert all(
        e["text"] == "" and len(e["heading_path"]) == 3 for e in packet["evidence"]
    )
    assert "999" not in json.dumps(packet)
    assert packet["structure"][0]["verified_ranges"] == [[231, 232]]


def test_returned_file_uses_its_own_structure_when_duplicate_titles_resolve() -> None:
    from onyx.regulatory.catalog_scope import CatalogSearchScope

    page = _page(2)
    first = page.items[0].file.id
    other = uuid4()
    page.scope = CatalogSearchScope(
        file_ids=[first, other],
        heading_prefixes=[["Gümrük Kanunu", "Cezalar"]],
        heading_paths=[
            ["Gümrük Kanunu", "Cezalar", "Madde 1"],
            ["Gümrük Kanunu", "Cezalar", "Madde 99"],
        ],
        heading_paths_by_file={
            str(first): [
                ["Gümrük Kanunu", "Cezalar", "Madde 1"],
                ["Gümrük Kanunu", "Cezalar", "Madde 2"],
            ],
            str(other): [["Gümrük Kanunu", "Cezalar", "Madde 99"]],
        },
    )
    packet = catalog_overview_payload("cezalar", page)
    assert all(e["text"] == "" for e in packet["evidence"])
    assert len(packet["structure"]) == 1
    assert packet["structure"][0]["verified_ranges"] == [[1, 2]]
    assert "99" not in json.dumps(packet)
