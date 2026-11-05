import pytest

from onyx.regulatory.source_document_identity import derive_document_identity

TITLE = "PAN-AVRUPA-AKDENİZ TERCİHLİ MENŞE KURALLARINA DAİR BÖLGESEL KONVANSİYON ORTAK KOMİTESİNİN 1/2023 SAYILI KARARI KAPSAMI TERCİHLİ MENŞE KURALLARI HAKKINDA YÖNETMELİK"


def test_own_regulation_heading_never_inherits_a_cited_decision_number() -> None:
    result = derive_document_identity(
        user_file_id="file",
        source_chunk_id="opening",
        text=f"**{TITLE}**\n*Ticaret Bakanlığından:20.12.2025 tarih ve33113 sayılı R.G.*",
        root_heading=TITLE,
    )
    assert result is not None
    assert result.fields == {
        "document_title": TITLE,
        "document_type": "yonetmelik",
        "document_number": None,
    }


def test_official_communique_number_must_be_adjacent_to_own_title() -> None:
    title = "ÖRNEK ÜRÜNLERİN DENETİMİNE DAİR TEBLİĞ"
    result = derive_document_identity(
        user_file_id="file",
        source_chunk_id="opening",
        text=f"5 Ocak 2025 PAZAR\nResmî Gazete\nSayı :32773\nTEBLİĞ\nSanayi ve Teknoloji Bakanlığından:\n{title}\n(2023/826/AB) (SGM-2024/5)\nAmaç",
        root_heading=title,
    )
    assert result is not None and result.fields["document_number"] == "SGM-2024/5"


@pytest.mark.parametrize(
    "prefix",
    ["MADDE 1- Bu Yönetmelikte atıf yapılan ", "EK I\n", "Başka bir belge başlığı\n"],
)
def test_a_cited_or_subordinate_title_is_not_own_identity(prefix: str) -> None:
    assert (
        derive_document_identity(
            user_file_id="file",
            source_chunk_id="opening",
            text=prefix + TITLE,
            root_heading=TITLE,
        )
        is None
    )


def test_new_clause_metadata_uses_bound_own_identity_instead_of_neighbor_citation() -> (
    None
):
    from uuid import UUID

    from onyx.regulatory.amendments import pipeline
    from onyx.regulatory.amendments.models import (
        AmendmentInstruction,
        ChunkFieldsDraft,
        DateResolution,
        DraftResult,
        MatchResult,
    )

    evidence = derive_document_identity(
        user_file_id="00000000-0000-0000-0000-000000000123",
        source_chunk_id="opening",
        text=f"**{TITLE}**\n*Ticaret Bakanlığından:20.12.2025 tarih ve33113 sayılı R.G.*",
        root_heading=TITLE,
    )
    assert evidence is not None
    match = MatchResult(
        old_chunk_id=None, confidence=1, rationale="verified own source"
    )
    context = pipeline.InstructionDraftContext(
        match=match,
        old_chunk_snapshot={},
        target_user_file_id=UUID(evidence.user_file_id),
        target_position=3,
        sibling_reference={
            "metadata": {"article_no": "2"},
            "heading_path": [TITLE, "MADDE 2", "(1) Kapsam"],
        },
        base_metadata={},
        base_heading_path=[],
        expected_new_article_no="2",
        own_document_identity=evidence,
    )
    instruction = AmendmentInstruction(
        instruction_text="Aynı Yönetmeliğin 2 nci maddesinin birinci fıkrasına aşağıdaki bent eklenmiştir. “m) Yeni hüküm,”"
    )
    draft = DraftResult(
        new_chunk=ChunkFieldsDraft(
            text="m) Yeni hüküm,",
            chunk_type="clause",
            metadata_changes={
                "document_type": "karar",
                "document_number": "2017/10411",
            },
        ),
        dates=DateResolution(rationale="publication"),
    )
    proposal = pipeline._build_proposal_draft(
        instruction_indices=[0],
        instructions=[instruction],
        matches=[match],
        context=context,
        draft=draft,
    )
    metadata = proposal.new_chunk_draft["metadata"]
    assert (
        metadata["document_type"] == "yonetmelik"
        and metadata.get("document_number") is None
    )
    assert metadata["amendment_document_identity"] == evidence.model_dump(mode="json")


@pytest.mark.parametrize(
    "continuation",
    [
        "TE DEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK",
        "\nDEĞİŞİKLİK YAPILMASINA DAİR YÖNETMELİK",
    ],
)
def test_official_amending_title_cannot_be_certified_as_its_base_title(
    continuation: str,
) -> None:
    title = "ÖRNEK USULLER HAKKINDA YÖNETMELİK"
    body = f"5 Ocak 2025 PAZAR\nResmî Gazete\nSayı :32773\nYÖNETMELİK\nTicaret Bakanlığından:\n{title}{continuation}\nAmaç"
    assert (
        derive_document_identity(
            user_file_id="file",
            source_chunk_id="opening",
            text=body,
            root_heading=title,
        )
        is None
    )


def test_own_number_survives_a_separate_issuer_line() -> None:
    title = "ÖRNEK ÜRÜNLERİN DENETİMİNE DAİR TEBLİĞ"
    evidence = derive_document_identity(
        user_file_id="file",
        source_chunk_id="opening",
        root_heading=title,
        text=f"{title}\nSanayi ve Teknoloji Bakanlığından:\n(2023/826/AB) (SGM-2024/5)\nMADDE 1- Amaç.",
    )
    assert evidence is not None and evidence.fields["document_number"] == "SGM-2024/5"


def test_unsupported_numbered_document_format_does_not_prove_number_absence() -> None:
    title = "ÖRNEK ÜRÜNLERİN DENETİMİNE DAİR TEBLİĞ"
    evidence = derive_document_identity(
        user_file_id="file",
        source_chunk_id="opening",
        root_heading=title,
        text=f"{title}\nSanayi ve Teknoloji Bakanlığından:\nTebliğ No: 2024/5\nMADDE 1- Amaç.",
    )
    assert evidence is not None and "document_number" not in evidence.fields
