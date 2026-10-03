import json

import pytest

from onyx.asv3.authority import (
    authority_obligations,
    statute_references,
    unresolved_authority_gap,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.working_memory import WorkingMemory
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc


def source(kind: str, instrument: str, article: str, text: str) -> EvidenceItem:
    metadata = {
        "document_type": kind,
        "title": instrument,
        "article_no": article,
        "heading_path": [instrument, f"Madde {article}"],
    }
    return EvidenceItem(
        source_id=f"{kind}-{instrument}",
        chunk_id=f"{kind}-{instrument}-{article}",
        text=text,
        metadata={"canonical_metadata": metadata},
        search_doc=SearchDoc(
            document_id=f"{kind}-{instrument}",
            chunk_ind=0,
            semantic_identifier=instrument,
            link="https://example.test/provision",
            blurb=text,
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata=metadata,
            match_highlights=[],
        ),
    )


@pytest.mark.parametrize("number,article", [("8917", "27"), ("7251", "45")])
def test_secondary_reference_cannot_substitute_for_named_original(
    number: str, article: str
) -> None:
    ledger, context = EvidenceLedger(), RunContext()
    answer = f"{number} sayılı Faaliyet Kanunu’nun {article} inci maddesi uyarınca izin gerekir [1]."
    ledger.add([source("tebliğ", "Uygulama Tebliği", "3", answer)], context)
    missing = unresolved_authority_gap(answer, ledger)
    assert missing is not None
    needs = missing["missing"]
    assert isinstance(needs, list) and isinstance(needs[0], dict)
    assert needs[0]["instrument_number"] == number
    # A related statute, or the right statute's wrong article, does not close the need.
    ledger.add(
        [
            source("kanun", f"{number} sayılı Faaliyet Kanunu", "99", "Başka hüküm."),
            source("kanun", "9923 sayılı Başka Kanun", article, "Farklı koşul."),
        ],
        context,
    )
    assert unresolved_authority_gap(answer, ledger) is not None
    correct = ledger.add(
        [source("kanun", f"{number} sayılı Faaliyet Kanunu", article, "İzin gerekir.")],
        context,
    )[0]
    obligation = authority_obligations(answer, ledger)[0]
    assert obligation["matching_original_evidence"] == [correct]
    assert obligation["status"] == "unresolved_original"
    assert (
        unresolved_authority_gap(answer.replace("[1]", f"[{correct}]"), ledger) is None
    )


def test_explicit_subunit_reference_is_a_locator_not_invented_rule() -> None:
    refs = statute_references(
        "7251 sayılı Kanunun (45/2-ç) maddesi uyarınca özel usul uygulanır."
    )
    assert [(ref.number, ref.article) for ref in refs] == [("7251", "45")]
    assert [(ref.paragraph, ref.clause) for ref in refs] == [("2", "ç")]
    assert statute_references("Bedel 7251 TL, süre 45 gün.") == ()
    assert statute_references("2024/17 sayılı Genelge.") == ()


@pytest.mark.parametrize(
    "article",
    ["27 nci maddesi", "27. maddesi", "Madde 27", "m. 27", "(27/2-ç) maddesi"],
)
def test_later_implementing_reference_never_relabels_a_statute(article: str) -> None:
    answer = (
        f"8917 sayılı Faaliyet Kanunu’nun {article} [1], "
        "Uygulama Yönetmeliği’nin 55 inci maddesi [2] ve "
        "Faaliyet Tebliği m. 3/4 [3] uyarınca işlem yapılır."
    )
    references = statute_references(answer)
    assert [(r.number, r.article) for r in references] == [("8917", "27")]
    expected_subunit = ("2", "ç") if article.startswith("(") else (None, None)
    assert [(r.paragraph, r.clause) for r in references] == [expected_subunit]
    ledger = EvidenceLedger()
    original = source("kanun", "8917 sayılı Faaliyet Kanunu", "27", "İşlem yapılır.")
    if article.startswith("("):
        original.metadata.update(paragraph_no="2", clause_label="ç")
    ledger.add([original], RunContext())
    assert unresolved_authority_gap(answer, ledger) is None


def test_unqualified_statute_does_not_acquire_another_instruments_article() -> None:
    references = statute_references(
        "8917 sayılı Faaliyet Kanunu [1] ve Uygulama Yönetmeliği m. 3/4 [2] uygulanır."
    )
    assert [(r.number, r.article, r.paragraph) for r in references] == [
        ("8917", None, None)
    ]


def test_adjacent_statutes_keep_their_own_article_and_clause() -> None:
    references = statute_references(
        "8917 sayılı Faaliyet Kanunu’nun 27 nci maddesi [1] ile "
        "7251 sayılı Veri Kanunu m. 45/2-ç [2] uygulanır."
    )
    assert [(r.number, r.article, r.paragraph, r.clause) for r in references] == [
        ("8917", "27", None, None),
        ("7251", "45", "2", "ç"),
    ]


def test_later_sentence_cannot_supply_a_missing_statutory_article() -> None:
    references = statute_references(
        "8917 sayılı Faaliyet Kanunu uygulanır. 27 nci maddesi ayrıca incelenir."
    )
    assert [(r.number, r.article) for r in references] == [("8917", None)]


def test_english_reference_does_not_use_the_later_regulations_subunit() -> None:
    references = statute_references(
        "Law No. 8917 Article 27 [1] and Implementation Regulation art. 3/4 [2] apply."
    )
    assert [(r.number, r.article, r.paragraph) for r in references] == [
        ("8917", "27", None)
    ]


def test_wrong_paragraph_or_clause_cannot_close_explicit_original_need() -> None:
    ledger, context = EvidenceLedger(), RunContext()
    answer = "8917 sayılı Faaliyet Kanunu m. 27/1-ç gereğince izin gerekir [1]."
    wrong = source("kanun", "8917 sayılı Faaliyet Kanunu", "27", "Başka hüküm.")
    wrong.metadata["paragraph_no"] = "2"
    wrong.metadata["clause_label"] = "ç"
    ledger.add([wrong], context)
    assert unresolved_authority_gap(answer, ledger) is not None
    another = source("kanun", "8917 sayılı Faaliyet Kanunu", "27", "Farklı bent.")
    another.metadata["paragraph_no"] = "1"
    another.metadata["clause_label"] = "c"
    ledger.add([another], context)
    assert unresolved_authority_gap(answer.replace("[1]", "[2]"), ledger) is not None
    correct = source("kanun", "8917 sayılı Faaliyet Kanunu", "27", "İzin gerekir.")
    correct.metadata["clause_label"] = "ç"
    correct.metadata["heading_path"] = [
        "8917 sayılı Faaliyet Kanunu",
        "Madde 27",
        "1. Başvuru",
        "ç) İzin",
    ]
    ledger.add([correct], context)
    assert unresolved_authority_gap(answer.replace("[1]", "[3]"), ledger) is None


def test_temporary_article_is_not_the_ordinary_article() -> None:
    ledger, context = EvidenceLedger(), RunContext()
    ledger.add(
        [source("kanun", "8917 sayılı Faaliyet Kanunu", "27", "Normal hüküm")], context
    )
    answer = "8917 sayılı Faaliyet Kanununun geçici 27 nci maddesi uygulanır [1]."
    assert unresolved_authority_gap(answer, ledger) is not None


def test_read_reference_is_durable_without_becoming_target_source_or_legal_evidence() -> (
    None
):
    memory = WorkingMemory({})
    text = "7251 sayılı Kanunun (45/2-ç) maddesi uyarınca özel usul uygulanır."
    receipt = ToolReceipt(
        call=CapabilityCall(name="read_evidence", arguments={"citation": 8}),
        elapsed_seconds=0.01,
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original evidence",
            data={"text": text, "source_id": "secondary-source", "chunk_id": "origin"},
        ),
    )
    memory.observe(receipt)
    exported = memory.export()
    encoded = json.dumps(exported, ensure_ascii=False)
    assert "7251" in encoded and "reference_lead_not_original" in encoded
    restored = WorkingMemory({})
    restored.restore(exported)
    locators = restored.view()["locators"]
    assert isinstance(locators, list)
    lead = next(
        row for row in locators if isinstance(row, dict) and row["kind"] == "reference"
    )
    assert lead["origin_source_id"] == "secondary-source"
    assert "source_id" not in lead
    assert lead["article"] == "45"
    reference_text = lead["reference_text"]
    assert isinstance(reference_text, str) and len(reference_text) <= 240


def test_authority_identity_does_not_copy_original_body() -> None:
    ledger = EvidenceLedger()
    ledger.add(
        [
            source(
                "kanun", "8917 sayılı Faaliyet Kanunu", "27", "LARGE ORIGINAL " * 4000
            )
        ],
        RunContext(),
    )
    metadata = ledger.authority_metadata()
    assert "LARGE ORIGINAL" not in json.dumps(metadata)
    assert metadata[0]["heading_path"] == ["8917 sayılı Faaliyet Kanunu"]
