"""Canonical identity and retained dependencies for the isolated Cheap workflow."""

import copy
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.cheap_authority import CheapAuthorityPolicy
from onyx.asv3.models import EvidenceItem, RunContext
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_shared_originals import full_record

REQUEST = "İçtima ve gümrük kontrolü hükümlerini incele."
OPTIONS = {"strict_reference_boundaries": True, "syntactic_reference_binding": True}
FIXTURE = Path(__file__).with_name("fixtures") / "cheap_authority_5077.json"
PUNCTUATED_CAPTURES: dict[int, tuple[str, str, str, str, str]] = {
    477: (
        "8",
        "rc_808cb96d6b57934c8ef8fcaa7299299111e40cb0",
        "afda8b6b32c26dae833157a5155e61d335e6915b22d0c8605d2df45f17c8ce27",
        "Organ veya temsilcinin davranışından dolayı sorumluluk",
        "**MADDE 8.-** (1) Organ veya temsilcilik görevi yapan ya da organ veya temsilci "
        "olmamakla birlikte, tüzel kişinin faaliyeti çerçevesinde görev üstlenen kişinin "
        "bu görevi kapsamında işlemiş bulunduğu kabahatten dolayı tüzel kişi hakkında "
        "da idarî yaptırım uygulanabilir.",
    ),
    480: (
        "9",
        "rc_417c9cba7e61aba5464fc26061fc34e15e5bc772",
        "506f22e3930a433c7a9d526f3356ff5ca8e1b2280e9d58a7274a1a7f506d8401",
        "Kast veya taksir",
        "**MADDE 9.-** (1) Kabahatler, kanunda açıkça hüküm bulunmayan hallerde, "
        "hem kasten hem de taksirle işlenebilir.",
    ),
}


class CapturedOriginal(TypedDict):
    source_id: str
    chunk_id: str
    text_hash: str
    text: str
    metadata: dict[str, JsonValue]


def captured(citation: int) -> EvidenceItem:
    data = json.loads(FIXTURE.read_text())
    row: CapturedOriginal
    if citation in PUNCTUATED_CAPTURES:
        article, chunk_id, text_hash, heading, text = PUNCTUATED_CAPTURES[citation]
        row = {
            "source_id": "75a82b64-d74c-4c69-852e-b189cef210cb",
            "chunk_id": chunk_id,
            "text_hash": text_hash,
            "text": text,
            "metadata": {
                "title": "Diğer Kanunlar/5326_sayili_kanun.md",
                "document_date": None,
                "document_type": "kanun",
                "read_as_of_date": "2026-10-12",
                "version_unknown": True,
                "provision_identifiers": [f"madde {article}"],
                "legal_dates": ["2005-03-30", "2005-03-31"],
                "article_no": None,
                "clause_label": None,
                "paragraph_no": None,
                "heading_path": [
                    "KABAHATLER KANUNU",
                    "Genel Hükümler",
                    "İKİNCİ BÖLÜM",
                    heading,
                ],
            },
        }
    else:
        row = cast(
            CapturedOriginal,
            next(row for row in data["rows"] if row["citation"] == citation),
        )
    assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_hash"]
    title = row["metadata"]["title"]
    assert isinstance(title, str)
    return EvidenceItem(
        source_id=row["source_id"],
        chunk_id=row["chunk_id"],
        text=row["text"],
        metadata={"canonical_metadata": row["metadata"]},
        search_doc=SearchDoc(
            document_id=row["source_id"],
            chunk_ind=0,
            semantic_identifier=title,
            blurb=row["text"],
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": row["chunk_id"]},
            match_highlights=[],
        ),
    )


def context_and_policy() -> tuple[RunContext, CheapAuthorityPolicy]:
    context = RunContext(
        scope={"authorized": "owned"}, services={"asv3_exp_cheap": True}
    )
    return context, CheapAuthorityPolicy(context, REQUEST)


@pytest.mark.parametrize(
    ("citation", "claim", "article", "previous"),
    [
        (291, "5326 sayılı Kabahatler Kanunu m.15 incelenmiştir [1].", "15", None),
        (574, "4458 sayılı Gümrük Kanunu m.10/A incelenmiştir [1].", "10/A", "10"),
        (477, "5326 sayılı Kabahatler Kanunu m.8/1 incelenmiştir [1].", "8", None),
        (480, "5326 sayılı Kabahatler Kanunu m.9/1 incelenmiştir [1].", "9", None),
    ],
)
def test_captured_own_heading_resolves_identity_without_mutating_original(
    citation: int, claim: str, article: str, previous: str | None
) -> None:
    context, policy = context_and_policy()
    ledger = ledger_with(captured(citation))
    saved = copy.deepcopy(ledger.export())
    assert native_named_authority_gap(claim, ledger, **OPTIONS)
    assert policy.publication_gap(claim, "writer", context, ledger) is None
    witnesses = policy.identity_witnesses(ledger)
    assert len(witnesses) == 1
    witness = witnesses[0]
    item = ledger.get(1)
    assert item is not None
    assert witness["article_no"] == article
    assert witness["previous_article_no"] == previous
    assert (witness["source_id"], witness["chunk_id"], witness["text_hash"]) == (
        item.source_id,
        item.chunk_id,
        item.text_hash,
    )
    assert witness["heading_start"] == 0
    heading = {
        291: "**MADDE 15**",
        574: "**MADDE 10/A-**",
        477: "**MADDE 8.-**",
        480: "**MADDE 9.-**",
    }[citation]
    assert witness["heading_end"] == len(heading)
    assert witness["heading_hash"] == hashlib.sha256(heading.encode()).hexdigest()
    assert ledger.export() == saved
    assert native_named_authority_gap(claim, ledger, **OPTIONS)


@pytest.mark.parametrize(
    "heading",
    [
        "**MADDE 8.**",
        "**MADDE 8.-**",
        "**MADDE 8. –**",
        "**MADDE 8—**",
        "**MADDE 8:**",
    ],
)
def test_own_heading_punctuation_keeps_exact_heading_witness(heading: str) -> None:
    context, policy = context_and_policy()
    item = captured(477)
    item.text = item.text.replace("**MADDE 8.-**", heading, 1)
    item.text_hash = hashlib.sha256(item.text.encode()).hexdigest()
    ledger = ledger_with(item)
    assert (
        policy.publication_gap(
            "5326 sayılı Kabahatler Kanunu m.8/1 uygulanır [1].",
            "writer",
            context,
            ledger,
        )
        is None
    )
    witnesses = policy.identity_witnesses(ledger)
    assert len(witnesses) == 1
    assert witnesses[0]["heading_end"] == len(heading)
    assert witnesses[0]["heading_hash"] == hashlib.sha256(heading.encode()).hexdigest()
    assert (
        policy.identity_ledger(ledger).authority_metadata()[0]["paragraph_no"] is None
    )


def test_witness_keeps_paragraph_and_closure_unknown_and_does_not_inherit_neighbors() -> (
    None
):
    _, policy = context_and_policy()
    ledger = ledger_with(captured(280), captured(291), captured(574), captured(575))
    rows = policy.identity_ledger(ledger).authority_metadata()
    assert [(row["citation"], row["article_no"]) for row in rows] == [
        (1, None),
        (2, "15"),
        (3, "10/A"),
        (4, "10"),
    ]
    assert rows[1]["paragraph_no"] is None
    assert rows[2]["paragraph_no"] == "2"
    paragraph_one = ledger.get(2)
    assert paragraph_one is not None and "(1)" in paragraph_one.text
    metadata = paragraph_one.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    assert metadata["article_closure_complete"] is False
    # Article identity witnesses are not a proof of paragraph 3 or article closure.
    assert [row["citation"] for row in policy.identity_witnesses(ledger)] == [2, 3]


@pytest.mark.parametrize(
    "claim",
    [
        "5326 sayılı Kabahatler Kanunu’nun **“İçtima” başlıklı 15. maddesinin üçüncü fıkrası** uygulanır [1].",
        "4458 sayılı Gümrük Kanunu’nun özgün metninde **“MADDE 10/A”** başlığıyla başlayan hüküm uygulanır [1].",
    ],
)
def test_captured_possessive_locators_cannot_hide_missing_actual_article(
    claim: str,
) -> None:
    context, policy = context_and_policy()
    ledger = ledger_with(captured(280) if "5326" in claim else captured(575))
    gap = policy.publication_gap(claim, "writer", context, ledger)
    assert gap is not None
    entries = gap["named_authority_gaps"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    assert entries[0]["article"] == ("15" if "5326" in claim else "10/A")
    assert entries[0]["unit_id"] == assertion_inventory(claim)[0]["unit_id"]
    retained = policy.view(context)["retained_authority_requirements"]
    assert isinstance(retained, list) and isinstance(retained[0], dict)
    assert retained[0]["article"] == entries[0]["article"]
    # Deleting the explicit locator does not discard the rejected dependency.
    assert policy.publication_gap("Bu hüküm uygulanır [1].", "writer", context, ledger)


@pytest.mark.parametrize("citation", [291, 477, 480])
@pytest.mark.parametrize(
    "change",
    [
        "cross_reference",
        "quoted",
        "bare_heading",
        "decision",
        "amendment",
        "external",
        "derived",
        "source",
        "chunk",
        "missing_identifiers",
        "conflicting_identifiers",
        "conflicting_article",
        "authoritative_path",
        "explicit_target",
        "truncated",
        "untrusted",
        "multiple_headings",
        "decimal_article",
        "article_range",
        "heading_without_operative_paragraph",
    ],
)
def test_own_heading_recovery_fails_closed_for_noncanonical_or_conflicting_sources(
    change: str,
    citation: int,
) -> None:
    context, policy = context_and_policy()
    item = captured(citation)
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    if change == "cross_reference":
        item.text = "Bu karar 5326 sayılı Kanunun MADDE 15 hükmüne atıf yapar."
    elif change == "quoted":
        item.text = '"' + item.text + '"'
    elif change == "bare_heading":
        item.text = "MADDE 15 kapsamında karar verilmiştir."
    elif change == "decision":
        metadata["document_type"] = "karar"
    elif change == "amendment":
        item.text += " Bu madde aşağıdaki şekilde değiştirilmiştir."
    elif change in {"external", "derived"}:
        item.metadata[change] = True
    elif change == "source":
        assert item.search_doc is not None
        item.search_doc.document_id = "another-source"
    elif change == "chunk":
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_chunk_id"] = "another-chunk"
    elif change == "missing_identifiers":
        metadata.pop("provision_identifiers")
    elif change == "conflicting_identifiers":
        metadata["provision_identifiers"] = ["madde 14"]
    elif change == "conflicting_article":
        metadata["article_no"] = "14"
    elif change == "authoritative_path":
        metadata["original_heading_path"] = ["KABAHATLER KANUNU", "MADDE 14"]
    elif change == "explicit_target":
        metadata["target_article_no"] = "15"
    elif change in {"truncated", "untrusted"}:
        metadata[change] = True
    elif change == "decimal_article":
        item.text = "**MADDE 8.1** (1) Bir şart gerekir."
    elif change == "article_range":
        item.text = "**MADDE 8.-9** (1) Bir şart gerekir."
    elif change == "heading_without_operative_paragraph":
        item.text = "**MADDE 8.-** Bu başlık bir hüküm alıntısıdır."
    else:
        item.text += "\n**MADDE 16** .- (1) Başka hüküm."
    item.text_hash = hashlib.sha256(item.text.encode()).hexdigest()
    ledger = ledger_with(item)
    assert policy.identity_witnesses(ledger) == []
    article = {291: "15", 477: "8", 480: "9"}[citation]
    assert policy.publication_gap(
        f"5326 sayılı Kabahatler Kanunu m.{article} uygulanır [1].",
        "writer",
        context,
        ledger,
    )


@pytest.mark.parametrize("change", ["base", "authoritative", "qualifier", "clause"])
def test_inserted_article_exception_is_limited_to_unqualified_navigation_base(
    change: str,
) -> None:
    _, policy = context_and_policy()
    item = captured(574)
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    if change == "base":
        metadata["article_no"] = "9"
        metadata["heading_path"] = ["4458 SAYILI GÜMRÜK KANUNU", "MADDE 9"]
    elif change == "authoritative":
        metadata["asv3_navigation_position_authority"] = "canonical"
    elif change == "qualifier":
        metadata["heading_path"] = ["4458 SAYILI GÜMRÜK KANUNU", "GEÇİCİ MADDE 10"]
    else:
        metadata["clause_label"] = "a"
    assert policy.identity_witnesses(ledger_with(item)) == []


def test_retained_article_requires_actual_delivery_or_exact_missing_original_disclosure() -> (
    None
):
    context, policy = context_and_policy()
    ledger = ledger_with(captured(575))
    claim = "4458 sayılı Gümrük Kanunu m.10/A uygulanır [1]."
    assert policy.publication_gap(claim, "first", context, ledger)
    ledger.add([captured(574)], context)
    ledger.record_delivery("next", "asv3_coordinator", [full_record(ledger, 2)])
    assert policy.publication_gap("Hüküm uygulanır [2].", "first", context, ledger)
    assert policy.publication_gap("Hüküm uygulanır [1].", "next", context, ledger)
    assert (
        policy.publication_gap("Hüküm uygulanır [2].", "next", context, ledger) is None
    )
    assert policy.publication_gap("Hüküm uygulanır [1].", "first", context, ledger)
    notice = "4458 sayılı Gümrük Kanunu'nun 10/A maddesinin özgün metni incelenemedi."
    assert policy.publication_gap(notice, "first", context, ledger) is None
    assert policy.publication_gap(
        notice.replace("10/A", "10"), "first", context, ledger
    )


def test_checkpoint_restores_actual_dependency_and_rejects_policy_scope_request_tampering() -> (
    None
):
    context, policy = context_and_policy()
    ledger = ledger_with(captured(280))
    claim = "5326 sayılı Kabahatler Kanunu m.15 uygulanır [1]."
    assert policy.publication_gap(claim, "writer", context, ledger)
    snapshot = policy.export()
    restored = CheapAuthorityPolicy(context, REQUEST)
    restored.restore(snapshot, context, REQUEST)
    assert restored.export() == snapshot
    assert restored.publication_gap("Hüküm uygulanır [1].", "writer", context, ledger)
    for key, value in (
        ("policy", "legacy"),
        ("version", 2),
        ("version", True),
        ("extra", 1),
    ):
        changed = copy.deepcopy(snapshot)
        changed[key] = value
        with pytest.raises(ValueError):
            restored.restore(changed, context, REQUEST)
    changed = copy.deepcopy(snapshot)
    requirements = changed["requirements"]
    assert isinstance(requirements, dict)
    records = requirements["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    records[0]["article"] = "14"
    with pytest.raises(ValueError):
        restored.restore(changed, context, REQUEST)
    with pytest.raises(ValueError):
        restored.restore(snapshot, context, REQUEST + "changed")
    context.scope = {"authorized": "another"}
    with pytest.raises(ValueError):
        restored.restore(snapshot, context, REQUEST)
    with pytest.raises(ValueError):
        restored.publication_gap(claim, "writer", context, ledger)


@pytest.mark.parametrize("flag", [None, False, "true", 1])
def test_noncheap_context_cannot_enable_policy(flag: object) -> None:
    with pytest.raises(ValueError, match="ASv3Exp Cheap"):
        CheapAuthorityPolicy(RunContext(services={"asv3_exp_cheap": flag}), REQUEST)


def test_unbound_quoted_or_titled_reference_does_not_invent_ownership() -> None:
    context, policy = context_and_policy()
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ"))
    answer = "Başlıklı 15. madde ve “MADDE 10/A” başlığı aktarılmıştır [1]."
    assert policy.publication_gap(answer, "writer", context, ledger) is None
    assert policy.view(context)["retained_authority_requirements"] == []


def test_owner_outcome_scoping_and_run_fences_are_preserved() -> None:
    context, policy = context_and_policy()
    ledger = ledger_with(captured(280))
    context.services.update(task_id="worker-one", task_outcome_ids=["içtima"])
    claim = "5326 sayılı Kabahatler Kanunu m.15 uygulanır [1]."
    assert policy.publication_gap(claim, "writer", context, ledger)
    rows = cast(
        list[dict[str, JsonValue]],
        policy.view(context)["retained_authority_requirements"],
    )
    assert rows[0]["outcome_ids"] == ["içtima"]
    context.services["task_id"] = "worker-two"
    assert policy.view(context)["retained_authority_requirements"] == []
    context.run_id = "another-run"
    with pytest.raises(ValueError):
        policy.view(context)
