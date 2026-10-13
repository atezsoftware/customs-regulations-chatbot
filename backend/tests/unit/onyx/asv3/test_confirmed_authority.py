"""Verified own headings preserve original bytes, numeric paragraphs and delivery."""

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.canonical_authority import (
    canonical_cited_lower_statute_gap,
    canonical_identity_ledger,
    canonical_matching_originals,
    canonical_named_authority_gap,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunStopped
from onyx.asv3.workflow_variant import ASV3_STANDARD_VARIANT, ASV3_TUNED_VARIANT
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db.asv3_corpus import CorpusSource
from onyx.llm.model_response import ModelResponse
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
    user_payload,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record
from tests.unit.onyx.asv3.test_spawned_researcher_terminal import capture_harnesses

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


OPTIONS = {"strict_reference_boundaries": True, "syntactic_reference_binding": True}
FIXTURE = Path(__file__).with_name("fixtures") / "confirmed_authority_5077.json"


def captured(citation: int) -> EvidenceItem:
    data = json.loads(FIXTURE.read_text())
    row = next(row for row in data["rows"] if row["citation"] == citation)
    assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_hash"]
    return EvidenceItem(
        source_id=row["source_id"],
        chunk_id=row["chunk_id"],
        text=row["text"],
        metadata={"canonical_metadata": row["metadata"]},
        search_doc=SearchDoc(
            document_id=row["source_id"],
            chunk_ind=0,
            semantic_identifier=row["metadata"]["title"],
            blurb=row["text"],
            source_type=DocumentSource.USER_FILE,
            boost=0,
            hidden=False,
            metadata={"regulatory_chunk_id": row["chunk_id"]},
            match_highlights=[],
        ),
    )


def gap(answer: str, ledger: EvidenceLedger) -> dict[str, JsonValue] | None:
    return canonical_named_authority_gap(
        answer, ledger, strict_numeric_paragraphs=True, **OPTIONS
    )


@pytest.mark.parametrize(
    ("citation", "claim"),
    [
        (
            477,
            "5326 sayılı Kabahatler Kanunu'nun 8/1. maddesi incelenmiştir [1].",
        ),
        (
            574,
            "4458 sayılı Gümrük Kanunu'nun 10/A. maddesi incelenmiştir [1].",
        ),
    ],
)
def test_actual_own_heading_recovery_preserves_original_and_does_not_forge_delivery(
    citation: int, claim: str
) -> None:
    ledger = ledger_with(captured(citation))
    saved = copy.deepcopy(ledger.export())
    native = native_named_authority_gap(claim, ledger, **OPTIONS)
    assert native is not None
    assert gap(claim, ledger) is None
    projection = canonical_identity_ledger(ledger)
    assert projection.get(1) == ledger.get(1)
    assert projection.completely_delivered("unread-writer") == set()
    ledger.record_delivery("actual-writer", "asv3_final", [full_record(ledger, 1)])
    assert projection.completely_delivered("actual-writer") == {1}
    # The projection never replaces the actual canonical metadata or source bytes.
    current = ledger.export()
    assert current["records"] == saved["records"]
    assert native_named_authority_gap(claim, ledger, **OPTIONS) == native


@pytest.mark.parametrize(
    ("citation", "article", "paragraph"), [(477, "8", "2"), (291, "15", "3")]
)
def test_recovered_article_does_not_cover_an_unread_numeric_paragraph(
    citation: int, article: str, paragraph: str
) -> None:
    ledger = ledger_with(captured(citation))
    claim = f"5326 sayılı Kabahatler Kanunu m.{article}/{paragraph} incelenmiştir [1]."
    result = gap(claim, ledger)
    assert result is not None
    entries = result["named_authority_gaps"]
    assert isinstance(entries, list) and len(entries) == 1
    entry = entries[0]
    assert isinstance(entry, dict)
    assert entry["paragraph"] == paragraph
    assert entry["matching_original_evidence"] == []


@pytest.mark.parametrize(
    "suffix",
    [
        " Aynı fıkradaki açıklama (2) numaralı belgeye atıf yapar.",
        '\n"(2) Başka bir metnin alıntısıdır."',
        "\n> (2) Alıntı metindir.",
        "\n2. İşlem listesi maddesidir.",
    ],
)
def test_unrelated_numbers_or_quoted_lists_are_not_own_paragraphs(suffix: str) -> None:
    item = captured(477)
    item.text += suffix
    item.text_hash = hashlib.sha256(item.text.encode()).hexdigest()
    assert gap("5326 sayılı Kabahatler Kanunu m.8/2 uygulanır [1].", ledger_with(item))


def test_complete_own_article_can_supply_multiple_literal_paragraphs() -> None:
    item = captured(477)
    item.text += "\n\n(2) Kendi ikinci fıkrasında ek şart bulunur."
    item.text_hash = hashlib.sha256(item.text.encode()).hexdigest()
    ledger = ledger_with(item)
    assert gap("5326 sayılı Kabahatler Kanunu m.8/2 incelenmiştir [1].", ledger) is None
    assert gap("5326 sayılı Kabahatler Kanunu m.8/3 incelenmiştir [1].", ledger)
    assert (
        canonical_identity_ledger(ledger).authority_metadata()[0]["paragraph_no"]
        is None
    )


def test_precise_missing_paragraph_notice_never_asserts_its_legal_effect() -> None:
    ledger = ledger_with(captured(477))
    notice = (
        "5326 sayılı Kabahatler Kanunu'nun 8/2 maddesinin özgün metni incelenemedi."
    )
    assert gap(notice, ledger) is None
    assert gap(notice + " Bu nedenle ikinci fıkraya göre izin gerekmez.", ledger)


def test_conflicting_paragraph_metadata_fails_closed_when_numeric_guard_is_enabled() -> (
    None
):
    item = captured(477)
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    metadata["paragraph_no"] = "2"
    ledger = ledger_with(item)
    claim = "5326 sayılı Kabahatler Kanunu m.8/1 incelenmiştir [1]."
    assert gap(claim, ledger)
    assert canonical_named_authority_gap(claim, ledger, **OPTIONS) is None


def test_normalized_possessive_locators_keep_original_answer_unit_ids() -> None:
    claim = (
        "5326 sayılı Kabahatler Kanunu’nun **“İçtima” başlıklı 15. maddesinin üçüncü "
        "fıkrası** uygulanır [1].\n\n"
        "4458 sayılı Gümrük Kanunu’nun özgün metninde **“MADDE 10/A”** başlığıyla "
        "başlayan hüküm uygulanır [2]."
    )
    result = gap(claim, ledger_with(captured(291), captured(575)))
    assert result is not None
    entries = result["named_authority_gaps"]
    assert isinstance(entries, list)
    assert {row["unit_id"] for row in entries if isinstance(row, dict)} == {
        row["unit_id"] for row in assertion_inventory(claim)
    }


def test_blank_neighbor_and_lower_instrument_cannot_supply_own_article_identity() -> (
    None
):
    assert gap(
        "5326 sayılı Kabahatler Kanunu m.15/3 uygulanır [1].",
        ledger_with(captured(280)),
    )
    item = captured(477)
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    metadata["document_type"] = "tebliğ"
    assert gap("5326 sayılı Kabahatler Kanunu m.8/1 uygulanır [1].", ledger_with(item))


def test_lower_statute_dependency_still_requires_its_actual_delivered_original() -> (
    None
):
    lower = original(
        "Uygulama Tebliği",
        "1",
        kind="tebliğ",
        text="5326 sayılı Kabahatler Kanunu m.8/1 kapsamında başvuru alınır.",
    )
    ledger = ledger_with(lower, captured(477))
    claim = "Başvuru koşulu ayrıca incelenmiştir [1]."
    assert canonical_cited_lower_statute_gap(claim, ledger, {1})
    assert canonical_cited_lower_statute_gap(claim, ledger, {1, 2}) is None
    lower.text = lower.text.replace("m.8/1", "m.15/3")
    lower.text_hash = hashlib.sha256(lower.text.encode()).hexdigest()
    assert canonical_cited_lower_statute_gap(
        claim, ledger_with(lower, captured(291)), {1, 2}
    )


def _install_capture(broker: Any, index: int, item: EvidenceItem) -> None:
    old_id = str(broker.sources[index].id)
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    source = CorpusSource(UUID(item.source_id), cast(str, metadata["title"]), "capture")
    chunk = broker.chunks.pop(old_id)
    broker.sources[index] = source
    broker.chunks[item.source_id] = replace(
        chunk,
        id=item.chunk_id,
        source_id=source.id,
        text=item.text,
        heading_path=tuple(cast(list[str], metadata["heading_path"])),
        metadata=copy.deepcopy(metadata),
    )


@pytest.mark.parametrize(
    ("profile", "variant", "recovers"),
    [
        ("normal", ASV3_TUNED_VARIANT, True),
        ("normal", ASV3_STANDARD_VARIANT, False),
        ("experimental", ASV3_STANDARD_VARIANT, False),
        ("deep", ASV3_STANDARD_VARIANT, False),
    ],
)
def test_actual_runtime_only_tuned_recovers_captured_identity(
    monkeypatch: pytest.MonkeyPatch, profile: str, variant: str, recovers: bool
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(
        research_profile=profile,
        workflow_variant=variant,
    )
    _install_capture(broker, 0, captured(477))
    _install_capture(broker, 1, captured(574))
    harnesses = capture_harnesses(monkeypatch)
    calls = 0
    body = ""

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal calls, body
        calls += 1
        if calls == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(source.id)})
                    for source in broker.sources
                ]
            )
        if calls == 2:
            from onyx.asv3.source_metadata_transport import expand_source_metadata

            catalogue = (
                expand_source_metadata(user_payload(arguments["prompt"][-1]))
                if recovers
                else delivered_originals(arguments)
            )
            numbers = {row["chunk_id"]: row["citation"] for row in catalogue}
            first, second = captured(477), captured(574)
            body = (
                "5326 sayılı Kabahatler Kanunu m.8/1 incelenmiştir "
                f"[{numbers[first.chunk_id]}].\n\n"
                "4458 sayılı Gümrük Kanunu m.10/A incelenmiştir "
                f"[{numbers[second.chunk_id]}]."
            )
            return response(body)
        assert not recovers
        footer = user_payload(arguments["prompt"][-1])
        assert footer["publication_gap"]["named_authority_gaps"]
        broker.cancelled.set()
        return response(body)

    llm.invoke.side_effect = scripted
    if recovers:
        runtime.run_asv3_loop(**kwargs)
        assert calls == 2
        assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
        assert checkpoints[-1]["final_publication_gap"] is None
        assert harnesses[0].context.services.get("asv3_exp_cheap") is not True
        assert "cheap_authority_policy" not in harnesses[0].context.services
    else:
        with pytest.raises(RunStopped, match="Research cancelled"):
            runtime.run_asv3_loop(**kwargs)
        assert calls == 3
        assert harnesses[0].publication_gap is not None
        assert "named_authority_gaps" in harnesses[0].publication_gap.data
        assert (
            harnesses[0].context.services.get(
                "asv3_workflow_variant", ASV3_STANDARD_VARIANT
            )
            == variant
        )
    assert body
    assert all(
        hashlib.sha256(chunk.text.encode()).hexdigest()
        == captured(477 if index == 0 else 574).text_hash
        for index, chunk in enumerate(broker.chunks.values())
    )


@pytest.mark.parametrize("citation", [291, 477])
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
    assert (
        canonical_identity_ledger(ledger).authority_metadata()
        == ledger.authority_metadata()
    )
    article = {291: "15", 477: "8", 480: "9"}[citation]
    assert gap(f"5326 sayılı Kabahatler Kanunu m.{article} uygulanır [1].", ledger)


@pytest.mark.parametrize("change", ["base", "authoritative", "qualifier", "clause"])
def test_inserted_article_exception_is_limited_to_unqualified_navigation_base(
    change: str,
) -> None:
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
    ledger = ledger_with(item)
    assert (
        canonical_identity_ledger(ledger).authority_metadata()
        == ledger.authority_metadata()
    )


@pytest.mark.parametrize("invalid", ["unknown", "undelivered", "numeric_paragraph"])
def test_tuned_runtime_keeps_unknown_delivery_and_numeric_source_guards(
    monkeypatch: pytest.MonkeyPatch, invalid: str
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    _install_capture(broker, 0, captured(477))
    _install_capture(broker, 1, captured(574))
    harnesses = capture_harnesses(monkeypatch)
    calls = 0
    body = ""

    def scripted(**arguments: Any) -> ModelResponse:
        nonlocal calls, body
        calls += 1
        if calls == 1:
            return response(
                calls=[
                    ("read_source_range", {"source_id": str(source.id)})
                    for source in broker.sources
                ]
            )
        if calls == 2:
            if invalid == "unknown":
                body = "5326 sayılı Kabahatler Kanunu m.8/1 incelenmiştir [999]."
            elif invalid == "undelivered":
                root = harnesses[0]
                assert root.evidence.add([captured(291)], root.context) == [3]
                body = "5326 sayılı Kabahatler Kanunu m.15 incelenmiştir [3]."
            else:
                body = "5326 sayılı Kabahatler Kanunu m.8/2 incelenmiştir [1]."
            return response(body)
        footer = user_payload(arguments["prompt"][-1])
        expected = {
            "unknown": "unknown_citations",
            "undelivered": "undelivered_citations",
            "numeric_paragraph": "named_authority_gaps",
        }[invalid]
        assert footer["publication_gap"][expected]
        broker.cancelled.set()
        return response(body)

    llm.invoke.side_effect = scripted
    with pytest.raises(RunStopped, match="Research cancelled"):
        runtime.run_asv3_loop(**kwargs)
    assert calls == 3
    assert not any(
        row.get("publication_stop_reason") == "native_answer_published"
        for row in checkpoints
    )


def test_exact_candidates_exclude_other_articles_and_unread_paragraphs() -> None:
    ledger = ledger_with(captured(477), captured(291), captured(574), captured(575))
    saved = copy.deepcopy(ledger.export())
    for probe, expected in [
        ("5326 sayılı Kabahatler Kanunu m.8/1 uygulanır.", [1]),
        ("5326 sayılı Kabahatler Kanunu m.8/2 uygulanır.", []),
        ("5326 sayılı Kabahatler Kanunu m.15/1 uygulanır.", [2]),
        ("5326 sayılı Kabahatler Kanunu m.15/3 uygulanır.", []),
        ("4458 sayılı Gümrük Kanunu m.10/A uygulanır.", [3]),
        ("4458 sayılı Gümrük Kanunu m.10 uygulanır.", [4]),
        ("Bir hüküm uygulanır.", []),
        ("5326 sayılı Kabahatler Kanunu m.8 ve m.15 uygulanır.", []),
    ]:
        assert canonical_matching_originals(probe, ledger) == expected
    assert ledger.export() == saved
    assert ledger.completely_delivered("unread-call") == set()


def test_gap_preserves_paragraph_and_qualified_reference_identity() -> None:
    result = gap(
        "5326 sayılı Kabahatler Kanunu m.15/3 uygulanır [1].",
        ledger_with(captured(280)),
    )
    assert result is not None
    rows = result["named_authority_gaps"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["paragraph"] == "3"
    assert rows[0]["article_qualifier"] is None
    qualified = "5326 sayılı Kabahatler Kanunu geçici m.8 uygulanır [1]."
    ledger = ledger_with(captured(477))
    result = gap(qualified, ledger)
    assert result is not None
    rows = result["named_authority_gaps"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["article_qualifier"] == "gecici"
    assert canonical_matching_originals(qualified, ledger) == []


def test_formal_name_only_numeric_gap_remains_a_named_acquisition_target() -> None:
    item = captured(477)
    metadata = item.metadata["canonical_metadata"]
    assert isinstance(metadata, dict)
    metadata["title"] = "Kabahatler Kanunu"
    result = gap("Kabahatler Kanunu m.8/2 uygulanır [1].", ledger_with(item))
    assert result is not None
    rows = result["named_authority_gaps"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["instrument_number"] is None
    assert rows[0]["formal_name"] == "kabahatler kanun"
    assert rows[0]["paragraph"] == "2"


@pytest.mark.parametrize("owner", ["Kanun'un", "Kanun’un", "Kanunu'nun", "Kanunu’nun"])
def test_both_possessive_forms_keep_inserted_article_identity(owner: str) -> None:
    claim = (
        f"4458 sayılı {owner} özgün metninde **“MADDE 10/A”** başlığıyla "
        "başlayan hüküm uygulanır [1]."
    )
    missing = gap(claim, ledger_with(captured(575)))
    assert missing is not None
    entries = missing["named_authority_gaps"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    assert entries[0]["article"] == "10/A"
    assert entries[0]["matching_original_evidence"] == []
    assert entries[0]["unit_id"] == assertion_inventory(claim)[0]["unit_id"]
    assert gap(claim, ledger_with(captured(574))) is None


@pytest.mark.parametrize(
    "owner", ["Kanun'un", "Kanun’un", "Kabahatler Kanunu'nun", "Kabahatler Kanunu’nun"]
)
def test_titled_possessive_forms_preserve_unread_paragraph_guard(owner: str) -> None:
    claim = (
        f"5326 sayılı {owner} **“İçtima” başlıklı 15. maddesinin birinci fıkrası** "
        "uygulanır [1]."
    )
    assert gap(claim, ledger_with(captured(291))) is None
    missing = gap(claim.replace("birinci", "üçüncü"), ledger_with(captured(291)))
    assert missing is not None
    entries = missing["named_authority_gaps"]
    assert isinstance(entries, list) and isinstance(entries[0], dict)
    assert entries[0]["article"] == "15"
    assert entries[0]["paragraph"] == "3"
    assert entries[0]["matching_original_evidence"] == []
