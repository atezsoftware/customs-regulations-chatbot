"""Cropped own-heading recovery keeps canonical text, paragraphs and delivery guards."""

import copy
import hashlib
from dataclasses import replace
from typing import Any, cast
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3 import runtime
from onyx.asv3.assertions import assertion_inventory
from onyx.asv3.authority import cited_lower_statute_gap, native_named_authority_gap
from onyx.asv3.cheap_authority import (
    canonical_identity_ledger,
    canonical_named_authority_gap,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem
from onyx.asv3.workflow_variant import ASV3_EXP_CROPPED_VARIANT, ASV3_TUNED_VARIANT
from onyx.db.asv3_corpus import CorpusSource
from onyx.llm.model_response import ModelResponse
from tests.unit.onyx.asv3.test_cheap_authority import OPTIONS, captured
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_runtime import response, setup_run, user_payload
from tests.unit.onyx.asv3.test_shared_originals import full_record
from tests.unit.onyx.asv3.test_spawned_researcher_terminal import capture_harnesses

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


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


def test_conflicting_paragraph_metadata_fails_closed_without_changing_cheap_default() -> (
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
    projected = canonical_identity_ledger(ledger)
    assert cited_lower_statute_gap(claim, projected, {1})
    assert cited_lower_statute_gap(claim, projected, {1, 2}) is None
    lower.text = lower.text.replace("m.8/1", "m.15/3")
    lower.text_hash = hashlib.sha256(lower.text.encode()).hexdigest()
    assert cited_lower_statute_gap(
        claim, canonical_identity_ledger(ledger_with(lower, captured(477))), {1, 2}
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


@pytest.mark.parametrize("cropped", [True, False])
def test_actual_runtime_cropped_and_experimental_recover_with_isolated_policies(
    monkeypatch: pytest.MonkeyPatch, cropped: bool
) -> None:
    kwargs, broker, llm, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(
        research_profile="normal",
        workflow_variant=ASV3_EXP_CROPPED_VARIANT if cropped else ASV3_TUNED_VARIANT,
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
            from onyx.asv3.cheap_metadata_transport import expand_cheap_metadata

            catalogue = expand_cheap_metadata(user_payload(arguments["prompt"][-1]))
            numbers = {row["chunk_id"]: row["citation"] for row in catalogue}
            first, second = captured(477), captured(574)
            body = (
                "5326 sayılı Kabahatler Kanunu'nun 8/1. maddesi incelenmiştir "
                f"[{numbers[first.chunk_id]}].\n\n"
                "4458 sayılı Gümrük Kanunu'nun 10/A. maddesi incelenmiştir "
                f"[{numbers[second.chunk_id]}]."
            )
            return response(body)
        raise AssertionError(
            "Both workflows must accept the acquired own-heading originals"
        )

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert calls == 2
    assert checkpoints[-1]["publication_stop_reason"] == "native_answer_published"
    assert checkpoints[-1]["final_publication_gap"] is None
    services = harnesses[0].context.services
    assert services.get("asv3_exp_cheap") is not True
    assert "cheap_authority_policy" not in services
    if cropped:
        assert services.get("asv3_exp_cropped") is True
        assert "cropped_article_bindings" in services
        assert "cropped_repair_state" in services
        assert "tuned_article_bindings" not in services
        assert "tuned_governing_acquisitions" not in services
    else:
        assert services.get("asv3_exp_cropped") is not True
        assert "tuned_article_bindings" in services
        assert "tuned_governing_acquisitions" in services
        assert "cropped_article_bindings" not in services
        assert "cropped_repair_state" not in services
    assert body
    assert all(
        hashlib.sha256(chunk.text.encode()).hexdigest()
        == captured(477 if index == 0 else 574).text_hash
        for index, chunk in enumerate(broker.chunks.values())
    )
