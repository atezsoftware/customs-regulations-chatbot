"""Catalogue-only references survive real model rendering without becoming evidence."""

import copy
import json
from typing import cast
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.source_notices import (
    SOURCE_NOTICE_STATE_SERVICE,
    SOURCE_NOTICE_USAGE_RULE,
    SourceNoticeState,
)
from onyx.db.asv3_corpus import CorpusScopeUnavailable
from onyx.regulatory.legistation_metadata import LegistationMetadata
from onyx.regulatory.source_notices import SourceNotice
from tests.unit.onyx.asv3.test_native_cache_projection import payloads
from tests.unit.onyx.asv3.test_native_model_adapter import model, turn, view
from tests.unit.onyx.asv3.test_shared_originals import original


def notice() -> SourceNotice:
    return SourceNotice(
        id=str(uuid4()),
        document_set_id=15,
        identity_key="circular:2012:7",
        instrument_type="circular",
        instrument_year=2012,
        instrument_number="7",
        catalog_title="2012/7 sayılı Genelge (Mülga)",
        legistation_metadata=LegistationMetadata(validity_status="Mülga"),
        provenance={
            "source": "mülga liste 20.10.2026.xlsx",
            "sheet": "Sayfa1",
            "rows": [500],
        },
    )


class Catalogue:
    def __init__(self) -> None:
        self.row: SourceNotice | None = notice()
        self.calls: list[tuple[list[str] | None, list[str] | None]] = []
        self.denied = False

    def lookup(
        self, queries: list[str] | None, ids: list[str] | None
    ) -> list[SourceNotice]:
        self.calls.append((queries, ids))
        if self.denied:
            raise CorpusScopeUnavailable("Current scope unavailable")
        return (
            [self.row]
            if self.row is not None
            and (self.row.identity_key in (queries or []) or self.row.id in (ids or []))
            else []
        )


def setup() -> tuple[RunContext, Catalogue, SourceNoticeState]:
    context = RunContext(scope={"document_set": ["PC Külliyatı"]})
    catalogue = Catalogue()
    state = SourceNoticeState(context, catalogue.lookup)
    context.services[SOURCE_NOTICE_STATE_SERVICE] = state
    return context, catalogue, state


def canonical(
    text: str = "Uygulamada 2012/7 sayılı Genelge dikkate alınır.",
    *,
    source: str = "real-source",
) -> EvidenceItem:
    item = original("canonical", text, source=source)
    assert item.search_doc is not None
    return item.model_copy(
        update={
            "search_doc": item.search_doc.model_copy(
                update={"metadata": {"regulatory_chunk_id": item.chunk_id}}
            )
        }
    )


def test_automatic_first_hop_without_model_navigation_preserves_originals() -> None:
    context, catalogue, state = setup()
    item = canonical()
    before = copy.deepcopy(item.model_dump())
    ledger = EvidenceLedger()
    ledger.add([item], context)
    before_ledger = copy.deepcopy(ledger.export())
    state.observe_evidence([item])
    result = state.model_payload()
    assert len(result) == 1
    row = cast(dict[str, JsonValue], result[0])
    assert catalogue.row is not None
    assert row["content_available"] is False
    assert row["original_acquired"] is False
    assert row["acquisition_state"] == "metadata_only_content_unavailable"
    assert row["legistation_metadata"] == catalogue.row.legistation_metadata.model_dump(
        mode="json"
    )
    assert "text" not in row and "chunk_id" not in row and "citation" not in row
    assert item.model_dump() == before and ledger.export() == before_ledger
    state.observe_evidence([item])
    assert len([call for call in catalogue.calls if call[0]]) == 1


@pytest.mark.parametrize(
    "changed",
    [
        {"derived": True},
        {"external": True},
        {"truncated": True},
        {"asv3_rerank_excerpt_truncated": True},
    ],
)
def test_navigation_snippets_and_derived_text_never_trigger_notice_acquisition(
    changed: dict[str, JsonValue],
) -> None:
    _, catalogue, state = setup()
    item = canonical()
    state.observe_evidence(
        [item.model_copy(update={"metadata": {**item.metadata, **changed}})]
    )
    assert catalogue.calls == []


def test_reference_target_is_not_scanned_recursively() -> None:
    _, catalogue, state = setup()
    item = canonical(source="first-hop-source")
    state.observe_receipt(
        ToolReceipt(
            call=CapabilityCall(name="follow_reference"),
            outcome=ToolOutcome(status=OutcomeStatus.FOUND, summary="Actual original"),
            elapsed_seconds=0,
        ),
        [item],
    )
    state.observe_evidence([item])
    assert catalogue.calls == []


def test_scope_checkpoint_and_current_access_are_revalidated() -> None:
    context, catalogue, state = setup()
    state.observe_text("2012/7 sayılı Genelge")
    saved = state.export()
    restored = SourceNoticeState(context, catalogue.lookup)
    restored.restore(saved)
    assert restored.model_payload()
    catalogue.denied = True
    with pytest.raises(PermissionError, match="no longer accessible"):
        restored.model_payload()
    assert restored.lookup_queries(["circular:2012:7"]) == []
    context.scope = {"document_set": ["Another set"]}
    with pytest.raises(PermissionError):
        restored.model_payload()


def test_checkpoint_cannot_restore_forged_or_cross_run_identity() -> None:
    context, catalogue, state = setup()
    state.observe_text("2012/7 sayılı Genelge")
    saved = state.export()
    saved["notice_ids"] = [str(uuid4())]
    with pytest.raises(ValueError):
        SourceNoticeState(context, catalogue.lookup).restore(saved)
    other = RunContext(scope=context.scope)
    with pytest.raises(ValueError):
        SourceNoticeState(other, catalogue.lookup).restore(state.export())


def test_new_materialized_revision_replaces_cached_missing_content() -> None:
    _, catalogue, state = setup()
    assert catalogue.row is not None
    identifier = catalogue.row.id
    state.observe_text("2012/7 sayılı Genelge")
    assert (
        cast(dict[str, JsonValue], state.model_payload()[0])["content_available"]
        is False
    )
    linked = str(uuid4())
    catalogue.row = catalogue.row.model_copy(
        update={"revision": 2, "content_state": "materialized", "user_file_id": linked}
    )
    result = cast(dict[str, JsonValue], state.model_payload()[0])
    assert result["content_available"] is True and result["revision"] == 2
    assert result["source_id"] == linked
    assert state.by_id("notice:" + identifier) == catalogue.row
    catalogue.row = None
    assert state.by_id("notice:" + identifier) is None
    with pytest.raises(PermissionError, match="no longer accessible"):
        state.model_payload()


def test_paragraph_heading_reference_is_not_mistaken_for_source_identity() -> None:
    _, catalogue, state = setup()
    assert catalogue.row is not None
    catalogue.row = catalogue.row.model_copy(
        update={
            "identity_key": "law:4703",
            "instrument_type": "law",
            "instrument_number": "4703",
            "instrument_year": None,
            "catalog_title": "4703 sayılı Kanun (Mülga)",
        }
    )
    paragraph = (
        "(2) Mevzuatta 4703 sayılı Kanuna yapılan atıflar bu Kanuna yapılmış sayılır."
    )
    item = canonical("7223 sayılı Kanun.\n" + paragraph)
    assert item.search_doc is not None
    item = item.model_copy(
        update={
            "search_doc": item.search_doc.model_copy(
                update={"semantic_identifier": "7223 sayılı Kanun"}
            ),
            "metadata": {
                **item.metadata,
                "heading_path": ["7223 sayılı Kanun", "MADDE 26", paragraph],
            },
        }
    )
    state.observe_evidence([item])
    queried = {key for queries, _ in catalogue.calls for key in queries or []}
    assert queried == {"law:4703"}
    assert (
        cast(dict[str, JsonValue], state.model_payload()[0])["identity_key"]
        == "law:4703"
    )


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("delivered", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_revoked_notice_never_reenters_model_in_retained_tool_history(
    cropped: bool,
    delivered: bool,
    native: bool,
) -> None:
    context, catalogue, state = setup()
    if cropped:
        context.services["asv3_exp_cropped"] = True
    state.observe_text("2012/7 sayılı Genelge")
    assert catalogue.row is not None
    cached = turn("notice-read", [])
    cached.results[0].content = json.dumps(
        {
            "outcome": {
                "status": "unavailable",
                "data": catalogue.row.model_dump(mode="json"),
            }
        },
        ensure_ascii=False,
    )
    if delivered:
        state.model_payload()
    catalogue.denied = True
    llm = model(limit=1_000_000)
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    with pytest.raises(PermissionError, match="no longer accessible"):
        if native:
            adapter._fit_native_decision(view(turns=[cached]))
        else:
            adapter._fit(
                "Research", "{}", [], max_tokens=100, research=True, turns=[cached]
            )
    llm.invoke.assert_not_called()


def test_first_lookup_without_matching_notice_remains_optional() -> None:
    _, catalogue, state = setup()
    catalogue.row = None
    assert state.observe_text("2012/7 sayılı Genelge") == []
    assert state.model_payload() == []
    state.assert_current_access()


def test_publication_fence_rejects_revoked_or_changed_delivered_notice() -> None:
    _, catalogue, state = setup()
    state.observe_text("2012/7 sayılı Genelge")
    state.model_payload()
    assert catalogue.row is not None
    catalogue.row = catalogue.row.model_copy(update={"revision": 2})
    with pytest.raises(PermissionError):
        state.assert_current_access()
    state.model_payload()
    state.assert_current_access()
    catalogue.denied = True
    with pytest.raises(PermissionError):
        state.assert_current_access()


def test_standard_named_read_of_first_hop_does_not_scan_second_hop() -> None:
    _, catalogue, state = setup()
    state.observe_evidence([canonical("2013/9 sayılı Genelge uygulanır.")])
    catalogue.calls.clear()
    item = canonical(source="target-generalge")
    assert item.search_doc is not None
    item = item.model_copy(
        update={
            "search_doc": item.search_doc.model_copy(
                update={"semantic_identifier": "Genelge 2013/9"}
            )
        }
    )
    state.observe_evidence([item])
    assert catalogue.calls == []


def test_direct_user_target_stays_independent_from_reference_depth() -> None:
    _, catalogue, state = setup()
    state.observe_text("2013/9 sayılı Genelgeyi değerlendir")
    state.observe_evidence([canonical("2013/9 sayılı Genelge uygulanır.")])
    item = canonical(source="target-generalge")
    assert item.search_doc is not None
    item = item.model_copy(
        update={
            "search_doc": item.search_doc.model_copy(
                update={"semantic_identifier": "Genelge 2013/9"}
            )
        }
    )
    state.observe_evidence([item])
    assert any(call[0] and "circular:2012:7" in call[0] for call in catalogue.calls)


@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize("name_only", [False, True])
def test_research_writer_and_verifier_receive_current_notice_after_compaction(
    cropped: bool,
    name_only: bool,
) -> None:
    context, catalogue, state = setup()
    if cropped:
        context.services["asv3_exp_cropped"] = True
    if name_only:
        assert catalogue.row is not None
        catalogue.row = catalogue.row.model_copy(
            update={
                "identity_key": "catalog:" + "a" * 64,
                "instrument_number": None,
                "instrument_year": None,
                "catalog_title": "İçeriği henüz yüklenmemiş Yönetmelik (Mülga)",
            }
        )
        state.observe_notices([catalogue.row])
    else:
        state.observe_evidence([canonical()])
    llm = model(limit=1_000_000)
    adapter = ResearchModel(llm, context, lean_native_mode=True)
    for writing in (False, True) if cropped else (False,):
        adapter._writing_answer = writing
        prompt, _, _ = adapter._fit_native_decision(view())
        data = next(row for row in payloads(prompt) if "source_notices" in row)
        assert (
            data["source_notices"][0]["legistation_metadata"]["validity_status"]
            == "Mülga"
        )
        assert SOURCE_NOTICE_USAGE_RULE in str(prompt[0].content)
        if name_only:
            assert data["source_notices"][0]["identity_key"].startswith("catalog:")
            assert data["source_notices"][0]["instrument_date"] is None
            assert data["source_notices"][0]["instrument_number"] is None
            assert "same historical version" in str(prompt[0].content)
    verifier_prompt, _, _ = adapter._fit(
        "Verify",
        json.dumps({"receipts": [{"old": True}], "evidence": []}),
        [],
        max_tokens=100,
    )
    parsed = next(row for row in payloads(verifier_prompt) if "source_notices" in row)
    assert parsed["source_notices"][0]["content_available"] is False
    assert "receipts" not in parsed
    llm.invoke.assert_not_called()


def test_absent_notices_add_no_model_field_or_lookup() -> None:
    context, catalogue, _ = setup()
    adapter = ResearchModel(model(limit=1_000_000), context, lean_native_mode=True)
    prompt, _, _ = adapter._fit_native_decision(view())
    assert catalogue.calls == []
    assert all("source_notices" not in row for row in payloads(prompt))
    assert SOURCE_NOTICE_USAGE_RULE not in str(prompt[0].content)
