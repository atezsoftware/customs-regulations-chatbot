"""One-hop NAV provenance cannot become original delivery or legal approval."""

import copy
import hashlib
import json
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import cited_lower_statute_gap, native_named_authority_gap
from onyx.asv3.cropped_crossrefs import CroppedCrossReferencePolicy
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original

SERVICE = "cropped_cross_references"


def make(
    request: str = "Compare the original governing provisions.",
) -> tuple[RunContext, CroppedCrossReferencePolicy, EvidenceLedger]:
    context = RunContext(scope={"corpus": "owned"}, services={"asv3_exp_cropped": True})
    policy = CroppedCrossReferencePolicy(context, request)
    context.services[SERVICE] = policy
    ledger = ledger_with(
        original(
            "Uygulama Tebliği",
            "1",
            kind="tebliğ",
            text="1001 sayılı İlk Kanun m.7 uygulanır.",
        ),
        original(
            "1001 sayılı İlk Kanun",
            "7",
            text="1002 sayılı İkinci Kanun m.11 incelenir.",
        ),
        original(
            "1002 sayılı İkinci Kanun",
            "11",
            text="Özgün şart ve istisna birlikte incelenir.",
        ),
    )
    return context, policy, ledger


def ref(number: str, article: str, origin: int) -> dict[str, JsonValue]:
    return {
        "instrument_number": number,
        "formal_name": None,
        "article": article,
        "qualifier": None,
        "origin_citations": [origin],
        "matching_original_evidence": [],
    }


def receipt(
    call: CapabilityCall,
    numbers: list[int],
    *,
    status: OutcomeStatus = OutcomeStatus.FOUND,
    data: dict[str, JsonValue] | None = None,
) -> ToolReceipt:
    return ToolReceipt(
        call=call,
        outcome=ToolOutcome(
            status=status, summary="Actual canonical source operation", data=data or {}
        ),
        elapsed_seconds=0.0,
        evidence_ids=numbers,
    )


def first(
    context: RunContext, policy: CroppedCrossReferencePolicy, ledger: EvidenceLedger
) -> CapabilityCall:
    root = CapabilityCall(
        name="read_provision",
        arguments={"source_id": "Uygulama Tebliği", "article": "1"},
    )
    policy.observe(receipt(root, [1]), context, ledger)
    call = CapabilityCall(
        name="read_provision",
        arguments={"source_id": "1001 sayılı İlk Kanun", "article": "7"},
    )
    assert policy.reserve_acquisition(call, ref("1001", "7", 1), context, ledger)
    assert policy.admit(call, context, ledger) is None
    policy.observe(receipt(call, [2]), context, ledger)
    return call


def test_root_to_first_reference_allowed_then_nested_reference_unread() -> None:
    context, policy, ledger = make()
    saved = copy.deepcopy(ledger.export())
    first(context, policy, ledger)
    assert policy.filter_references([ref("1001", "7", 1)], ledger, context)
    nested = ref("1002", "11", 2)
    assert policy.filter_references([nested], ledger, context) == []
    assert not policy.reserve_acquisition(
        CapabilityCall(
            name="read_named_provision",
            arguments={"source_name": "1002 sayılı Kanun", "article": "11"},
        ),
        nested,
        context,
        ledger,
    )
    assert policy.view(context)["unread_reference_navigation"]
    assert policy.view(context)["legal_approval"] is False
    assert ledger.export() == saved
    assert ledger.completely_delivered("no-writer") == set()


@pytest.mark.parametrize("caller_depth", [0, 1, 4])
def test_follow_reference_cannot_reset_host_depth(caller_depth: int) -> None:
    context, policy, ledger = make()
    first(context, policy, ledger)
    item = ledger.get(2)
    assert item is not None
    call = CapabilityCall(
        name="follow_reference",
        arguments={
            "source_id": item.source_id,
            "chunk_id": item.chunk_id,
            "target_source_id": "1002 sayılı İkinci Kanun",
            "article": "11",
            "depth": caller_depth,
        },
    )
    denied = policy.admit(call, context, ledger)
    assert denied is not None and denied.status == OutcomeStatus.DENIED
    assert denied.data["not_read"] is True
    assert denied.data["absence_proven"] is False
    assert denied.data["legal_approval"] is False
    policy.observe(receipt(call, [], status=OutcomeStatus.DENIED), context, ledger)


@pytest.mark.parametrize(
    "name,arguments",
    [
        ("read_provision", {"source_id": "1002 sayılı İkinci Kanun", "article": "11"}),
        (
            "read_named_provision",
            {"source_name": "1002 sayılı İkinci Kanun", "article": "11"},
        ),
        ("read_named_provision", {"source_name": "1002", "article": "11"}),
    ],
)
def test_follow_nav_then_switching_tools_does_not_erase_nested_lineage(
    name: str, arguments: dict[str, JsonValue]
) -> None:
    context, policy, ledger = make()
    first(context, policy, ledger)
    item = ledger.get(2)
    assert item is not None
    nav = CapabilityCall(
        name="follow_reference",
        arguments={"source_id": item.source_id, "chunk_id": item.chunk_id},
    )
    assert policy.admit(nav, context, ledger) is not None
    attempted = CapabilityCall(name=name, arguments=arguments)
    assert policy.admit(attempted, context, ledger) is not None
    with pytest.raises(ValueError, match="another tool"):
        policy.observe(receipt(attempted, [3]), context, ledger)


def test_first_hop_ambiguous_nav_origin_is_not_promoted_or_failed() -> None:
    context, policy, ledger = make()
    root = CapabilityCall(
        name="read_provision",
        arguments={"source_id": "Uygulama Tebliği", "article": "1"},
    )
    policy.observe(receipt(root, [1]), context, ledger)
    item = ledger.get(1)
    assert item is not None
    nav = CapabilityCall(
        name="follow_reference",
        arguments={"source_id": item.source_id, "chunk_id": item.chunk_id},
    )
    assert policy.admit(nav, context, ledger) is None
    policy.observe(receipt(nav, [1], status=OutcomeStatus.AMBIGUOUS), context, ledger)
    assert policy.reference_depth(ref("1001", "7", 1), ledger) == 1
    follow = CapabilityCall(
        name="read_named_provision",
        arguments={"source_name": "1001 sayılı Kanun", "article": "7"},
    )
    assert policy.admit(follow, context, ledger) is None
    policy.observe(receipt(follow, [2]), context, ledger)
    assert policy.reference_depth(ref("1002", "11", 2), ledger) == 2


@pytest.mark.parametrize(
    "method,arguments",
    [
        ("read_evidence", {"numbers": [2]}),
        (
            "read_source_range",
            {"source_id": "1001 sayılı İlk Kanun", "start": 50, "limit": 20},
        ),
        (
            "read_provision",
            {
                "source_id": "1001 sayılı İlk Kanun",
                "article": "7",
                "start_position": 50,
            },
        ),
    ],
)
def test_actual_reread_and_continuation_cannot_demote_first_hop(
    method: str, arguments: dict[str, JsonValue]
) -> None:
    context, policy, ledger = make()
    first(context, policy, ledger)
    call = CapabilityCall(name=method, arguments=arguments)
    assert policy.admit(call, context, ledger) is None
    policy.observe(receipt(call, [2], status=OutcomeStatus.PARTIAL), context, ledger)
    assert policy.reference_depth(ref("1002", "11", 2), ledger) == 2
    item = ledger.get(2)
    assert item is not None and item.text == "1002 sayılı İkinci Kanun m.11 incelenir."


@pytest.mark.parametrize(
    "user_question",
    [
        "1002 sayılı İkinci Kanun m.11 sonucunu da açıkla.",
        "İkinci Kanun'un 11. maddesinin sonucunu da açıkla.",
    ],
)
def test_explicit_user_target_remains_independent(user_question: str) -> None:
    context, policy, ledger = make(user_question)
    first(context, policy, ledger)
    assert policy.reference_depth(ref("1002", "11", 2), ledger) == 0
    call = CapabilityCall(
        name="read_named_provision",
        arguments={"source_name": "1002 sayılı Kanun", "article": "11"},
    )
    assert policy.admit(call, context, ledger) is None
    policy.observe(receipt(call, [3]), context, ledger)
    assert (
        next(
            row
            for row in cast(list[dict[str, JsonValue]], policy.export()["sources"])
            if row["citation"] == 3
        )["depth"]
        == 0
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"source_name": "1003 sayılı Başka Kanun", "article": "11"},
        {"source_name": "1002 sayılı İkinci Kanun", "article": "12"},
        {"source_name": "1002 sayılı İkinci Kanun", "article": "geçici 11"},
    ],
)
def test_other_direct_targets_are_not_keyword_blacklisted(
    arguments: dict[str, JsonValue],
) -> None:
    context, policy, ledger = make()
    first(context, policy, ledger)
    assert (
        policy.admit(
            CapabilityCall(name="read_named_provision", arguments=arguments),
            context,
            ledger,
        )
        is None
    )


def judicial_case() -> EvidenceItem:
    # Captured5172 original404: span_736c7df... native tool message14.
    text = "4. 492 sayılı Harçlar Kanunu'nun 13. maddesinin (j) bendi parantez içi hükmü uyarınca alınması gereken harç dahil olmak üzere, yargılama giderlerinin yeniden verilecek kararda karşılanması gerektiğine, 01/12/2025 tarihinde kesin olarak oybirliğiyle karar verildi.\nSE"
    title = "danistay_7_daire_e_2024-2801_k_2025-3985_transit_rejiminde_t1_belgesinde_beyanda_farklilikta_gk_235_gore_ceza_tatbik_edilebilecegi.md"
    item = original(title, "", kind="unknown", text=text)
    item.source_id = "1c00b547-b9e6-4a8f-9ec5-6752d630d893"
    item.chunk_id = "rc_ae5bc6427c47efd6c43d8ffe0e4adb31a595346c"
    assert item.search_doc is not None
    item.search_doc.document_id = item.source_id
    item.search_doc.metadata["regulatory_chunk_id"] = item.chunk_id
    item.metadata = {
        "canonical_metadata": {
            "title": title,
            "article_no": None,
            "clause_label": None,
            "document_date": None,
            "document_type": "unknown",
            "read_as_of_date": "2026-10-13",
            "version_unknown": True,
            "decision_numbers": ["2024/2801", "2025/3985"],
            "legal_dates": ["2025-12-01"],
            "paragraph_no": "4",
            "heading_path": ["D A N I Ş T A Y", "4. 492 sayılı Harçlar Kanunu'nun 13"],
        }
    }
    assert (
        item.text_hash
        == "3710c4e991f8148548065d4012414e2d2b7aef3aa12c32e66756082af90d9fd8"
    )
    return item


def test_actual_unknown_kind_judicial_cost_reference_not_automatic_dependency() -> None:
    context, policy, _ = make()
    ledger = ledger_with(judicial_case())
    saved = copy.deepcopy(ledger.export())
    answer = "Danıştay transit beyan farklılığında ceza koşulları ve tutarı için kararı bozmuştur; bu ihracat vakası için statü şartını ortadan kaldırmaz [1]."
    default = cited_lower_statute_gap(answer, ledger, {1})
    assert default and default["unread_cited_statute_references"]
    assert policy.publication_gap(answer, ledger, {1}) is None
    assert policy.view(context)["unread_reference_navigation"]
    assert cited_lower_statute_gap(answer, ledger, {1}) == default
    assert ledger.export() == saved
    assert native_named_authority_gap(
        "492 sayılı Harçlar Kanunu m.13 gereğince harç alınmaz [1].", ledger
    )


@pytest.mark.parametrize(
    "change", ["no_decision_identity", "not_court_root", "lower_type"]
)
def test_unknown_source_not_silently_relabelled_judicial(change: str) -> None:
    context, policy, _ = make()
    item = judicial_case()
    meta = cast(dict[str, JsonValue], item.metadata["canonical_metadata"])
    if change == "no_decision_identity":
        meta["decision_numbers"] = []
    elif change == "not_court_root":
        meta["heading_path"] = ["Uygulama Tebliği"]
    else:
        meta["document_type"] = "tebliğ"
    assert policy.publication_gap(
        "Kararda bu değerlendirme yapılır [1].", ledger_with(item), {1}
    )


def test_direct_m15_requirement_and_actual_delivery_are_not_bypassed() -> None:
    context, policy, _ = make()
    ledger = ledger_with(
        original(
            "Uygulama Tebliği",
            "1",
            kind="tebliğ",
            text="5326 sayılı Kabahatler Kanunu m.15 uygulanır.",
        )
    )
    answer = "Suç-kabahat önceliği bu hükümde değerlendirilir [1]."
    assert policy.publication_gap(answer, ledger, {1}) == cited_lower_statute_gap(
        answer, ledger, {1}
    )
    assert native_named_authority_gap(
        "5326 sayılı Kabahatler Kanunu m.15/3 uygulanır [1].", ledger
    )
    assert ledger.completely_delivered("writer") == set()


def test_noncanonical_search_receipt_is_ignored_but_reserved_original_is_strict() -> (
    None
):
    context, policy, _ = make()
    external = original("External preview", "1", external=True)
    ledger = ledger_with(external)
    policy.observe(
        receipt(
            CapabilityCall(
                name="search_corpus", arguments={"query": "independent topic"}
            ),
            [1],
        ),
        context,
        ledger,
    )
    assert policy.export()["sources"] == []


def test_named_user_target_allows_explicit_follow_without_depth_reset() -> None:
    context, policy, ledger = make("İkinci Kanun'un 11. maddesi hakkında cevap ver.")
    first(context, policy, ledger)
    item = ledger.get(2)
    assert item is not None
    call = CapabilityCall(
        name="follow_reference",
        arguments={
            "source_id": item.source_id,
            "chunk_id": item.chunk_id,
            "target_source_id": "1002 sayılı İkinci Kanun",
            "article": "11",
            "depth": 4,
        },
    )
    assert policy.admit(call, context, ledger) is None
    policy.observe(
        receipt(call, [3], data={"origin_chunk_id": item.chunk_id}), context, ledger
    )
    assert (
        next(
            row
            for row in cast(list[dict[str, JsonValue]], policy.export()["sources"])
            if row["citation"] == 3
        )["depth"]
        == 0
    )


def test_reserved_noncanonical_original_fails_without_fake_depth_or_delivery() -> None:
    context, policy, ledger = make()
    root = CapabilityCall(
        name="read_provision",
        arguments={"source_id": "Uygulama Tebliği", "article": "1"},
    )
    policy.observe(receipt(root, [1]), context, ledger)
    numbers = ledger.add(
        [original("External preview", "7", external=True)], RunContext()
    )
    call = CapabilityCall(
        name="read_provision",
        arguments={"source_id": "1001 sayılı İlk Kanun", "article": "7"},
    )
    assert policy.reserve_acquisition(call, ref("1001", "7", 1), context, ledger)
    with pytest.raises(ValueError, match="not canonical"):
        policy.observe(receipt(call, numbers), context, ledger)


def test_same_source_different_article_follow_cannot_reset_to_root() -> None:
    context, policy, _ = make()
    ledger = ledger_with(
        original(
            "1001 sayılı İlk Kanun",
            "1",
            text="**MADDE 1-** 1001 sayılı İlk Kanun m.2 incelenir.",
        ),
        original(
            "1001 sayılı İlk Kanun",
            "2",
            text="**MADDE 2-** 1001 sayılı İlk Kanun m.3 incelenir.",
        ),
        original(
            "1001 sayılı İlk Kanun",
            "3",
            text="**MADDE 3-** Özgün şart ve istisna incelenir.",
        ),
    )
    root = CapabilityCall(
        name="read_provision",
        arguments={"source_id": "1001 sayılı İlk Kanun", "article": "1"},
    )
    policy.observe(receipt(root, [1]), context, ledger)
    item = ledger.get(1)
    assert item is not None
    call = CapabilityCall(
        name="follow_reference",
        arguments={
            "source_id": item.source_id,
            "chunk_id": item.chunk_id,
            "target_source_id": item.source_id,
            "article": "2",
            "depth": 0,
        },
    )
    assert policy.admit(call, context, ledger) is None
    policy.observe(
        receipt(call, [2], data={"origin_chunk_id": item.chunk_id}), context, ledger
    )
    assert policy.reference_depth(ref("1001", "3", 2), ledger) == 2
    assert policy.admit(
        CapabilityCall(
            name="read_provision",
            arguments={"source_id": item.source_id, "article": "3"},
        ),
        context,
        ledger,
    )


def test_schema_valid_checkpoint_cannot_relabel_nested_lead_depth() -> None:
    context, policy, ledger = make()
    first(context, policy, ledger)
    saved = policy.export()
    nested = next(
        row
        for row in cast(list[dict[str, JsonValue]], saved["leads"])
        if row["depth"] == 2
    )
    nested["depth"] = 1
    raw = {key: value for key, value in saved.items() if key != "integrity"}
    saved["integrity"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="lead depth changed"):
        policy.restore(saved, context, ledger)


def test_restore_preserves_nested_target_and_actual_source_fence() -> None:
    context, policy, ledger = make()
    first(context, policy, ledger)
    saved = policy.export()
    restored = CroppedCrossReferencePolicy(context, policy.request)
    context.services[SERVICE] = restored
    restored.restore(saved, context, ledger)
    assert restored.export() == saved
    assert restored.admit(
        CapabilityCall(
            name="read_named_provision",
            arguments={"source_name": "1002 sayılı Kanun", "article": "11"},
        ),
        context,
        ledger,
    )
    broken = copy.deepcopy(saved)
    cast(list[dict[str, JsonValue]], broken["sources"])[0]["text_hash"] = "changed"
    raw = {key: value for key, value in broken.items() if key != "integrity"}
    broken["integrity"] = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="original changed"):
        restored.restore(broken, context, ledger)


@pytest.mark.parametrize("change", ["scope", "run", "service", "cheap", "disabled"])
def test_same_run_host_policy_fence(change: str) -> None:
    context, policy, ledger = make()
    if change == "scope":
        context.scope = {"corpus": "foreign"}
    elif change == "run":
        context.run_id = "foreign"
    elif change == "service":
        context.services[SERVICE] = object()
    elif change == "cheap":
        context.services["asv3_exp_cheap"] = True
    else:
        context.services["asv3_exp_cropped"] = False
    with pytest.raises(ValueError):
        policy.filter_references([ref("1001", "7", 1)], ledger, context)


@pytest.mark.parametrize(
    "services",
    [{}, {"asv3_exp_cheap": True}, {"asv3_exp_cropped": True, "asv3_exp_cheap": True}],
)
def test_protected_workflows_cannot_create_policy(services: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CroppedCrossReferencePolicy(RunContext(services=services), "request")
