"""Retain rejected source dependencies without approving their legal conclusions."""

import copy

import pytest

from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext
from tests.unit.onyx.asv3.test_native_authority import ledger_with, original
from tests.unit.onyx.asv3.test_shared_originals import full_record

CLAIM = "8917 sayılı Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
ANONYMOUS_CLAIM = "İzin gerekir [1]."
NOTICE = "8917 sayılı Faaliyet Kanunu'nun 27. maddesinin özgün metni incelenemedi."


def setup() -> tuple[RunContext, EvidenceLedger, AuthorityRequirements]:
    context = RunContext(scope={"authorized": "owned"})
    ledger = ledger_with(original("Uygulama Tebliği", "3", kind="tebliğ", text=CLAIM))
    ledger.record_delivery("first", "asv3_coordinator", [full_record(ledger, 1)])
    state = AuthorityRequirements(context, "What permission is required?")
    return context, ledger, state


def read_original(
    ledger: EvidenceLedger, context: RunContext, item: EvidenceItem
) -> int:
    ledger.add([item], context)
    citation = max(ledger.citation_mapping())
    ledger.record_delivery("next", "asv3_coordinator", [full_record(ledger, citation)])
    return citation


def test_rejected_governing_basis_survives_name_deletion_and_is_deduplicated() -> None:
    context, ledger, state = setup()
    for answer in (CLAIM, CLAIM, ANONYMOUS_CLAIM):
        gap = state.publication_gap(answer, "first", context, ledger)
        assert gap is not None
        retained = gap["retained_authority_requirements"]
        assert isinstance(retained, list)
        assert len(retained) == 1
    records = state.export()["records"]
    assert isinstance(records, list) and len(records) == 1


def test_existing_matching_original_must_be_inline_and_fully_delivered() -> None:
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    citation = read_original(
        ledger, context, original("8917 SAYILI FAALİYET KANUNU", "27")
    )
    supported = f"İzin gerekir [{citation}]."
    assert state.publication_gap(supported, None, context, ledger)
    assert state.publication_gap(supported, "first", context, ledger)
    assert state.publication_gap(ANONYMOUS_CLAIM, "next", context, ledger)
    assert state.publication_gap(supported, "next", context, ledger) is None
    # A later unsupported revision reopens the same dependency.
    assert state.publication_gap(ANONYMOUS_CLAIM, "first", context, ledger)


def test_verified_formal_identity_does_not_require_a_number_in_source_heading() -> None:
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    citation = read_original(ledger, context, original("FAALİYET KANUNU", "27"))
    assert (
        state.publication_gap(f"İzin gerekir [{citation}].", "next", context, ledger)
        is None
    )


def test_alias_only_rejection_retains_its_recognized_instrument_identity() -> None:
    context, ledger, state = setup()
    alias = "Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
    assert state.publication_gap(alias, "first", context, ledger)
    citation = read_original(ledger, context, original("FAALİYET KANUNU", "27"))
    assert (
        state.publication_gap(f"İzin gerekir [{citation}].", "next", context, ledger)
        is None
    )


def test_numberless_formal_dependency_can_close_with_new_numbered_original() -> None:
    context = RunContext(scope={"authorized": "owned"})
    ledger = ledger_with(
        original("Uygulama Tebliği", "3", kind="tebliğ"),
        original("FAALİYET KANUNU", "28"),
    )
    ledger.record_delivery("first", "asv3_coordinator", [full_record(ledger, 1)])
    state = AuthorityRequirements(context, "What permission is required?")
    alias = "Faaliyet Kanunu'nun 27. maddesi uyarınca izin gerekir [1]."
    assert state.publication_gap(alias, "first", context, ledger)
    rows = state.export()["records"]
    assert isinstance(rows, list) and isinstance(rows[0], dict)
    assert rows[0]["instrument_number"] is None
    citation = read_original(
        ledger, context, original("8917 SAYILI FAALİYET KANUNU", "27")
    )
    assert (
        state.publication_gap(f"İzin gerekir [{citation}].", "next", context, ledger)
        is None
    )


def test_explicit_numbered_dependency_rejects_conflicting_number_with_same_name() -> (
    None
):
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    citation = read_original(
        ledger, context, original("8918 SAYILI FAALİYET KANUNU", "27")
    )
    assert state.publication_gap(f"İzin gerekir [{citation}].", "next", context, ledger)


@pytest.mark.parametrize("wrong", ["article", "instrument", "external", "derived"])
def test_an_unrelated_or_noncanonical_original_cannot_close_requirement(
    wrong: str,
) -> None:
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    item = original(
        "4451 SAYILI BAŞKA KANUN"
        if wrong == "instrument"
        else "8917 SAYILI FAALİYET KANUNU",
        "26" if wrong == "article" else "27",
        external=wrong == "external",
    )
    if wrong == "derived":
        item.metadata["derived"] = True
    citation = read_original(ledger, context, item)
    assert state.publication_gap(f"İzin gerekir [{citation}].", "next", context, ledger)


def test_withdrawn_unsupported_result_can_disclose_gap_and_retain_other_detail() -> (
    None
):
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    partial = "Uygulama Tebliği'nde başvuru belgesi belirtilmiştir [1].\n\n" + NOTICE
    assert state.publication_gap(partial, "first", context, ledger) is None
    assert state.view(context)["retained_authority_requirements"]


@pytest.mark.parametrize(
    "notice",
    [
        "Kaynaklar incelenemedi.",
        NOTICE.replace("27", "28"),
        NOTICE.replace("8917", "8918"),
        NOTICE + " [1]",
        "İzin gerekir; " + NOTICE,
        NOTICE + " İzin zorunludur.",
    ],
)
def test_generic_wrong_or_claim_combined_notice_does_not_close_requirement(
    notice: str,
) -> None:
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    assert state.publication_gap(
        "Belge bilgisi [1].\n\n" + notice, "first", context, ledger
    )


def test_unrejected_names_and_conversation_do_not_create_requirements() -> None:
    context, ledger, state = setup()
    assert state.publication_gap("Merhaba!", "first", context, ledger) is None
    assert state.view(context)["retained_authority_requirements"] == []


def test_requirements_are_owner_scoped_with_actual_assignment_outcome_ids() -> None:
    context, ledger, state = setup()
    context.services.update(task_id="worker-one", task_outcome_ids=["permission"])
    assert state.publication_gap(CLAIM, "first", context, ledger)
    visible = state.view(context)["retained_authority_requirements"]
    assert isinstance(visible, list) and isinstance(visible[0], dict)
    assert visible[0]["outcome_ids"] == ["permission"]
    context.services["task_id"] = "worker-two"
    assert state.view(context)["retained_authority_requirements"] == []
    assert state.publication_gap(ANONYMOUS_CLAIM, "first", context, ledger) is None


def test_checkpoint_restores_retained_gap_without_past_draft_copy() -> None:
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    snapshot = state.export()
    restored = AuthorityRequirements(context, "What permission is required?")
    restored.restore(snapshot, context, "What permission is required?")
    assert restored.publication_gap(ANONYMOUS_CLAIM, "first", context, ledger)
    assert restored.export() == snapshot
    assert "draft" not in snapshot


@pytest.mark.parametrize("change", ["run", "scope", "request", "id", "formal_name"])
def test_checkpoint_and_runtime_fences_reject_changed_identity(change: str) -> None:
    context, ledger, state = setup()
    assert state.publication_gap(CLAIM, "first", context, ledger)
    snapshot = copy.deepcopy(state.export())
    if change == "run":
        context.run_id = "another-run"
    elif change == "scope":
        context.scope = {"authorized": "different"}
    elif change == "request":
        snapshot["request_hash"] = "changed"
    else:
        rows = snapshot["records"]
        assert isinstance(rows, list) and isinstance(rows[0], dict)
        if change == "formal_name":
            rows[0]["formal_name"] = "veri kanun"
        else:
            rows[0]["requirement_id"] = "authority_" + "0" * 64
    with pytest.raises(ValueError):
        state.restore(snapshot, context, "What permission is required?")


def test_generic_unqualified_notice_cannot_close_a_qualified_article() -> None:
    context, ledger, state = setup()
    qualified = CLAIM.replace("27. maddesi", "GEÇİCİ MADDE 27")
    assert state.publication_gap(qualified, "first", context, ledger)
    assert state.publication_gap("Belge [1].\n\n" + NOTICE, "first", context, ledger)
