"""Hosted authority identities use explicit syntax, including fenced legacy resumes."""

import copy
from typing import Any, cast

import pytest
from pydantic import JsonValue

from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext
from tests.unit.onyx.asv3.test_native_authority import original
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    FOCUS,
    outer,
    serial_run,
    session,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record

pytestmark = pytest.mark.usefixtures("empty_source_inventory")

CLAUSE = "6183 sayılı Kanun gecikme zammı oranında gecikme faizi ve m.241/1 kapsamında işlem uygulanır [1][2]."
GENERIC = "8917 sayılı Faaliyet Kanunu hükümlerine göre faiz hesaplanır ve m.47/1 uyarınca işlem uygulanır [1][2]."


def _context(*, scoped: bool = True) -> RunContext:
    return RunContext(
        run_id="owned-authority-resume",
        scope={"tenant": "A", "document_sets": [15]},
        services={
            "research_profile": "experimental",
            "experimental_parallel": False,
            "serial_session_diagnostics": scoped,
            "lean_native_mode": True,
            "task_id": "owned-task",
            "task_outcome_ids": ["material-outcome"],
        },
    )


@pytest.mark.parametrize(
    "clause,number,article", [(CLAUSE, "6183", "241"), (GENERIC, "8917", "47")]
)
def test_fenced_legacy_migration_recomputes_false_article_without_erasing_law(
    clause: str, number: str, article: str
) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    ledger.add([original("Uygulama Tebliği", "3", kind="tebliğ")], ctx)
    old = AuthorityRequirements(ctx, FOCUS)
    assert old.publication_gap(clause, None, ctx, ledger)
    saved = old.export()
    assert any(
        row["article"] == article
        for row in cast(list[dict[str, JsonValue]], saved["records"])
    )
    unchanged = copy.deepcopy(saved)
    migrated = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    migrated.restore(
        saved,
        ctx,
        FOCUS,
        allow_legacy_upgrade=True,
        ledger=ledger,
        retained_answer=clause,
    )
    assert saved == unchanged
    assert migrated.export()["reference_policy"] == "syntactic-v1"
    rows = migrated.view(ctx)["retained_authority_requirements"]
    assert isinstance(rows, list) and rows
    assert all(
        isinstance(row, dict)
        and row["instrument_number"] == number
        and row["article"] is None
        and row["owner"] == "owned-task"
        and row["outcome_ids"] == ["material-outcome"]
        for row in rows
    )
    assert migrated.publication_gap("The law name was deleted [1].", None, ctx, ledger)
    assert any(
        row["instrument_number"] == number
        for row in cast(
            list[dict[str, JsonValue]],
            migrated.view(ctx)["retained_authority_requirements"],
        )
    )


@pytest.mark.parametrize(
    "change", ["scope", "request", "integrity", "identity", "unscoped", "no_opt_in"]
)
def test_upgrade_cannot_bypass_checkpoint_fences_or_profile(change: str) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    old = AuthorityRequirements(ctx, FOCUS)
    assert old.publication_gap(GENERIC, None, ctx, ledger)
    snapshot = old.export()
    target = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    if change == "scope":
        snapshot["scope_hash"] = "wrong"
    elif change == "request":
        snapshot["request_hash"] = "wrong"
    elif change == "integrity":
        snapshot["record_integrity"] = {}
    elif change == "identity":
        rows = cast(list[dict[str, JsonValue]], snapshot["records"])
        rows[0]["requirement_id"] = "authority_" + "0" * 64
    elif change == "unscoped":
        ctx.services["serial_session_diagnostics"] = False
    with pytest.raises(ValueError):
        target.restore(
            snapshot,
            ctx,
            FOCUS,
            allow_legacy_upgrade=change != "no_opt_in",
            ledger=ledger,
            retained_answer=GENERIC,
        )
    assert target.view(ctx)["retained_authority_requirements"] == []


def test_migration_preserves_genuine_explicit_previous_article_after_name_deleted() -> (
    None
):
    ctx, ledger = _context(), EvidenceLedger()
    old = AuthorityRequirements(ctx, FOCUS)
    claim = "4458 sayılı Gümrük Kanunu m.241 uygulanır [1].\n\n" + CLAUSE
    assert old.publication_gap(claim, None, ctx, ledger)
    migrated = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    migrated.restore(
        old.export(),
        ctx,
        FOCUS,
        allow_legacy_upgrade=True,
        ledger=ledger,
        retained_answer=CLAUSE,
    )
    rows = cast(
        list[dict[str, JsonValue]],
        migrated.view(ctx)["retained_authority_requirements"],
    )
    assert {(row["instrument_number"], row["article"]) for row in rows} == {
        ("4458", "241"),
        ("6183", None),
    }
    gap = migrated.publication_gap("The old names were removed.", None, ctx, ledger)
    assert gap and gap["retained_authority_requirements"]


def test_explicit_article_remains_required_even_for_same_instrument_as_old_false_binding() -> (
    None
):
    ctx, ledger = _context(), EvidenceLedger()
    old = AuthorityRequirements(ctx, FOCUS)
    assert old.publication_gap("6183 sayılı Kanun m.241 uygulanır.", None, ctx, ledger)
    upgraded = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    upgraded.restore(old.export(), ctx, FOCUS, allow_legacy_upgrade=True, ledger=ledger)
    rows = cast(
        list[dict[str, JsonValue]],
        upgraded.view(ctx)["retained_authority_requirements"],
    )
    assert len(rows) == 1 and rows[0]["article"] == "241"
    assert upgraded.publication_gap("Name removed.", None, ctx, ledger)


def test_integrity_valid_but_unmappable_legacy_reference_fails_closed() -> None:
    from onyx.asv3.authority_requirements import _digest

    ctx, ledger = _context(), EvidenceLedger()
    old = AuthorityRequirements(ctx, FOCUS)
    assert old.publication_gap(GENERIC, None, ctx, ledger)
    snapshot = old.export()
    rows = cast(list[dict[str, JsonValue]], snapshot["records"])
    rows[0]["reference_text"] = "An unspecified former reference."
    integrity = cast(dict[str, JsonValue], snapshot["record_integrity"])
    integrity[str(rows[0]["requirement_id"])] = _digest(rows[0])
    migrated = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    with pytest.raises(ValueError, match="cannot be recomputed"):
        migrated.restore(snapshot, ctx, FOCUS, allow_legacy_upgrade=True, ledger=ledger)
    assert migrated.view(ctx)["retained_authority_requirements"] == []


@pytest.mark.parametrize("merge", [False, True])
def test_rebuilt_identity_preserves_actual_original_answer_unit_and_first_merge_origin(
    merge: bool,
) -> None:
    ctx, ledger = _context(), EvidenceLedger()
    old = AuthorityRequirements(ctx, FOCUS)
    later = GENERIC.replace("m.47", "m.48")
    if not merge:
        later = later.replace("8917", "7251").replace("Faaliyet", "Veri")
    answer = "\n\n".join(
        ["A supplied fact."] * 14
        + [GENERIC]
        + ["Another supplied fact."] * 18
        + [later]
    )
    assert old.publication_gap(answer, None, ctx, ledger)
    snapshot = old.export()
    old_rows = cast(list[dict[str, JsonValue]], snapshot["records"])
    assert str(old_rows[0]["origin_unit_id"]).startswith("au14-")
    assert str(old_rows[1]["origin_unit_id"]).startswith("au33-")
    migrated = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    migrated.restore(snapshot, ctx, FOCUS, allow_legacy_upgrade=True, ledger=ledger)
    rows = cast(list[dict[str, JsonValue]], migrated.export()["records"])
    assert rows[0]["origin_unit_id"] == old_rows[0]["origin_unit_id"]
    if merge:
        assert len(rows) == 1
    else:
        assert len(rows) == 2
        assert rows[1]["origin_unit_id"] == old_rows[1]["origin_unit_id"]


def test_integrity_valid_but_different_recognized_law_cannot_replace_legacy_dependency() -> (
    None
):
    from onyx.asv3.authority_requirements import _digest

    ctx, ledger = _context(), EvidenceLedger()
    old = AuthorityRequirements(ctx, FOCUS)
    assert old.publication_gap(CLAUSE, None, ctx, ledger)
    snapshot = old.export()
    rows = cast(list[dict[str, JsonValue]], snapshot["records"])
    rows[0]["reference_text"] = "4458 sayılı Kanun m.27 uygulanır."
    integrity = cast(dict[str, JsonValue], snapshot["record_integrity"])
    integrity[str(rows[0]["requirement_id"])] = _digest(rows[0])
    migrated = AuthorityRequirements(ctx, FOCUS, syntactic_reference_binding=True)
    with pytest.raises(ValueError, match="identity cannot be recomputed"):
        migrated.restore(snapshot, ctx, FOCUS, allow_legacy_upgrade=True, ledger=ledger)
    assert migrated.view(ctx)["retained_authority_requirements"] == []


def test_real_hosted_guard_preserves_actual_statute_article_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    ledger.add(
        [
            original("6183 sayılı Kanun", "51"),
            original("4458 sayılı Gümrük Kanunu", "241"),
        ],
        child.context,
    )
    ledger.record_delivery(
        "actual-source-call",
        "owned-task",
        [full_record(ledger, 1), full_record(ledger, 2)],
    )
    assert child.requirements.syntactic_reference_binding is True
    assert child.requirements.export()["reference_policy"] == "syntactic-v1"
    assert child._authority_gap(CLAUSE, "actual-source-call") is None
    assert child.publication_gap(CLAUSE, "actual-source-call") is None
    explicit = "4458 sayılı Gümrük Kanunu m.241 uygulanır [1]."
    gap = child.publication_gap(explicit, "actual-source-call")
    assert gap is not None
    assert any(
        row["instrument_number"] == "4458" and row["article"] == "241"
        for row in cast(list[dict[str, JsonValue]], gap.data["named_authority_gaps"])
    )
    assert (
        child.publication_gap(
            "4458 sayılı Gümrük Kanunu m.241 uygulanır [2].", "actual-source-call"
        )
        is None
    )


def test_legacy_accepted_state_restores_without_changing_immutable_seal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    child.requirements = AuthorityRequirements(child.context, FOCUS)
    child.context.services["authority_requirements"] = child.requirements
    run.calls.clear()
    result = child.run()
    snapshot = child.snapshot()
    old_state = copy.deepcopy(child.source_state())
    old_snap = copy.deepcopy(snapshot)
    run.selected.invoke.side_effect = AssertionError(
        "Accepted replay must not run the model"
    )
    restored = session(run, child.outer_context, ledger)
    try:
        restored.restore(snapshot)
        assert restored.requirements.syntactic_reference_binding is True
        assert restored.requirements.export()["reference_policy"] == "syntactic-v1"
        assert restored.source_state() == old_state
        assert restored.run().summary == result.summary
        assert (
            restored.validate_accepted(
                result.summary, restored.model.last_call_id or "", result.status
            )
            is None
        )
        assert snapshot == old_snap
    finally:
        restored.workers.close()


def test_unaccepted_hosted_resume_exports_rebuilt_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    ledger = EvidenceLedger()
    child = session(run, outer(run), ledger)
    old = AuthorityRequirements(child.context, FOCUS)
    assert old.publication_gap(GENERIC, None, child.context, ledger)
    child.requirements = old
    child.harness.last_draft = GENERIC
    snapshot = child.snapshot()
    restored = session(run, child.outer_context, ledger)
    try:
        restored.restore(snapshot)
        state = cast(
            dict[str, JsonValue], restored.snapshot()["serial_experimental_session"]
        )
        requirements = cast(dict[str, JsonValue], state["authority_requirements"])
        assert requirements["reference_policy"] == "syntactic-v1"
        rows = cast(list[dict[str, JsonValue]], requirements["records"])
        assert rows and all(row["article"] is None for row in rows)
    finally:
        restored.workers.close()


def test_actual_parallel_root_uses_strict_binding_and_preserves_explicit_source_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from onyx.asv3 import runtime
    from onyx.asv3.harness import Harness
    from tests.unit.onyx.asv3.test_experimental_parallel_runtime import parallel_setup
    from tests.unit.onyx.asv3.test_runtime import response

    kwargs, _broker, selected, _checkpoints, _queue, _fence = parallel_setup(
        monkeypatch
    )
    roots: list[Harness] = []
    real_harness = runtime.Harness

    def capture(**arguments: Any) -> Harness:
        harness = real_harness(**arguments)
        roots.append(harness)
        return harness

    monkeypatch.setattr(runtime, "Harness", capture)
    selected.invoke.side_effect = None
    selected.invoke.return_value = response(
        calls=[
            (
                "submit_answer",
                {"answer": "Merhaba!", "basis": "conversation", "_language": "tr"},
            )
        ]
    )
    runtime.run_asv3_loop(**kwargs)
    root = roots[0]
    requirements = root.context.services["authority_requirements"]
    assert isinstance(requirements, AuthorityRequirements)
    assert requirements.syntactic_reference_binding is True
    root.evidence.add(
        [
            original("6183 sayılı Kanun", "51"),
            original("4458 sayılı Gümrük Kanunu", "241"),
        ],
        root.context,
    )
    call = str(root.context.services["last_model_call_id"])
    root.evidence.record_delivery(
        call,
        "coordinator",
        [full_record(root.evidence, 1), full_record(root.evidence, 2)],
    )
    assert root.draft_guard is not None
    assert root.draft_guard(CLAUSE) is None
    gap = root.draft_guard("4458 sayılı Gümrük Kanunu m.241 uygulanır [1].")
    assert gap is not None
    rows = cast(list[dict[str, JsonValue]], gap.data["named_authority_gaps"])
    assert any(
        row["instrument_number"] == "4458" and row["article"] == "241" for row in rows
    )
    assert root.draft_guard("4458 sayılı Gümrük Kanunu m.241 uygulanır [2].") is None
