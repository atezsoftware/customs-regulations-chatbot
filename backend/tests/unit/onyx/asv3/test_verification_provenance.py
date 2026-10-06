"""Auxiliary verification never replaces a parallel decision's source owner."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from onyx.asv3 import runtime
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.models import OutcomeStatus, RunContext, RunStopped
from tests.unit.onyx.asv3.test_native_cache_projection import actual_originals
from tests.unit.onyx.asv3.test_runtime import response, user_payload
from tests.unit.onyx.asv3.test_serial_experimental_session_parity import (
    declaration,
    outer,
    serial_run,
    session,
)
from tests.unit.onyx.asv3.test_shared_originals import original

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


def test_runtime_verify_and_same_decision_terminal_keep_distinct_actual_deliveries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = serial_run(monkeypatch, "conversation", "Merhaba!")
    run.selected.reset_mock()
    ledger = run.harness.evidence
    root_verify = run.harness.context.services["verify_claim"]
    assert callable(root_verify)
    child = session(
        run,
        outer(run),
        ledger,
        verify=lambda context, args: root_verify(args, context),
    )
    first = original("first", "The applicable rule.")
    second = original("second", "The qualifying condition.", source="second")
    ledger.add([first, second], child.context)
    child.harness.evidence_working_set.remember(1, 0, len(first.text))
    child.harness.evidence_working_set.remember(2, 0, len(second.text))
    body = "The rule applies [1], subject to the separate condition [2]."
    semantic_call = ""
    calls = 0

    def invoke(**kwargs: Any) -> Any:
        nonlocal calls, semantic_call
        calls += 1
        if calls == 1:
            assert {row["citation"] for row in actual_originals(kwargs["prompt"])} == {
                1,
                2,
            }
            return response(
                calls=[
                    (
                        "verify_claim",
                        {"claim": "The rule applies [1].", "citations": [1]},
                    ),
                    (
                        "submit_answer",
                        {
                            "answer": body,
                            "basis": "originals",
                            "_language": "tr",
                            "_outcomes": [declaration()],
                        },
                    ),
                ]
            )
        assert calls == 2
        semantic_call = child.model.last_call_id or ""
        assert child.context.services["last_model_call_id"] == semantic_call
        evidence = json.loads(user_payload(kwargs["prompt"][-1])["evidence"])
        assert [row["citation"] for row in evidence] == [1]
        return response(
            json.dumps(
                {
                    "status": "supported",
                    "explanation": "The selected original supports the claim.",
                    "required_conditions": [],
                    "missing_conditions": [],
                    "evidence_numbers": [1],
                    "safe_to_publish": True,
                }
            )
        )

    run.selected.invoke.side_effect = invoke
    result = child.run()
    assert result.status == OutcomeStatus.FOUND and result.summary == body
    assert (
        child.model.last_call_id
        == child.context.services["last_model_call_id"]
        == semantic_call
    )
    assert ledger.completely_delivered(semantic_call) == {1, 2}
    rows = ledger.export()["deliveries"]
    assert isinstance(rows, list)
    verifier_calls = {
        row["call_id"]
        for row in rows
        if isinstance(row, dict) and row["call_id"] != semantic_call
    }
    assert len(verifier_calls) == 1
    assert ledger.completely_delivered(str(next(iter(verifier_calls)))) == {1}
    assert child.validate_accepted(body, semantic_call, OutcomeStatus.FOUND) is None
    assert run.selected.invoke.call_count == 2


@pytest.mark.parametrize("hosted", [False, True])
def test_concurrent_verification_contexts_preserve_shared_objects_and_cancel(
    hosted: bool,
) -> None:
    ledger = EvidenceLedger()
    context = RunContext(
        services={
            "research_profile": "experimental",
            "experimental_parallel": not hosted,
            "serial_session_diagnostics": hosted,
            "lean_native_mode": True,
            "task_id": "owned" if hosted else "",
            "last_model_call_id": "semantic-owner",
            "evidence": ledger,
        }
    )
    reviews = LegalSourceReviews(context, "Explain the condition.")
    context.services["legal_source_reviews"] = reviews
    barrier = threading.Barrier(2)

    def worker(identity: str) -> str:
        local = runtime._verification_context(context)
        assert local is not context and local.services is not context.services
        assert local.scope is context.scope and local.budget is context.budget
        assert local.deadline == context.deadline and local.depth == context.depth
        assert local.services["evidence"] is ledger
        assert local.services["legal_source_reviews"] is reviews
        local.services["last_model_call_id"] = identity
        barrier.wait(2)
        assert context.services["last_model_call_id"] == "semantic-owner"
        return str(local.services["last_model_call_id"])

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(worker, identity) for identity in ("verify-a", "verify-b")
        ]
        assert [future.result(timeout=3) for future in futures] == [
            "verify-a",
            "verify-b",
        ]
    local = runtime._verification_context(context)
    context.cancel()
    with pytest.raises(RunStopped, match="cancel"):
        local.check_active()
    assert context.services["last_model_call_id"] == "semantic-owner"


@pytest.mark.parametrize("profile", ["experimental", "normal", "deep"])
def test_plain_verification_context_is_unchanged(profile: str) -> None:
    context = RunContext(
        services={"research_profile": profile, "experimental_parallel": False}
    )
    assert runtime._verification_context(context) is context
