"""New actual delivery may revalidate a cached terminal delivery rejection."""

from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.retained_answer import bind_retained_answer, resolve_retained_answer
from tests.unit.onyx.asv3.test_experimental_workflow import experimental_context
from tests.unit.onyx.asv3.test_legal_source_reviews import (
    deliver,
    lead_id,
    navigation,
    review,
    seen,
)
from tests.unit.onyx.asv3.test_shared_originals import full_record

REQUEST = "Can the rule be applied?"
BODY = (
    "The particular changed wording may affect this outcome. [2]\n\n"
    "The underlying obligation and applicable dates remain distinct. [2]\n\n"
    "The supplied general rule remains available. [1]"
)


def setup(
    *, hosted: bool = False, parallel: bool = True
) -> tuple[Harness, RunContext, EvidenceLedger, LegalSourceReviews, list[str]]:
    context, ledger, reviews = experimental_context()
    context.services.update(
        experimental_parallel=parallel and not hosted,
        serial_session_diagnostics=hosted,
        task_id="owned-question" if hosted else "coordinator",
    )
    seen(context, ledger, reviews)
    handled: list[str] = []

    def submit(arguments: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        handled.append(str(arguments["answer"]))
        current = str(child.services["last_model_call_id"])
        gap = reviews.publication_gap(str(arguments["answer"]), current, child, ledger)
        return gap or ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Validated complete answer"
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description="Validate the complete candidate",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {"type": "string", "enum": ["originals"]},
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
                handler=submit,
            )
            for name in ("submit_answer", "submit_partial_answer")
        ]
    )
    harness = Harness(
        request=REQUEST,
        context=context,
        registry=registry,
        decide=lambda _: Decision(),
        evidence=ledger,
    )
    return harness, context, ledger, reviews, handled


def terminal(
    identifier: str,
    *,
    name: str = "submit_answer",
    body: str = BODY,
    assessment: dict[str, JsonValue] | None = None,
) -> CapabilityCall:
    return CapabilityCall(
        call_id=identifier,
        name=name,
        arguments={
            "answer": body,
            "basis": "originals",
            "_related_source_reviews": [review() if assessment is None else assessment],
        },
    )


def begin(context: RunContext, ledger: EvidenceLedger) -> None:
    deliver(ledger, "before", [1])
    context.services["last_model_call_id"] = "before"


@pytest.mark.parametrize("hosted", [False, True])
@pytest.mark.parametrize("name", ["submit_answer", "submit_partial_answer"])
def test_exact_body_and_review_retry_runs_current_validation_after_full_delivery(
    hosted: bool, name: str
) -> None:
    harness, context, ledger, reviews, handled = setup(hosted=hosted)
    begin(context, ledger)
    before = reviews.export()
    rejected = harness._dispatch([terminal("before-tool", name=name)])[0]
    assert rejected.outcome.status == OutcomeStatus.INVALID
    diagnostic = cast(
        dict[str, JsonValue], rejected.outcome.data["related_source_review_error"]
    )
    assert diagnostic["code"] == "not_fully_delivered"
    assert reviews.export() == before and handled == []
    assert harness.last_draft == BODY

    deliver(ledger, "after", [1, 2])
    context.services["last_model_call_id"] = "after"
    retry = terminal("after-tool", name=name)
    assert retry.arguments == rejected.call.arguments
    accepted = harness._dispatch([retry])[0]
    assert accepted.outcome.status == OutcomeStatus.FOUND
    assert handled == [BODY]
    assert ledger.completely_delivered("after") == {1, 2}
    assert reviews.export() != before


@pytest.mark.parametrize("hosted", [False, True])
def test_persisted_failure_and_cached_refusal_allow_owned_retained_body_retry(
    hosted: bool,
) -> None:
    harness, context, ledger, _, handled = setup(hosted=hosted)
    begin(context, ledger)
    harness._dispatch([terminal("before-tool")])
    still_missing = harness._dispatch([terminal("still-missing")])[0]
    assert still_missing.outcome.summary.startswith("Repeated failed call:")
    snapshot = harness.snapshot()
    restored = Harness(
        request=REQUEST,
        context=context,
        registry=harness.registry,
        decide=lambda _: Decision(),
        evidence=ledger,
    )
    restored.restore(snapshot)
    deliver(ledger, "after", [1, 2])
    context.services["last_model_call_id"] = "after"
    _, descriptor = bind_retained_answer(
        restored.registry.definitions(context),
        context,
        restored.last_draft,
        request=REQUEST,
    )
    assert descriptor is not None
    raw = CapabilityCall(
        call_id="after-tool",
        name="submit_answer",
        arguments={
            "retained_answer_id": descriptor["retained_answer_id"],
            "basis": "originals",
            "_related_source_reviews": [review()],
        },
    )
    decision = resolve_retained_answer(
        Decision(calls=[raw]), context, restored.last_draft, request=REQUEST
    )
    assert decision.calls[0].arguments == terminal("unused").arguments
    assert raw.arguments.get("answer") is None
    assert restored._dispatch(decision.calls)[0].outcome.status == OutcomeStatus.FOUND
    assert handled == [BODY]


@pytest.mark.parametrize("partial", [False, True])
def test_history_only_or_partial_current_delivery_keeps_failed_call_cache(
    partial: bool,
) -> None:
    harness, context, ledger, _, handled = setup(hosted=True)
    begin(context, ledger)
    harness._dispatch([terminal("before-tool")])
    deliver(ledger, "historical-full", [1, 2])
    records = [full_record(ledger, 1)]
    if partial:
        witness = full_record(ledger, 2)
        witness["text"] = str(witness["text"])[:10]
        records.append(witness)
    ledger.record_delivery("current", "asv3_coordinator", records)
    context.services["last_model_call_id"] = "current"
    blocked = harness._dispatch([terminal("after-tool")])[0]
    assert blocked.outcome.summary.startswith("Repeated failed call:")
    assert handled == []


@pytest.mark.parametrize(
    "changes",
    [
        {"gap": "The interaction remains unresolved."},
        {"witnesses": []},
        {"witnesses": [{"citation": 3, "start_char": 0, "end_char": 10}]},
        {"witnesses": [{"citation": 2, "start_char": 0, "end_char": 500}]},
    ],
)
def test_mixed_or_metadata_errors_remain_cached_when_originals_arrive(
    changes: dict[str, JsonValue],
) -> None:
    harness, context, ledger, _, handled = setup(hosted=True)
    begin(context, ledger)
    assessment = review(**changes)
    first = harness._dispatch([terminal("before-tool", assessment=assessment)])[0]
    assert first.outcome.status == OutcomeStatus.INVALID
    deliver(ledger, "after", [1, 2, 3])
    context.services["last_model_call_id"] = "after"
    second = harness._dispatch([terminal("after-tool", assessment=assessment)])[0]
    assert second.outcome.summary.startswith("Repeated failed call:")
    assert handled == []


def test_unknown_persisted_diagnostic_cannot_bypass_failed_call_cache() -> None:
    harness, context, ledger, _, handled = setup(hosted=True)
    begin(context, ledger)
    first = harness._dispatch([terminal("before-tool")])[0]
    diagnostic = cast(
        dict[str, JsonValue], first.outcome.data["related_source_review_error"]
    )
    diagnostic.pop("validation_errors")
    deliver(ledger, "after", [1, 2])
    context.services["last_model_call_id"] = "after"
    assert harness._dispatch([terminal("after-tool")])[0].outcome.summary.startswith(
        "Repeated failed call:"
    )
    assert handled == []


def test_retry_keeps_current_owner_and_answer_retention_validation() -> None:
    harness, context, ledger, _, handled = setup(hosted=True)
    begin(context, ledger)
    harness._dispatch([terminal("before-tool")])
    deliver(ledger, "after", [1, 2])
    context.services["last_model_call_id"] = "after"
    context.services["task_id"] = "foreign-question"
    rejected = harness._dispatch([terminal("after-tool")])[0]
    assert rejected.outcome.data["invalid_related_source_review"] is True
    assert handled == []
    context.services["task_id"] = "owned-question"
    unsupported = "The supplied general rule remains available. [1]"
    gap = harness._dispatch([terminal("missing-effect", body=unsupported)])[0]
    assert gap.outcome.status == OutcomeStatus.PARTIAL
    assert handled == [unsupported]
    assert gap.outcome.data["unretained_examined_source_effects"]


def test_unresolved_sibling_review_without_supplied_witness_does_not_block_retry() -> (
    None
):
    harness, context, ledger, reviews, handled = setup(hosted=True)
    begin(context, ledger)
    reviews.record_delivery("law-call", context, navigation("other-decision"), ledger)
    unresolved = review(
        lead_id=lead_id("other-decision"),
        status="unresolved",
        source_role="unknown",
        gap="The other candidate's operative passage is not yet available.",
    )
    unresolved.pop("witnesses")
    first = terminal("before-tool")
    first.arguments["_related_source_reviews"] = [review(), unresolved]
    assert harness._dispatch([first])[0].outcome.status == OutcomeStatus.INVALID
    deliver(ledger, "after", [1, 2])
    context.services["last_model_call_id"] = "after"
    second = first.model_copy(update={"call_id": "after-tool"})
    outcome = harness._dispatch([second])[0].outcome
    assert outcome.status == OutcomeStatus.PARTIAL
    assert outcome.data["undisclosed_related_source_gaps"]
    assert handled == [BODY]


def test_same_call_id_is_not_replayed_and_plain_serial_cache_is_unchanged() -> None:
    for hosted, parallel in ((True, True), (False, False)):
        harness, context, ledger, _, handled = setup(hosted=hosted, parallel=parallel)
        begin(context, ledger)
        harness._dispatch([terminal("before-tool")])
        deliver(ledger, "after", [1, 2])
        context.services["last_model_call_id"] = "after"
        identifier = "before-tool" if hosted else "after-tool"
        assert harness._dispatch([terminal(identifier)])[0].outcome.summary.startswith(
            "Repeated failed call:"
        )
        assert handled == []
