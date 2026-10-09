"""Pending-call estimates use acquisition identity without running acquisition."""

from collections import Counter
from collections.abc import Callable
from math import ceil
from uuid import UUID

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.models import (
    ResearchNeed,
    ResearchPlan,
    SourceAction,
    WorkflowPolicy,
)
from onyx.legal_composite.routing import SourceLaneRouter

SOURCE = UUID("00000000-0000-0000-0000-000000000001")
Handler = Callable[[dict[str, JsonValue], RunContext], ToolOutcome]


def plan() -> ResearchPlan:
    return ResearchPlan(
        language="en",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id=need,
                question=f"What supports {need}?",
                governing_source="Fictional rule",
                conditions_to_check=[],
            )
            for need in ("a", "b")
        ],
        initial_actions=[],
        missing_user_facts=[],
    )


def action(query: str = "Fictional rule", need: str = "a") -> SourceAction:
    return SourceAction(
        need_ids=[need], tool="search_corpus", arguments={"query": query}
    )


def registry(handler: Handler, *, external: bool = False) -> CapabilityRegistry:
    return CapabilityRegistry(
        [
            ToolSpec(
                name="search_corpus",
                description="Synthetic search only.",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "minLength": 1},
                        "expand_query": {"type": "boolean"},
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
                handler=handler,
                external=external,
            ),
            ToolSpec(
                name="read_provision",
                description="Synthetic exact provision only.",
                parameters={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string"},
                        "article": {"type": "string"},
                    },
                    "required": ["source_id", "article"],
                    "additionalProperties": False,
                },
                handler=handler,
            ),
        ]
    )


def lane_acquirer(
    handler: Handler,
) -> tuple[CanonicalAcquirer, list[SourceKind]]:
    built_kinds: list[SourceKind] = []

    def build(kind: SourceKind) -> CapabilityRegistry:
        built_kinds.append(kind)
        return registry(handler)

    catalogue = SourceLaneCatalogue(
        user_id=SOURCE,
        scope_sha256="synthetic-frozen-scope",
        records=(
            SourceClassification(
                source_id=SOURCE,
                name="Fictional unclassified source",
                kind=SourceKind.UNKNOWN,
                method="synthetic",
                uncertain=True,
                observed_document_types=(),
            ),
        ),
        complete=True,
    )
    router = SourceLaneRouter(catalogue, build)
    return (
        CanonicalAcquirer(
            registry(handler),
            RunContext(),
            EvidenceLedger(),
            WorkflowPolicy(max_parallel_tools=12, max_search_calls=48, max_tools=60),
            registry_for_action=router.registry,
            expand_actions=router.expand,
        ),
        built_kinds,
    )


def not_found(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
    return ToolOutcome(
        status=OutcomeStatus.NOT_FOUND,
        summary="Synthetic result; not evidence of actual corpus absence.",
    )


def forbidden(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("Pending-call inspection performed acquisition work")


def test_three_queries_keep_all_twelve_lanes_and_three_worker_waves_without_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquirer, built_kinds = lane_acquirer(not_found)
    actions = [action(query) for query in ("basis", "procedure", "exception")]
    frozen_actions = [item.model_dump(mode="json") for item in actions]
    frozen_plan = plan()
    frozen_plan_dump = frozen_plan.model_dump(mode="json")
    budget = acquirer.context.budget.snapshot()
    with monkeypatch.context() as guard:
        guard.setattr(acquirer, "_dispatch_action", forbidden)
        guard.setattr(acquirer, "_scheduled_calls", forbidden)
        guard.setattr(acquirer, "_bind_originals", forbidden)
        guard.setattr(acquirer.ledger, "add", forbidden)
        guard.setattr(acquirer.ledger, "get", forbidden)
        guard.setattr(acquirer.ledger, "provision_metadata", forbidden)
        guard.setattr(acquirer.context.budget, "snapshot", forbidden)
        counts = acquirer.pending_call_counts(actions, frozen_plan)
    assert counts == {"search_corpus": 36}
    assert ceil(counts["search_corpus"] / acquirer.policy.max_parallel_tools) == 3
    assert len(built_kinds) == 12 and set(built_kinds) == set(SourceKind)
    assert acquirer.context.budget.snapshot() == budget
    assert acquirer.search_calls == 0 and acquirer.last_receipts == []
    assert acquirer.ledger.citation_numbers() == () and acquirer._completed == {}
    assert [item.model_dump(mode="json") for item in actions] == frozen_actions
    assert frozen_plan.model_dump(mode="json") == frozen_plan_dump


def test_estimate_matches_actual_expansion_and_duplicate_scope_merging() -> None:
    executed: list[tuple[dict[str, JsonValue], tuple[str, ...]]] = []

    def execute(arguments: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        stage = context.services["legal_composite_original_stage"]
        executed.append((arguments, stage.need_ids))
        return not_found(arguments, context)

    acquirer, _ = lane_acquirer(execute)
    first = action("same", "a")
    first.arguments.update({"expand_query": True, "_public_update": "First"})
    duplicate = action("same", "b")
    duplicate.arguments.update({"_public_update": "Second", "expand_query": False})
    actions = [first, duplicate, action("distinct")]
    expected = acquirer.pending_call_counts(actions, plan())
    assert executed == [] and expected == {"search_corpus": 24}
    receipts = acquirer.acquire(actions, plan())
    assert Counter(str(row["tool"]) for row in receipts) == expected
    assert len(executed) == 24
    assert all(arguments["expand_query"] is False for arguments, _ in executed)
    assert all(
        scopes == ("a", "b")
        for arguments, scopes in executed
        if arguments["query"] == "same"
    )


def test_completed_calls_are_ignored_without_rebinding_or_receipt_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquirer, _ = lane_acquirer(not_found)
    acquirer.acquire([action("completed")], plan())
    original_receipts = [dict(row) for row in acquirer.last_receipts]
    completed = dict(acquirer._completed)
    budget = acquirer.context.budget.snapshot()
    searches = acquirer.search_calls
    with monkeypatch.context() as guard:
        guard.setattr(acquirer, "_bind_originals", forbidden)
        guard.setattr(acquirer, "_dispatch_action", forbidden)
        guard.setattr(acquirer.ledger, "add", forbidden)
        guard.setattr(acquirer.ledger, "get", forbidden)
        counts = acquirer.pending_call_counts(
            [action("completed", "b"), action("new")], plan()
        )
    assert counts == {"search_corpus": 12}
    assert acquirer._completed == completed
    assert acquirer.last_receipts == original_receipts
    assert acquirer.context.budget.snapshot() == budget
    assert acquirer.search_calls == searches


def test_signature_preserves_source_kind_tool_and_nonpublic_arguments() -> None:
    acquirer, _ = lane_acquirer(not_found)
    provision = SourceAction(
        need_ids=["a"],
        tool="read_provision",
        source_kind=SourceKind.STATUTE,
        arguments={"source_id": str(SOURCE), "article": "7"},
    )
    actions = [
        provision,
        provision.model_copy(update={"source_kind": SourceKind.REGULATION}),
        provision.model_copy(update={"need_ids": ["b"]}),
        provision.model_copy(
            update={"arguments": {"article": "8", "source_id": str(SOURCE)}}
        ),
        action("same"),
        action("same").model_copy(
            update={"arguments": {"query": "same", "_language": "tr"}}
        ),
    ]
    assert acquirer.pending_call_counts(actions, plan()) == {
        "read_provision": 3,
        "search_corpus": 24,
    }


@pytest.mark.parametrize(
    "change",
    [
        {"need_ids": ["unknown"]},
        {"tool": "unknown"},
        {"arguments": {}},
        {"arguments": {"query": 17}},
        {"arguments": {"query": "valid", "unexpected": True}},
    ],
)
def test_invalid_actions_fail_before_dispatch_or_acquisition_mutation(
    change: dict[str, object],
) -> None:
    acquirer, _ = lane_acquirer(not_found)
    malformed = action().model_copy(update=change)
    budget = acquirer.context.budget.snapshot()
    with pytest.raises(InvalidSourceAction):
        acquirer.pending_call_counts([action("valid"), malformed], plan())
    assert acquirer.context.budget.snapshot() == budget
    assert acquirer.search_calls == 0 and acquirer.last_receipts == []
    assert acquirer._completed == {} and acquirer.ledger.citation_numbers() == ()


@pytest.mark.parametrize("flag", ["external", "orchestrates"])
def test_noncanonical_tools_cannot_be_accepted_as_deferred_work(flag: str) -> None:
    tools = registry(not_found)
    spec = tools.get("search_corpus")
    assert spec is not None
    blocked = CapabilityRegistry([spec.model_copy(update={flag: True})])
    acquirer = CanonicalAcquirer(
        blocked, RunContext(), EvidenceLedger(), WorkflowPolicy()
    )
    with pytest.raises(InvalidSourceAction, match="allowlist"):
        acquirer.pending_call_counts([action()], plan())
    assert acquirer.search_calls == 0 and acquirer._completed == {}


def test_no_pending_work_remains_empty_and_does_not_require_available_budget() -> None:
    acquirer, _ = lane_acquirer(not_found)
    acquirer.search_calls = acquirer.policy.max_search_calls
    assert acquirer.pending_call_counts([], plan()) == {}
    assert acquirer.search_calls == acquirer.policy.max_search_calls
