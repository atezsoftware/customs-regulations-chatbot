"""Queue priority uses dispatcher schemas without modifying a capability call."""

from collections.abc import Callable
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext, ToolOutcome
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import (
    AcquisitionCall,
    CanonicalAcquirer,
    InvalidSourceAction,
)
from onyx.legal_composite.models import SourceAction, WorkflowPolicy
from onyx.legal_composite.routing import SourceLaneRouter
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    CHUNK_ID,
    SOURCE_ID,
    original,
    plan,
    read_action,
    registry,
    search_action,
)

CONTROL_METADATA: dict[str, JsonValue] = {
    "_public_update": ["Reading the located rule", "Checking its original condition"],
    "_need_id": "condition",
    "_language": "en",
    "_notifications": True,
    "_external_requested": False,
    "_outcomes": [],
    "_coverage": {},
}


def fixture(*, prioritize: bool = True) -> CanonicalAcquirer:
    def forbidden(
        _arguments: dict[str, JsonValue], _context: RunContext
    ) -> ToolOutcome:
        raise AssertionError("Priority validation executed a capability")

    ledger = EvidenceLedger()
    context = RunContext()
    ledger.add([original(CHUNK_ID, "Observed immutable fictional original.")], context)
    return CanonicalAcquirer(
        registry(forbidden, forbidden),
        context,
        ledger,
        WorkflowPolicy(),
        prioritize_observed_reads=prioritize,
    )


def calls(actions: list[SourceAction]) -> list[AcquisitionCall]:
    return [
        (
            action,
            CapabilityCall(name=action.tool, arguments=dict(action.arguments)),
            f"existing-signature-{index}",
        )
        for index, action in enumerate(actions)
    ]


@pytest.mark.parametrize("metadata_key", list(CONTROL_METADATA))
@pytest.mark.parametrize("tool", ["read_provision", "read_chunk", "read_chunk_context"])
def test_permitted_control_metadata_keeps_exact_read_priority_and_call_identity(
    metadata_key: str, tool: str
) -> None:
    acquirer = fixture()
    read = read_action()
    if tool != "read_provision":
        read.tool = tool
        read.arguments = {"source_id": SOURCE_ID, "chunk_id": CHUNK_ID}
    read.arguments[metadata_key] = CONTROL_METADATA[metadata_key]
    batch = calls([search_action(), read])
    before = [
        (action.model_dump(mode="json"), call.model_dump(mode="json"), signature)
        for action, call, signature in batch
    ]
    budget = acquirer.context.budget.snapshot()
    pending = acquirer.pending_call_counts([batch[0][0], read], plan())

    result = acquirer._scheduled_calls(batch)

    assert result == [batch[1], batch[0]]
    assert result[0] is batch[1] and result[1] is batch[0]
    assert [
        (action.model_dump(mode="json"), call.model_dump(mode="json"), signature)
        for action, call, signature in batch
    ] == before
    assert pending == {"search_corpus": 1, tool: 1}
    assert read.need_ids == ["condition"]
    assert acquirer.context.budget.snapshot() == budget
    assert acquirer.ledger.citation_numbers() == (1,) and acquirer._completed == {}


def test_unknown_argument_still_prevents_priority_and_dispatch_execution() -> None:
    acquirer = fixture()
    read = read_action()
    read.arguments.update(
        {
            "_public_update": CONTROL_METADATA["_public_update"],
            "unknown_argument": "not permitted",
        }
    )
    batch = calls([search_action(), read])
    assert acquirer._scheduled_calls(batch) == batch
    outcome = acquirer.registry.dispatch(batch[1][1], acquirer.context)
    assert outcome.status == OutcomeStatus.INVALID
    assert acquirer.ledger.citation_numbers() == (1,)


@pytest.mark.parametrize("enabled", [False, True])
def test_related_review_control_metadata_is_stripped_only_when_feature_enabled(
    enabled: bool,
) -> None:
    acquirer = fixture()
    if enabled:
        acquirer.context.services["research_profile"] = "experimental"
    read = read_action()
    read.arguments["_related_source_reviews"] = []
    batch = calls([search_action(), read])
    assert acquirer._scheduled_calls(batch) == (
        [batch[1], batch[0]] if enabled else batch
    )
    # Ordering never permits related-review metadata on a nonterminal source action.
    assert (
        acquirer.registry.dispatch(batch[1][1], acquirer.context).status
        == OutcomeStatus.INVALID
    )
    assert acquirer.ledger.citation_numbers() == (1,)


def test_default_off_order_is_the_exact_original_batch_even_with_control_metadata() -> (
    None
):
    acquirer = fixture(prioritize=False)
    read = read_action()
    read.arguments.update(CONTROL_METADATA)
    batch = calls([search_action(), read])
    assert acquirer._scheduled_calls(batch) is batch


def test_unobserved_source_does_not_gain_priority_from_control_metadata() -> None:
    acquirer = fixture()
    read = read_action()
    read.arguments.update({"source_id": str(uuid4()), **CONTROL_METADATA})
    batch = calls([search_action(), read])
    assert acquirer._scheduled_calls(batch) == batch


def test_actual_dispatch_preserves_full_original_bindings_and_all_twelve_searches() -> (
    None
):
    context = RunContext()
    ledger = EvidenceLedger()
    body = "Complete immutable fictional original.\n" * 20
    ledger.add([original(CHUNK_ID, body)], context)
    received: list[tuple[str, dict[str, JsonValue]]] = []

    def search_for(
        kind: SourceKind,
    ) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
        def search(
            arguments: dict[str, JsonValue], _context: RunContext
        ) -> ToolOutcome:
            received.append((kind.value, arguments))
            return ToolOutcome(
                status=OutcomeStatus.NOT_FOUND, summary="Bounded lane result."
            )

        return search

    def read(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        received.append(("read", arguments))
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Exact original.",
            evidence=[original(CHUNK_ID, body)],
        )

    catalogue = SourceLaneCatalogue(
        user_id=uuid4(),
        scope_sha256="synthetic-control-priority",
        records=(
            SourceClassification(
                source_id=UUID(SOURCE_ID),
                name="Fictional original",
                kind=SourceKind.UNKNOWN,
                method="synthetic",
                uncertain=True,
                observed_document_types=(),
            ),
        ),
        complete=True,
    )
    router = SourceLaneRouter(catalogue, lambda kind: registry(search_for(kind), read))
    acquirer = CanonicalAcquirer(
        registry(search_for(SourceKind.UNKNOWN), read),
        context,
        ledger,
        WorkflowPolicy(max_parallel_tools=1, max_search_calls=12),
        expand_actions=router.expand,
        registry_for_action=router.registry,
        prioritize_observed_reads=True,
    )
    action = read_action()
    action.arguments.update(
        {
            key: value
            for key, value in CONTROL_METADATA.items()
            if key not in {"_outcomes", "_coverage"}
        }
    )
    actions = [search_action(), action]
    before = [item.model_dump(mode="json") for item in actions]

    receipts = acquirer.acquire(actions, plan())

    assert received[0] == ("read", {"source_id": SOURCE_ID, "article": "17"})
    assert {kind for kind, _arguments in received[1:]} == {
        kind.value for kind in SourceKind
    }
    assert len(received) == 13 and len(receipts) == 13 and acquirer.search_calls == 12
    assert all(
        arguments == {"query": actions[0].arguments["query"], "expand_query": False}
        for _kind, arguments in received[1:]
    )
    assert [item.model_dump(mode="json") for item in actions] == before
    retained = ledger.get(1)
    assert retained is not None and retained.text == body
    assert retained.source_id == SOURCE_ID and retained.chunk_id == CHUNK_ID
    assert retained.question_ids == ["condition"]
    assert ledger.citation_numbers() == (1,)
    assert next(row for row in receipts if row["tool"] == "read_provision")[
        "need_ids"
    ] == ["condition"]
    wrong_scope = action.model_copy(deep=True)
    wrong_scope.arguments["source_id"] = str(uuid4())
    with pytest.raises(InvalidSourceAction, match="authorized candidate scope"):
        acquirer.acquire([wrong_scope], plan())
    wrong_need = action.model_copy(deep=True, update={"need_ids": ["unplanned"]})
    with pytest.raises(InvalidSourceAction, match="unknown need"):
        acquirer.acquire([wrong_need], plan())
    assert len(received) == 13 and acquirer.search_calls == 12
    assert ledger.citation_numbers() == (1,)
