"""Exact read navigation requires delivered canonical identity, never legal approval."""

import json
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import RunContext, ToolOutcome
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
)
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.models import SourceAction, WorkflowPolicy
from onyx.legal_composite.routing import SourceLaneRouter
from tests.unit.onyx.legal_composite.test_acquisition_read_priority import (
    CHUNK_ID,
    SOURCE_ID,
    original,
    plan,
    registry,
    search_action,
)


def fixture() -> tuple[CanonicalAcquirer, EvidenceLedger, list[SourceKind], list[str]]:
    invoked: list[str] = []

    def forbidden(
        _arguments: dict[str, JsonValue], _context: RunContext
    ) -> ToolOutcome:
        invoked.append("handler")
        raise AssertionError("Read partition dispatched a capability")

    built: list[SourceKind] = []

    def build(kind: SourceKind) -> CapabilityRegistry:
        built.append(kind)
        return registry(forbidden, forbidden)

    catalogue = SourceLaneCatalogue(
        user_id=uuid4(),
        scope_sha256="synthetic-read-partition",
        records=(
            SourceClassification(
                source_id=UUID(SOURCE_ID),
                name="Fictional immutable rules",
                kind=SourceKind.UNKNOWN,
                method="synthetic",
                uncertain=True,
                observed_document_types=(),
            ),
        ),
        complete=True,
    )
    router = SourceLaneRouter(catalogue, build)
    ledger = EvidenceLedger()
    context = RunContext()
    ledger.add(
        [original(CHUNK_ID, "Observed original without the requested article.")],
        context,
    )
    return (
        CanonicalAcquirer(
            registry(forbidden, forbidden),
            context,
            ledger,
            WorkflowPolicy(max_parallel_tools=12, max_tools=60, max_search_calls=48),
            registry_for_action=router.registry,
            expand_actions=router.expand,
        ),
        ledger,
        built,
        invoked,
    )


def read(tool: str, target: str = CHUNK_ID, *, need: str = "condition") -> SourceAction:
    return SourceAction(
        need_ids=[need],
        tool=tool,
        arguments={
            "source_id": SOURCE_ID,
            "article" if tool == "read_provision" else "chunk_id": target,
        },
    )


def test_mixed_followup_preserves_all_queries_and_exact_read_proposals_without_work() -> (
    None
):
    acquirer, ledger, built, invoked = fixture()
    actions = [
        search_action("Fictional basis"),
        read("read_provision", "27"),
        search_action("Fictional exception"),
        read("read_chunk"),
        read("read_chunk_context", need="exception"),
        search_action("Fictional procedure"),
    ]
    frozen = [item.model_dump(mode="json") for item in actions]
    evidence = ledger.serialize_records(ledger.citation_numbers())
    budget = acquirer.context.budget.snapshot()
    selected, remaining = acquirer.safe_observed_read_actions(actions, plan(), {1})
    assert selected == [actions[1], actions[3], actions[4]]
    assert remaining == [actions[0], actions[2], actions[5]]
    assert all(a is b for a, b in zip(selected, [actions[1], actions[3], actions[4]]))
    assert len(built) == 12 and set(built) == set(SourceKind)
    assert acquirer.pending_call_counts(remaining, plan()) == {"search_corpus": 36}
    assert [item.model_dump(mode="json") for item in actions] == frozen
    assert ledger.serialize_records(ledger.citation_numbers()) == evidence
    assert acquirer.context.budget.snapshot() == budget
    assert not invoked and acquirer.search_calls == 0 and acquirer._completed == {}
    assert acquirer.last_receipts == []
    retained = ledger.get(1)
    assert retained is not None and "27" not in retained.text


@pytest.mark.parametrize("delivered", [set(), {9}])
def test_retained_but_not_delivered_original_cannot_authorize_partition(
    delivered: set[int],
) -> None:
    acquirer, _ledger, _built, invoked = fixture()
    actions = [read("read_provision", "27"), read("read_chunk")]
    assert acquirer.safe_observed_read_actions(actions, plan(), delivered) == (
        [],
        actions,
    )
    assert not invoked


def test_context_target_requires_same_delivered_source_and_chunk_pair() -> None:
    acquirer, ledger, _built, invoked = fixture()
    unseen = read("read_chunk_context", str(uuid4()))
    actions = [unseen, read("read_chunk"), read("read_provision", "GEÇİCİ 9")]
    selected, remaining = acquirer.safe_observed_read_actions(actions, plan(), {1})
    assert selected == actions[1:] and remaining == [unseen]
    assert ledger.citation_numbers() == (1,) and not invoked


def test_chunk_identity_on_another_source_does_not_authorize_the_requested_pair() -> (
    None
):
    acquirer, ledger, _built, invoked = fixture()
    other = original(str(uuid4()), "Another source's context.")
    other.source_id = str(uuid4())
    assert other.search_doc is not None
    other.search_doc.document_id = other.source_id
    ledger.add([other], acquirer.context)
    assert other.chunk_id is not None
    action = read("read_chunk_context", other.chunk_id)
    assert acquirer.safe_observed_read_actions([action], plan(), {1, 2}) == (
        [],
        [action],
    )
    assert not invoked


@pytest.mark.parametrize("article", ["", "not an article", "MADDE 27 ve MADDE 28"])
def test_non_single_article_navigation_remains_unexecuted(article: str) -> None:
    acquirer, _ledger, _built, invoked = fixture()
    action = read("read_provision", article)
    assert acquirer.safe_observed_read_actions([action], plan(), {1}) == ([], [action])
    assert not invoked


@pytest.mark.parametrize(
    "fault",
    ["source_binding", "chunk_binding", "hash", "citable", "chunk_missing"],
)
def test_canonical_identity_defects_cannot_select_any_source_read(fault: str) -> None:
    acquirer, ledger, _built, invoked = fixture()
    item = ledger._items[1]
    assert item.search_doc is not None
    if fault == "source_binding":
        item.search_doc.document_id = str(uuid4())
    elif fault == "chunk_binding":
        item.search_doc.metadata["regulatory_chunk_id"] = str(uuid4())
    elif fault == "hash":
        item.text_hash = "0" * 64
    elif fault == "citable":
        item.search_doc = None
    else:
        item.chunk_id = None
    actions = [read("read_provision", "27"), read("read_chunk")]
    assert acquirer.safe_observed_read_actions(actions, plan(), {1}) == ([], actions)
    assert not invoked


@pytest.mark.parametrize("layer", ["original", "search", "canonical"])
@pytest.mark.parametrize("flag", ["external", "derived", "untrusted", "truncated"])
def test_unsafe_flags_in_every_original_layer_prevent_selection(
    layer: str, flag: str
) -> None:
    acquirer, ledger, _built, invoked = fixture()
    item = ledger._items[1]
    assert item.search_doc is not None
    if layer == "original":
        item.metadata[flag] = True
    elif layer == "search":
        item.search_doc.metadata[flag] = "true"
    else:
        item.metadata["canonical_metadata"] = {flag: True}
    action = read("read_chunk_context")
    assert acquirer.safe_observed_read_actions([action], plan(), {1}) == ([], [action])
    assert not invoked


@pytest.mark.parametrize("defect", ["need", "schema", "tool", "scope"])
def test_full_proposal_validation_precedes_any_partition(defect: str) -> None:
    acquirer, ledger, _built, invoked = fixture()
    bad = search_action()
    if defect == "need":
        bad.need_ids = ["unknown"]
    elif defect == "schema":
        bad.arguments["query"] = 27
    elif defect == "tool":
        bad.tool = "invented_capability"
    else:
        bad = read("read_chunk")
        bad.arguments["source_id"] = str(uuid4())
    before = ledger.serialize_records(ledger.citation_numbers())
    with pytest.raises(InvalidSourceAction):
        acquirer.safe_observed_read_actions([read("read_chunk"), bad], plan(), {1})
    assert ledger.serialize_records(ledger.citation_numbers()) == before
    assert not invoked and acquirer.last_receipts == [] and acquirer._completed == {}


def test_completed_and_duplicate_reads_preserve_need_bindings_without_reuse_side_effects() -> (
    None
):
    acquirer, ledger, _built, invoked = fixture()
    first = read("read_chunk")
    first.source_kind = SourceKind.UNKNOWN
    duplicate = first.model_copy(deep=True, update={"need_ids": ["exception"]})
    signature = json.dumps(
        {
            "tool": first.tool,
            "source_kind": first.source_kind,
            "arguments": first.arguments,
        },
        sort_keys=True,
    )
    acquirer._completed[signature] = {"tool": first.tool, "citations": [1]}
    before = ledger.serialize_records(ledger.citation_numbers())
    selected, remaining = acquirer.safe_observed_read_actions(
        [first, duplicate], plan(), {1}
    )
    assert selected == [first, duplicate] and remaining == []
    assert acquirer.pending_call_counts(selected, plan()) == {}
    assert ledger.serialize_records(ledger.citation_numbers()) == before
    assert not invoked and acquirer.last_receipts == []


def test_no_dispatched_read_or_broad_search_is_created_when_no_proposal_exists() -> (
    None
):
    acquirer, ledger, _built, invoked = fixture()
    acquirer.context.research_deadline = 0
    assert acquirer.safe_observed_read_actions([], plan(), {1}) == ([], [])
    assert ledger.citation_numbers() == (1,) and not invoked
