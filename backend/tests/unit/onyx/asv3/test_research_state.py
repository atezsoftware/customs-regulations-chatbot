"""Research obligations and original working context survive adaptive decisions."""

import json
import threading

import pytest
from pydantic import JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import (
    NeedVerification,
    QuestionVerification,
    VerificationResult,
    publication_review_inventory,
)
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    TaskStatus,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.publication import publication_gap
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.research_state import (
    EvidenceWorkingSet,
    ResearchNeed,
    ResearchState,
    ResearchUpdate,
    build_research_specs,
)
from onyx.asv3.workers import WorkerPool
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc


def state_pair() -> tuple[RunContext, EvidenceLedger, ResearchState]:
    context = RunContext(scope={"source": "owned"})
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="law",
                chunk_id="a",
                text="Condition, exception and later settlement.",
                search_doc=SearchDoc(
                    document_id="law",
                    chunk_ind=1,
                    semantic_identifier="Original rule",
                    blurb="",
                    source_type=DocumentSource.USER_FILE,
                    boost=1,
                    hidden=False,
                    metadata={},
                    match_highlights=[],
                ),
            ),
            EvidenceItem(source_id="law", chunk_id="b", text="Other operative rule."),
        ],
        context,
    )
    state = ResearchState(["Result?", "Procedure and later settlement?"], context)
    context.services.update(evidence=ledger, research_state=state)
    return context, ledger, state


def need(key: str = "basis", **changes: object) -> dict[str, object]:
    return {
        "need_id": key,
        "question_ids": ["q0"],
        "purpose": "Find the governing basis",
        "completion_test": "Operative original plus applicable exception",
        **changes,
    }


def test_determination_binding_cannot_move_to_another_question() -> None:
    context, ledger, state = state_pair()
    before = state.export()
    with pytest.raises(ValueError, match="original question IDs"):
        state.update(
            ResearchUpdate.model_validate(
                {"needs": [need(determination_ids=["q1:d0"])]}
            ),
            ledger,
        )
    assert state.export() == before
    state.update(
        ResearchUpdate.model_validate({"needs": [need(determination_ids=["q0:d0"])]}),
        ledger,
    )
    restored = ResearchState(list(state.questions), context)
    restored.restore(state.export(), ledger)
    assert restored.export() == state.export()


@pytest.mark.parametrize("binding", [None, "unknown", "excluded", "incidental"])
def test_source_action_binding_is_checked_before_io_and_budget(
    binding: str | None,
) -> None:
    context, ledger, old_state = state_pair()
    state = ResearchState(
        list(old_state.questions), context, require_need_bindings=True
    )
    context.services["research_state"] = state
    state.update(
        ResearchUpdate.model_validate(
            {
                "needs": [
                    need(
                        "excluded",
                        status="out_of_scope",
                        gap="Unrelated to this request",
                    ),
                    need("incidental", material=False),
                ]
            }
        ),
        ledger,
    )
    executed: list[str] = []
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_original",
                description="Read an original",
                parameters={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
                requires_research_need=True,
                handler=lambda _args, _context: (
                    executed.append("read")
                    or ToolOutcome(status=OutcomeStatus.FOUND, summary="Original")
                ),
            )
        ]
    )
    before = context.budget.snapshot()
    result = registry.dispatch(
        CapabilityCall(
            name="read_original", arguments={"_need_id": binding} if binding else {}
        ),
        context,
    )
    assert result.status == OutcomeStatus.INVALID
    assert "completion test" in str(result.data)
    assert not executed and context.budget.snapshot() == before
    definition = registry.definitions(context)[0]["function"]
    assert isinstance(definition, dict) and isinstance(definition["parameters"], dict)
    assert "_need_id" in definition["parameters"]["required"]


def test_need_update_and_parallel_original_reads_share_one_decision() -> None:
    context, ledger, old_state = state_pair()
    state = ResearchState(
        list(old_state.questions), context, require_need_bindings=True
    )
    context.services["research_state"] = state
    registry = CapabilityRegistry(build_research_specs(state, ledger))
    barrier = threading.Barrier(2)
    entered: list[str] = []

    def read(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        assert state.has_need("basis")
        entered.append(str(args["anchor"]))
        barrier.wait(timeout=2)
        return ToolOutcome(status=OutcomeStatus.FOUND, summary="Original retained")

    registry.register(
        ToolSpec(
            name="read_original",
            description="Read selected original",
            requires_research_need=True,
            parameters={
                "type": "object",
                "properties": {"anchor": {"type": "string"}},
                "required": ["anchor"],
                "additionalProperties": False,
            },
            handler=read,
        )
    )
    decisions = iter(
        [
            Decision(
                calls=[
                    CapabilityCall(
                        name="read_original",
                        arguments={"anchor": "a", "_need_id": "basis"},
                    ),
                    CapabilityCall(
                        name="update_research",
                        arguments={"needs": [need(question_ids=["q0", "q1"])]},
                    ),
                    CapabilityCall(
                        name="read_original",
                        arguments={"anchor": "b", "_need_id": "basis"},
                    ),
                ]
            ),
            Decision(answer="Research completed"),
        ]
    )
    result = Harness(
        request="Resolve both questions",
        context=context,
        registry=registry,
        decide=lambda _view: next(decisions),
        evidence=ledger,
    ).run()
    assert result.status == OutcomeStatus.FOUND
    assert sorted(entered) == ["a", "b"]
    assert state.uncovered_questions() == []
    assert [r.call.name for r in result.receipts] == [
        "update_research",
        "read_original",
        "read_original",
    ]


def test_cached_original_read_cannot_bypass_current_need_binding() -> None:
    context, ledger, old_state = state_pair()
    state = ResearchState(
        list(old_state.questions), context, require_need_bindings=True
    )
    context.services["research_state"] = state
    state.update(ResearchUpdate.model_validate({"needs": [need()]}), ledger)
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    for spec in build_research_specs(state, ledger):
        registry.register(spec)
    decisions = iter(
        [
            Decision(
                calls=[
                    CapabilityCall(
                        name="read_evidence",
                        arguments={"citation": 1, "_need_id": "basis"},
                    )
                ]
            ),
            Decision(
                calls=[
                    CapabilityCall(
                        name="update_research",
                        arguments={
                            "needs": [
                                need(status="out_of_scope", gap="No longer material")
                            ]
                        },
                    ),
                    CapabilityCall(
                        name="read_evidence",
                        arguments={"citation": 1, "_need_id": "basis"},
                    ),
                ]
            ),
            Decision(answer="Done"),
        ]
    )
    result = Harness(
        request="Read an original",
        context=context,
        registry=registry,
        decide=lambda _view: next(decisions),
        evidence=ledger,
    ).run()
    reads = [r for r in result.receipts if r.call.name == "read_evidence"]
    assert [r.outcome.status for r in reads] == [
        OutcomeStatus.FOUND,
        OutcomeStatus.INVALID,
    ]
    assert reads[-1].evidence_ids == []


def test_research_binding_failure_can_recover_after_the_need_is_recorded() -> None:
    context, ledger, old_state = state_pair()
    state = ResearchState(
        list(old_state.questions), context, require_need_bindings=True
    )
    context.services["research_state"] = state
    registry = CapabilityRegistry()
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    for spec in build_research_specs(state, ledger):
        registry.register(spec)
    arguments: dict[str, JsonValue] = {"citation": 1, "_need_id": "basis"}
    decisions = iter(
        [
            Decision(calls=[CapabilityCall(name="read_evidence", arguments=arguments)]),
            Decision(
                calls=[
                    CapabilityCall(
                        name="update_research", arguments={"needs": [need()]}
                    ),
                    CapabilityCall(name="read_evidence", arguments=arguments),
                ]
            ),
            Decision(answer="Done"),
        ]
    )
    result = Harness(
        request="Read an original",
        context=context,
        registry=registry,
        decide=lambda _view: next(decisions),
        evidence=ledger,
    ).run()
    reads = [r for r in result.receipts if r.call.name == "read_evidence"]
    assert [r.outcome.status for r in reads] == [
        OutcomeStatus.INVALID,
        OutcomeStatus.FOUND,
    ]
    assert reads[-1].evidence_ids == [1]


@pytest.mark.parametrize(
    "update",
    [
        {"needs": [need(question_ids=["q9"])]},
        {"needs": [need(depends_on=["missing"])]},
        {
            "needs": [
                need(depends_on=["procedure"]),
                need("procedure", depends_on=["basis"]),
            ]
        },
        {"needs": [need(evidence_numbers=[999])]},
        {"needs": [need(status="out_of_scope")]},
        {
            "needs": [need()],
            "findings": [
                {
                    "finding_id": "f",
                    "need_id": "basis",
                    "statement": "Rule",
                    "kind": "rule",
                    "witnesses": [{"citation": 1, "end_char": 9000}],
                }
            ],
        },
    ],
)
def test_rejected_update_is_atomic(update: dict[str, object]) -> None:
    _, ledger, state = state_pair()
    before = state.export()
    with pytest.raises(ValueError):
        state.update(ResearchUpdate.model_validate(update), ledger)
    assert state.export() == before


def test_original_question_ids_and_witnesses_survive_large_checkpoint() -> None:
    context, ledger, state = state_pair()
    for start in (0, 20):
        state.update(
            ResearchUpdate(
                needs=[
                    ResearchNeed.model_validate(need(f"n{i}"))
                    for i in range(start, start + 20)
                ]
            ),
            ledger,
        )
    restored = ResearchState(list(state.questions), context)
    restored.restore(state.export(), ledger)
    assert restored.export() == state.export()
    wrong = ResearchState(["different request"], context)
    with pytest.raises(ValueError, match="mismatch"):
        wrong.restore(state.export(), ledger)
    with pytest.raises(ValueError, match="different questions"):
        state.update(
            ResearchUpdate.model_validate({"needs": [need("n0", question_ids=["q1"])]}),
            ledger,
        )


def test_working_set_stores_locators_and_recovers_complete_original_units() -> None:
    _, ledger, _ = state_pair()
    working = EvidenceWorkingSet()
    first, second = ledger.get(1), ledger.get(2)
    assert first is not None and second is not None
    working.remember(1, 0, len(first.text))
    working.remember(2, 0, len(second.text))
    working.remember(1, 0, 10)
    assert working.export() == [
        [2, 0, len(second.text)],
        [1, 0, len(first.text)],
    ]
    current = working.view(ledger, preferred=[1])
    records = current["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    assert records[0]["text"] == first.text
    assert records[0]["truncated"] is False
    limited = working.view(ledger, preferred=[1], max_chars=20)
    omitted = limited["omitted"]
    assert limited["records"] == [] and isinstance(omitted, list) and len(omitted) == 2


def test_repair_originals_precede_recent_reads_under_context_pressure() -> None:
    _, ledger, _ = state_pair()
    working = EvidenceWorkingSet()
    recent = ledger.get(2)
    assert recent is not None
    working.remember(2, 0, len(recent.text))
    full = working.view(ledger, preferred=[], required=[1])
    records = full["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    capacity = len(json.dumps(records[0], ensure_ascii=False)) + 2
    bounded = working.view(ledger, preferred=[], required=[1], max_chars=capacity)
    assert bounded["records"] == records[:1]
    assert bounded["omitted"] == [
        {"citation": 2, "start_char": 0, "end_char": len(recent.text)}
    ]


def test_intervening_tools_do_not_erase_originals_or_change_questions() -> None:
    context, ledger, state = state_pair()
    registry = CapabilityRegistry(build_research_specs(state, ledger))
    for spec in build_core_specs(registry, ledger, lambda: {}):
        registry.register(spec)
    decisions = iter(
        [
            Decision(
                calls=[
                    CapabilityCall(
                        name="read_evidence",
                        arguments={"citation": 1, "num_chars": 2000},
                    )
                ]
            ),
            Decision(
                calls=[CapabilityCall(name="read_evidence", arguments={"citation": 2})]
            ),
            Decision(
                calls=[
                    CapabilityCall(
                        name="read_evidence",
                        arguments={"citation": 1, "num_chars": 4000},
                    )
                ]
            ),
            Decision(answer="Result [1].", questions=["Paraphrased new question"]),
        ]
    )
    harness = Harness(
        request="scenario",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _: next(decisions),
    )
    result = harness.run()
    assert result.questions == list(state.questions)
    assert result.receipts[-1].outcome.data["reused_recorded_read"] is True
    assert {record["citation"] for record in harness.view().original_evidence} == {1, 2}
    snapshot = harness.snapshot()
    restored = Harness(
        request="scenario",
        context=context,
        registry=registry,
        evidence=ledger,
        decide=lambda _: Decision(answer="Result [1]."),
    )
    restored.restore(snapshot)
    assert restored.view().original_evidence == harness.view().original_evidence


def test_selected_original_range_survives_a_large_attached_candidate_prefix() -> None:
    context = RunContext(scope={"source": "owned"})
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id="original",
                chunk_id=str(index),
                text=f"Original {index}. " + "Operative text. " * 60,
            )
            for index in range(50)
        ],
        context,
    )
    working = EvidenceWorkingSet()
    for number in range(1, 51):
        item = ledger.get(number)
        assert item is not None
        working.remember(number, 0, len(item.text))
    selected = ledger.get(41)
    assert selected is not None
    working.remember(41, 0, len(selected.text))
    # The first need may have dozens of attached candidates. It must not evict
    # the exact original the model just selected from a different need.
    preferred = list(range(1, 41))
    current = working.view(ledger, preferred=preferred, max_chars=2500)
    records = current["records"]
    assert isinstance(records, list) and isinstance(records[0], dict)
    assert records[0]["citation"] == 41
    assert records[0]["text"] == selected.text
    assert current["omitted"]
    restored = EvidenceWorkingSet()
    restored.restore(working.export(), ledger)
    assert restored.view(ledger, preferred=preferred, max_chars=2500) == current


def test_need_verification_cannot_be_bypassed_by_omitting_the_norm_name() -> None:
    _, ledger, state = state_pair()
    state.update(
        ResearchUpdate.model_validate({"needs": [need(evidence_numbers=[1])]}), ledger
    )
    review = VerificationResult(
        status="supported",
        explanation="True statements",
        required_conditions=[],
        missing_conditions=[],
        evidence_numbers=[1],
        safe_to_publish=True,
        question_results=[
            QuestionVerification(
                question_id="q0",
                status="supported",
                evidence_numbers=[1],
                missing_conditions=[],
            ),
            QuestionVerification(
                question_id="q1",
                status="supported",
                evidence_numbers=[1],
                missing_conditions=[],
            ),
        ],
    )
    gap = publication_gap(
        "Result [1].", review, list(state.questions), ledger, research_state=state
    )
    assert gap is not None and "completion test" in json.dumps(gap.data)
    reviewed = review.model_copy(
        update={
            "need_results": [
                NeedVerification(
                    need_id="basis",
                    status="supported",
                    evidence_numbers=[1],
                    missing_conditions=[],
                )
            ]
        }
    )
    assert (
        publication_gap(
            "Result [1].", reviewed, list(state.questions), ledger, research_state=state
        )
        is None
    )


def test_publication_rejects_an_uncovered_original_question() -> None:
    context, ledger, old_state = state_pair()
    state = ResearchState(
        list(old_state.questions), context, require_need_bindings=True
    )
    state.update(ResearchUpdate.model_validate({"needs": [need()]}), ledger)
    review = VerificationResult(
        status="supported",
        explanation="Original assertions verified",
        required_conditions=[],
        missing_conditions=[],
        evidence_numbers=[1],
        safe_to_publish=True,
        question_results=[
            QuestionVerification(
                question_id=f"q{i}",
                status="supported",
                evidence_numbers=[1],
                missing_conditions=[],
            )
            for i in range(2)
        ],
        need_results=[
            NeedVerification(
                need_id="basis",
                status="supported",
                evidence_numbers=[1],
                missing_conditions=[],
            )
        ],
    )
    gap = publication_gap(
        "Rule and procedure [1].",
        review,
        list(state.questions),
        ledger,
        research_state=state,
    )
    assert gap is not None and "q1" in str(gap.data)
    state.update(
        ResearchUpdate.model_validate(
            {"needs": [need("procedure", question_ids=["q1"])]}
        ),
        ledger,
    )
    review.need_results.append(
        NeedVerification(
            need_id="procedure",
            status="supported",
            evidence_numbers=[1],
            missing_conditions=[],
        )
    )
    assert (
        publication_gap(
            "Rule and procedure [1].",
            review,
            list(state.questions),
            ledger,
            research_state=state,
        )
        is None
    )


def test_review_inventory_uses_only_material_in_scope_needs() -> None:
    inventory = publication_review_inventory(
        json.dumps(
            {
                "assertion_units": [],
                "research_state": {
                    "needs": [
                        need("material", material=True),
                        need("excluded", material=True, status="out_of_scope"),
                        need("incidental", material=False),
                    ]
                },
            }
        )
    )
    assert inventory is not None and inventory["need_results"] == {"material"}


def test_worker_binds_and_reuses_an_already_assigned_need() -> None:
    context, ledger, state = state_pair()
    state.update(ResearchUpdate.model_validate({"needs": [need()]}), ledger)
    entered, release = threading.Event(), threading.Event()
    focused: list[object] = []

    def runner(_task: str, child: RunContext, _updates: object) -> ToolOutcome:
        focused.append(child.services["task_need_ids"])
        entered.set()
        assert release.wait(2)
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Original found; exception remains open",
        )

    pool = WorkerPool(context, runner)
    try:
        identifier = pool.spawn("Find the basis", need_ids=["basis"])
        assert entered.wait(2)
        assert (
            pool.spawn("Same need phrased differently", need_ids=["basis"])
            == identifier
        )
        assert len(pool.list()) == 1
        release.set()
        pool.wait_for_change(identifier)
        assert focused == [["basis"]]
        assert pool.list()[0].status == TaskStatus.COMPLETED
        outcome = pool.list()[0].outcome
        assert outcome is not None and outcome.status == OutcomeStatus.PARTIAL
    finally:
        release.set()
        pool.close()


def test_research_schema_is_usable_by_vertex() -> None:
    from litellm.llms.vertex_ai.common_utils import _build_vertex_schema

    _, ledger, state = state_pair()
    for tool in build_research_specs(state, ledger):
        normalized = _build_vertex_schema(tool.parameters)
        assert normalized["type"] == "object"
