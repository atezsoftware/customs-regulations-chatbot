"""Research obligations and original working context survive adaptive decisions."""

import json
import threading

import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.llm_adapter import (
    NeedVerification,
    QuestionVerification,
    VerificationResult,
)
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    TaskStatus,
    ToolOutcome,
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
