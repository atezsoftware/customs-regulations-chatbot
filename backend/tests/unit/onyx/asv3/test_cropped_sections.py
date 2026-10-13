import copy
import json
import re
from typing import cast

import pytest
from pydantic import JsonValue

from onyx.asv3.assertions import AssertionWitness
from onyx.asv3.cropped_sections import (
    CroppedAssemblyReview,
    CroppedAssemblyReviewInput,
    CroppedSectionGeneration,
    CroppedSectionMaterialContext,
    CroppedSectionPlan,
    CroppedSectionStore,
    assemble_cropped_sections,
    plan_cropped_sections,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.llm_adapter import MaterialSourceOmission
from onyx.asv3.models import (
    Decision,
    EvidenceItem,
    HarnessView,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
)
from onyx.asv3.outcome_map import (
    OutcomeCondition,
    OutcomeMap,
    OutcomeUpdate,
    OutcomeWitness,
    RequestedOutcome,
)
from onyx.asv3.research_state import ResearchState
from onyx.asv3.scenario import question_determinations
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.llm.models import ChatCompletionMessage

REQUEST = "Which duties and exceptions apply? Which dates remain unknown?"


def original(
    number: int, *, source_id: str | None = None, article: str | None = "1"
) -> EvidenceItem:
    text = f"Duty {number} requires its own condition; exception {number} remains conditional."
    return EvidenceItem(
        source_id=source_id or f"law-{number}",
        chunk_id=f"article-{number}",
        text=text,
        metadata={
            "title": f"Official law {number}",
            "article_no": article,
            "publication_revision": 1,
        },
        search_doc=SearchDoc(
            document_id=source_id or f"law-{number}",
            chunk_ind=number,
            semantic_identifier=f"Official law {number}",
            link="https://example.test/law",
            blurb=text,
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={},
            match_highlights=[],
        ),
    )


def fixture(count: int = 3) -> tuple[RunContext, EvidenceLedger, HarnessView]:
    context = RunContext(
        run_id="cropped-run",
        scope={"document_set": 15},
        services={
            "asv3_exp_cropped": True,
            "user_id": "owner",
            "cropped_user_context": "The service date is not supplied.",
        },
    )
    ledger = EvidenceLedger()
    ledger.add([original(number) for number in range(1, count + 1)], context)
    context.services["evidence"] = ledger
    view = HarnessView(
        request=REQUEST,
        questions=[REQUEST],
        facts=["The service date is not supplied."],
        receipts=[],
        evidence=[],
        tools=[],
        original_evidence=cast(
            list[dict[str, JsonValue]],
            json.loads(ledger.serialize_records(range(1, count + 1), max_chars=None)),
        ),
        required_evidence_numbers=list(range(1, count + 1)),
        research_state={
            "remaining_scope": "All exceptions and uncertainty remain material."
        },
    )
    return context, ledger, view


def count_input(view: HarnessView, _plan: CroppedSectionPlan) -> int:
    return 130_000 * len(view.original_evidence)


def terminal_body(decision: Decision) -> str:
    assert len(decision.calls) == 1
    assert decision.calls[0].name in {"submit_answer", "submit_partial_answer"}
    return cast(str, decision.calls[0].arguments["answer"])


class Calls:
    def __init__(self, context: RunContext, ledger: EvidenceLedger) -> None:
        self.context, self.ledger = context, ledger
        self.writes: list[tuple[HarnessView, CroppedSectionPlan]] = []
        self.reviews: list[CroppedAssemblyReviewInput] = []
        self.guards: list[tuple[str, str]] = []

    def writer(
        self, view: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        self.writes.append((view.model_copy(deep=True), plan))
        call = f"writer-{plan.ordinal}"
        self.ledger.record_delivery(call, "asv3_coordinator", view.original_evidence)
        self.context.services["last_model_call_id"] = call
        return CroppedSectionGeneration(
            answer=f"Duty {plan.ordinal + 1} retains its condition and exception [{plan.evidence_numbers[0]}].",
            model_call_id=call,
        )

    def reviewer(self, value: CroppedAssemblyReviewInput) -> CroppedAssemblyReview:
        self.reviews.append(value)
        self.context.services["last_model_call_id"] = "review"
        return CroppedAssemblyReview(
            model_call_id="review",
            review_scope="section_coverage_and_consistency",
            answer_hash=value.answer_hash,
            assessed_section_ids=[
                cast(str, cast(dict[str, JsonValue], row["plan"])["section_id"])
                for row in value.sections
            ],
            assessed_determination_ids=[
                cast(str, row["determination_id"]) for row in value.determinations
            ],
            complete=True,
            consistent=True,
            missing=[],
            contradictions=[],
        )

    def guard(self, answer: str, call: str) -> None:
        self.guards.append((answer, call))
        assert set(int(row) for row in re.findall(r"\[(\d+)\]", answer)).issubset(
            self.ledger.completely_delivered(call)
        )


def test_bounded_sections_keep_full_scenario_and_merge_exact_writer_bytes() -> None:
    context, ledger, view = fixture()
    before = view.model_dump_json()
    calls = Calls(context, ledger)
    result = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    assert len(calls.writes) == 3
    assert len(calls.reviews) == 1
    assert terminal_body(result) == "\n\n".join(
        f"Duty {index + 1} retains its condition and exception [{index + 1}]."
        for index in range(3)
    )
    assert view.model_dump_json() == before
    for prepared, plan in calls.writes:
        assert prepared.request == view.request
        assert prepared.questions == view.questions
        assert prepared.facts == view.facts
        assert (
            prepared.research_state["remaining_scope"]
            == view.research_state["remaining_scope"]
        )
        assert prepared.required_evidence_numbers == plan.evidence_numbers
        assert len(prepared.original_evidence) == 1
        assert count_input(prepared, plan) < 240_000
    assert calls.reviews[0].facts == view.facts
    assert calls.reviews[0].user_context == "The service date is not supplied."
    assert ledger.completely_delivered("review") == set()
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    assert terminal_body(result) is not None
    store.verify_assembly(context, view, terminal_body(result), ledger, calls.guard)
    assert store.assembly_original_numbers(terminal_body(result)) == {1, 2, 3}
    assert len(calls.reviews) == 1
    assert ledger.export()["pinned_delivery_calls"] == [
        "writer-0",
        "writer-1",
        "writer-2",
    ]


def test_full_retained_condition_survives_bounded_navigation_and_final_review() -> None:
    context, ledger, view = fixture(4)
    state = ResearchState(view.questions, context)
    requirement = MaterialSourceOmission(
        witness=AssertionWitness(
            citation=4, source_quote="exception 4 remains conditional"
        ),
        determination_ids=[
            str(row["determination_id"])
            for row in question_determinations(view.questions)
        ],
        detail="Preserve the source's conditional exception in every application.",
        applicability="The requested duties depend on this exception.",
    )
    state.source_conditions.remember([requirement], ledger)
    context.services["research_state"] = state
    view.original_evidence = [
        row for row in view.original_evidence if row["citation"] != 4
    ]
    view.required_evidence_numbers = [1, 2, 3]
    view.research_state = {"source_conditions": [], "source_conditions_omitted": 1}
    calls = Calls(context, ledger)
    result = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    assert [plan.evidence_numbers for _, plan in calls.writes] == [[1], [2], [3], [4]]
    assert (
        calls.writes[-1][0].research_state["source_conditions"]
        == state.source_conditions.required_conditions()
    )
    assert (
        calls.reviews[0].retained_requirements["retained_source_conditions"]
        == state.source_conditions.required_conditions()
    )
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    state.source_conditions.remember(
        [
            requirement.model_copy(
                update={"detail": "Preserve an additional material actor."}
            )
        ],
        ledger,
    )
    with pytest.raises(ValueError, match="unowned or changed"):
        store.verify_assembly(context, view, terminal_body(result), ledger, calls.guard)


def test_omitted_navigation_original_is_restored_whole_before_section_assignment() -> (
    None
):
    context, ledger, view = fixture()
    view.original_evidence = [view.original_evidence[0]]
    view.required_evidence_numbers = [1]
    view.original_evidence_omitted = [{"citation": 2, "start_char": 8, "end_char": 15}]
    planned = plan_cropped_sections(view, context, ledger, count_input)
    assert [plan.evidence_numbers for plan, _ in planned] == [[1], [2]]
    assert (
        planned[-1][1].original_evidence[0]["text"]
        == cast(EvidenceItem, ledger.get(2)).text
    )
    assert planned[-1][1].original_evidence[0].get("start_char", 0) == 0
    assert planned[-1][1].original_evidence[0]["truncated"] is False


def test_section_outer_whitespace_cannot_change_owned_assembled_bytes() -> None:
    context, ledger, view = fixture()
    calls = Calls(context, ledger)

    def writer(
        prepared: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        result = calls.writer(prepared, plan)
        return result.model_copy(update={"answer": "\n" + result.answer})

    with pytest.raises(ValueError, match="publication guard"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
    assert calls.reviews == []


def test_one_writer_receipt_cannot_be_reused_for_distinct_section_bodies() -> None:
    context, ledger, view = fixture()
    calls = Calls(context, ledger)
    ledger.record_delivery("same-call", "asv3_coordinator", view.original_evidence)

    def writer(
        prepared: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        generated = calls.writer(prepared, plan)
        context.services["last_model_call_id"] = "same-call"
        return generated.model_copy(update={"model_call_id": "same-call"})

    with pytest.raises(ValueError, match="distinct owned writer"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
    assert len(calls.writes) == 2
    assert calls.reviews == []


def test_connected_outcome_sources_are_not_split_to_force_the_input_price_tier() -> (
    None
):
    context, ledger, view = fixture()
    outcomes = OutcomeMap(view.questions, context)
    outcomes.update(
        OutcomeUpdate(
            outcomes=[
                RequestedOutcome(
                    outcome_id="permission",
                    question_ids=["q0"],
                    detail="Permission and its exception.",
                )
            ],
            conditions=[
                OutcomeCondition(
                    condition_id="condition",
                    outcome_ids=["permission"],
                    detail="Both operative originals are material.",
                    witnesses=[
                        OutcomeWitness(citation=1, end_char=10),
                        OutcomeWitness(citation=2, end_char=10),
                    ],
                )
            ],
        ),
        ledger,
    )
    context.services["outcome_map"] = outcomes
    calls = Calls(context, ledger)
    with pytest.raises(RunStopped, match="connected source obligation.*1,2"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=calls.writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
    assert calls.writes == []
    assert calls.reviews == []


def test_existing_cited_assertion_is_an_indivisible_source_dependency() -> None:
    context, ledger, view = fixture()
    view.draft_to_repair = "Duty depends on both the prerequisite and exception [1][2]."
    with pytest.raises(RunStopped, match="connected source obligation.*1,2"):
        plan_cropped_sections(view, context, ledger, count_input)


def test_blank_article_locator_does_not_invent_a_safe_provision_boundary() -> None:
    context, ledger, view = fixture(1)
    ledger = EvidenceLedger()
    ledger.add(
        [
            original(1, source_id="same-law", article="8"),
            original(2, source_id="same-law", article=None),
            original(3),
        ],
        context,
    )
    view.original_evidence = cast(
        list[dict[str, JsonValue]],
        json.loads(ledger.serialize_records([1, 2, 3], max_chars=None)),
    )
    view.required_evidence_numbers = [1, 2, 3]
    with pytest.raises(RunStopped, match="connected source obligation.*1,2"):
        plan_cropped_sections(view, context, ledger, count_input)


@pytest.mark.parametrize(
    "failure", ["partial-original", "wrong-writer", "outside-assignment", "guard"]
)
def test_section_receipt_requires_exact_owned_complete_delivery_and_existing_guard(
    failure: str,
) -> None:
    context, ledger, view = fixture()
    plan, prepared = plan_cropped_sections(view, context, ledger, count_input)[0]
    records = copy.deepcopy(prepared.original_evidence)
    if failure == "partial-original":
        records[0]["text"] = cast(str, records[0]["text"])[:10]
    ledger.record_delivery("writer", "asv3_coordinator", records)
    context.services["last_model_call_id"] = (
        "other" if failure == "wrong-writer" else "writer"
    )
    generation = CroppedSectionGeneration(
        answer="An operative condition is required [2]."
        if failure == "outside-assignment"
        else "An operative condition is required [1].",
        model_call_id="writer",
    )
    store = CroppedSectionStore(context, REQUEST, user_id="owner")

    def guard(_answer: str, _call: str) -> ToolOutcome | None:
        return (
            ToolOutcome(
                status=OutcomeStatus.PARTIAL, summary="Missing governing original"
            )
            if failure == "guard"
            else None
        )

    with pytest.raises(ValueError):
        store.seal_section(context, view, plan, generation, ledger, guard)
    assert store.export()["sections"] == []
    assert ledger.export()["pinned_delivery_calls"] == []


@pytest.mark.parametrize(
    "failure",
    [
        "incomplete",
        "contradiction",
        "wrong-answer",
        "missing-question",
        "missing-section",
        "wrong-call",
    ],
)
def test_final_assessment_cannot_approve_omission_or_changed_assembly(
    failure: str,
) -> None:
    context, ledger, view = fixture()
    calls = Calls(context, ledger)

    def review(value: CroppedAssemblyReviewInput) -> CroppedAssemblyReview:
        result = calls.reviewer(value)
        choices: dict[str, dict[str, object]] = {
            "incomplete": {
                "complete": False,
                "missing": ["A material exception is absent."],
            },
            "contradiction": {
                "consistent": False,
                "contradictions": ["Two sections disagree."],
            },
            "wrong-answer": {"answer_hash": "0" * 64},
            "missing-question": {"assessed_determination_ids": []},
            "missing-section": {
                "assessed_section_ids": result.assessed_section_ids[:1]
            },
            "wrong-call": {"model_call_id": "writer-0"},
        }
        return result.model_copy(update=choices[failure])

    with pytest.raises(RunStopped, match="completeness or consistency"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=calls.writer,
            review_assembly=review,
            publication_gap=calls.guard,
        )
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    assert store.export()["assemblies"] == []


def test_registered_assembly_rejects_changed_fact_body_source_metadata_or_scope() -> (
    None
):
    context, ledger, view = fixture()
    calls = Calls(context, ledger)
    result = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore) and terminal_body(result) is not None
    with pytest.raises(ValueError, match="unowned or changed"):
        store.verify_assembly(
            context,
            view,
            terminal_body(result) + "\nNew unsupported connection [1].",
            ledger,
            calls.guard,
        )
    changed_view = view.model_copy(deep=True)
    changed_view.facts.append("Service date supplied later.")
    with pytest.raises(ValueError, match="unowned or changed"):
        store.verify_assembly(
            context, changed_view, terminal_body(result), ledger, calls.guard
        )
    changed_context = RunContext(
        run_id=context.run_id, scope={"document_set": 16}, services=context.services
    )
    with pytest.raises(ValueError, match="scope or request"):
        store.verify_assembly(
            changed_context, view, terminal_body(result), ledger, calls.guard
        )
    snapshot = ledger.export()
    records = cast(list[dict[str, JsonValue]], snapshot["records"])
    metadata = cast(
        dict[str, JsonValue], cast(dict[str, JsonValue], records[0]["item"])["metadata"]
    )
    metadata["publication_revision"] = 2
    changed = EvidenceLedger()
    changed.restore(snapshot, context)
    with pytest.raises(ValueError, match="original version or receipt"):
        store.verify_assembly(
            context, view, terminal_body(result), changed, lambda _body, _call: None
        )


def test_checkpoint_restores_exact_pinned_sections_and_rejects_owner_or_tampering() -> (
    None
):
    context, ledger, view = fixture()
    calls = Calls(context, ledger)
    result = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore) and terminal_body(result) is not None
    restored_ledger = EvidenceLedger()
    restored_ledger.restore(ledger.export(), context)
    restored = CroppedSectionStore(context, REQUEST, user_id="owner")
    restored.restore(store.export(), context, REQUEST, restored_ledger)
    restored.verify_assembly(
        context, view, terminal_body(result), restored_ledger, lambda _body, _call: None
    )
    wrong_owner = CroppedSectionStore(context, REQUEST, user_id="other")
    with pytest.raises(ValueError, match="ownership or integrity"):
        wrong_owner.restore(store.export(), context, REQUEST, restored_ledger)
    tampered = store.export()
    sections = cast(list[dict[str, JsonValue]], tampered["sections"])
    generation = cast(dict[str, JsonValue], sections[0]["generation"])
    generation["answer"] = "Rewritten condition [1]."
    with pytest.raises(ValueError, match="ownership or integrity"):
        CroppedSectionStore(context, REQUEST, user_id="owner").restore(
            tampered, context, REQUEST, restored_ledger
        )


def test_original_and_cheap_workflows_cannot_enter_section_assembly() -> None:
    context, ledger, view = fixture()
    for flags in (
        {},
        {"asv3_exp_cheap": True},
        {"asv3_exp_cropped": True, "asv3_exp_cheap": True},
    ):
        context.services = flags
        with pytest.raises(ValueError, match="isolated Cropped"):
            plan_cropped_sections(view, context, ledger, count_input)


@pytest.mark.parametrize("change", ["run", "task", "depth", "user", "request"])
def test_owned_assembly_cannot_transfer_between_tasks_or_owners(change: str) -> None:
    context, ledger, view = fixture()
    context.services["task_id"] = "task-a"
    calls = Calls(context, ledger)
    decision = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    body = terminal_body(decision)
    assert store.owns_assembly(body)
    assert not store.owns_assembly(body + " ")
    other = RunContext(
        run_id="other" if change == "run" else context.run_id,
        scope=context.scope,
        services=context.services,
        depth=1 if change == "depth" else 0,
    )
    if change in {"task", "user"}:
        other.services["task_id" if change == "task" else "user_id"] = "other"
    other_view = view.model_copy(deep=True)
    if change == "request":
        other_view.request += " A different question."
    with pytest.raises(ValueError, match="owner, scope or request"):
        store.verify_assembly(other, other_view, body, ledger, calls.guard)


def test_partial_section_status_survives_exact_assembly_without_inventing_missing_law() -> (
    None
):
    context, ledger, view = fixture()
    notice = "6455 sayılı Kanun 11. maddesinin özgün metni elde edilemedi."
    view.draft_to_repair = notice + "\n\nA separately supported condition remains [1]."
    view.publication_gap = {
        "named_authority_gaps": [
            {
                "instrument_number": "6455",
                "article": "11",
                "matching_original_evidence": [],
            }
        ]
    }
    calls = Calls(context, ledger)

    def writer(
        prepared: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        result = calls.writer(prepared, plan)
        if plan.ordinal == 0:
            assert (
                prepared.draft_to_repair is not None
                and notice in prepared.draft_to_repair
            )
            return result.model_copy(
                update={
                    "status": OutcomeStatus.PARTIAL,
                    "answer": notice + "\n\n" + result.answer,
                }
            )
        assert notice not in (prepared.draft_to_repair or "")
        return result

    decision = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    assert decision.calls[0].name == "submit_partial_answer"
    assert set(decision.calls[0].arguments) == {"answer"}
    assert notice in terminal_body(decision)
    assert "no law" not in terminal_body(decision)
    assert calls.reviews[0].publication_gap == view.publication_gap


def test_interrupted_checkpoint_reuses_only_the_exact_previously_validated_section() -> (
    None
):
    context, ledger, view = fixture()
    calls = Calls(context, ledger)

    def stopped_writer(
        prepared: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        if plan.ordinal == 1:
            raise RunStopped("Budget before provider call")
        return calls.writer(prepared, plan)

    with pytest.raises(RunStopped, match="Budget before"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=stopped_writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
    saved_store = context.services["cropped_section_store"]
    assert isinstance(saved_store, CroppedSectionStore)
    assert len(saved_store.export()["sections"]) == 1
    assert saved_store.export()["assemblies"] == []
    restored_ledger = EvidenceLedger()
    restored_ledger.restore(ledger.export(), context)
    restored = CroppedSectionStore(context, REQUEST, user_id="owner")
    restored.restore(saved_store.export(), context, REQUEST, restored_ledger)
    resumed = Calls(context, restored_ledger)
    decision = assemble_cropped_sections(
        view,
        context,
        restored_ledger,
        count_input=count_input,
        write_section=resumed.writer,
        review_assembly=resumed.reviewer,
        publication_gap=resumed.guard,
        store=restored,
    )
    assert [plan.ordinal for _, plan in resumed.writes] == [1, 2]
    assert len(resumed.reviews) == 1
    restored.verify_assembly(
        context, view, terminal_body(decision), restored_ledger, resumed.guard
    )


@pytest.mark.parametrize("cross_section", [False, True])
def test_new_terminal_requirements_are_retained_and_must_stay_with_one_writer(
    cross_section: bool,
) -> None:
    context, ledger, view = fixture()
    outcomes = OutcomeMap(view.questions, context)
    context.services["outcome_map"] = outcomes
    calls = Calls(context, ledger)

    def writer(
        prepared: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        result = calls.writer(prepared, plan)
        if plan.ordinal == 0:
            witnesses = [OutcomeWitness(citation=1, end_char=10)]
            if cross_section:
                witnesses.append(OutcomeWitness(citation=2, end_char=10))
            outcomes.update(
                OutcomeUpdate(
                    outcomes=[
                        RequestedOutcome(
                            outcome_id="permission",
                            question_ids=["q0"],
                            detail="Permission and its material exception.",
                        )
                    ],
                    conditions=[
                        OutcomeCondition(
                            condition_id="exception",
                            outcome_ids=["permission"],
                            detail="This exception is mandatory, not removable by paraphrase.",
                            witnesses=witnesses,
                        )
                    ],
                ),
                ledger,
            )
        return result

    if cross_section:
        with pytest.raises(ValueError, match="cross-section source obligation"):
            assemble_cropped_sections(
                view,
                context,
                ledger,
                count_input=count_input,
                write_section=writer,
                review_assembly=calls.reviewer,
                publication_gap=calls.guard,
            )
        assert calls.reviews == []
    else:
        decision = assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
        assert (
            calls.reviews[0].retained_requirements["conditions"]
            == outcomes.view()["conditions"]
        )
        store = context.services["cropped_section_store"]
        assert isinstance(store, CroppedSectionStore)
        store.verify_assembly(
            context, view, terminal_body(decision), ledger, calls.guard
        )
        outcomes.update(
            OutcomeUpdate(
                outcomes=[
                    RequestedOutcome(
                        outcome_id="new-condition",
                        question_ids=["q0"],
                        detail="New material condition.",
                    )
                ]
            ),
            ledger,
        )
        with pytest.raises(ValueError, match="unowned or changed"):
            store.verify_assembly(
                context, view, terminal_body(decision), ledger, calls.guard
            )


def material_context(context: RunContext) -> CroppedSectionMaterialContext:
    return CroppedSectionMaterialContext.capture(
        context,
        REQUEST,
        research_tasks=[
            {"task": "Assess the requested exceptions", "status": "partial"}
        ],
        repair_observations=[
            {
                "name": "read_named_provision",
                "arguments_json": '{"article":"48"}',
                "result": {
                    "status": "not_found",
                    "absence_proven": False,
                    "cursor": None,
                },
                "untrusted_assistant_assessment": "The service date is unknown.",
            }
        ],
    )


def test_material_context_is_immutable_and_identical_for_counts_writes_and_review() -> (
    None
):
    context, ledger, view = fixture()
    material = material_context(context)
    context.services["cropped_section_material_context"] = material
    expected = material.payload()
    changed = material.repair_observations()
    changed[0]["result"] = {"status": "found"}
    assert material.payload() == expected
    counted: dict[str, dict[str, JsonValue]] = {}
    calls = Calls(context, ledger)

    def count(prepared: HarnessView, plan: CroppedSectionPlan) -> int:
        counted[plan.section_id] = material.payload()
        return count_input(prepared, plan)

    def writer(
        prepared: HarnessView, plan: CroppedSectionPlan
    ) -> CroppedSectionGeneration:
        assert counted[plan.section_id] == material.payload() == expected
        return calls.writer(prepared, plan)

    result = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count,
        write_section=writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    assert len(calls.writes) == 3
    assert calls.reviews[0].retained_requirements["actual_material_context"] == expected
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    snapshot = store.export()
    assert snapshot["material_context"] == material.export()
    context.services.pop("cropped_section_material_context")
    restored = CroppedSectionStore(context, REQUEST, user_id="owner")
    restored.restore(snapshot, context, REQUEST, ledger)
    assert context.services["cropped_section_material_context"] == material
    restored.verify_assembly(context, view, terminal_body(result), ledger, calls.guard)


def test_changed_material_status_invalidates_owned_assembly_and_resume() -> None:
    context, ledger, view = fixture()
    initial = material_context(context)
    context.services["cropped_section_material_context"] = initial
    calls = Calls(context, ledger)
    result = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=calls.reviewer,
        publication_gap=calls.guard,
    )
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    observations = initial.repair_observations()
    observations[0]["result"] = {"status": "partial", "has_more": True}
    context.services["cropped_section_material_context"] = (
        CroppedSectionMaterialContext.capture(
            context,
            REQUEST,
            research_tasks=initial.research_tasks(),
            repair_observations=observations,
        )
    )
    with pytest.raises(ValueError, match="material context changed"):
        store.verify_assembly(context, view, terminal_body(result), ledger, calls.guard)
    with pytest.raises(ValueError, match="material context changed"):
        CroppedSectionStore(context, REQUEST, user_id="owner").restore(
            store.export(), context, REQUEST, ledger
        )
    changed_plans = plan_cropped_sections(view, context, ledger, count_input)
    assert changed_plans[0][0].section_id != calls.writes[0][1].section_id


@pytest.mark.parametrize(
    "problem",
    [
        {"citation": True},
        {"citation": 99},
        {"identity_ref": 99},
        {"citation": 1, "source_id": "wrong-source"},
        {"cropped_worker_handoff": "opaque"},
        {"cropped_worker_handoff": {}},
        {"untrusted_assistant_assessment": "The exception relies on [99]."},
    ],
)
def test_unknown_material_references_fail_before_count_or_writer(
    problem: dict[str, JsonValue],
) -> None:
    context, ledger, view = fixture()
    context.services["cropped_section_material_context"] = (
        CroppedSectionMaterialContext.capture(
            context,
            REQUEST,
            research_tasks=[],
            repair_observations=[problem],
        )
    )
    calls = Calls(context, ledger)
    counts: list[str] = []

    def count(prepared: HarnessView, plan: CroppedSectionPlan) -> int:
        counts.append(plan.section_id)
        return count_input(prepared, plan)

    with pytest.raises(RunStopped, match="context"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count,
            write_section=calls.writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
    assert counts == [] and calls.writes == [] and calls.reviews == []


def test_shared_original_dependency_cannot_be_hidden_from_other_sections() -> None:
    context, ledger, view = fixture()
    source = copy.deepcopy(view.original_evidence[0])
    source["identity_ref"] = 1
    material = CroppedSectionMaterialContext.capture(
        context,
        REQUEST,
        research_tasks=[],
        repair_observations=[{"result": source}],
    )
    assert material.original_dependencies(context, REQUEST, ledger) == {1}
    context.services["cropped_section_material_context"] = material
    calls = Calls(context, ledger)
    with pytest.raises(RunStopped, match="outside its assignment"):
        assemble_cropped_sections(
            view,
            context,
            ledger,
            count_input=count_input,
            write_section=calls.writer,
            review_assembly=calls.reviewer,
            publication_gap=calls.guard,
        )
    assert calls.writes == [] and calls.reviews == []
    assert ledger.export()["pinned_delivery_calls"] == []


def test_material_snapshot_checkpoint_rejects_changed_owner_and_tampered_bytes() -> (
    None
):
    context, _ledger, _view = fixture()
    material = material_context(context)
    snapshot = material.export()
    bad = copy.deepcopy(snapshot)
    bad["observations_json"] = "[]"
    with pytest.raises(ValueError, match="checkpoint changed"):
        CroppedSectionMaterialContext.restore(bad, context, REQUEST)
    child = context.child()
    child.services["task_id"] = "other-task"
    with pytest.raises(ValueError, match="owner or scope changed"):
        material.validate(child, REQUEST)


def test_material_context_verifies_actual_worker_hashes_and_source_dependencies() -> (
    None
):
    from tests.unit.onyx.asv3.test_cropped_worker_handoff import build

    handoff, context, ledger, *_rest = build()
    snapshot = CroppedSectionMaterialContext.capture(
        context,
        REQUEST,
        research_tasks=[{"cropped_worker_handoff": handoff}],
        repair_observations=[],
    )
    assert snapshot.original_dependencies(context, REQUEST, ledger) == {1}
    snapshot.require_assignment(context, REQUEST, ledger, [1])
    with pytest.raises(RunStopped, match="outside its assignment"):
        snapshot.require_assignment(context, REQUEST, ledger, [])
    corrupted = copy.deepcopy(handoff)
    bindings = corrupted["source_bindings"]
    assert isinstance(bindings, list) and isinstance(bindings[0], dict)
    bindings[0]["text_hash"] = "0" * 64
    changed = CroppedSectionMaterialContext.capture(
        context,
        REQUEST,
        research_tasks=[{"cropped_worker_handoff": corrupted}],
        repair_observations=[],
    )
    with pytest.raises(ValueError):
        changed.original_dependencies(context, REQUEST, ledger)


@pytest.mark.parametrize("overflow", [False, True])
def test_source_writer_snapshots_original_observations_before_turns_clear(
    monkeypatch: pytest.MonkeyPatch,
    overflow: bool,
) -> None:
    from onyx.asv3.cheap_repair_packet import cheap_repair_observations
    from onyx.asv3.cropped_transport import (
        CroppedInputOverflow,
        prepare_cropped_writer_view,
    )
    from tests.unit.onyx.asv3.test_cropped_repair_model import current
    from tests.unit.onyx.asv3.test_cropped_repair_model import (
        fixture as adapter_fixture,
    )

    adapter, ledger, selected = adapter_fixture()
    view = current(ledger).model_copy(
        update={"draft_to_repair": None, "publication_gap": None}
    )
    tasks: list[dict[str, JsonValue]] = [
        {"task": "Preserve exact facts", "status": "partial"}
    ]
    adapter.pending_tasks = lambda: copy.deepcopy(tasks)
    pending_callback = adapter.pending_tasks
    original_observations = cheap_repair_observations(view)
    assert original_observations and view.turns
    full = prepare_cropped_writer_view(
        view, adapter.context, adapter._native_original_records(view.turns)
    )
    adapter._writing_answer = True
    expected_prompt, expected_tools, expected_output = adapter._fit_native_decision(
        full,
        repair_observations=original_observations,
    )
    adapter._writing_answer = False
    monkeypatch.setattr(adapter, "_decide_cropped_read_plan", lambda _view: None)
    captured: list[CroppedSectionMaterialContext] = []

    def invoke(
        _prepared: HarnessView,
        prompt: list[ChatCompletionMessage],
        tools: object,
        output: int,
        **_kwargs: object,
    ) -> Decision:
        assert [message.model_dump_json() for message in prompt] == [
            message.model_dump_json() for message in expected_prompt
        ]
        assert tools == expected_tools and output == expected_output
        assert adapter.pending_tasks() == tasks
        if overflow:
            raise CroppedInputOverflow(240_001)
        return Decision(answer="Supported text [1].")

    def sections(
        prepared: HarnessView, *, material_context: CroppedSectionMaterialContext
    ) -> Decision:
        assert prepared.turns == []
        assert {row["citation"] for row in prepared.original_evidence} == {1, 2, 3}
        assert material_context.research_tasks() == tasks
        assert material_context.repair_observations() == original_observations
        captured.append(material_context)
        return Decision(answer="Captured bounded section input.")

    monkeypatch.setattr(adapter, "_invoke_decision", invoke)
    monkeypatch.setattr(adapter, "_assemble_cropped_sections", sections)
    monkeypatch.setattr(
        adapter,
        "_experimental_terminal_envelope",
        lambda decision, *_args, **_kwargs: decision,
    )
    adapter._decide_cropped_source_answer(view)
    assert bool(captured) is overflow
    assert (
        adapter.pending_tasks is pending_callback and adapter._writing_answer is False
    )
    selected.invoke.assert_not_called()
