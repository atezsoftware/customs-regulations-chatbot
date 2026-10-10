import pytest
from pydantic import ValidationError

from onyx.legal_review.models import (
    Issue,
    IssuePlan,
    LegalDimension,
    PassageSupport,
    ReviewResearchTask,
    SourceAction,
)
from onyx.legal_review.research import ResearchLedger


def plan() -> IssuePlan:
    return IssuePlan(
        language="tr",
        issues=[
            Issue(
                issue_id="parent",
                question="Which condition applies?",
                requested_outcome="Explain the condition",
            )
        ],
    )


def diagnosis(*, existing: str | None = None) -> ReviewResearchTask:
    return ReviewResearchTask(
        task_id="task-1",
        subject="Application condition",
        supports=[PassageSupport(citation=1, span_number=1)],
        question="Was the condition changed?",
        query=None if existing else "condition amendment",
        existing_need_id=existing,
        dimension=LegalDimension.VALIDITY,
    )


def test_empty_or_failed_search_consumes_the_attempt_and_a_paraphrase_does_not_retry() -> (
    None
):
    ledger = ResearchLedger()
    need = ledger.bind_task(diagnosis(), ["parent"])
    action = SourceAction(
        issue_ids=["parent"],
        tool="search_corpus",
        arguments={"query": "condition amendment"},
        research_need_ids=[need.need_id],
    )
    admitted = ledger.admit([action], plan())
    ledger.record_results(
        admitted,
        [
            {
                "tool": "search_corpus",
                "arguments": action.arguments,
                "status": "missing",
                "call_id": "once",
                "evidence_ids": [],
            }
        ],
        set(),
    )
    later = diagnosis(existing=need.need_id)
    later.question = "What is the present version of this condition?"
    assert ledger.bind_task(later, ["parent"]) is need
    action.arguments["query"] = "current operative application condition"
    assert ledger.admit([action], plan()) == []
    assert need.receipt_ids == ["once"] and need.new_evidence_ids == []
    assert ledger.skipped


def test_new_source_child_gets_its_own_single_attempt_and_reading_is_still_available() -> (
    None
):
    ledger = ResearchLedger()
    parent = plan()
    first = SourceAction(
        issue_ids=["parent"],
        tool="search_corpus",
        arguments={"query": "application condition"},
    )
    admitted = ledger.admit([first], parent)
    ledger.record_results(
        admitted,
        [
            {
                "tool": "search_corpus",
                "arguments": first.arguments,
                "call_id": "parent-search",
                "evidence_ids": [1],
            }
        ],
        set(),
    )
    child = Issue(
        issue_id="child",
        question="Does the newly found exception apply?",
        requested_outcome="Establish exception scope",
        origin="source",
        parent_issue_id="parent",
        trigger_dimension=LegalDimension.EXCEPTIONS,
        supporting_requirement_ids=["r1"],
        material_reason="The returned condition is subject to this exception",
        closure_criteria=["Read the exception prerequisites"],
    )
    parent.issues.append(child)
    action = SourceAction(
        issue_ids=["child"],
        tool="search_corpus",
        arguments={"query": "exception prerequisites"},
    )
    assert len(ledger.admit([action], parent)) == 1
    assert ledger.needs["issue:child"].parent_need_id == "issue:parent"
    assert ledger.admit([action], parent) == []
    read = SourceAction(
        issue_ids=["child"],
        tool="read_provision",
        arguments={"source_id": "found-source", "article": "2"},
    )
    assert ledger.admit([read], parent) == [read]


def test_one_gap_cannot_supply_multiple_discovery_queries() -> None:
    payload = diagnosis().model_dump()
    payload["query"] = ["first wording", "second wording"]
    with pytest.raises(ValidationError):
        ReviewResearchTask.model_validate(payload)


def test_review_can_keep_an_attempted_gap_open_without_a_new_query() -> None:
    ledger = ResearchLedger()
    need = ledger.bind_task(diagnosis(), ["parent"])
    need.attempted = True
    later = diagnosis(existing=need.need_id)
    assert ledger.bind_task(later, ["parent"]) is need
    assert later.query is None


def test_initial_discovery_does_not_consume_a_new_material_gaps_search() -> None:
    ledger = ResearchLedger()
    parent = plan()
    ledger.admit(
        [
            SourceAction(
                issue_ids=["parent"],
                tool="search_corpus",
                arguments={"query": "application condition"},
            )
        ],
        parent,
    )
    need = ledger.bind_task(diagnosis(), ["parent"])
    assert need.need_id != "issue:parent" and not need.attempted
    action = SourceAction(
        issue_ids=["parent"],
        tool="search_corpus",
        arguments={"query": "condition amendment"},
        research_need_ids=[need.need_id],
    )
    assert ledger.admit([action], parent)
    with pytest.raises(ValueError, match="Initial issue discovery"):
        ledger.bind_task(diagnosis(existing="issue:parent"), ["parent"])


def test_distinct_questions_in_same_dimension_are_not_merged_by_scope() -> None:
    ledger = ResearchLedger()
    first = ledger.bind_task(diagnosis(), ["parent"])
    first.attempted = True
    different = diagnosis()
    different.question = "When does the newly discovered exception take effect?"
    different.query = "exception effective date transitional rule"
    second = ledger.bind_task(different, ["parent"])
    assert second.need_id != first.need_id and not second.attempted


def test_other_subject_or_dimension_does_not_consume_a_search() -> None:
    ledger = ResearchLedger()
    first = ledger.bind_task(diagnosis(), ["parent"])
    first.attempted = True
    other_subject = diagnosis()
    other_subject.subject = "Another enabling norm"
    second = ledger.bind_task(other_subject, ["parent"])
    other_dimension = diagnosis()
    other_dimension.dimension = LegalDimension.CASE_LAW
    third = ledger.bind_task(other_dimension, ["parent"])
    assert len({first.need_id, second.need_id, third.need_id}) == 3
    assert not second.attempted and not third.attempted
    explicit_wrong_scope = diagnosis(existing=first.need_id)
    explicit_wrong_scope.dimension = LegalDimension.CASE_LAW
    with pytest.raises(ValueError, match="another dimension"):
        ledger.bind_task(explicit_wrong_scope, ["parent"])


def test_shared_search_consumes_each_explicitly_covered_dimension_but_no_others() -> (
    None
):
    ledger = ResearchLedger()
    task = diagnosis()
    task.question = "What are its conditions and judicial effects?"
    task.query = "application condition current text judgments annulment"
    need = ledger.bind_task(
        task,
        ["parent"],
        covered_dimensions=[LegalDimension.PENALTIES, LegalDimension.CASE_LAW],
    )
    action = SourceAction(
        issue_ids=["parent"],
        tool="search_corpus",
        arguments={"query": task.query},
        research_need_ids=[need.need_id],
    )
    assert len(ledger.admit([action], plan())) == 1
    for dimension in [LegalDimension.PENALTIES, LegalDimension.CASE_LAW]:
        later = diagnosis(existing=need.need_id)
        later.dimension = dimension
        assert ledger.bind_task(later, ["parent"]) is need
    assert ledger.admit([action], plan()) == []
    unrelated = diagnosis(existing=need.need_id)
    unrelated.dimension = LegalDimension.PROCEDURE
    with pytest.raises(ValueError, match="another dimension"):
        ledger.bind_task(unrelated, ["parent"])
    assert len(ledger.needs) == 1
