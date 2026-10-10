import json
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from openai.types.responses import (
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
)
from openai.types.responses.response_text_config_param import ResponseTextConfigParam
from pydantic import JsonValue, ValidationError

from onyx.legal_review.diagnostics import (
    OpenAIReviewDiagnoser,
    _check_scope,
    _compile_diagnoses,
    _response_model,
)
from onyx.legal_review.models import ReviewCheck, ReviewDiagnosisBatch, StrictModel
from onyx.legal_review.review_scope import scope_review_state


def test_repair_diagnosis_receives_the_matching_confirmed_publication_defect() -> None:
    check = ReviewCheck(
        id="all_answer_claims",
        instructions="Check all assertions",
        dimension="procedure_and_deadlines",
    )
    defect: dict[str, JsonValue] = {
        "check_id": check.id,
        "disposition": "defect",
        "target": "assertion",
        "answer_quotes": ["Approval automatically releases the security."],
        "reason": "Approval and release have different conditions.",
        "required_change": "Preserve the release conditions.",
        "supports": [],
    }
    state: dict[str, JsonValue] = {
        "request": "When is the security released?",
        "draft": {"answer": "Approval automatically releases the security."},
        "draft_adjudication": {
            "findings": [
                defect,
                {
                    **defect,
                    "check_id": "unrelated",
                    "disposition": "rebutted",
                },
            ]
        },
    }

    def complete(
        packet: dict[str, JsonValue], response_model: type[StrictModel], *_args: object
    ) -> StrictModel:
        checks = packet["flagged_checks"]
        assert isinstance(checks, list) and isinstance(checks[0], dict)
        assert checks[0]["publication_finding"] == defect
        assert "unrelated" not in json.dumps(packet)
        return response_model.model_validate(
            {
                "q0001": {
                    "kind": "correction",
                    "dimension": check.dimension,
                    "assertion": "Release is conditional.",
                    "reason": "The original requires settlement.",
                    "required_change": "Keep that condition.",
                    "supports": [],
                }
            }
        )

    examiner = OpenAIReviewDiagnoser(api_key="test")
    with patch.object(examiner, "_complete", side_effect=complete) as call:
        examiner.diagnose(state, [check], 10)
    call.assert_called_once()
    assert state["draft_adjudication"] is not None


def _mixed_research_batch(fresh_count: int) -> ReviewDiagnosisBatch:
    tasks = [
        {
            "subject": f"Subject {index}",
            "question": f"Which effect {index}?",
            "dimension": "procedure_and_deadlines",
            "supports": [],
            "query": f"specific effect {index}" if index < fresh_count else None,
            "existing_need_id": None if index < fresh_count else "needed_prior",
        }
        for index in range(fresh_count + 1)
    ]
    return _compile_diagnoses(
        {
            "q0001": {
                "kind": "research",
                "assertion": "Conditions remain open.",
                "reason": "Need these sources.",
                "required_change": "Read the missing effect.",
                "supports": [],
                "dimension": "procedure_and_deadlines",
                "research_tasks": tasks,
            }
        },
        {"q0001": ReviewCheck(id="check", instructions="Check conditions")},
    )


def test_one_fresh_search_with_prior_attempts_needs_no_replanning_call() -> None:
    batch = _mixed_research_batch(1)
    examiner = OpenAIReviewDiagnoser(api_key="test")
    with patch.object(examiner, "_complete") as call:
        result = examiner.consolidate({}, batch, 10)
    assert result is batch
    call.assert_not_called()


def test_consolidation_registry_cannot_import_unrelated_old_work() -> None:
    batch = _mixed_research_batch(2)
    tasks = {row.task_id: row for row in batch.research_tasks}
    examiner = OpenAIReviewDiagnoser(api_key="test")

    def complete(
        packet: dict[str, JsonValue], response_model: type[StrictModel], *_args: object
    ) -> StrictModel:
        assert "unrelated_prior" not in json.dumps(packet)
        assert "unrelated_prior" not in json.dumps(response_model.model_json_schema())
        assert "needed_prior" in json.dumps(response_model.model_json_schema())
        rows = packet["accepted_gaps"]
        assert isinstance(rows, list)
        return response_model.model_validate(
            {
                str(row["slot"]): {
                    "coverage_reason": "Preserve this accepted obligation.",
                    "investigations": [
                        tasks[str(row["task_id"])].model_dump(exclude={"task_id"})
                    ],
                }
                for row in rows
                if isinstance(row, dict)
            }
        )

    state: dict[str, JsonValue] = {
        "request": "What conditions apply?",
        "research_needs": [
            {
                "need_id": identity,
                "attempted": True,
                "origin": "review",
                "dimension": "procedure_and_deadlines",
            }
            for identity in ("needed_prior", "unrelated_prior")
        ],
    }
    with patch.object(examiner, "_complete", side_effect=complete):
        result = examiner.consolidate(state, batch, 10)
    assert len(result.research_tasks) == 3
    assert {task.existing_need_id for task in result.research_tasks} == {
        None,
        "needed_prior",
    }


def test_final_diagnosis_keeps_all_outcome_bindings_without_private_conclusions() -> (
    None
):
    state: dict[str, JsonValue] = {
        "draft": {"answer": "Başvuru için belgenin ibrazı gereklidir."},
        "dimension_assessments": [
            {
                "issue_id": "i1",
                "requirement_ids": ["r1", "r2"],
                "reason": "Old opinion",
            },
            {"issue_id": "i2", "requirement_ids": ["r2"], "reason": "Other opinion"},
        ],
        "requirements": [
            {"requirement_id": identity, "rule": "Private extraction", "supports": []}
            for identity in ("r1", "r2")
        ],
        "research_needs": [{"need_id": "existing_attempt", "attempted": True}],
    }
    scoped = scope_review_state(state)
    assert _check_scope(
        scoped, ReviewCheck(id="dimension", instructions="Check", issue_id="i1")
    ) == ["r1", "r2"]
    assert _check_scope(
        scoped, ReviewCheck(id="dimension", instructions="Check", issue_id="i2")
    ) == ["r2"]
    assert scoped["research_needs"] == state["research_needs"]
    assert scoped["draft"] == state["draft"]
    assert "Private extraction" not in str(scoped) and "Old opinion" not in str(scoped)
    assert "requirements" in state and "dimension_assessments" in state


@pytest.mark.parametrize("prior_gap", [None, "review_need_4"])
def test_code_owned_slots_cover_every_flag_in_one_batched_request(
    prior_gap: str | None,
) -> None:
    checks = [
        ReviewCheck(id="i1:özgün", instructions="Check one"),
        ReviewCheck(id="claim:two", instructions="Check two"),
    ]
    usage = MagicMock()
    client = MagicMock()
    client.__enter__.return_value = client

    def streaming(**kwargs: object) -> MagicMock:
        assert kwargs["model"] == "gpt-6.1-sol"
        assert kwargs["store"] is False
        packet = json.loads(str(kwargs["input"]))
        assert len(packet["flagged_checks"]) == 2
        text = cast(ResponseTextConfigParam, kwargs["text"])
        output_format = text["format"]
        assert output_format["type"] == "json_schema"
        schema = output_format["schema"]
        assert schema["required"] == ["q0001", "q0002"]
        assert output_format["strict"] is True
        assert kwargs["stream"] is True
        definitions = json.loads(json.dumps(schema))["$defs"]
        assert (
            definitions["q0001ResearchFinding"]["properties"]["research_tasks"][
                "minItems"
            ]
            == 1
        )
        assert (
            "research_tasks"
            not in definitions["q0001_NonResearchFinding"]["properties"]
        )
        assert "issue:initial" not in json.dumps(schema)
        assert packet["state"]["initial_discovery"][0]["need_id"] == "issue:initial"
        assert all(
            row["origin"] != "question" for row in packet["state"]["research_needs"]
        )
        need_schema = definitions[
            "AttemptedQuestion_validity_and_timing"
            if prior_gap
            else "FreshResearchQuestion"
        ]["properties"]["existing_need_id"]
        if prior_gap:
            assert prior_gap in json.dumps(need_schema)
        else:
            assert need_schema["type"] == "null"
        tasks = [
            {
                "subject": "Application permission",
                "question": "Which permission applies?",
                "query": "application permission conditions",
                "existing_need_id": None,
                "supports": [{"citation": 1, "span_number": 1}],
                "dimension": "procedure_and_deadlines",
            },
            {
                "subject": "Eligibility exception",
                "question": "What limits the exception?",
                "query": "eligibility exception scope",
                "existing_need_id": None,
                "supports": [{"citation": 1, "span_number": 1}],
                "dimension": "exceptions_and_exemptions",
            },
        ]
        data = {
            "q0001": {
                "kind": "research",
                "assertion": "The application is treated as unconditional.",
                "reason": "The original requires a separate permission.",
                "required_change": "Read the permission conditions.",
                "supports": [{"citation": 1, "span_number": 1}],
                "research_tasks": tasks,
                "dimension": "procedure_and_deadlines",
            },
            "q0002": {"same_as": "q0001"},
        }
        response = SimpleNamespace(
            status="completed",
            output=[],
            output_text=json.dumps(data),
            usage=SimpleNamespace(input_tokens=42, output_tokens=12),
        )
        stream = MagicMock()
        stream.__enter__.return_value = stream
        stream.__iter__.return_value = iter(
            [
                SimpleNamespace(type="response.created"),
                ResponseCompletedEvent.model_construct(response=response),
            ]
        )
        return stream

    client.responses.create.side_effect = streaming
    state: dict[str, JsonValue] = {
        "research_needs": [
            {
                "need_id": "issue:initial",
                "question": "Which procedure applies?",
                "origin": "question",
                "attempted": True,
            },
            *(
                [
                    {
                        "need_id": prior_gap,
                        "question": "Was another condition amended?",
                        "origin": "review",
                        "attempted": True,
                        "dimension": "validity_and_timing",
                    }
                ]
                if prior_gap
                else []
            ),
        ],
        "original_evidence": [
            {
                "source_id": "law",
                "citation": 1,
                "metadata": {},
                "passages": [{"span_number": 1, "text": "Permission is required."}],
            }
        ],
    }
    with patch("openai.OpenAI", return_value=client) as provider:
        result = OpenAIReviewDiagnoser(api_key="test", record_usage=usage)._diagnose(
            state, checks, 10
        )
    assert result.diagnoses[0].check_ids == [check.id for check in checks]
    assert result.diagnoses[0].research_task_ids == ["task_1", "task_2"]
    assert len(result.research_tasks) == 2
    usage.assert_called_once_with(42, 12)
    client.responses.create.assert_called_once()
    assert provider.call_args.kwargs["max_retries"] == 0
    assert client.responses.create.call_args.kwargs["max_output_tokens"] == 128_000


def test_judicial_check_cannot_be_replaced_by_a_penalty_diagnosis() -> None:
    slots = {
        "q0001": ReviewCheck(
            id="penalty",
            instructions="Check conditions",
            dimension="penalties_and_reductions",
        ),
        "q0002": ReviewCheck(
            id="judicial",
            instructions="Check judicial effect",
            dimension="case_law_and_rulings",
        ),
        "q0003": ReviewCheck(
            id="judicial2",
            instructions="Check another judicial effect",
            dimension="case_law_and_rulings",
        ),
    }
    model = _response_model(slots, [])
    diagnosis = {
        "kind": "disputed",
        "assertion": "A condition is established.",
        "reason": "The passage supplies the condition.",
        "required_change": "None.",
        "supports": [],
        "dimension": "penalties_and_reductions",
    }
    payload = {
        "q0001": diagnosis,
        "q0002": {"same_as": "q0001"},
        "q0003": {"same_as": "q0002"},
    }
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    payload["q0002"] = diagnosis
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    payload["q0002"] = {**diagnosis, "dimension": "case_law_and_rulings"}
    assert model.model_validate(payload)
    references = model.model_json_schema()["$defs"]["q0003Reference"]["properties"][
        "same_as"
    ]
    assert references["const"] == "q0002"


def test_judicial_task_cannot_reuse_a_conditions_search_in_provider_schema() -> None:
    needs: list[dict[str, JsonValue]] = [
        {
            "need_id": "conditions",
            "dimension": "penalties_and_reductions",
            "attempted": True,
        }
    ]
    slots = {"q0001": ReviewCheck(id="judicial", instructions="Check judicial effect")}
    model = _response_model(slots, needs)
    task = {
        "subject": "Enabling norm",
        "question": "Was the enabling power annulled?",
        "dimension": "case_law_and_rulings",
        "supports": [],
        "query": None,
        "existing_need_id": "conditions",
    }
    payload = {
        "q0001": {
            "kind": "research",
            "assertion": "The power applies.",
            "reason": "Judicial effects have not been investigated.",
            "required_change": "Investigate changes to the enabling norm.",
            "supports": [],
            "research_tasks": [task],
            "dimension": "case_law_and_rulings",
        },
    }
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    needs[0]["covered_dimensions"] = [
        "case_law_and_rulings",
        "penalties_and_reductions",
    ]
    assert _response_model(slots, needs).model_validate(payload)
    task.update(existing_need_id=None, query="enabling norm annulment judicial effect")
    assert model.model_validate(payload)


def test_duplicate_review_inventory_does_not_call_provider() -> None:
    check = ReviewCheck(id="one", instructions="Check")
    with patch("openai.OpenAI") as provider:
        with pytest.raises(ValueError, match="unique"):
            OpenAIReviewDiagnoser(api_key="test").diagnose({}, [check, check], 10)
    provider.assert_not_called()


@pytest.mark.parametrize(
    ("status", "reason", "body", "message"),
    [
        (
            "incomplete",
            "max_output_tokens",
            '{"q0001":',
            "incomplete: max_output_tokens",
        ),
        ("completed", None, '{"q0001":', "Invalid JSON"),
        ("failed", "server_error", "", "failed: server_error"),
    ],
)
def test_terminal_failure_records_usage_before_parsing_without_retry(
    status: str, reason: str | None, body: str, message: str
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    usage = MagicMock()
    response = SimpleNamespace(
        status=status,
        output=[],
        output_text=body,
        incomplete_details=SimpleNamespace(reason=reason)
        if status == "incomplete"
        else None,
        error=SimpleNamespace(code=reason) if status == "failed" else None,
        usage=SimpleNamespace(input_tokens=123, output_tokens=8192),
    )
    stream = client.responses.create.return_value
    stream.__enter__.return_value = stream
    event_model = {
        "completed": ResponseCompletedEvent,
        "incomplete": ResponseIncompleteEvent,
        "failed": ResponseFailedEvent,
    }[status]
    stream.__iter__.return_value = iter(
        [event_model.model_construct(response=response)]
    )
    with patch("openai.OpenAI", return_value=client):
        with pytest.raises(ValueError, match=message):
            OpenAIReviewDiagnoser(api_key="test", record_usage=usage).diagnose(
                {}, [ReviewCheck(id="one", instructions="Check")], 10
            )
    usage.assert_called_once_with(123, 8192)
    client.responses.create.assert_called_once()


def test_stream_without_terminal_response_is_not_accepted() -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    stream = client.responses.create.return_value
    stream.__enter__.return_value = stream
    stream.__iter__.return_value = iter([SimpleNamespace(type="response.created")])
    with patch("openai.OpenAI", return_value=client):
        with pytest.raises(ValueError, match="no terminal response"):
            OpenAIReviewDiagnoser(api_key="test").diagnose(
                {}, [ReviewCheck(id="one", instructions="Check")], 10
            )


def test_independent_dimension_scope_includes_findings_from_other_dimensions() -> None:
    state: dict[str, JsonValue] = {
        "dimension_assessments": [
            {
                "issue_id": "i1",
                "dimension": "case_law_and_rulings",
                "requirement_ids": ["financial_rule"],
            },
            {
                "issue_id": "i1",
                "dimension": "penalties_and_reductions",
                "requirement_ids": ["statutory_penalty"],
            },
            {
                "issue_id": "other",
                "dimension": "procedure_and_deadlines",
                "requirement_ids": ["unrelated"],
            },
        ]
    }
    assert _check_scope(
        state,
        ReviewCheck(
            id="i1:court",
            issue_id="i1",
            dimension="case_law_and_rulings",
            instructions="Review all controlling bases",
        ),
    ) == ["financial_rule", "statutory_penalty"]


def test_inline_questions_get_unique_host_ids_and_shared_supports_without_orphans() -> (
    None
):
    question = {
        "subject": "Primary power",
        "question": "Has the power changed?",
        "dimension": "case_law_and_rulings",
        "query": "primary power annulment decision",
        "existing_need_id": None,
        "supports": [{"citation": 1, "span_number": 1}],
    }
    common = {
        "kind": "research",
        "assertion": "The power applies.",
        "reason": "The authority has not been verified.",
        "required_change": "Read authoritative changes.",
        "supports": [],
        "dimension": "case_law_and_rulings",
    }
    data = {
        "q0001": {**common, "research_tasks": [question]},
        "q0002": {
            **common,
            "research_tasks": [
                {**question, "supports": [{"citation": 2, "span_number": 1}]}
            ],
        },
        "q0003": {"same_as": "q0002"},
    }
    slots = {slot: ReviewCheck(id=slot, instructions="Check") for slot in data}
    result = _compile_diagnoses(data, slots)
    assert len(result.research_tasks) == 1
    assert result.research_tasks[0].task_id == "task_1"
    assert [s.citation for s in result.research_tasks[0].supports] == [1, 2]
    assert [d.research_task_ids for d in result.diagnoses] == [["task_1"], ["task_1"]]
    assert result.diagnoses[1].check_ids == ["q0002", "q0003"]


def test_two_independent_gaps_can_share_one_search_without_losing_their_checks() -> (
    None
):
    from onyx.legal_review.models import LegalDimension
    from onyx.tracing.flows import LLMFlow

    checks = [
        ReviewCheck(
            id="conditions",
            issue_id="i1",
            instructions="Check conditions",
            dimension=LegalDimension.PENALTIES,
        ),
        ReviewCheck(
            id="judgments",
            issue_id="i1",
            instructions="Check judicial effects",
            dimension=LegalDimension.CASE_LAW,
        ),
    ]
    statement = {
        "kind": "research",
        "assertion": "This provision applies.",
        "reason": "Its decisive effects are not established.",
        "required_change": "Research the controlling provision.",
        "supports": [],
    }
    question = {
        "subject": "Enabling provision",
        "question": "What are the operative conditions and judicial effects?",
        "query": "enabling provision conditions amount judgments annulment",
        "existing_need_id": None,
        "supports": [],
        "dimension": "penalties_and_reductions",
    }
    native = {
        f"q{index:04d}": {
            **statement,
            "dimension": check.dimension,
            "research_tasks": [{**question, "dimension": check.dimension}],
        }
        for index, check in enumerate(checks, 1)
    }
    calls: list[LLMFlow] = []

    def complete(
        packet: dict[str, JsonValue],
        response_model: type[StrictModel],
        _prompt: str,
        flow: LLMFlow,
        _timeout_seconds: float,
        _max_output_tokens: int,
    ) -> StrictModel:
        calls.append(flow)
        if flow == LLMFlow.LEGAL_REVIEW_DIAGNOSIS:
            return response_model.model_validate(native)
        accepted = packet["accepted_gaps"]
        assert isinstance(accepted, list)
        slots = [row["slot"] for row in accepted if isinstance(row, dict)]
        assert all(isinstance(slot, str) for slot in slots)
        return response_model.model_validate(
            {
                slots[0]: {
                    "coverage_reason": "Shared norm and complete scope",
                    "investigations": [{"reuse_group": slots[1]}],
                },
                slots[1]: {
                    "coverage_reason": "Conditions and judgments in one search",
                    "investigations": [question],
                },
            }
        )

    examiner = OpenAIReviewDiagnoser(api_key="test")
    with patch.object(examiner, "_complete", side_effect=complete):
        result = examiner.diagnose({"request": "Which provision applies?"}, checks, 10)
    assert calls == [LLMFlow.LEGAL_REVIEW_DIAGNOSIS, LLMFlow.LEGAL_REVIEW_RESEARCH_PLAN]
    assert len(result.research_tasks) == 1
    assert {row.check_ids[0] for row in result.diagnoses} == {"conditions", "judgments"}
    assert all(row.research_task_ids == ["investigation_1"] for row in result.diagnoses)
    assert set(result.research_coverage["investigation_1"]) == {
        LegalDimension.PENALTIES,
        LegalDimension.CASE_LAW,
    }


def test_provider_schema_requires_query_for_new_work_and_known_attempt_for_reuse() -> (
    None
):
    import jsonschema

    slot = {
        "q0001": ReviewCheck(id="gap", instructions="Investigate a missing condition")
    }
    needs: list[dict[str, JsonValue]] = [
        {
            "need_id": "prior",
            "dimension": "procedure_and_deadlines",
            "attempted": True,
            "origin": "review",
        }
    ]
    model = _response_model(slot, needs)
    validate = jsonschema.Draft202012Validator(model.model_json_schema()).validate
    question = {
        "subject": "Application rule",
        "question": "What starts the deadline?",
        "dimension": "procedure_and_deadlines",
        "supports": [],
        "query": None,
        "existing_need_id": None,
    }
    payload = {
        "q0001": {
            "kind": "research",
            "assertion": "The deadline applies",
            "reason": "The trigger is unread",
            "required_change": "Read the trigger",
            "supports": [],
            "dimension": "procedure_and_deadlines",
            "research_tasks": [question],
        }
    }
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)
    question["query"] = "application deadline trigger"
    validate(payload)
    question.update(query=None, existing_need_id="prior")
    validate(payload)
    question["existing_need_id"] = "invented"
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)
    question.update(existing_need_id="prior", dimension="case_law_and_rulings")
    with pytest.raises(jsonschema.ValidationError):
        validate(payload)


def test_publication_schema_selects_literal_answer_passages_instead_of_retyping() -> (
    None
):
    from onyx.legal_review.adjudication import publication_response_model

    response = publication_response_model(
        [ReviewCheck(id="condition", instructions="Check source condition")],
        {"a0001": "İbraz şarttır.", "a0002": "Bilinmeyen husus açık bırakılmıştır."},
    )
    proposal = {
        "disposition": "rebutted",
        "target": "assertion",
        "reason": "Condition present",
        "required_change": None,
        "supports": [],
        "answer_spans": ["a0001"],
    }
    assert response.model_validate({"q0001": proposal}).model_dump()["q0001"][
        "answer_spans"
    ] == ["a0001"]
    with pytest.raises(ValidationError):
        response.model_validate({"q0001": {**proposal, "answer_spans": ["invented"]}})
    with pytest.raises(ValidationError):
        response.model_validate({"q0001": {**proposal, "answer_spans": []}})


def test_cited_source_checks_preserve_each_basis_without_creating_issues() -> None:
    from onyx.legal_review.adjudication import cited_source_checks
    from onyx.legal_review.models import AnswerClaim, DraftAnswer, PassageSupport

    draft = DraftAnswer(
        answer="The general entitlement and narrower procedure have different conditions.",
        claims=[
            AnswerClaim(
                claim_id="general",
                issue_ids=["eligibility"],
                answer_excerpt="General entitlement.",
                supports=[PassageSupport(citation=2, span_number=1)],
            ),
            AnswerClaim(
                claim_id="procedure",
                issue_ids=["procedure"],
                answer_excerpt="Narrower procedure.",
                supports=[
                    PassageSupport(citation=2, span_number=1),
                    PassageSupport(citation=7, span_number=1),
                    PassageSupport(citation=7, span_number=2),
                ],
            ),
        ],
    )
    before = draft.model_dump()
    checks = cited_source_checks(draft)
    assert [check.source_citation for check in checks] == [2, 7]
    assert len({check.id for check in checks}) == 2
    assert checks[0].issue_id is None
    assert checks[1].issue_id == "procedure"
    assert "general, procedure" in checks[0].instructions
    assert draft.model_dump() == before
