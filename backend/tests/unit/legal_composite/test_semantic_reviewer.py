"""Native review identity, complete evidence, uncertainty, and provider access fences."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from threading import Barrier, Event, Lock, get_ident
from typing import Any, cast
from unittest.mock import Mock

import httpx
import pytest

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, RunStopped
from onyx.asv3.witnesses import original_witness_spans
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.legal_composite import providers
from onyx.legal_composite.budget import WorkflowBudget
from onyx.legal_composite.gateway import BudgetedGateway
from onyx.legal_composite.models import (
    AnswerSection,
    AuthorityDependency,
    DependencyOrigin,
    DraftClaim,
    GapResolution,
    PassageSupport,
    SourceRequirement,
    SpanSupport,
    WorkflowPolicy,
)
from onyx.legal_composite.models import (
    IssueResearchNeed as ResearchNeed,
)
from onyx.legal_composite.models import (
    IssueResearchPlan as ResearchPlan,
)
from onyx.legal_composite.models import (
    StructuredDraftAnswer as DraftAnswer,
)
from onyx.legal_composite.reviewer import (
    REVIEW_DIMENSIONS,
    DecisionsAnswerReviewer,
    GatewayAnswerReviewer,
    ReviewQuestion,
)
from onyx.llm.interfaces import LLMConfig


def inputs(
    *texts: str,
) -> tuple[EvidenceLedger, ResearchPlan, DraftAnswer, list[SourceRequirement]]:
    ledger = EvidenceLedger()
    ledger.add(
        [
            EvidenceItem(
                source_id=f"source-{index}",
                chunk_id=f"chunk-{index}",
                text=text,
                question_ids=["n1"],
                search_doc=SearchDoc(
                    document_id=f"source-{index}",
                    chunk_ind=index,
                    semantic_identifier="Original",
                    blurb="Navigation",
                    source_type=DocumentSource.FILE,
                    boost=0,
                    hidden=False,
                    metadata={"regulatory_chunk_id": f"chunk-{index}"},
                    match_highlights=[],
                ),
            )
            for index, text in enumerate(
                texts or ("A permit is required only if the goods are controlled.",), 1
            )
        ],
        RunContext(),
    )
    plan = ResearchPlan(
        language="en",
        requires_sources=True,
        needs=[
            ResearchNeed(
                need_id="n1",
                question="What permit is required?",
                governing_source="Law",
                conditions_to_check=["controlled goods"],
            )
        ],
        initial_actions=[],
        missing_user_facts=[],
    )
    text = "A permit is required for controlled goods [1]."
    draft = DraftAnswer(
        unresolved_need_ids=[],
        sections=[AnswerSection(section_id="s1", need_ids=["n1"], text=text)],
        claims=[
            DraftClaim(
                claim_id="c1",
                section_id="s1",
                need_ids=["n1"],
                answer_excerpt=text,
                requirement_ids=["r1"],
            )
        ],
    )
    original = ledger.get(1)
    assert original is not None
    requirements = [
        SourceRequirement(
            requirement_id="r1",
            need_id="n1",
            dimension="legal_basis_and_hierarchy",
            rule="A permit is required for controlled goods.",
            application="Check the goods' control status.",
            supports=[SpanSupport(citation=1, quotation=original.text)],
        )
    ]
    return ledger, plan, draft, requirements


def reviewer(
    ledger: EvidenceLedger,
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    provider: str = "openrouter",
    policy: WorkflowPolicy | None = None,
) -> DecisionsAnswerReviewer:
    return DecisionsAnswerReviewer(
        config=LLMConfig(
            model_provider=provider,
            model_name="typesafe/jev-1.13"
            if provider == "openrouter"
            else "gpt-6-luna",
            temperature=1,
            api_key="test-not-real",
            max_input_tokens=32_000,
        ),
        budget=WorkflowBudget(
            policy or WorkflowPolicy(max_model_calls=100, max_input_tokens=1_000_000)
        ),
        ledger=ledger,
        transport=httpx.MockTransport(handler),
        token_counter=lambda text: len(text) // 4,
    )


def response(
    payload: dict[str, Any], *, status: str = "addressed", probability: float = 0.99
) -> dict[str, Any]:
    questions = payload["questions"]
    rows = (
        questions.items()
        if isinstance(questions, dict)
        else ((q["name"], q) for q in questions)
    )
    answers: dict[str, Any] = {}
    for identity, question in rows:
        labels = (
            list(question["criteria"])
            if "criteria" in question
            else [c["value"] for c in question["choices"]]
        )
        chosen = status if status in labels else "addressed"
        values = {
            label: probability
            if label == chosen
            else (1 - probability) / (len(labels) - 1)
            for label in labels
        }
        answers[identity] = {
            "type": "choice",
            "choice": chosen,
            "confidence": 0.99,
            "probabilities": values,
        }
    if payload["model"] == "gpt-6-luna":
        native = []
        for identity, answer in answers.items():
            native.append(
                {
                    **answer,
                    "name": identity,
                    "probabilities": [
                        {"value": value, "probability": probability}
                        for value, probability in answer["probabilities"].items()
                    ],
                }
            )
        output: Any = native
    else:
        output = answers
    return {
        "id": "native-review",
        "model": payload["model"] + "-20260917",
        "answers": output,
        "usage": {"input_tokens": 476, "output_tokens": 70, "cost": 0.000019992},
    }


@pytest.mark.parametrize(
    "provider,endpoint,rate",
    [
        ("openrouter", "https://openrouter.ai/api/alpha/decisions", 0.042),
        ("openai", "https://api.openai.com/v1/decisions", 0.10),
    ],
)
def test_one_typed_review_preserves_origins_host_ids_and_input_only_accounting(
    provider: str, endpoint: str, rate: float
) -> None:
    ledger, plan, draft, requirements = inputs()
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == endpoint
        body = json.loads(request.content)
        calls.append(body)
        state = body.get("state") or json.loads(body["input"])
        original = ledger.get(1)
        assert original is not None
        assert state["originals"][0]["text"] == original.text
        assert state["originals"][0]["text_hash"] == original.text_hash
        assert state["claims"][0]["requirement_ids"] == ["r1"]
        assert "response_format" not in body and "reasoning" not in body
        questions = body["questions"]
        rows = questions.values() if isinstance(questions, dict) else questions
        assert all("checks[" in q["instructions"] for q in rows)
        return httpx.Response(200, json=response(body))

    model = reviewer(ledger, handler, provider=provider)
    expected = model.expected_checks("Question", plan, draft, requirements, [], {1})
    assert len(expected) == len(REVIEW_DIMENSIONS) + 7
    assert "evidence:n1" in expected
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert result.failure is None
    assert {check.check_id for check in result.checks} == set(expected)
    assert all(check.status == "addressed" for check in result.checks)
    assert len(calls) == 1
    assert model.budget.snapshot()["estimated_cost_usd"] == pytest.approx(
        476 * rate / 1_000_000, abs=5e-9
    )
    assert model.budget.snapshot()["output_tokens"] == 0
    assert (
        ledger.delivery_flow(next(iter(model.budget._reservations)))
        == "legal_composite_review"
    )


def test_omitted_original_and_dependency_have_independent_checks() -> None:
    ledger, plan, draft, requirements = inputs(
        "A permit applies.", "A favorable exemption displaces the permit."
    )
    origin = ledger.get(1)
    assert origin is not None
    dependency = AuthorityDependency(
        edge_id="edge",
        need_ids=["n1"],
        instrument_name="Exemption",
        article="2",
        origins=[
            DependencyOrigin(
                citation=1,
                source_id=origin.source_id,
                chunk_id=origin.chunk_id,
                text_hash=origin.text_hash,
            )
        ],
        candidate_citations=[2],
    )
    model = reviewer(
        ledger, lambda req: httpx.Response(200, json=response(json.loads(req.content)))
    )
    checks = model.expected_checks(
        "Question", plan, draft, requirements, [dependency], {1, 2}
    )
    assert checks["dependency:edge"].citations == [1, 2]
    assert checks["original:2"].citations == [2]
    assert "favorable" in checks["original:2"].question
    assert not checks["claim:c1"].allow_not_applicable


@pytest.mark.parametrize(
    "defect", ["missing", "unexpected", "nan", "duplicate", "wrong_model"]
)
def test_invalid_native_judgment_cannot_approve_and_does_not_retry(defect: str) -> None:
    ledger, plan, draft, requirements = inputs()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        body = response(json.loads(request.content))
        key = next(iter(body["answers"]))
        if defect == "missing":
            del body["answers"][key]
        elif defect == "unexpected":
            body["answers"]["model_invented"] = body["answers"][key]
        elif defect == "nan":
            body["answers"][key]["confidence"] = True
        elif defect == "duplicate":
            body["answers"][key]["probabilities"]["unexpected"] = 0
        elif defect == "wrong_model":
            body["model"] = "other/model"
        return httpx.Response(200, json=body)

    model = reviewer(ledger, handler)
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert result.failure
    assert all(check.status == "uncertain" for check in result.checks)
    assert calls == 1


def test_low_probability_remains_uncertain_and_prior_approval_is_rechecked() -> None:
    ledger, plan, draft, requirements = inputs()
    probability = 0.99
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json=response(json.loads(request.content), probability=probability)
        )

    model = reviewer(ledger, handler)
    prior = model.review("Question", plan, draft, requirements, [], {1})
    probability = 0.70
    result = model.review(
        "Question",
        plan,
        draft,
        requirements,
        [],
        {1},
        previous=prior,
        affected_sections=set(),
    )
    assert calls == 2
    assert all(check.status == "uncertain" for check in result.checks)


def test_missing_or_oversized_original_is_never_clipped_or_certified() -> None:
    ledger, plan, draft, requirements = inputs("A complete required rule. " * 10_000)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("Oversized required context must not be sent")

    model = reviewer(ledger, handler)
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert calls == 0 and result.failure
    assert all(check.status == "uncertain" for check in result.checks)


def test_parallel_partitions_are_bounded_to_four_and_originals_stay_whole(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, plan, draft, requirements = inputs(
        "Required original.", "Additional exception."
    )
    barrier = Barrier(4)
    lock = Lock()
    active = peak = total = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal active, peak, total
        with lock:
            active += 1
            total += 1
            peak = max(peak, active)
            first = total <= 4
        if first:
            barrier.wait(timeout=3)
        body = json.loads(request.content)
        for original in body["state"]["originals"]:
            item = ledger.get(original["citation"])
            assert item is not None and original["text"] == item.text
        with lock:
            active -= 1
        return httpx.Response(200, json=response(body))

    monkeypatch.setattr(
        "onyx.legal_composite.reviewer._estimated_decision_input_tokens",
        lambda body, _counter: len(body["questions"]) * 1000,
    )
    model = reviewer(
        ledger,
        handler,
        policy=WorkflowPolicy(
            max_context_tokens=1500, max_model_calls=100, max_input_tokens=1_000_000
        ),
    )
    result = model.review("Question", plan, draft, requirements, [], {1, 2})
    assert result.failure is None
    assert peak == 4 and total > 4
    assert all(check.status == "addressed" for check in result.checks)


def test_timeout_keeps_unknown_spend_reserved_and_returns_uncertain() -> None:
    ledger, plan, draft, requirements = inputs()
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("private provider body must not leak")

    model = reviewer(ledger, handler)
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert attempts == 1 and result.failure
    assert "private" not in result.failure
    assert all(check.status == "uncertain" for check in result.checks)
    assert model.budget.snapshot()["estimated_cost_usd"] > 0


def test_provider_factory_uses_authorized_openrouter_without_global_model_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_config = Mock(is_visible=True, max_input_tokens=1050000)
    model_config.name = "openai/gpt-5.6-luna"
    public = Mock(id=3, provider="openrouter", model_configurations=[model_config])
    public.name = "OpenRouter "
    private = Mock(
        provider="openrouter",
        api_key="test-not-real",
        api_base="https://openrouter.ai/api/v1",
    )
    fetch = Mock(return_value=private)
    cost = Mock()
    monkeypatch.setattr(
        providers,
        "fetch_all_accessible_llm_providers",
        lambda _session, _user: [public],
    )
    monkeypatch.setattr(providers, "fetch_accessible_llm_provider_by_id", fetch)
    monkeypatch.setattr(providers, "is_onyx_managed_api_key", lambda _key: False)
    monkeypatch.setattr(providers, "check_llm_cost_limit_for_provider", cost)
    ledger, *_ = inputs()
    session, user = Mock(), Mock(id="original-actor")
    result = providers.build_answer_reviewer(
        session=session,
        user=user,
        budget=WorkflowBudget(
            WorkflowPolicy(max_context_tokens=128_000, max_input_tokens=500_000)
        ),
        ledger=ledger,
        check_active=lambda: None,
        token_counter=None,
        run_id="run",
        scope={},
    )
    assert isinstance(result, GatewayAnswerReviewer)
    assert result.mode == "structured_generation"
    assert result.config.model_name == "openai/gpt-5.6-luna"
    assert result.config.max_input_tokens == 128_000
    fetch.assert_called_once_with(session, user, 3)
    cost.assert_called_once()
    assert public.model_configurations == [model_config]
    sessions: list[Mock] = []
    main_thread = get_ident()

    @contextmanager
    def fresh_pricing_session() -> Any:
        assert get_ident() == main_thread
        fresh = Mock(closed=False)
        sessions.append(fresh)
        try:
            yield fresh
        finally:
            fresh.closed = True

    def construct_gateway(**kwargs: Any) -> Mock:
        assert kwargs["db_session"] is sessions[-1]
        assert kwargs["db_session"] is not session
        assert sessions[-1].closed is False
        assert kwargs["user_identity"].user_id == "original-actor"
        assert kwargs["reserve_finalization"] is False
        return Mock(spec=BudgetedGateway)

    monkeypatch.setattr(
        providers, "get_session_with_current_tenant", fresh_pricing_session
    )
    monkeypatch.setattr(providers, "get_llm", Mock())
    monkeypatch.setattr(providers, "BudgetedGateway", construct_gateway)
    session.close()
    user.id = "changed-after-original-session-close"
    result.gateway_factory()
    result.gateway_factory()
    assert len(sessions) == 2 and sessions[0] is not sessions[1]
    assert all(fresh.closed for fresh in sessions)


def test_unsupported_openrouter_endpoint_does_not_invoke_an_alternate_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    public = [
        Mock(id=3, name="OpenRouter", provider="openrouter", model_configurations=[]),
        Mock(
            id=7,
            name="Embedding",
            provider="openai",
            model_configurations=[Mock(name="gpt-6-luna", is_visible=True)],
        ),
    ]
    # Mock's name argument names the mock, so expose the real model-name attribute explicitly.
    public[1].model_configurations[0].name = "gpt-6-luna"
    configured_model = Mock(is_visible=True, max_input_tokens=1050000)
    configured_model.name = "openai/gpt-5.6-luna"
    public[0].model_configurations = [configured_model]
    public[0].name = "OpenRouter"
    public[1].name = "Embedding"
    configs = {
        3: Mock(
            provider="openrouter",
            api_key="or-secret",
            api_base="https://unapproved.example/api/v1",
        ),
        7: Mock(
            provider="openai",
            api_key="openai-secret",
            api_base="https://api.openai.com/v1",
        ),
    }
    monkeypatch.setattr(
        providers, "fetch_all_accessible_llm_providers", lambda _session, _user: public
    )
    fetch = Mock(side_effect=lambda _session, _user, identity: configs[identity])
    monkeypatch.setattr(
        providers,
        "fetch_accessible_llm_provider_by_id",
        fetch,
    )
    monkeypatch.setattr(providers, "is_onyx_managed_api_key", lambda _key: False)
    monkeypatch.setattr(
        providers, "check_llm_cost_limit_for_provider", lambda **_kwargs: None
    )
    ledger, *_ = inputs()
    result = providers.build_answer_reviewer(
        session=Mock(),
        user=Mock(),
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
        check_active=lambda: None,
        token_counter=None,
        run_id="run",
        scope={},
    )
    assert result is None
    assert fetch.call_args.args[2] == 3
    assert fetch.call_count == 1


def test_different_need_original_is_not_claimed_read_for_this_issue() -> None:
    ledger, plan, draft, requirements = inputs(
        "Required rule.", "Another issue's rule."
    )
    ledger._items[2].question_ids = ["n2"]
    plan.needs.append(
        ResearchNeed(
            need_id="n2",
            question="Another issue?",
            governing_source="Law",
            conditions_to_check=[],
        )
    )
    model = reviewer(
        ledger, lambda req: httpx.Response(200, json=response(json.loads(req.content)))
    )
    checks: dict[str, ReviewQuestion] = model.expected_checks(
        "Question", plan, draft, requirements, [], {1, 2}
    )
    assert checks["original:1"].need_ids == ["n1"]
    assert checks["original:2"].need_ids == ["n2"]


def test_shared_original_is_checked_once_across_all_bound_issues_and_sections() -> None:
    ledger, plan, draft, requirements = inputs()
    ledger._items[1].question_ids = ["n1", "n2"]
    plan.needs.append(
        ResearchNeed(
            need_id="n2",
            question="Related effect?",
            governing_source="Law",
            conditions_to_check=[],
        )
    )
    draft.sections.append(
        AnswerSection(
            section_id="s2",
            need_ids=["n2"],
            text="Related condition remains explicit [1].",
        )
    )
    model = reviewer(
        ledger, lambda req: httpx.Response(200, json=response(json.loads(req.content)))
    )
    checks = model.expected_checks("Question", plan, draft, requirements, [], {1})
    originals = [
        check for check in checks.values() if check.check_id.startswith("original:")
    ]
    assert len(originals) == 1
    assert originals[0].need_ids == ["n1", "n2"]
    assert originals[0].section_ids == ["s1", "s2"]
    assert "ANY bound issue" in originals[0].question
    state = cast(
        dict[str, Any],
        model._payload("Question", plan, draft, requirements, [], originals, {1})[
            "state"
        ],
    )
    assert state["originals"][0]["text"] == ledger._items[1].text
    assert {need["need_id"] for need in state["needs"]} == {"n1", "n2"}


@pytest.mark.parametrize(
    "invalid",
    ["document_id", "chunk_id", "derived", "external", "untrusted", "truncated"],
)
def test_original_omission_checks_cannot_certify_derived_or_misbound_sources(
    invalid: str,
) -> None:
    ledger, plan, draft, requirements = inputs()
    original = ledger._items[1]
    assert original.search_doc is not None
    if invalid == "document_id":
        original.search_doc.document_id = "different-source"
    elif invalid == "chunk_id":
        original.search_doc.metadata["regulatory_chunk_id"] = "different-chunk"
    else:
        original.metadata[invalid] = True
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("Unverified evidence must not reach reviewer")

    result = reviewer(ledger, handler).review(
        "Question", plan, draft, requirements, [], {1}
    )
    assert calls == 0 and result.failure
    assert all(check.status == "uncertain" for check in result.checks)


def test_precise_evidence_gap_and_unresolved_issue_reach_judge_without_becoming_missing_fact() -> (
    None
):
    ledger, plan, draft, requirements = inputs()
    plan.needs[0].evidence_gaps = ["The adverse amendment has not been read."]
    draft.unresolved_need_ids = ["n1"]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state = body["state"]
        assert state["needs"][0]["evidence_gaps"] == plan.needs[0].evidence_gaps
        assert state["unresolved_need_ids"] == ["n1"]
        assert state["missing_user_facts"] == []
        assert "specifically disclosed" in body["questions"]["issue:n1"]["instructions"]
        evidence_question = body["questions"]["evidence:n1"]
        assert (
            "even when the draft correctly discloses it"
            in evidence_question["instructions"]
        )
        assert "not_applicable" not in evidence_question["criteria"]
        assert (
            state["canonical_witnesses"][0]["quotation"]
            == requirements[0].supports[0].quotation
        )
        return httpx.Response(200, json=response(body, status="gap"))

    result = reviewer(ledger, handler).review(
        "Question", plan, draft, requirements, [], {1}
    )
    assert result.failure is None
    assert (
        next(check for check in result.checks if check.check_id == "issue:n1").status
        == "gap"
    )


def test_reviewer_preserves_selected_occurrences_of_repeated_original_text() -> None:
    repeated = "x" * 800
    ledger, plan, draft, requirements = inputs(repeated * 2)
    spans = original_witness_spans(1, repeated * 2)
    requirements[0].supports = [
        SpanSupport(citation=1, quotation=repeated, span_id=span["witness_id"])
        for span in spans
    ]
    judge = reviewer(ledger, lambda _: httpx.Response(500))
    checks = judge.expected_checks("Question", plan, draft, requirements, [], {1})
    payload = judge._payload(
        "Question", plan, draft, requirements, [], list(checks.values()), {1}
    )
    state = cast(dict[str, Any], payload["state"])
    witnesses = state["canonical_witnesses"]
    assert [(w["start_char"], w["end_char"]) for w in witnesses] == [
        (0, 800),
        (800, 1600),
    ]
    assert all(w["quotation"] == repeated for w in witnesses)
    assert (
        judge._witness(PassageSupport(citation=1, quotation=repeated), {1})[
            "start_char"
        ]
        == 0
    )


@pytest.mark.parametrize("defect", ["invented_id", "other_citation", "quotation"])
def test_reviewer_rejects_a_false_span_even_with_a_real_original_quotation(
    defect: str,
) -> None:
    text = "x" * 800 + "y" * 800
    ledger, _, _, _ = inputs(text)
    spans = original_witness_spans(1, text)
    span_id = spans[1]["witness_id"]
    quotation = "y" * 800
    if defect == "invented_id":
        span_id = "invented"
    elif defect == "other_citation":
        span_id = original_witness_spans(2, text)[1]["witness_id"]
    else:
        quotation = "x" * 800
    judge = reviewer(ledger, lambda _: httpx.Response(500))
    with pytest.raises(ValueError, match="span differs"):
        judge._witness(
            SpanSupport(citation=1, quotation=quotation, span_id=span_id), {1}
        )


def test_answer_omission_is_distinct_from_supplied_evidence_sufficiency() -> None:
    ledger, plan, draft, requirements = inputs()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert (
            "already supplied evidence is not an evidence gap"
            in body["questions"]["evidence:n1"]["instructions"]
        )
        native = response(body)
        native["answers"]["issue:n1"] = {
            "type": "choice",
            "choice": "gap",
            "confidence": 1.0,
            "probabilities": {
                label: float(label == "gap")
                for label in body["questions"]["issue:n1"]["criteria"]
            },
        }
        return httpx.Response(200, json=native)

    result = reviewer(ledger, handler).review(
        "Question", plan, draft, requirements, [], {1}
    )
    by_id = {check.check_id: check for check in result.checks}
    assert by_id["evidence:n1"].status == "addressed"
    assert by_id["issue:n1"].status == "gap"


def test_gap_resolution_retains_exact_old_interaction_and_active_original_support() -> (
    None
):
    ledger, plan, draft, requirements = inputs()
    prior_gap = "The exception in the later rule was not read."
    plan.needs[0].evidence_gap_resolutions = [
        GapResolution(need_id="n1", gap=prior_gap, requirement_ids=["r1"])
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        state = body["state"]
        assert state["needs"][0]["evidence_gap_resolutions"][0]["gap"] == prior_gap
        original = ledger.get(1)
        assert original is not None
        assert state["originals"][0]["text"] == original.text
        assert (
            "exact named prior unread law interaction"
            in body["questions"]["gap-resolution:n1:0"]["instructions"]
        )
        return httpx.Response(200, json=response(body))

    result = reviewer(ledger, handler).review(
        "Question", plan, draft, requirements, [], {1}
    )
    assert result.failure is None
    assert (
        next(
            check for check in result.checks if check.check_id == "gap-resolution:n1:0"
        ).status
        == "addressed"
    )


def test_superseded_gap_resolution_does_not_guess_current_source_support() -> None:
    ledger, plan, draft, requirements = inputs()
    plan.needs[0].evidence_gap_resolutions = [
        GapResolution(
            need_id="n1",
            gap="Prior unread rule interaction",
            requirement_ids=["old-requirement"],
        )
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "gap-resolution:n1:0" not in body["questions"]
        return httpx.Response(200, json=response(body))

    result = reviewer(ledger, handler).review(
        "Question", plan, draft, requirements, [], {1}
    )
    assert result.failure is not None
    assert (
        next(
            check for check in result.checks if check.check_id == "gap-resolution:n1:0"
        ).status
        == "uncertain"
    )


def test_only_latest_explicit_refresh_of_same_named_gap_is_active_proof() -> None:
    ledger, plan, draft, requirements = inputs()
    gap = "Prior unread exception"
    plan.needs[0].evidence_gap_resolutions = [
        GapResolution(need_id="n1", gap=gap, requirement_ids=["superseded"]),
        GapResolution(need_id="n1", gap=gap, requirement_ids=["r1"]),
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "gap-resolution:n1:0" not in body["questions"]
        assert "gap-resolution:n1:1" in body["questions"]
        assert len(body["state"]["needs"][0]["evidence_gap_resolutions"]) == 2
        return httpx.Response(200, json=response(body))

    model = reviewer(ledger, handler)
    expected = model.expected_checks("Question", plan, draft, requirements, [], {1})
    assert "gap-resolution:n1:0" not in expected
    assert expected["gap-resolution:n1:1"].citations == [1]
    assert model.review("Question", plan, draft, requirements, [], {1}).failure is None


@pytest.mark.parametrize("invalid", [None, "missing", "need_id", "not_applicable"])
def test_generation_reviewer_keeps_fixed_identity_whole_originals_and_no_alternate(
    invalid: str | None,
) -> None:
    ledger, plan, draft, requirements = inputs()
    calls: list[dict[str, Any]] = []
    gateways: list[Mock] = []
    main_thread = get_ident()
    config = LLMConfig(
        model_provider="openrouter",
        model_name="openai/gpt-5.6-luna",
        temperature=1,
        api_key="test-not-real",
        max_input_tokens=32_000,
    )

    def factory() -> BudgetedGateway:
        assert get_ident() == main_thread
        gateway = Mock(spec=BudgetedGateway)
        gateway.selected_llm = Mock(config=config)
        gateway.last_call_id = None
        gateway._fit_messages.return_value = ([], 100, [])

        def complete(
            _system: str,
            payload: dict[str, Any],
            response_type: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> Any:
            calls.append(payload)
            original = ledger.get(1)
            assert original is not None
            assert payload["original_evidence"][0]["text"] == original.text
            assert "text" not in payload["review_context"]["originals"][0]
            assert payload["required_evidence_numbers"] == [1]
            assert "unresolved_need_ids" in payload["review_context"]
            gateway.last_call_id = "generation-call"
            ledger.record_delivery(
                gateway.last_call_id,
                "legal_composite_review",
                payload["original_evidence"],
            )
            rows = [
                {
                    "check_id": check["check_id"],
                    "need_ids": check["need_ids"],
                    "section_ids": check["section_ids"],
                    "status": "addressed",
                    "confidence": 0.99,
                }
                for check in payload["expected_checks"]
            ]
            if invalid == "missing":
                rows.pop()
            elif invalid == "need_id":
                rows[0]["need_ids"] = ["different-need"]
            elif invalid == "not_applicable":
                rows[0]["status"] = "not_applicable"
            return response_type.model_validate({"checks": rows})

        gateway.complete.side_effect = complete
        gateways.append(gateway)
        return gateway

    model = GatewayAnswerReviewer(
        config=config,
        gateway_factory=factory,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
    )
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert len(calls) == 1 and len(gateways) == 2
    assert ledger.completely_delivered("generation-call") == {1}
    if invalid is None:
        assert result.failure is None
        assert all(check.status == "addressed" for check in result.checks)
    else:
        assert result.failure is not None
        assert all(check.status == "uncertain" for check in result.checks)


def test_generation_batches_have_four_worker_overlap_and_independent_delivery_state() -> (
    None
):
    ledger, plan, draft, requirements = inputs()
    config = LLMConfig(
        model_provider="openrouter",
        model_name="openai/gpt-5.6-luna",
        temperature=1,
        api_key="test-not-real",
        max_input_tokens=32_000,
    )
    barrier, lock = Barrier(4), Lock()
    main_thread = get_ident()
    active = peak = count = created = 0
    gateway_ids: set[int] = set()

    def factory() -> BudgetedGateway:
        nonlocal created
        assert get_ident() == main_thread
        created += 1
        gateway = Mock(spec=BudgetedGateway)
        gateway.selected_llm = Mock(config=config)
        gateway.last_call_id = None

        def fit(
            _system: str, payload: dict[str, Any], *_args: Any, **_kwargs: Any
        ) -> Any:
            if len(payload["expected_checks"]) > 1:
                raise RunStopped("Whole context does not fit combined batch")
            return [], 100, payload["original_evidence"]

        def complete(
            _system: str,
            payload: dict[str, Any],
            response_type: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> Any:
            nonlocal active, peak, count
            with lock:
                active += 1
                count += 1
                peak = max(peak, active)
                first = count <= 4
                assert id(gateway) not in gateway_ids
                gateway_ids.add(id(gateway))
            if first:
                barrier.wait(timeout=3)
            check = payload["expected_checks"][0]
            gateway.last_call_id = check["check_id"]
            with lock:
                active -= 1
            return response_type.model_validate(
                {
                    "checks": [
                        {
                            "check_id": check["check_id"],
                            "need_ids": check["need_ids"],
                            "section_ids": check["section_ids"],
                            "status": "addressed",
                            "confidence": 0.99,
                        }
                    ]
                }
            )

        gateway._fit_messages.side_effect = fit
        gateway.complete.side_effect = complete
        return gateway

    model = GatewayAnswerReviewer(
        config=config,
        gateway_factory=factory,
        budget=WorkflowBudget(WorkflowPolicy(max_model_calls=100)),
        ledger=ledger,
    )
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert result.failure is None
    assert peak == 4 and count == len(result.checks)
    assert created == count + 1


def test_generation_unfit_or_unverified_whole_context_never_calls_provider() -> None:
    ledger, plan, draft, requirements = inputs("Whole decisive original")
    config = LLMConfig(
        model_provider="openrouter",
        model_name="openai/gpt-5.6-luna",
        temperature=1,
        api_key="test-not-real",
        max_input_tokens=32_000,
    )
    gateway = Mock(spec=BudgetedGateway)
    gateway.selected_llm = Mock(config=config)
    gateway._fit_messages.side_effect = RunStopped("Whole context unavailable")
    model = GatewayAnswerReviewer(
        config=config,
        gateway_factory=lambda: gateway,
        budget=WorkflowBudget(WorkflowPolicy()),
        ledger=ledger,
    )
    result = model.review("Question", plan, draft, requirements, [], {1})
    gateway.complete.assert_not_called()
    assert result.failure is not None
    assert all(check.status == "uncertain" for check in result.checks)


def test_dependencies_can_be_source_grounded_nonmaterial_and_user_facts_need_no_legal_citation() -> (
    None
):
    ledger, plan, draft, requirements = inputs()
    origin = ledger.get(1)
    assert origin is not None
    dependency = AuthorityDependency(
        edge_id="edge",
        need_ids=["n1"],
        instrument_name="Unrelated procedure",
        article="2",
        origins=[
            DependencyOrigin(
                citation=1,
                source_id=origin.source_id,
                chunk_id=origin.chunk_id,
                text_hash=origin.text_hash,
            )
        ],
    )
    model = reviewer(
        ledger,
        lambda req: httpx.Response(
            200, json=response(json.loads(req.content), status="not_applicable")
        ),
    )
    checks = model.expected_checks(
        "Question", plan, draft, requirements, [dependency], {1}
    )
    assert checks["dependency:edge"].allow_not_applicable
    assert (
        "user-supplied facts do not require"
        in checks["section:s1:claim_inventory"].question
    )
    result = model.review("Question", plan, draft, requirements, [dependency], {1})
    assert (
        next(
            check for check in result.checks if check.check_id == "dependency:edge"
        ).status
        == "not_applicable"
    )


def test_positive_ambiguity_requires_98_percent_mass_without_weakening_claims() -> None:
    ledger, plan, draft, requirements = inputs()
    model = reviewer(ledger, lambda _request: httpx.Response(500))
    checks = model.expected_checks("Question", plan, draft, requirements, [], {1})
    dimension = checks["dimension:n1:legal_basis_and_hierarchy"]
    row = {
        "type": "choice",
        "choice": "addressed",
        "confidence": 0.41,
        "probabilities": {
            "addressed": 0.53,
            "not_applicable": 0.46,
            "gap": 0.01,
            "incorrect": 0,
            "uncertain": 0,
        },
    }
    decoded = model._decode({"answers": {dimension.check_id: row}}, [dimension])
    assert decoded[0].status == "addressed" and decoded[0].confidence == 0.99
    row["probabilities"] = {
        "addressed": 0.53,
        "not_applicable": 0.44,
        "gap": 0.03,
        "incorrect": 0,
        "uncertain": 0,
    }
    assert (
        model._decode({"answers": {dimension.check_id: row}}, [dimension])[0].status
        == "uncertain"
    )
    claim = checks["claim:c1"]
    row["probabilities"] = {
        "addressed": 0.53,
        "gap": 0.46,
        "incorrect": 0.01,
        "uncertain": 0,
    }
    assert (
        model._decode({"answers": {claim.check_id: row}}, [claim])[0].status
        == "uncertain"
    )


def test_native_confidence_and_endpoint_price_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger, plan, draft, requirements = inputs()
    spans: list[Mock] = []

    @contextmanager
    def trace_call(**kwargs: Any) -> Any:
        span = Mock()
        span.span_data.model_config = kwargs["extra_config"]
        spans.append(span)
        yield span

    monkeypatch.setattr("onyx.legal_composite.reviewer.traced_llm_call", trace_call)
    model = reviewer(
        ledger, lambda req: httpx.Response(200, json=response(json.loads(req.content)))
    )
    result = model.review("Question", plan, draft, requirements, [], {1})
    assert result.failure is None
    assert all(check.confidence == 0.99 for check in result.checks)
    config = spans[0].span_data.model_config
    assert config["legal_composite_decisions_reported_cost_usd"] == "1.9992e-05"
    assert config["legal_composite_decisions_input_rate_usd_per_million"] == "0.042"
    assert config["legal_composite_decisions_reported_output_tokens"] == "70"
    assert spans[0].span_data.usage["output_tokens"] == 0
    assert "test-not-real" not in json.dumps(config)


def test_cancel_does_not_wait_for_transport_or_certify_unfinished_call() -> None:
    ledger, plan, draft, requirements = inputs()
    started, release = Event(), Event()

    def handler(request: httpx.Request) -> httpx.Response:
        started.set()
        release.wait(timeout=3)
        return httpx.Response(200, json=response(json.loads(request.content)))

    model = reviewer(ledger, handler)

    def check_active() -> None:
        if started.is_set():
            raise RunStopped("cancelled")

    model.check_active = check_active
    before = time.monotonic()
    try:
        result = model.review("Question", plan, draft, requirements, [], {1})
        assert time.monotonic() - before < 1
        assert result.failure
        assert all(check.status == "uncertain" for check in result.checks)
        assert model.budget.snapshot()["unsettled_calls"] == 1
    finally:
        release.set()


def _unregistered_multi_issue_inputs() -> tuple[
    EvidenceLedger, ResearchPlan, DraftAnswer
]:
    ledger, plan, _draft, _requirements = inputs(
        "Complete original for the first issue, including its exception.",
        "Complete original for the second issue, including its deadline.",
        "Complete original for the third issue, including its remedy.",
        "Complete original whose issue binding is absent.",
        "Complete original whose issue binding is unknown to this plan.",
        "Canonical original acquired but never delivered to the writer.",
    )
    bindings = [["n1"], ["n2"], ["n3"], [], ["unknown"], ["n1"]]
    for citation, identities in enumerate(bindings, 1):
        ledger._items[citation].question_ids = identities
    plan.needs = [
        ResearchNeed(
            need_id=identity,
            question=f"Explain the requested outcome for {identity}",
            governing_source="",
            conditions_to_check=[],
        )
        for identity in ("n1", "n2", "n3")
    ]
    draft = DraftAnswer(
        unresolved_need_ids=[],
        sections=[
            AnswerSection(
                section_id=f"s{index}", need_ids=[identity], text=f"Outcome {identity}."
            )
            for index, identity in enumerate(("n1", "n2", "n3"), 1)
        ],
        claims=[],
    )
    return ledger, plan, draft


@pytest.mark.parametrize("identity,citation", [("n1", 1), ("n2", 2), ("n3", 3)])
def test_aggregate_checks_have_complete_delivered_issue_originals_before_requirements(
    identity: str, citation: int
) -> None:
    ledger, plan, draft = _unregistered_multi_issue_inputs()
    model = reviewer(ledger, lambda _request: httpx.Response(500))
    checks = model.expected_checks("Question", plan, draft, [], [], {1, 2, 3, 4, 5})
    for prefix in ["evidence", "issue"]:
        assert checks[f"{prefix}:{identity}"].citations == [citation, 4, 5]
    for dimension in REVIEW_DIMENSIONS:
        assert checks[f"dimension:{identity}:{dimension}"].citations == [citation, 4, 5]
    body = model._payload(
        "Question",
        plan,
        draft,
        [],
        [],
        [checks[f"evidence:{identity}"]],
        {1, 2, 3, 4, 5},
    )
    state = cast(dict[str, Any], body["state"])
    assert [row["citation"] for row in state["originals"]] == [citation, 4, 5]
    for row in state["originals"]:
        original = ledger.get(row["citation"])
        assert original is not None
        assert row["text"] == original.text and row["text_hash"] == original.text_hash
    assert state["requirements"] == [] and state["canonical_witnesses"] == []
    assert checks["request:coverage"].citations == [1, 2, 3, 4, 5]


def test_explicit_requirement_and_claim_support_expand_the_actual_original_issue_scope() -> (
    None
):
    ledger, plan, draft = _unregistered_multi_issue_inputs()
    original = ledger.get(2)
    assert original is not None
    requirement = SourceRequirement(
        requirement_id="cross-issue-rule",
        need_id="n1",
        dimension="legal_basis_and_hierarchy",
        rule="The second source also limits the first requested outcome.",
        application="Apply only its supported condition.",
        supports=[SpanSupport(citation=2, quotation=original.text)],
    )
    draft.claims = [
        DraftClaim(
            claim_id="third-issue-rule",
            section_id="s3",
            need_ids=["n3"],
            answer_excerpt=draft.sections[2].text,
            supports=[SpanSupport(citation=1, quotation=ledger._items[1].text)],
        )
    ]
    model = reviewer(ledger, lambda _request: httpx.Response(500))
    checks = model.expected_checks(
        "Question", plan, draft, [requirement], [], {1, 2, 3}
    )
    assert checks["issue:n1"].citations == [1, 2]
    assert checks["issue:n2"].citations == [2]
    assert checks["issue:n3"].citations == [1, 3]
    assert checks["original:1"].need_ids == ["n1", "n3"]
    assert checks["original:2"].need_ids == ["n1", "n2"]
    assert len([identity for identity in checks if identity == "original:2"]) == 1


def test_first_generation_issue_batch_receives_whole_originals_without_registered_rules() -> (
    None
):
    ledger, plan, draft = _unregistered_multi_issue_inputs()
    calls: list[dict[str, Any]] = []
    config = LLMConfig(
        model_provider="openrouter",
        model_name="openai/gpt-5.6-luna",
        temperature=1,
        api_key="test-not-real",
        max_input_tokens=128_000,
    )

    def factory() -> BudgetedGateway:
        gateway = Mock(spec=BudgetedGateway)
        gateway.selected_llm = Mock(config=config)
        gateway.last_call_id = None
        gateway._fit_messages.return_value = ([], 100, [])

        def complete(
            _system: str,
            payload: dict[str, Any],
            response_type: Any,
            *_args: Any,
            **_kwargs: Any,
        ) -> Any:
            calls.append(payload)
            assert len(payload["expected_checks"]) <= 32
            assert payload["original_evidence"]
            assert all(
                row["text"] == ledger._items[row["citation"]].text
                for row in payload["original_evidence"]
            )
            return response_type.model_validate(
                {
                    "checks": [
                        {
                            "check_id": check["check_id"],
                            "need_ids": check["need_ids"],
                            "section_ids": check["section_ids"],
                            "status": "addressed",
                            "confidence": 0.99,
                        }
                        for check in payload["expected_checks"]
                    ]
                }
            )

        gateway.complete.side_effect = complete
        return gateway

    model = GatewayAnswerReviewer(
        config=config,
        gateway_factory=factory,
        budget=WorkflowBudget(WorkflowPolicy(max_context_tokens=128_000)),
        ledger=ledger,
    )
    result = model.review("Question", plan, draft, [], [], {1, 2, 3, 4, 5})
    assert result.failure is None
    first_issue_batch = next(
        body
        for body in calls
        if body["expected_checks"][0]["check_id"] == "evidence:n1"
    )
    identities = [check["check_id"] for check in first_issue_batch["expected_checks"]]
    assert len(identities) == 32
    assert all(
        identity.startswith(("issue:", "evidence:", "dimension:"))
        for identity in identities
    )
    assert first_issue_batch["required_evidence_numbers"] == [1, 2, 3, 4, 5]
    assert len(first_issue_batch["original_evidence"]) == 5
    assert first_issue_batch["review_context"]["requirements"] == []
    assert first_issue_batch["review_context"]["canonical_witnesses"] == []
    assert not any(
        row["citation"] == 6 for body in calls for row in body["original_evidence"]
    )
