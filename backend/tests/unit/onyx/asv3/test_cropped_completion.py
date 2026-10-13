"""Optional followups never turn an open source gap into a verified legal answer."""

import json
from typing import cast

import jsonschema
import pytest
from pydantic import JsonValue, ValidationError

from onyx.asv3.cropped_completion import (
    CroppedCompletionContext,
    CroppedRequiredFollowup,
    CroppedValidatedFollowups,
    append_cropped_required_followups,
    build_cropped_completion_context,
    cropped_required_followups_schema,
    validate_cropped_required_followups,
)
from onyx.asv3.cropped_sections import (
    CroppedAssemblyReview,
    CroppedAssemblyReviewInput,
    CroppedSectionStore,
    assemble_cropped_sections,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeMap, OutcomeUpdate
from onyx.asv3.registry import CapabilityRegistry, native_research_bindings
from onyx.asv3.research_state import ResearchNeed, ResearchState, ResearchUpdate
from onyx.asv3.source_answer_transport import (
    bind_source_answer_body,
    source_answer_wire_tools,
)
from onyx.asv3.terminal_wire_schema import strict_terminal_tools
from tests.unit.onyx.asv3.test_cropped_sections import (
    Calls,
    count_input,
    terminal_body,
)
from tests.unit.onyx.asv3.test_cropped_sections import (
    fixture as section_fixture,
)
from tests.unit.onyx.asv3.test_runtime import response

REQUEST = "Muafiyet koşulları ve itiraz yolunu açıklayın. Başvuru tarihi bilinmiyor."
FACTS = ["Başvuru tarihi bilinmiyor."]
ANSWER = (
    "Eldeki kaynağın belge koşulu korunur [1]. Tarih ve eksik özgün metin açık kalır."
)


def fixture() -> tuple[RunContext, EvidenceLedger, ResearchState, OutcomeMap]:
    context = RunContext(
        scope={"document_set_id": 15, "as_of_date": "2026-10-13"},
        services={"asv3_exp_cropped": True},
    )
    ledger = EvidenceLedger()
    ledger.add(
        [EvidenceItem(source_id="law", chunk_id="a", text="A document is required.")],
        context,
    )
    state = ResearchState(["Muafiyet?", "İtiraz?"], context)
    state.update(
        ResearchUpdate(
            needs=[
                ResearchNeed(
                    need_id="basis",
                    question_ids=["q0"],
                    purpose="Özgün dayanağı edin.",
                    completion_test="Tam özgün hüküm ve maddi istisnaları.",
                    status="blocked",
                    gap="Özgün metin edinilemedi; yokluk kanıtlanmadı.",
                ),
                ResearchNeed(
                    need_id="settled",
                    question_ids=["q0"],
                    purpose="Belge koşulunu değerlendir.",
                    completion_test="Belge koşulu ve kapsamı.",
                    status="candidate",
                ),
                ResearchNeed(
                    need_id="excluded",
                    question_ids=["q0"],
                    purpose="Başka ürün.",
                    completion_test="Ürün kapsamı.",
                    status="out_of_scope",
                    gap="Soru bu ürünü içermiyor.",
                ),
                ResearchNeed(
                    need_id="incidental",
                    question_ids=["q0"],
                    purpose="Genel tarihi araştır.",
                    completion_test="Tarihçe.",
                    material=False,
                ),
            ]
        ),
        ledger,
    )
    outcomes = OutcomeMap(list(state.questions), context, factual_context=REQUEST)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "outcomes": [
                    {
                        "outcome_id": "result",
                        "question_ids": ["q0"],
                        "detail": "Muafiyet sonucu.",
                    },
                    {
                        "outcome_id": "closed",
                        "question_ids": ["q0"],
                        "detail": "Belge koşulu.",
                    },
                    {
                        "outcome_id": "other",
                        "question_ids": ["q1"],
                        "detail": "İtiraz yolu.",
                    },
                ],
                "resolutions": [
                    {
                        "outcome_id": "result",
                        "status": "unresolved",
                        "gap": "Özgün dayanak açık.",
                    },
                    {
                        "outcome_id": "closed",
                        "status": "supported",
                        "evidence_numbers": [1],
                    },
                ],
            }
        ),
        ledger,
    )
    context.services.update(research_state=state, outcome_map=outcomes, evidence=ledger)
    return context, ledger, state, outcomes


def item(**changes: JsonValue) -> dict[str, JsonValue]:
    return {
        "need_id": "basis",
        "outcome_id": "result",
        "question": "Eksik özgün dayanak ve tarihi kesinleştirecek belgeler incelensin mi?",
        "material_rationale": "Açık dayanak ve tarih, istenen sonucu değiştirebilir.",
        "definitely_necessary": True,
        "material": True,
        **changes,
    }


def validate(
    context: RunContext, raw: JsonValue, *, answer: str = ANSWER
) -> CroppedValidatedFollowups:
    snapshot = build_cropped_completion_context(context, REQUEST, facts=FACTS)
    return validate_cropped_required_followups(
        raw, context, REQUEST, snapshot, answer=answer, facts=FACTS
    )


def render(context: RunContext, raw: JsonValue) -> str:
    return append_cropped_required_followups(
        ANSWER, validate(context, raw), context, REQUEST, facts=FACTS
    )


def test_genuine_open_material_followup_renders_only_original_question_after_exact_body() -> (
    None
):
    context, ledger, state, outcomes = fixture()
    before = (ledger.export(), state.export(), outcomes.export())
    row = item()
    result = render(context, [row])
    assert result.startswith(ANSWER + "\n\n")
    assert result.endswith(str(row["question"]))
    assert str(row["material_rationale"]) not in result
    assert "İstersen şu açık soruları da araştırabilirim:" in result
    assert "$" not in result
    assert (ledger.export(), state.export(), outcomes.export()) == before
    assert ledger.completely_delivered("nonexistent-review") == set()


@pytest.mark.parametrize("raw", [None, []])
def test_absent_followup_preserves_entire_answer_without_banner(raw: JsonValue) -> None:
    context, _, _, _ = fixture()
    assert render(context, raw) == ANSWER


@pytest.mark.parametrize(
    "changes",
    [
        {"definitely_necessary": False},
        {"material": False},
        {"need_id": "excluded"},
        {"need_id": "incidental"},
        {"need_id": None, "outcome_id": "closed"},
        {"need_id": "settled", "outcome_id": "closed"},
    ],
)
def test_speculative_resolved_incidental_and_excluded_items_are_not_offered(
    changes: dict[str, JsonValue],
) -> None:
    context, _, _, _ = fixture()
    assert render(context, [item(**changes)]) == ANSWER


def test_supported_outcome_does_not_hide_an_explicit_linked_material_need_gap() -> None:
    context, _, _, _ = fixture()
    assert render(context, [item(outcome_id="closed")]) != ANSWER


@pytest.mark.parametrize(
    "changes",
    [
        {"need_id": "invented"},
        {"outcome_id": "invented"},
        {"need_id": "basis", "outcome_id": "other"},
        {"need_id": None, "outcome_id": None},
        {"material_rationale": " "},
        {"definitely_necessary": "true"},
        {"question": "Eksik hüküm kesin olarak uygulanır."},
    ],
)
def test_invalid_or_unbound_items_fail_instead_of_inventing_a_new_question(
    changes: dict[str, JsonValue],
) -> None:
    context, ledger, state, outcomes = fixture()
    before = (ledger.export(), state.export(), outcomes.export())
    with pytest.raises((ValueError, ValidationError)):
        validate(context, [item(**changes)])
    assert (ledger.export(), state.export(), outcomes.export()) == before


def test_explicit_question_cannot_add_an_unvalidated_citation() -> None:
    context, _, _, _ = fixture()
    with pytest.raises(ValueError, match="citations"):
        validate(context, [item(question="Yeni metnin [9] koşulları incelensin mi?")])
    row = item(question="Eldeki [1] belgesinin tarih bilgisi doğrulansın mı?")
    assert render(context, [row]).endswith(str(row["question"]))


def test_optional_question_does_not_replace_a_missing_answer() -> None:
    context, _, _, _ = fixture()
    validated = validate(context, [item()], answer="")
    with pytest.raises(ValueError, match="missing user answer"):
        append_cropped_required_followups("", validated, context, REQUEST, facts=FACTS)


def test_duplicate_identical_questions_are_not_appended_twice() -> None:
    context, _, _, _ = fixture()
    row = item()
    assert render(context, [row, row]).count(str(row["question"])) == 1


def test_writer_snapshot_is_complete_detached_and_preserves_source_gap_uncertainty() -> (
    None
):
    context, _, state, outcomes = fixture()
    gap: dict[str, JsonValue] = {
        "missing_governing_original": "actual-target",
        "absence_proven": False,
    }
    snapshot = build_cropped_completion_context(
        context, REQUEST, facts=FACTS, publication_gap=gap
    )
    view = snapshot.writer_view()
    assert view["full_question"] == REQUEST
    assert view["facts"] == FACTS
    assert len(cast(list[JsonValue], view["research_needs"])) == 4
    assert len(cast(list[JsonValue], view["requested_outcomes"])) == 3
    assert view["publication_gap"] == gap
    assert "not proof that the law is absent" in str(view["instruction"])
    assert "source approval" in str(view["instruction"])
    cast(list[JsonValue], view["research_needs"]).clear()
    gap.clear()
    assert len(cast(list[JsonValue], snapshot.writer_view()["research_needs"])) == 4
    assert json.loads(snapshot.research_json) == state.export()
    assert json.loads(snapshot.outcomes_json) == outcomes.export()


def test_state_revision_change_rejects_retained_metadata_before_validation() -> None:
    context, ledger, _, outcomes = fixture()
    snapshot = build_cropped_completion_context(context, REQUEST, facts=FACTS)
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "resolutions": [
                    {
                        "outcome_id": "result",
                        "status": "supported",
                        "evidence_numbers": [1],
                    }
                ]
            }
        ),
        ledger,
    )
    with pytest.raises(ValueError, match="stale"):
        validate_cropped_required_followups(
            [item()], context, REQUEST, snapshot, answer=ANSWER, facts=FACTS
        )


def test_own_terminal_resolution_filters_prevalidated_followup_after_body_guard() -> (
    None
):
    context, ledger, _, outcomes = fixture()
    validated = validate(context, [item(need_id=None)])
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "resolutions": [
                    {
                        "outcome_id": "result",
                        "status": "supported",
                        "evidence_numbers": [1],
                    }
                ]
            }
        ),
        ledger,
    )
    assert (
        append_cropped_required_followups(
            ANSWER, validated, context, REQUEST, facts=FACTS
        )
        == ANSWER
    )


def test_live_out_of_scope_need_is_filtered_after_terminal_state_update() -> None:
    context, ledger, state, _ = fixture()
    validated = validate(context, [item()])
    need = ResearchNeed.model_validate(
        cast(list[JsonValue], state.export()["needs"])[0]
    )
    state.update(
        ResearchUpdate(
            needs=[
                need.model_copy(
                    update={"status": "out_of_scope", "gap": "Olay kapsamı değişti."}
                )
            ]
        ),
        ledger,
    )
    assert (
        append_cropped_required_followups(
            ANSWER, validated, context, REQUEST, facts=FACTS
        )
        == ANSWER
    )


@pytest.mark.parametrize(
    "change", ["facts", "request", "body", "scope", "owner", "depth"]
)
def test_followup_cannot_transfer_to_a_changed_answer_or_scenario(change: str) -> None:
    context, _, _, _ = fixture()
    validated = validate(context, [item()])
    answer, request, facts = ANSWER, REQUEST, FACTS
    if change == "facts":
        facts = ["Başvuru tarihi artık biliniyor."]
    elif change == "request":
        request += " Yeni olay."
    elif change == "body":
        answer += " Yeni sonuç."
    elif change == "scope":
        context.scope["as_of_date"] = "2024-01-01"
    elif change == "owner":
        context.services["task_id"] = "other-task"
    else:
        context.depth = 1
    with pytest.raises(ValueError, match="changed answer or scenario"):
        append_cropped_required_followups(
            answer, validated, context, request, facts=facts
        )


def test_changed_need_meaning_is_not_reused_after_validation() -> None:
    context, ledger, state, _ = fixture()
    validated = validate(context, [item()])
    need = ResearchNeed.model_validate(
        cast(list[JsonValue], state.export()["needs"])[0]
    )
    state.update(
        ResearchUpdate(
            needs=[need.model_copy(update={"purpose": "Başka sonucu incele."})]
        ),
        ledger,
    )
    with pytest.raises(ValueError, match="meaning changed"):
        append_cropped_required_followups(
            ANSWER, validated, context, REQUEST, facts=FACTS
        )


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"asv3_exp_cropped": False},
        {"asv3_exp_cropped": True, "asv3_exp_cheap": True},
    ],
)
def test_protected_workflows_cannot_enter_cropped_completion(
    flags: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValueError, match="isolated Cropped"):
        build_cropped_completion_context(RunContext(services=dict(flags)), REQUEST)


def test_schema_is_inline_optional_metadata_with_explicit_model_decisions() -> None:
    schema = cropped_required_followups_schema()
    jsonschema.Draft202012Validator.check_schema(schema)
    assert "$ref" not in json.dumps(schema)
    jsonschema.Draft202012Validator(schema).validate([item()])
    jsonschema.Draft202012Validator(schema).validate([])
    required = cast(dict[str, JsonValue], schema["items"])["required"]
    assert isinstance(required, list)
    assert {
        "question",
        "material_rationale",
        "definitely_necessary",
        "material",
    } <= set(required)


def terminal_registry(context: RunContext, executed: list[str]) -> CapabilityRegistry:
    def submit(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
        assert args == {"answer": ANSWER}
        validated = child.services.get("cropped_validated_required_followups")
        executed.append(
            append_cropped_required_followups(
                ANSWER,
                cast(CroppedValidatedFollowups, validated),
                child,
                REQUEST,
                facts=FACTS,
            )
            if validated is not None
            else ANSWER
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Actual terminal handler executed"
        )

    context.services["lean_native_mode"] = True
    context.services["cropped_completion_context"] = build_cropped_completion_context(
        context, REQUEST, facts=FACTS
    )
    return CapabilityRegistry(
        [
            ToolSpec(
                name="submit_answer",
                description="Actual terminal action",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=submit,
            )
        ]
    )


@pytest.mark.parametrize("with_coverage", [False, True])
def test_registry_validates_invalid_binding_before_any_coverage_or_handler(
    with_coverage: bool,
) -> None:
    context, ledger, state, outcomes = fixture()
    executed: list[str] = []
    registry = terminal_registry(context, executed)
    args: dict[str, JsonValue] = {
        "answer": ANSWER,
        "_required_followups": [item(outcome_id="invented")],
    }
    if with_coverage:
        args["_coverage"] = {
            "resolutions": [
                {"outcome_id": "result", "status": "supported", "evidence_numbers": [1]}
            ]
        }
    before = (
        ledger.export(),
        state.export(),
        outcomes.export(),
        context.budget.snapshot(),
    )
    outcome = registry.dispatch(
        CapabilityCall(name="submit_answer", arguments=args), context
    )
    assert outcome.status == OutcomeStatus.INVALID
    assert outcome.data["invalid_required_followups"] is True
    assert not executed
    assert (
        ledger.export(),
        state.export(),
        outcomes.export(),
        context.budget.snapshot(),
    ) == before


def test_registry_own_coverage_closes_prevalidated_followup_before_append() -> None:
    context, ledger, _, outcomes = fixture()
    executed: list[str] = []
    registry = terminal_registry(context, executed)
    call = CapabilityCall(
        name="submit_answer",
        arguments={
            "answer": ANSWER,
            "_required_followups": [item(need_id=None)],
            "_coverage": {
                "resolutions": [
                    {
                        "outcome_id": "result",
                        "status": "supported",
                        "evidence_numbers": [1],
                    }
                ]
            },
        },
    )
    assert registry.dispatch(call, context).status == OutcomeStatus.FOUND
    assert executed == [ANSWER]
    assert (
        cast(
            dict[str, JsonValue],
            cast(list[JsonValue], outcomes.export()["resolutions"])[0],
        )["status"]
        == "supported"
    )
    assert ledger.completely_delivered("fake-writer") == set()


def test_harness_style_preapplied_coverage_consumes_only_this_call_validation() -> None:
    context, _, _, _ = fixture()
    executed: list[str] = []
    registry = terminal_registry(context, executed)
    call = CapabilityCall(
        name="submit_answer",
        arguments={
            "answer": ANSWER,
            "_required_followups": [item(need_id=None)],
            "_coverage": {
                "resolutions": [
                    {
                        "outcome_id": "result",
                        "status": "supported",
                        "evidence_numbers": [1],
                    }
                ]
            },
        },
    )
    assert registry.research_binding_gap(call, context) is None
    assert registry.outcome_metadata_gap(call, context) is None
    tool_context = context.child()
    tool_context.depth = context.depth
    tool_context.services["applied_outcome_metadata_call"] = call
    assert registry.dispatch(call, tool_context).status == OutcomeStatus.FOUND
    assert executed == [ANSWER]


def test_cached_validation_cannot_ignore_later_state_edits_or_foreign_worker() -> None:
    context, ledger, _, outcomes = fixture()
    executed: list[str] = []
    registry = terminal_registry(context, executed)
    call = CapabilityCall(
        name="submit_answer",
        arguments={"answer": ANSWER, "_required_followups": [item()]},
    )
    assert registry.research_binding_gap(call, context) is None
    worker = context.child()
    worker.services["task_id"] = "other-task"
    assert registry.dispatch(call, worker).status == OutcomeStatus.INVALID
    outcomes.update(
        OutcomeUpdate.model_validate(
            {
                "resolutions": [
                    {
                        "outcome_id": "result",
                        "status": "supported",
                        "evidence_numbers": [1],
                    }
                ]
            }
        ),
        ledger,
    )
    assert registry.dispatch(call, context).status == OutcomeStatus.INVALID
    assert not executed


def test_registry_no_followup_does_not_require_completion_snapshot_or_append_text() -> (
    None
):
    context, _, _, _ = fixture()
    executed: list[str] = []
    registry = terminal_registry(context, executed)
    context.services.pop("cropped_completion_context")
    assert (
        registry.dispatch(
            CapabilityCall(name="submit_answer", arguments={"answer": ANSWER}), context
        ).status
        == OutcomeStatus.FOUND
    )
    assert executed == [ANSWER]


def test_cropped_registry_exposes_optional_followup_field_and_valid_unnumbered_q0() -> (
    None
):
    context, _, _, _ = fixture()
    context.services["research_state"] = ResearchState([REQUEST], context)
    executed: list[str] = []
    registry = terminal_registry(context, executed)
    bindings = native_research_bindings(context)
    assert bindings is not None and bindings["question_ids"] == ["q0"]
    function = cast(dict[str, JsonValue], registry.definitions(context)[0]["function"])
    parameters = cast(dict[str, JsonValue], function["parameters"])
    assert "_required_followups" in cast(dict[str, JsonValue], parameters["properties"])
    assert "_required_followups" not in cast(list[JsonValue], parameters["required"])
    context.services["asv3_exp_cropped"] = False
    assert native_research_bindings(context) is None
    ordinary = cast(dict[str, JsonValue], registry.definitions(context)[0]["function"])
    assert "_required_followups" not in cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], ordinary["parameters"])["properties"],
    )
    assert (
        registry.dispatch(
            CapabilityCall(name="submit_answer", arguments={"answer": ANSWER}), context
        ).status
        == OutcomeStatus.FOUND
    )
    assert executed == [ANSWER]


def test_english_heading_does_not_rewrite_the_model_question() -> None:
    context, _, _, _ = fixture()
    context.language = "en-US"
    question = "Should the original provision and the filing date be verified?"
    result = render(context, [item(question=question)])
    assert "If you'd like, I can research these open questions too:" in result
    assert result.endswith(question)


def test_source_answer_body_transport_and_strict_wire_keep_exact_followup_rows() -> (
    None
):
    context, _, _, _ = fixture()
    registry = terminal_registry(context, [])
    tools = registry.definitions(context)
    body_wire = source_answer_wire_tools(tools)
    body_properties = cast(
        dict[str, JsonValue],
        cast(
            dict[str, JsonValue],
            cast(dict[str, JsonValue], body_wire[0]["function"])["parameters"],
        )["properties"],
    )
    assert "answer" not in body_properties
    assert "_required_followups" in body_properties
    context.services.update(research_profile="experimental", experimental_parallel=True)
    constrained = strict_terminal_tools(tools, context, "openai", "gpt-6.1-sol")
    function = cast(dict[str, JsonValue], constrained[0]["function"])
    assert function["strict"] is True
    assert "_required_followups" in cast(
        dict[str, JsonValue],
        cast(dict[str, JsonValue], function["parameters"])["properties"],
    )
    row = item()
    bound = bind_source_answer_body(
        response(ANSWER, calls=[("submit_answer", {"_required_followups": [row]})]),
        lambda text: cast(dict[str, JsonValue], json.loads(text)),
    )
    assert bound.choice.message.tool_calls is not None
    assert json.loads(bound.choice.message.tool_calls[0].function.arguments or "") == {
        "_required_followups": [row],
        "answer": ANSWER,
    }


def test_existing_section_review_reports_durable_followups_without_a_new_model_stage() -> (
    None
):
    context, ledger, view = section_fixture()
    state = ResearchState(view.questions, context)
    state.update(
        ResearchUpdate(
            needs=[
                ResearchNeed(
                    need_id="date",
                    question_ids=["q0"],
                    purpose="Verify the user's missing service date.",
                    completion_test="Actual service date supplied.",
                    status="blocked",
                    gap="The service date is not supplied.",
                )
            ]
        ),
        ledger,
    )
    context.services["research_state"] = state
    calls = Calls(context, ledger)
    reported = CroppedRequiredFollowup(
        need_id="date",
        question="Should the actual service date be supplied to resolve the remaining time limit?",
        material_rationale="The requested time limit depends on that unknown date.",
        definitely_necessary=True,
        material=True,
    )

    def review(supplied: CroppedAssemblyReviewInput) -> CroppedAssemblyReview:
        snapshot = build_cropped_completion_context(
            context,
            supplied.request,
            facts=supplied.facts,
            publication_gap=supplied.publication_gap,
        )
        assert supplied.completion_context == snapshot.writer_view()
        return calls.reviewer(supplied).model_copy(
            update={
                "required_followups": [reported],
                "completion_snapshot": snapshot.export(),
            }
        )

    decision = assemble_cropped_sections(
        view,
        context,
        ledger,
        count_input=count_input,
        write_section=calls.writer,
        review_assembly=review,
        publication_gap=calls.guard,
    )
    assert len(calls.writes) == 3 and len(calls.reviews) == 1
    assert decision.calls[0].name == "submit_partial_answer"
    body = terminal_body(decision)
    assert "_required_followups" in decision.calls[0].arguments
    store = context.services["cropped_section_store"]
    assert isinstance(store, CroppedSectionStore)
    store.verify_assembly(context, view, body, ledger, calls.guard)
    validated = store.assembly_completion_followups(context, view, body, ledger)
    assert validated is not None and validated.items == (reported,)
    rendered = append_cropped_required_followups(
        body, validated, context, view.request, facts=view.facts
    )
    assert rendered.startswith(body + "\n\n") and rendered.endswith(reported.question)
    assert ledger.completely_delivered("review") == set()
    restored = CroppedSectionStore(context, view.request, user_id="owner")
    restored.restore(store.export(), context, view.request, ledger)
    restored.verify_assembly(context, view, body, ledger, calls.guard)
    recovered = restored.assembly_completion_followups(context, view, body, ledger)
    assert recovered == validated
    assert restored.assembly_status(body) == OutcomeStatus.PARTIAL
    assert len(calls.reviews) == 1
    changed = view.model_copy(deep=True)
    changed.facts.append("Service date supplied after the checkpoint.")
    with pytest.raises(ValueError, match="scope or sources changed"):
        restored.assembly_completion_followups(context, changed, body, ledger)


@pytest.mark.parametrize("change", ["unknown_need", "no_snapshot", "wrong_snapshot"])
def test_section_review_cannot_sneak_unbound_or_stale_followups_into_a_terminal(
    change: str,
) -> None:
    context, ledger, view = section_fixture()
    calls = Calls(context, ledger)

    def review(supplied: CroppedAssemblyReviewInput) -> CroppedAssemblyReview:
        snapshot = build_cropped_completion_context(
            context, supplied.request, facts=supplied.facts
        )
        row = CroppedRequiredFollowup(
            need_id="invented",
            question="Should this invented new task be researched?",
            material_rationale="Declared material, but no actual binding exists.",
            definitely_necessary=True,
            material=True,
        )
        payload = snapshot.export()
        if change == "wrong_snapshot":
            payload["owner"] = "foreign-worker"
        return calls.reviewer(supplied).model_copy(
            update={
                "required_followups": [row],
                "completion_snapshot": None if change == "no_snapshot" else payload,
            }
        )

    with pytest.raises((ValueError, RunStopped)):
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
    assert isinstance(store, CroppedSectionStore) and not store.export()["assemblies"]


@pytest.mark.parametrize(
    "change", ["boolean_depth", "missing_field", "nonliteral_fact", "noncanonical_json"]
)
def test_completion_checkpoint_types_and_literal_json_are_exact(change: str) -> None:
    context, _, _, _ = fixture()
    snapshot = build_cropped_completion_context(context, REQUEST, facts=FACTS)
    payload = snapshot.export()
    assert CroppedCompletionContext.restore(payload) == snapshot
    if change == "boolean_depth":
        payload["depth"] = True
    elif change == "missing_field":
        payload.pop("owner")
    elif change == "nonliteral_fact":
        payload["facts_json"] = "[3]"
    else:
        payload["research_json"] = "{} "
    with pytest.raises(ValueError):
        CroppedCompletionContext.restore(payload)
