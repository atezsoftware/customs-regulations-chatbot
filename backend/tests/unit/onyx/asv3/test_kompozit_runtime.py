"""Fixed Kompozit roles retain the ordinary ASv3 and Cropped execution policies."""

import copy
import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import llm_adapter, runtime
from onyx.asv3.cropped_cost_budget import (
    CroppedCostBudget,
    CroppedResearchBudgetReached,
)
from onyx.asv3.cropped_repair_model import fresh_model
from onyx.asv3.llm_adapter import ResearchModel, VerificationResult
from onyx.asv3.model_composition.models import (
    MODEL_COMPOSITION_SERVICE,
    CompositionProfileId,
    CompositionRole,
    ModelBinding,
    ModelComposition,
    allows_cropped_composition_implicit_cache,
)
from onyx.asv3.model_composition.profiles import COMPOSITION_PROFILES, MODEL_SPECS
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome
from onyx.asv3.performance import FAST_TIMING_SERVICE
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_FAST_CROPPED_VARIANT,
    ASV3_FAST_VARIANT,
    ASV3_KOMPOZIT_CROPPED_VARIANT,
    ASV3_KOMPOZIT_VARIANT,
    ASV3_STANDARD_VARIANT,
    ASV3_TUNED_VARIANT,
    checkpoint_variant_fields,
)
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import Usage
from onyx.llm.models import ChatCompletionMessage, ReasoningEffort, UserMessage
from onyx.prompts.asv3.fast import (
    FAST_RESPONSE_POLICY_SERVICE,
    FAST_TEK_ACTION_RESEARCH,
)
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from onyx.tracing.flows import LLMFlow
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.tracing.framework.spans import Span
from tests.unit.onyx.asv3.test_fast_primary_coordinator import names, prose
from tests.unit.onyx.asv3.test_model_composition_adapter import (
    composition_fixture,
    original_view,
)
from tests.unit.onyx.asv3.test_native_model_adapter import native_action
from tests.unit.onyx.asv3.test_runtime import response, setup_run
from tests.unit.onyx.llm.test_factorial_cropped_flash_cache import (
    composition_context,
    selected,
)

_VARIANTS = (ASV3_KOMPOZIT_VARIANT, ASV3_KOMPOZIT_CROPPED_VARIANT)


class SetupBoundaryReached(Exception):
    pass


def _runtime_composition(
    primary: MagicMock,
) -> tuple[ModelComposition, dict[CompositionRole, MagicMock]]:
    models: dict[CompositionRole, MagicMock] = {}
    bindings: dict[CompositionRole, ModelBinding] = {}
    for index, (role, alias) in enumerate(COMPOSITION_PROFILES["kompozit_v1"].items()):
        spec = MODEL_SPECS[alias]
        llm = primary if role == "coordinator" else MagicMock(spec=LLM)
        llm.config = primary.config.model_copy(
            update={"model_provider": spec.provider_type, "model_name": spec.model_name}
        )
        models[role] = llm
        bindings[role] = ModelBinding(llm, len, index + 101)
    return ModelComposition("kompozit_v1", 1, bindings), models


def _ordinary_adapter(
    monkeypatch: pytest.MonkeyPatch, *, cropped: bool
) -> tuple[ResearchModel, dict[CompositionRole, MagicMock], ModelComposition]:
    previous, models, composition = composition_fixture(
        monkeypatch, cropped=cropped, profile="kompozit_v1"
    )
    previous.context.services.pop("asv3_fast")
    previous.context.services.pop(FAST_RESPONSE_POLICY_SERVICE, None)
    previous.context.services["asv3_kompozit"] = True
    if cropped:
        previous.context.services["asv3_kompozit_cropped"] = True
    return (
        ResearchModel(models["coordinator"], previous.context, lean_native_mode=True),
        models,
        composition,
    )


@pytest.mark.parametrize("variant", _VARIANTS)
def test_runtime_wires_fixed_roles_without_fast_execution_or_benchmark_deadline(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, broker, primary, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    composition, models = _runtime_composition(primary)
    kwargs.update(
        research_profile="normal",
        workflow_variant=variant,
        model_composition=composition,
        reasoning_effort=ReasoningEffort.LOW,
    )
    vision: list[LLM] = []

    def create_broker(
        _user: Any, _scope: Any, *, vision_llm: LLM, **_options: Any
    ) -> Any:
        vision.append(vision_llm)
        return broker

    monkeypatch.setattr(runtime, "CorpusBroker", create_broker)
    fast_setup = MagicMock(
        side_effect=AssertionError("Fast policy must remain isolated")
    )
    monkeypatch.setattr(runtime, "install_fast_timing", fast_setup)
    monkeypatch.setattr(runtime, "fast_material_review_model", fast_setup)
    search, scoped = MagicMock(spec=SearchTool), MagicMock(spec=SearchTool)
    search.fork_for_independent_context.return_value = scoped
    kwargs["tools"] = [search]
    captured: list[RunContext] = []

    def capture(_llm: LLM, context: RunContext, **_options: Any) -> ResearchModel:
        captured.append(context)
        return ResearchModel(_llm, context, **_options)

    monkeypatch.setattr(runtime, "ResearchModel", capture)
    monkeypatch.setattr(
        runtime, "build_search_adapter", MagicMock(side_effect=SetupBoundaryReached)
    )
    with pytest.raises(SetupBoundaryReached):
        runtime.run_asv3_loop(**kwargs)
    context = captured[0]
    assert math.isinf(context.deadline)
    assert context.services[MODEL_COMPOSITION_SERVICE] is composition
    assert context.services["asv3_kompozit"] is True
    assert context.services.get("asv3_fast") is not True
    assert context.services.get("asv3_fast_retrieval") is not True
    assert FAST_TIMING_SERVICE not in context.services
    assert FAST_RESPONSE_POLICY_SERVICE not in context.services
    assert vision == [models["routine"]]
    assert scoped.llm is models["search"]
    assert scoped.llm.config.model_name == "gemini-3.5-flash-lite"
    assert models["routine"].config.model_name == "gemini-3.8-flash"
    assert models["author"].config.model_name == "claude-sonnet-5-5"
    assert models["repair"].config.model_name == "gpt-6.1-terra"
    assert models["verifier"].config.model_name == "gemini-4-argon"
    assert context.services.get("asv3_exp_cropped") is (
        True if variant == ASV3_KOMPOZIT_CROPPED_VARIANT else None
    )
    if variant == ASV3_KOMPOZIT_CROPPED_VARIANT:
        assert context.services["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
        assert isinstance(context.services["cropped_cost_budget"], CroppedCostBudget)
        assert allows_cropped_composition_implicit_cache(context)
    else:
        assert "cropped_cost_budget" not in context.services
        assert not allows_cropped_composition_implicit_cache(context)
    fast_setup.assert_not_called()
    for selected_model in models.values():
        selected_model.invoke.assert_not_called()


@pytest.mark.parametrize("variant", _VARIANTS)
@pytest.mark.parametrize("profile", [None, "factorial_v1_fna"])
def test_named_runtime_rejects_missing_or_benchmark_composition_before_corpus_access(
    monkeypatch: pytest.MonkeyPatch, variant: str, profile: CompositionProfileId | None
) -> None:
    kwargs, _broker, primary, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    composition, _models = _runtime_composition(primary)
    kwargs.update(
        research_profile="normal",
        workflow_variant=variant,
        model_composition=None
        if profile is None
        else ModelComposition(profile, 1, composition.role_bindings),
        reasoning_effort=ReasoningEffort.LOW,
    )
    corpus = MagicMock(
        side_effect=AssertionError("invalid configuration reached corpus")
    )
    monkeypatch.setattr(runtime, "bind_pc_corpus_scope", corpus)
    with pytest.raises(ValueError):
        runtime.run_asv3_loop(**kwargs)
    corpus.assert_not_called()
    primary.invoke.assert_not_called()
    assert not checkpoints


@pytest.mark.usefixtures("empty_source_inventory")
@pytest.mark.parametrize("variant", _VARIANTS)
def test_real_runtime_checkpoints_preserve_copy_and_composition_resume_identity(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, _broker, primary, checkpoints, _queue = setup_run(monkeypatch)
    monkeypatch.setattr(
        "onyx.asv3.cropped_cost_budget._rates",
        lambda *_args: (1e-9, 2e-9, 0.1e-9, 1.25e-9),
    )
    kwargs.pop("test_language")
    composition, models = _runtime_composition(primary)
    kwargs.update(
        research_profile="normal",
        workflow_variant=variant,
        model_composition=composition,
        reasoning_effort=ReasoningEffort.LOW,
    )
    primary.invoke.side_effect = None
    primary.invoke.return_value = response(
        calls=[("ask_user", {"question": "Makinenin kullanım amacı nedir?"})]
    )
    primary.invoke.return_value.usage = Usage(
        prompt_tokens=1,
        completion_tokens=1,
        total_tokens=2,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    runtime.run_asv3_loop(**kwargs)
    assert primary.invoke.call_count == 1
    models["verifier"].invoke.assert_not_called()
    models["author"].invoke.assert_not_called()
    assert checkpoints
    for saved in checkpoints:
        assert all(
            saved[key] == value
            for key, value in checkpoint_variant_fields(variant).items()
        )
        assert saved[MODEL_COMPOSITION_SERVICE] == composition.identity()
    previous = copy.deepcopy(checkpoints[-1])
    previous[MODEL_COMPOSITION_SERVICE] = {**composition.identity(), "version": 2}
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_options: previous)
    kwargs["resume_message_id"] = 42
    with pytest.raises(ValueError, match="same model composition"):
        runtime.run_asv3_loop(**kwargs)
    assert primary.invoke.call_count == 1


@pytest.mark.parametrize("cropped", [False, True])
def test_native_research_keeps_ordinary_tools_then_sticky_author_and_terra_repair(
    monkeypatch: pytest.MonkeyPatch, cropped: bool
) -> None:
    adapter, models, composition = _ordinary_adapter(monkeypatch, cropped=cropped)
    current = original_view(adapter)
    models["coordinator"].invoke.return_value = native_action(
        "read_provision", {"source_id": "law", "article": "7"}
    )
    models["routine"].invoke.return_value = prose(
        "Routine candidate is not the published answer [1][2]."
    )
    models["author"].invoke.side_effect = [
        native_action("read_provision", {"source_id": "law", "article": "8"}),
        prose("A AND B, subject to written-consent exception C [1][2]."),
    ]
    assert adapter.decide(current).calls[0].arguments["article"] == "7"
    assert adapter.decide(current).calls[0].arguments["article"] == "8"
    assert (
        adapter.decide(current).answer
        == "A AND B, subject to written-consent exception C [1][2]."
    )
    assert models["routine"].invoke.call_count == 1
    assert models["author"].invoke.call_count == 2
    routine = models["routine"].invoke.call_args.kwargs
    assert "request_primary_answer" not in names(routine["tools"])
    assert {"submit_answer", "ask_user"} <= names(routine["tools"])
    assert FAST_TEK_ACTION_RESEARCH not in str(routine["prompt"][0].content)
    assert FAST_RESPONSE_POLICY_SERVICE not in adapter.context.services
    checkpoint: dict[str, JsonValue] = {
        "native_coordinator_sampling": adapter.native_sampling_snapshot()
    }
    restored = ResearchModel(
        models["coordinator"], adapter.context, lean_native_mode=True
    )
    restored.restore_native_sampling(checkpoint)
    models["author"].invoke.side_effect = None
    models["author"].invoke.return_value = prose("Same author after resume [1][2].")
    assert restored.decide(current).answer == "Same author after resume [1][2]."
    assert models["routine"].invoke.call_count == 1
    models["repair"].invoke.return_value = prose(
        "Repaired answer retains condition C [1][2]."
    )
    repair = current.model_copy(
        update={
            "draft_to_repair": "Owned answer [1][2].",
            "publication_gap": {"missing": "condition C"},
        }
    )
    assert (
        restored.decide(repair).answer == "Repaired answer retains condition C [1][2]."
    )
    assert models["repair"].invoke.call_count == 1
    assert restored.llm is models["coordinator"]
    assert restored.token_counter is composition.binding("coordinator").token_counter


@pytest.mark.usefixtures("empty_source_inventory")
@pytest.mark.parametrize("variant", _VARIANTS)
def test_verifier_is_argon_only_when_existing_verify_claim_path_is_requested(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, _broker, primary, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    composition, models = _runtime_composition(primary)
    kwargs.update(
        research_profile="normal",
        workflow_variant=variant,
        model_composition=composition,
        reasoning_effort=ReasoningEffort.LOW,
    )
    contexts: list[RunContext] = []

    def boundary(*, context: RunContext, **_options: Any) -> Any:
        contexts.append(context)
        raise SetupBoundaryReached

    monkeypatch.setattr(runtime, "Harness", boundary)
    with pytest.raises(SetupBoundaryReached):
        runtime.run_asv3_loop(**kwargs)
    assert all(
        selected_model.invoke.call_count == 0 for selected_model in models.values()
    )
    verifier = MagicMock()
    verifier.invoke_verification.return_value = VerificationResult(
        status="supported",
        explanation="The supplied original establishes the condition.",
        required_conditions=[],
        missing_conditions=[],
        evidence_numbers=[],
    )
    factory = MagicMock(return_value=verifier)
    monkeypatch.setattr(runtime, "ResearchModel", factory)
    verify = cast(
        Callable[[dict[str, Any], RunContext], ToolOutcome],
        contexts[0].services["verify_claim"],
    )
    assert (
        verify(
            {"claim": "The condition applies.", "evidence_numbers": []}, contexts[0]
        ).status
        == OutcomeStatus.FOUND
    )
    assert factory.call_args.args[0] is models["verifier"]
    assert factory.call_args.args[0].config.model_name == "gemini-4-argon"
    assert factory.call_args.kwargs["composition_role"] == "verifier"
    assert (
        factory.call_args.kwargs["token_counter"]
        is composition.binding("verifier").token_counter
    )


@pytest.mark.parametrize(
    "profile,fast,kompozit,cropped,allowed",
    [
        ("kompozit_v1", False, True, True, True),
        ("kompozit_v1", False, True, False, False),
        ("kompozit_v1", False, False, True, False),
        ("factorial_v1_fna", True, False, True, True),
        ("factorial_v1_fna", False, True, True, False),
        ("flash_sol_v1", True, False, True, False),
    ],
)
def test_implicit_cache_permission_is_limited_to_its_existing_or_new_cropped_copy(
    profile: CompositionProfileId,
    fast: bool,
    kompozit: bool,
    cropped: bool,
    allowed: bool,
) -> None:
    context = composition_context(selected(), profile, fast)
    context.services["asv3_kompozit_cropped"] = kompozit
    context.services["asv3_exp_cropped"] = cropped
    assert allows_cropped_composition_implicit_cache(context) is allowed


def test_fresh_terra_repair_retains_scoped_implicit_cache_permission_without_fast_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, models, composition = _ordinary_adapter(monkeypatch, cropped=True)
    repaired = fresh_model(adapter, role="repair")
    assert repaired.llm is models["repair"]
    assert repaired.context.services[MODEL_COMPOSITION_SERVICE] is composition
    assert repaired.context.services["asv3_kompozit_cropped"] is True
    assert allows_cropped_composition_implicit_cache(repaired.context)
    assert repaired.context.services.get("asv3_fast") is not True
    assert repaired.context.services["cropped_cost_phase"] == "repair"
    assert COMPOSITION_PROFILES["kompozit_v1"]["repair"] == "terra"


@pytest.mark.parametrize("path", ["native", "search"])
@pytest.mark.parametrize("permission", [False, True])
def test_native_flash_cache_proof_reaches_both_budget_boundaries_without_fast(
    monkeypatch: pytest.MonkeyPatch, path: str, permission: bool
) -> None:
    monkeypatch.setattr(
        "onyx.asv3.cropped_cost_budget._rates",
        lambda *_args: (0.75e-6, 3.75e-6, 0.075e-6, None),
    )
    llm = selected()
    context = composition_context(llm, "kompozit_v1", False)
    context.services["asv3_kompozit_cropped"] = permission
    prompt: list[ChatCompletionMessage] = [UserMessage(content="Question")]
    if path == "search":
        scoped = ScopedSearchLLM(llm, context, None)
        if permission:
            _budget, ticket = scoped._cropped_reservation(prompt, [], None, None, 1024)
            assert ticket is not None and ticket.reservation_upper_verified
        else:
            with pytest.raises(CroppedResearchBudgetReached):
                scoped._cropped_reservation(prompt, [], None, None, 1024)
        return
    span = MagicMock()
    span.span_data = GenerationSpanData(model="gemini-3.8-flash")

    @contextmanager
    def resources(
        *_args: object, **_kwargs: object
    ) -> Iterator[Span[GenerationSpanData]]:
        yield cast(Span[GenerationSpanData], span)

    monkeypatch.setattr(llm_adapter, "_generation_resources", resources)
    invoked = MagicMock(side_effect=SetupBoundaryReached)
    monkeypatch.setattr(llm, "invoke", invoked)
    adapter = ResearchModel(llm, context, token_counter=len, lean_native_mode=True)
    with pytest.raises(
        SetupBoundaryReached if permission else CroppedResearchBudgetReached
    ):
        adapter._invoke_once(
            prompt,
            [],
            LLMFlow.ASV3_COORDINATOR,
            max_tokens=1024,
            research=True,
            structured=False,
        )
    assert invoked.call_count == int(permission)


@pytest.mark.parametrize(
    "variant,fast,cropped",
    [
        (ASV3_STANDARD_VARIANT, False, False),
        (ASV3_EXP_CROPPED_VARIANT, False, True),
        (ASV3_FAST_VARIANT, True, False),
        (ASV3_FAST_CROPPED_VARIANT, True, True),
    ],
)
def test_legacy_runtime_policy_selection_does_not_acquire_kompozit_services(
    monkeypatch: pytest.MonkeyPatch, variant: str, fast: bool, cropped: bool
) -> None:
    kwargs, _broker, primary, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    monkeypatch.setattr(runtime, "fast_material_review_model", lambda _user: primary)
    monkeypatch.setattr(runtime, "get_llm_token_counter", lambda _llm: len)
    contexts: list[RunContext] = []

    def capture(_llm: LLM, context: RunContext, **_options: Any) -> Any:
        contexts.append(context)
        raise SetupBoundaryReached

    monkeypatch.setattr(runtime, "ResearchModel", capture)
    with pytest.raises(SetupBoundaryReached):
        runtime.run_asv3_loop(**kwargs)
    context = contexts[0]
    assert context.services.get("asv3_fast", False) is fast
    assert context.services.get("asv3_exp_cropped", False) is cropped
    assert "asv3_kompozit" not in context.services
    assert "asv3_kompozit_cropped" not in context.services
    assert MODEL_COMPOSITION_SERVICE not in context.services
    assert math.isinf(context.deadline)
