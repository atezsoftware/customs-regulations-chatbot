"""Exercise the replacement selection and checkpoint fences at their real boundaries."""

import copy
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from onyx.asv3 import runtime
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.models import ASv3WorkflowSelection
from onyx.asv3.shared_reads import SharedReads
from onyx.asv3.workflow_variant import (
    ASV3_STANDARD_VARIANT,
    ASV3_TUNED_POLICY,
    ASV3_TUNED_VARIANT,
    checkpoint_variant_fields,
    resolve_asv3_workflow,
    validate_asv3_variant_resume,
)
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM
from onyx.prompts.asv3.tuned import TUNED_PROMPT_VERSION
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from tests.unit.onyx.asv3.test_runtime import response, setup_run

pytestmark = pytest.mark.usefixtures("empty_source_inventory")


@pytest.mark.parametrize("profile", ["normal", "deep", "experimental"])
def test_only_legacy_parallel_selection_is_replaced(profile: str) -> None:
    ordinary = resolve_asv3_workflow(profile, False)
    assert ordinary.research_profile == profile
    assert ordinary.parallel_research is False
    assert ordinary.workflow_variant == ASV3_STANDARD_VARIANT
    assert not ordinary.selected_model_only
    assert checkpoint_variant_fields(ordinary.workflow_variant) == {}
    tuned = resolve_asv3_workflow("experimental", True)
    assert tuned.research_profile == "normal"
    assert tuned.parallel_research is False
    assert tuned.workflow_variant == ASV3_TUNED_VARIANT
    assert tuned.selected_model_only


def test_workflow_selection_is_strict_and_immutable() -> None:
    selected = resolve_asv3_workflow("experimental", True)
    with pytest.raises(ValidationError):
        selected.parallel_research = True
    with pytest.raises(ValidationError):
        ASv3WorkflowSelection.model_validate(
            {"research_profile": "normal", "parallel_research": 0}
        )


@pytest.mark.parametrize(
    "saved",
    [
        {},
        {"research_profile": "normal", "parallel_research": False},
        {"research_profile": "experimental", "parallel_research": True},
        {"asv3_workflow_variant": "asv3_tuned"},
        {
            "asv3_workflow_variant": "asv3_tuned",
            "asv3_workflow_policy": ASV3_TUNED_POLICY,
            "research_profile": "normal",
            "parallel_research": 0,
        },
    ],
)
def test_tuned_resume_requires_its_exact_variant_policy(saved: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        validate_asv3_variant_resume(ASV3_TUNED_VARIANT, saved)


def test_standard_keeps_historical_checkpoints_but_rejects_tuned() -> None:
    validate_asv3_variant_resume("standard", {"research_profile": "normal"})
    validate_asv3_variant_resume("standard", None)
    with pytest.raises(ValueError, match="same workflow variant"):
        validate_asv3_variant_resume(
            "standard", checkpoint_variant_fields("asv3_tuned")
        )


def test_tuned_runtime_uses_normal_tools_and_selected_model_everywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    secondary = MagicMock(spec=LLM)
    secondary.invoke.side_effect = AssertionError("Lite handoff must not run")
    search = MagicMock(spec=SearchTool)
    fork = MagicMock(spec=SearchTool)
    search.llm = secondary
    search.fork_for_independent_context.return_value = fork
    contexts = []
    original_model = runtime.ResearchModel

    def make_model(llm: LLM, context: Any, **options: Any) -> ResearchModel:
        assert llm is selected
        assert options.get("research_llm") is None
        contexts.append(context)
        return original_model(llm, context, **options)

    def make_broker(_user: Any, scope: Any, **options: Any) -> Any:
        assert options["vision_llm"] is selected
        assert options["allow_numbered_title_fallback"] is True
        broker.scope = scope
        return broker

    monkeypatch.setattr(runtime, "ResearchModel", make_model)
    monkeypatch.setattr(runtime, "CorpusBroker", make_broker)
    kwargs.update(
        research_profile="normal",
        parallel_research=False,
        workflow_variant=ASV3_TUNED_VARIANT,
        research_llm=secondary,
        tools=[search],
    )
    runtime.run_asv3_loop(**kwargs)
    assert selected.invoke.call_count == 2
    secondary.invoke.assert_not_called()
    assert fork.llm is selected
    assert search.llm is secondary
    context = contexts[0]
    assert context.services["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
    assert context.services["research_profile"] == "normal"
    assert context.services["experimental_parallel"] is False
    assert context.services["independent_question_mode"] is False
    assert not {
        "authority_requirements",
        "parallel_execution_slots",
        "parallel_query_embeddings",
    }.intersection(context.services)
    assert isinstance(context.services["legal_source_reviews"], LegalSourceReviews)
    assert isinstance(context.services["shared_reads"], SharedReads)
    for invocation in selected.invoke.call_args_list:
        names = {tool["function"]["name"] for tool in invocation.kwargs["tools"]}
        assert not {
            "research_questions",
            "assemble_answers",
            "commit_retained_answer",
        }.intersection(names)
    assert checkpoints[-1]["prompt_version"] == TUNED_PROMPT_VERSION
    assert checkpoints[-1]["research_profile"] == "normal"
    assert checkpoints[-1]["parallel_research"] is False
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
    assert checkpoints[-1]["asv3_workflow_policy"] == ASV3_TUNED_POLICY


def test_actual_runtime_resume_fences_before_saved_mode_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, _broker, selected, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=ASV3_TUNED_VARIANT)
    kwargs["simple_chat_history"] = [
        ChatMessageSimple(
            message="Merhaba", token_count=1, message_type=MessageType.USER
        )
    ]
    selected.invoke.side_effect = lambda **_args: response(
        calls=[
            (
                "submit_answer",
                {"answer": "Merhaba!", "basis": "conversation", "_language": "tr"},
            )
        ]
    )
    runtime.run_asv3_loop(**kwargs)
    tuned = copy.deepcopy(checkpoints[-1])
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: tuned)
    kwargs["resume_message_id"] = 2
    selected.reset_mock()
    runtime.run_asv3_loop(**kwargs)
    assert selected.invoke.call_count == 1
    assert checkpoints[-1]["asv3_workflow_variant"] == ASV3_TUNED_VARIANT
    kwargs["workflow_variant"] = ASV3_STANDARD_VARIANT
    selected.reset_mock()
    with pytest.raises(ValueError, match="same workflow variant"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
    old_parallel = copy.deepcopy(tuned)
    old_parallel.pop("asv3_workflow_variant")
    old_parallel.pop("asv3_workflow_policy")
    old_parallel.update(research_profile="experimental", parallel_research=True)
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_args: old_parallel)
    kwargs["workflow_variant"] = ASV3_TUNED_VARIANT
    with pytest.raises(ValueError, match="retired parallel"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
    kwargs["workflow_variant"] = ASV3_STANDARD_VARIANT
    with pytest.raises(ValueError, match="retired parallel"):
        runtime.run_asv3_loop(**kwargs)
    selected.invoke.assert_not_called()
