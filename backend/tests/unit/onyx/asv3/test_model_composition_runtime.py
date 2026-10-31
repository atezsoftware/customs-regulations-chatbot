"""The real Fast runtime preserves composition identity and existing source guards."""

import copy
import json
import time
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3 import runtime
from onyx.asv3.cropped_cost_budget import CroppedCostBudget
from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.model_composition import (
    MODEL_COMPOSITION_SERVICE,
    ModelBinding,
    ModelComposition,
)
from onyx.asv3.model_composition.models import COMPOSITION_ROLES
from onyx.asv3.models import RunContext
from onyx.asv3.search_adapter import ScopedSearchLLM
from onyx.asv3.workflow_variant import ASV3_FAST_CROPPED_VARIANT, ASV3_FAST_VARIANT
from onyx.llm.interfaces import LLM
from onyx.llm.model_response import ModelResponse
from onyx.llm.models import ReasoningEffort, UserMessage
from onyx.prompts.asv3.fast import FAST_TEK_CONDITION_REVIEW
from onyx.tools.tool_implementations.search.search_tool import SearchTool
from tests.unit.onyx.asv3.test_fast_primary_coordinator import prose
from tests.unit.onyx.asv3.test_model_composition_adapter import (
    composition_fixture,
    original_view,
)
from tests.unit.onyx.asv3.test_native_model_adapter import native_action
from tests.unit.onyx.asv3.test_runtime import (
    delivered_originals,
    response,
    setup_run,
    user_payload,
)


def composition(primary: LLM) -> ModelComposition:
    binding = ModelBinding(primary, len, 101)
    return ModelComposition(
        "sol_reference_v1", 1, dict.fromkeys(COMPOSITION_ROLES, binding)
    )


def test_child_author_handoff_stays_author_after_further_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, models, _bundle = composition_fixture(monkeypatch)
    child = ResearchModel(root.llm, root.context.child(), lean_native_mode=True)
    models["routine"].invoke.return_value = native_action("request_primary_answer", {})
    models["author"].invoke.side_effect = [
        native_action(
            "read_provision", {"source_id": "authorized-statute", "article": "17"}
        ),
        prose("Author's source-bound result [1][2]."),
    ]
    current = original_view(root)
    assert child.decide(current).calls[0].name == "read_provision"
    assert child.decide(current).answer == "Author's source-bound result [1][2]."
    assert models["routine"].invoke.call_count == 1
    assert models["author"].invoke.call_count == 2
    models["coordinator"].invoke.assert_not_called()


def test_cropped_search_reservation_counts_actual_search_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, models, bundle = composition_fixture(monkeypatch, cropped=True)
    coordinator_counter = cast(MagicMock, bundle.binding("coordinator").token_counter)
    search_counter = cast(MagicMock, bundle.binding("search").token_counter)
    coordinator_counter.side_effect = AssertionError(
        "Coordinator must not count search input"
    )
    search_counter.side_effect = lambda _text: 1
    budget = MagicMock(spec=CroppedCostBudget)
    root.context.services["cropped_cost_budget"] = budget
    scoped = ScopedSearchLLM(models["search"], root.context, None)
    returned, _ticket = scoped._cropped_reservation(
        [UserMessage(content="Query")], None, None, None, 256
    )
    assert returned is budget
    assert budget.prepare.call_args.args[:3] == ("role-search", "openai", 145)
    assert search_counter.call_count > 0
    coordinator_counter.assert_not_called()
    models["search"].invoke.assert_not_called()


@pytest.mark.parametrize("variant", [ASV3_FAST_VARIANT, ASV3_FAST_CROPPED_VARIANT])
def test_runtime_wires_search_vision_and_child_services_before_any_model_call(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    kwargs, broker, primary, _checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    bundle = composition(primary)
    routine, search_llm = MagicMock(spec=LLM), MagicMock(spec=LLM)
    routine.config = search_llm.config = primary.config
    bindings = dict(bundle.role_bindings)
    bindings["routine"], bindings["search"] = (
        ModelBinding(routine, len, 202),
        ModelBinding(search_llm, len, 303),
    )
    bundle = ModelComposition(bundle.profile_id, bundle.version, bindings)
    kwargs.update(
        research_profile="normal",
        workflow_variant=variant,
        model_composition=bundle,
        reasoning_effort=ReasoningEffort.LOW,
    )
    vision: list[LLM] = []

    def create_broker(
        _user: Any, _scope: Any, *, vision_llm: LLM, **_options: Any
    ) -> Any:
        vision.append(vision_llm)
        return broker

    monkeypatch.setattr(runtime, "CorpusBroker", create_broker)
    monkeypatch.setattr(
        runtime,
        "fast_material_review_model",
        MagicMock(side_effect=AssertionError("legacy reviewer must not resolve")),
    )
    search, scoped = MagicMock(spec=SearchTool), MagicMock(spec=SearchTool)
    search.fork_for_independent_context.return_value = scoped
    kwargs["tools"] = [search]
    captured: list[RunContext] = []

    def boundary(_llm: LLM, context: RunContext, **_options: Any) -> Any:
        captured.append(context)
        # Root model setup precedes search wiring; inspect the same context below.
        return MagicMock()

    monkeypatch.setattr(runtime, "ResearchModel", boundary)
    monkeypatch.setattr(
        runtime,
        "build_search_adapter",
        MagicMock(side_effect=RuntimeError("search setup boundary")),
    )
    with pytest.raises(RuntimeError, match="search setup boundary"):
        runtime.run_asv3_loop(**kwargs)
    assert vision == [routine]
    assert scoped.llm is search_llm
    assert captured[0].services[MODEL_COMPOSITION_SERVICE] is bundle
    assert 479 < captured[0].deadline - time.monotonic() <= 480
    primary.invoke.assert_not_called()


@pytest.mark.parametrize("defect", ["original", "effort", "primary", "sampling"])
def test_invalid_composition_stops_before_paid_or_corpus_work(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    kwargs, _broker, primary, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(
        research_profile="normal",
        workflow_variant=ASV3_FAST_VARIANT,
        model_composition=composition(primary),
        reasoning_effort=ReasoningEffort.LOW,
    )
    if defect == "original":
        kwargs["workflow_variant"] = "standard"
    elif defect == "effort":
        kwargs["reasoning_effort"] = ReasoningEffort.AUTO
    else:
        selected = MagicMock(spec=LLM)
        selected.config = primary.config.model_copy(
            update={"model_name": "another-model"}
            if defect == "primary"
            else {"temperature": 0.8}
        )
        kwargs["llm"] = selected
    corpus = MagicMock()
    monkeypatch.setattr(runtime, "bind_pc_corpus_scope", corpus)
    with pytest.raises(ValueError):
        runtime.run_asv3_loop(**kwargs)
    corpus.assert_not_called()
    primary.invoke.assert_not_called()
    assert not checkpoints


@pytest.mark.usefixtures("empty_source_inventory")
def test_real_fast_tek_composition_handoff_review_and_checkpoint_fences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, broker, primary, checkpoints, _queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    primary.config = primary.config.model_copy(
        update={"model_provider": "vertex_ai", "model_name": "gemini-3.8-flash"}
    )
    routine, author = MagicMock(spec=LLM), MagicMock(spec=LLM)
    routine.config = primary.config.model_copy(
        update={"model_name": "gemini-3.5-flash-lite"}
    )
    author.config = primary.config.model_copy(
        update={"model_provider": "openai", "model_name": "gpt-6.1-sol"}
    )
    primary_binding, routine_binding, author_binding = (
        ModelBinding(primary, len, 101),
        ModelBinding(routine, len, 202),
        ModelBinding(author, len, 303),
    )
    bundle = ModelComposition(
        "flash_sol_v1",
        1,
        {
            "coordinator": primary_binding,
            "routine": routine_binding,
            "search": routine_binding,
            "author": author_binding,
            "repair": author_binding,
            "verifier": author_binding,
        },
    )
    kwargs.update(
        research_profile="normal",
        workflow_variant=ASV3_FAST_VARIANT,
        model_composition=bundle,
        reasoning_effort=ReasoningEffort.LOW,
    )
    monkeypatch.setattr(broker, "source_notices", lambda *_args: [], raising=False)
    legacy = MagicMock(side_effect=AssertionError("legacy resolver must not run"))
    monkeypatch.setattr(runtime, "fast_material_review_model", legacy)
    primary.invoke.side_effect = lambda **_arguments: response(
        calls=[("read_source_range", {"source_id": str(s.id)}) for s in broker.sources]
    )
    routine.invoke.return_value = response(calls=[("request_primary_answer", {})])
    author_packets: list[dict[str, Any]] = []

    def write_or_review(**arguments: Any) -> ModelResponse:
        prompt = arguments["prompt"]
        if str(prompt[0].content).startswith(FAST_TEK_CONDITION_REVIEW):
            payload = user_payload(prompt[-1])
            return response(
                json.dumps(
                    {
                        "status": "complete",
                        "candidate_sha256": payload["candidate_sha256"],
                        "source_set_sha256": payload["source_set_sha256"],
                        "examined_citations": payload["required_evidence_numbers"],
                        "findings": [],
                    }
                )
            )
        originals = delivered_originals(arguments)
        assert {row["text"] for row in originals} == {
            chunk.text for chunk in broker.chunks.values()
        }
        author_packets.append(arguments)
        return response(
            calls=[
                (
                    "submit_answer",
                    {
                        "answer": "Tamir sonucu [1]; değiştirme sonucu [2].",
                        "basis": "originals",
                    },
                )
            ]
        )

    author.invoke.side_effect = write_or_review
    runtime.run_asv3_loop(**kwargs)
    assert "Tamir sonucu" in (kwargs["state_container"].get_answer_tokens() or "")
    assert primary.invoke.call_count == routine.invoke.call_count == 1
    assert author.invoke.call_count == 2 and len(author_packets) == 1
    assert all(
        row[MODEL_COMPOSITION_SERVICE] == bundle.identity() for row in checkpoints
    )
    assert (
        checkpoints[-1]["native_coordinator_sampling"]["fast_primary_answer_mode"]
        is True
    )
    legacy.assert_not_called()
    for saved_identity in [None, {**bundle.identity(), "version": 2}]:
        previous = copy.deepcopy(checkpoints[-1])
        previous[MODEL_COMPOSITION_SERVICE] = saved_identity
        monkeypatch.setattr(
            runtime, "load_asv3_checkpoint", lambda **_options: previous
        )
        kwargs["resume_message_id"] = 42
        with pytest.raises(ValueError, match="same model composition"):
            runtime.run_asv3_loop(**kwargs)
    previous = copy.deepcopy(checkpoints[-1])
    monkeypatch.setattr(runtime, "load_asv3_checkpoint", lambda **_options: previous)
    kwargs["model_composition"] = None
    with pytest.raises(ValueError, match="same model composition"):
        runtime.run_asv3_loop(**kwargs)
    kwargs["model_composition"] = bundle
    monkeypatch.setattr(
        runtime,
        "ResearchModel",
        MagicMock(side_effect=RuntimeError("valid resume model boundary")),
    )
    with pytest.raises(RuntimeError, match="valid resume model boundary"):
        runtime.run_asv3_loop(**kwargs)
    assert author.invoke.call_count == 2
