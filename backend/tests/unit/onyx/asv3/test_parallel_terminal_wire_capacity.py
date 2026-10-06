"""Physical context fitting accounts for the provider's expanded terminal schema."""

import copy

import pytest

from onyx.asv3.llm_adapter import ResearchModel
from onyx.asv3.terminal_wire_schema import (
    bind_terminal_review_citations,
    strict_terminal_tools,
)
from onyx.llm.models import ChatCompletionMessage, UserMessage
from tests.unit.onyx.asv3.test_native_model_adapter import model
from tests.unit.onyx.asv3.test_terminal_review_citation_bindings import A, B, selected
from tests.unit.onyx.asv3.test_terminal_wire_schema import context


@pytest.mark.parametrize(
    "provider,name,expanded",
    [
        ("openai", "gpt-6-luna", True),
        ("openai", "compatible-custom-model", False),
        ("vertex_ai", "gpt-6-luna", False),
    ],
)
def test_input_accounting_uses_exact_wire_and_preserves_selected_host_schemas(
    provider: str, name: str, expanded: bool
) -> None:
    chosen = model()
    chosen.config = chosen.config.model_copy(
        update={"model_provider": provider, "model_name": name}
    )
    scoped = context()
    bound = bind_terminal_review_citations(selected(), scoped, {A: (1, 4), B: (2,)})
    before = copy.deepcopy(bound)
    wire = strict_terminal_tools(bound, scoped, provider, name)
    parallel = ResearchModel(chosen, scoped, token_counter=len, lean_native_mode=True)
    plain_accounting = ResearchModel(
        chosen, context(False), token_counter=len, lean_native_mode=True
    )
    prompt: list[ChatCompletionMessage] = [
        UserMessage(content="The complete original remains present.")
    ]
    assert parallel._input_cost(prompt, bound) == plain_accounting._input_cost(
        prompt, wire
    )
    host_cost = plain_accounting._input_cost(prompt, bound)
    if expanded:
        assert parallel._input_cost(prompt, bound) > host_cost
    else:
        assert parallel._input_cost(prompt, bound) == host_cost
    assert bound == before
