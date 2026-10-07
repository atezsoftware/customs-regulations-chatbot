from unittest.mock import MagicMock

import pytest

from onyx.asv3 import answer_model
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.db.models import Persona, User
from onyx.llm.interfaces import LLM, LLMConfig


@pytest.mark.parametrize(
    "variant,provider,name,expected",
    [
        (ASV3_TUNED_VARIANT, "vertex_ai", "gemini-3.8-flash", True),
        ("standard", "vertex_ai", "gemini-3.8-flash", False),
        (ASV3_TUNED_VARIANT, "vertex_ai", "gemini-3.7-flash", False),
        (ASV3_TUNED_VARIANT, "openai", "gpt-6-luna", False),
    ],
)
def test_source_answer_policy_is_scoped(
    variant: str, provider: str, name: str, expected: bool
) -> None:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider=provider,
        model_name=name,
        temperature=1,
        max_input_tokens=1000000,
    )
    assert answer_model.uses_source_answer_model(variant, llm) is expected


def test_source_answer_selection_uses_visible_model_factory_and_caller_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, persona = MagicMock(spec=User), MagicMock(spec=Persona)
    factory = MagicMock(return_value=MagicMock(spec=LLM))
    monkeypatch.setattr(answer_model, "default_asv3_persona", lambda: persona)
    monkeypatch.setattr(answer_model, "get_llm_for_persona", factory)
    result = answer_model.source_answer_model(user)
    assert result is factory.return_value
    args = factory.call_args
    assert args.args == (persona, user)
    selection = args.kwargs["llm_override"]
    assert selection.model_provider_type == "anthropic"
    assert selection.model_version == "claude-sonnet-5-5"
    assert selection.model_provider_id is None
    assert selection.temperature is None
