from unittest.mock import MagicMock

import pytest

from onyx.asv3 import answer_model
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.db import asv3_models
from onyx.db.models import Persona, User
from onyx.error_handling.exceptions import OnyxError
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


def test_fast_material_review_uses_authorized_sol_and_preflights_its_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    user, persona = MagicMock(spec=User), MagicMock(spec=Persona)
    reviewer = MagicMock(spec=LLM)
    reviewer.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=1,
        max_input_tokens=240000,
        api_key="synthetic-provider-key",
    )
    factory = MagicMock(return_value=reviewer)
    cost = MagicMock()
    provider = MagicMock(return_value=43)
    monkeypatch.setattr(answer_model, "default_asv3_persona", lambda: persona)
    monkeypatch.setattr(answer_model, "get_llm_for_persona", factory)
    monkeypatch.setattr(answer_model, "check_asv3_secondary_model_cost_limit", cost)
    monkeypatch.setattr(answer_model, "resolve_asv3_model_provider_id", provider)
    assert answer_model.fast_material_review_model(user) is reviewer
    assert factory.call_args.args == (persona, user)
    selection = factory.call_args.kwargs["llm_override"]
    assert selection.model_provider_type == "openai"
    assert selection.model_version == "gpt-6.1-sol"
    assert selection.model_provider_id == 43
    provider.assert_called_once_with("openai", "gpt-6.1-sol")
    cost.assert_called_once_with("synthetic-provider-key")


@pytest.mark.parametrize("stage", ["access", "cost"])
def test_fast_reviewer_access_and_cost_errors_propagate_without_fallback(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    reviewer = MagicMock(spec=LLM)
    reviewer.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=1,
        max_input_tokens=240000,
    )
    factory = MagicMock(return_value=reviewer)
    cost = MagicMock()
    error = PermissionError("Access or cost limit denied")
    if stage == "access":
        factory.side_effect = error
    else:
        cost.side_effect = error
    monkeypatch.setattr(
        answer_model, "default_asv3_persona", lambda: MagicMock(spec=Persona)
    )
    monkeypatch.setattr(answer_model, "get_llm_for_persona", factory)
    monkeypatch.setattr(answer_model, "check_asv3_secondary_model_cost_limit", cost)
    monkeypatch.setattr(answer_model, "resolve_asv3_model_provider_id", lambda *_: 43)
    with pytest.raises(PermissionError) as raised:
        answer_model.fast_material_review_model(MagicMock(spec=User))
    assert raised.value is error
    assert factory.call_count == 1 and cost.call_count == (stage == "cost")


def test_secondary_model_cost_preflight_uses_tenant_scoped_db_helper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = MagicMock()
    context = MagicMock()
    context.__enter__.return_value = session
    cost = MagicMock()
    monkeypatch.setattr(asv3_models, "get_session_with_current_tenant", lambda: context)
    monkeypatch.setattr(asv3_models, "get_current_tenant_id", lambda: "test-tenant")
    monkeypatch.setattr(asv3_models, "check_llm_cost_limit_for_provider", cost)
    asv3_models.check_asv3_secondary_model_cost_limit("synthetic-provider-key")
    cost.assert_called_once_with(
        db_session=session,
        tenant_id="test-tenant",
        llm_provider_api_key="synthetic-provider-key",
    )
    assert context.__exit__.call_count == 1


def test_reviewer_provider_lookup_includes_named_visible_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, context = MagicMock(), MagicMock()
    context.__enter__.return_value = session
    session.scalars.return_value = [43]
    monkeypatch.setattr(asv3_models, "get_session_with_current_tenant", lambda: context)

    assert asv3_models.resolve_asv3_model_provider_id("openai", "gpt-6.1-sol") == 43

    query = str(
        session.scalars.call_args.args[0].compile(
            compile_kwargs={"literal_binds": True}
        )
    )
    assert "SELECT DISTINCT llm_provider.id" in query
    assert "llm_provider.provider = 'openai'" in query
    assert "model_configuration.name = 'gpt-6.1-sol'" in query
    assert "model_configuration.is_visible IS true" in query
    assert "llm_provider.name" not in query
    assert "LIMIT 2" in query
    assert context.__exit__.call_count == 1


@pytest.mark.parametrize("identifiers", [[], [43, 44]])
def test_reviewer_provider_lookup_rejects_missing_or_ambiguous_configuration(
    monkeypatch: pytest.MonkeyPatch, identifiers: list[int]
) -> None:
    session, context = MagicMock(), MagicMock()
    context.__enter__.return_value = session
    session.scalars.return_value = identifiers
    monkeypatch.setattr(asv3_models, "get_session_with_current_tenant", lambda: context)
    factory = MagicMock()
    monkeypatch.setattr(answer_model, "get_llm_for_persona", factory)

    with pytest.raises(OnyxError, match="selected language model is unavailable"):
        answer_model.fast_material_review_model(MagicMock(spec=User))

    factory.assert_not_called()
