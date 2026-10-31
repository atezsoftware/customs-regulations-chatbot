"""Composition preflight uses configured identities without model or database calls."""

import json
from typing import cast
from unittest.mock import MagicMock

import pytest

from onyx.asv3.model_composition import (
    MODEL_COMPOSITION_SERVICE,
    CompositionProfileId,
    ModelBinding,
    ModelComposition,
    get_model_composition,
    resolver,
)
from onyx.asv3.model_composition.models import COMPOSITION_ROLES
from onyx.asv3.model_composition.profiles import COMPOSITION_PROFILES, MODEL_SPECS
from onyx.asv3.models import RunContext
from onyx.asv3.workflow_variant import ASV3_FAST_CROPPED_VARIANT, ASV3_FAST_VARIANT
from onyx.db.models import Persona, User
from onyx.llm.interfaces import LLM, LLMConfig
from onyx.llm.override_models import LLMOverride


@pytest.mark.parametrize("profile_id", list(COMPOSITION_PROFILES))
@pytest.mark.parametrize("variant", [ASV3_FAST_VARIANT, ASV3_FAST_CROPPED_VARIANT])
def test_every_profile_resolves_exact_configured_roles_with_caller_and_preflight(
    monkeypatch: pytest.MonkeyPatch, profile_id: CompositionProfileId, variant: str
) -> None:
    user, persona = MagicMock(spec=User), MagicMock(spec=Persona)
    provider_ids = {"openai": 101, "vertex_ai": 202, "anthropic": 303}
    provider = MagicMock(side_effect=lambda kind, _model: provider_ids[kind])

    def create(
        resolved_persona: Persona | None,
        resolved_user: User,
        *,
        llm_override: LLMOverride,
    ) -> LLM:
        assert (resolved_persona, resolved_user) == (persona, user)
        selection = llm_override
        assert selection.model_provider_type is not None
        assert selection.model_version is not None
        llm = MagicMock(spec=LLM)
        llm.config = LLMConfig(
            model_provider=selection.model_provider_type,
            model_name=selection.model_version,
            temperature=0.2,
            max_input_tokens=200000,
            api_key="synthetic-key",
        )
        assert (
            selection.model_provider_id == provider_ids[selection.model_provider_type]
        )
        return llm

    factory, cost, counter = (
        MagicMock(side_effect=create),
        MagicMock(),
        MagicMock(return_value=len),
    )
    monkeypatch.setattr(resolver, "resolve_asv3_model_provider_id", provider)
    monkeypatch.setattr(resolver, "get_llm_for_persona", factory)
    monkeypatch.setattr(resolver, "check_asv3_secondary_model_cost_limit", cost)
    monkeypatch.setattr(resolver, "get_llm_token_counter", counter)
    composition = resolver.resolve_model_composition(profile_id, user, persona, variant)

    profile = COMPOSITION_PROFILES[profile_id]
    assert (
        factory.call_count
        == provider.call_count
        == cost.call_count
        == counter.call_count
        == len(set(profile.values()))
    )
    for role, alias in profile.items():
        binding = composition.binding(role)
        expected = MODEL_SPECS[alias]
        assert binding.llm.config.model_name == expected.model_name
        assert binding.llm.config.model_provider == expected.provider_type
        assert binding.token_counter is len
        cast(MagicMock, binding.llm).invoke.assert_not_called()
        for other, other_alias in profile.items():
            if alias == other_alias:
                assert composition.binding(other) is binding
    assert composition.identity()["profile_id"] == profile_id
    assert "synthetic-key" not in json.dumps(composition.identity())


@pytest.mark.parametrize("stage", ["provider", "access", "cost", "tokenizer"])
def test_preflight_error_propagates_without_fallback_or_paid_calls(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=0,
        max_input_tokens=200000,
    )
    operations = {
        "provider": MagicMock(return_value=101),
        "access": MagicMock(return_value=llm),
        "cost": MagicMock(),
        "tokenizer": MagicMock(return_value=len),
    }
    error = PermissionError("configured model unavailable")
    operations[stage].side_effect = error
    for name, operation in [
        ("resolve_asv3_model_provider_id", "provider"),
        ("get_llm_for_persona", "access"),
        ("check_asv3_secondary_model_cost_limit", "cost"),
        ("get_llm_token_counter", "tokenizer"),
    ]:
        monkeypatch.setattr(resolver, name, operations[operation])
    with pytest.raises(PermissionError) as raised:
        resolver.resolve_model_composition(
            "sol_reference_v1",
            MagicMock(spec=User),
            MagicMock(spec=Persona),
            ASV3_FAST_VARIANT,
        )
    assert raised.value is error
    llm.invoke.assert_not_called()


@pytest.mark.parametrize("variant", ["standard", "asv3_tuned", "asv3_exp_cropped"])
def test_composition_rejects_original_variants_before_resolution(
    monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    provider = MagicMock()
    monkeypatch.setattr(resolver, "resolve_asv3_model_provider_id", provider)
    with pytest.raises(ValueError, match="explicit Fast"):
        resolver.resolve_model_composition(
            "sol_reference_v1", MagicMock(spec=User), None, variant
        )
    provider.assert_not_called()


def test_unknown_profile_and_incomplete_bindings_fail_closed() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        resolver.resolve_model_composition(
            cast(CompositionProfileId, "unknown"),
            MagicMock(spec=User),
            None,
            ASV3_FAST_VARIANT,
        )
    with pytest.raises(ValueError, match="every exact role"):
        ModelComposition("sol_reference_v1", 1, {})


def test_binding_identity_is_safe_and_changes_with_capacity_or_endpoint() -> None:
    llm = MagicMock(spec=LLM)
    llm.config = LLMConfig(
        model_provider="openai",
        model_name="gpt-6.1-sol",
        temperature=0,
        max_input_tokens=200000,
        api_key="synthetic-secret",
        api_base="https://example.invalid/private-endpoint",
    )
    binding = ModelBinding(llm, len, 101)
    composition = ModelComposition(
        "sol_reference_v1", 1, dict.fromkeys(COMPOSITION_ROLES, binding)
    )
    identity = composition.identity()
    assert "synthetic-secret" not in json.dumps(identity)
    assert "private-endpoint" not in json.dumps(identity)
    llm.config = llm.config.model_copy(update={"max_input_tokens": 199000})
    assert composition.identity() != identity
    context = RunContext(services={MODEL_COMPOSITION_SERVICE: composition})
    assert get_model_composition(context) is composition
    assert get_model_composition(RunContext()) is None
    with pytest.raises(ValueError, match="Invalid resolved"):
        get_model_composition(RunContext(services={MODEL_COMPOSITION_SERVICE: {}}))
