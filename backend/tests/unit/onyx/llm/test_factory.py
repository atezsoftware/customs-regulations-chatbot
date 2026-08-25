from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from onyx.db.llm import (
    fetch_existing_llm_provider,
    fetch_existing_llm_provider_by_type_nameless,
    fetch_llm_provider_for_model_selection,
)
from onyx.db.models import Persona, User
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError
from onyx.llm.constants import LlmProviderNames
from onyx.llm.factory import (
    _build_provider_extra_headers,
    _resolve_provider_and_model,
    get_llm,
    get_llm_for_persona,
    get_llm_tokenizer_encode_func,
    llm_from_provider,
)
from onyx.llm.override_models import LLMOverride
from onyx.llm.well_known_providers.constants import LM_STUDIO_API_KEY_CONFIG_KEY
from onyx.server.manage.llm.models import LLMProviderView, ModelConfigurationView
from shared_configs.enums import EmbeddingProvider


def test_build_provider_extra_headers_adds_bearer_for_lm_studio_api_key() -> None:
    headers = _build_provider_extra_headers(
        LlmProviderNames.LM_STUDIO,
        {LM_STUDIO_API_KEY_CONFIG_KEY: "  test-key  "},
    )

    assert headers == {"Authorization": "Bearer test-key"}


def test_build_provider_extra_headers_keeps_existing_bearer_prefix() -> None:
    headers = _build_provider_extra_headers(
        LlmProviderNames.LM_STUDIO,
        {LM_STUDIO_API_KEY_CONFIG_KEY: "bearer test-key"},
    )

    assert headers == {"Authorization": "bearer test-key"}


def test_build_provider_extra_headers_ignores_empty_lm_studio_api_key() -> None:
    headers = _build_provider_extra_headers(
        LlmProviderNames.LM_STUDIO,
        {LM_STUDIO_API_KEY_CONFIG_KEY: "   "},
    )

    assert headers == {}


def test_build_provider_extra_headers_ignores_legacy_ollama_custom_config() -> None:
    # Ollama now carries its key in the standard api_key field, which LiteLLM
    # turns into a Bearer header itself; custom_config must not add one.
    headers = _build_provider_extra_headers(
        LlmProviderNames.OLLAMA_CHAT,
        {"OLLAMA_API_KEY": "test-key"},
    )

    assert headers == {}


@pytest.mark.parametrize(
    ("provider", "expected_tokenizer_provider"),
    [
        (LlmProviderNames.OPENAI, EmbeddingProvider.OPENAI),
        (LlmProviderNames.ANTHROPIC, EmbeddingProvider.LITELLM),
        (LlmProviderNames.VERTEX_AI, EmbeddingProvider.LITELLM),
    ],
)
def test_llm_tokenizer_uses_lightweight_cloud_fallback(
    provider: str, expected_tokenizer_provider: EmbeddingProvider
) -> None:
    llm = MagicMock()
    llm.config.model_provider = provider
    llm.config.model_name = "new-cloud-model"
    tokenizer = MagicMock()

    with patch("onyx.llm.factory.get_tokenizer", return_value=tokenizer) as get:
        encode = get_llm_tokenizer_encode_func(llm)

    assert encode is tokenizer.encode
    get.assert_called_once_with(
        model_name="new-cloud-model",
        provider_type=expected_tokenizer_provider,
    )


def test_resolve_provider_uses_type_for_nameless_provider() -> None:
    provider = MagicMock()
    provider.model_configurations = [
        SimpleNamespace(name="gemini-3.6-flash", is_visible=True)
    ]
    db_session = MagicMock()
    persona = cast(Persona, SimpleNamespace(default_model_configuration_id=None))

    with (
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_legacy_selection",
            return_value=provider,
        ) as fetch_legacy,
    ):
        resolved = _resolve_provider_and_model(
            persona,
            LlmProviderNames.VERTEX_AI,
            "gemini-3.6-flash",
            db_session,
        )

    assert resolved == (provider, "gemini-3.6-flash")
    fetch_legacy.assert_called_once_with(LlmProviderNames.VERTEX_AI, db_session)


def test_resolve_provider_rejects_model_missing_from_provider_catalog() -> None:
    provider = MagicMock()
    provider.model_configurations = [
        SimpleNamespace(name="gemini-3.6-flash", is_visible=True)
    ]
    persona = cast(Persona, SimpleNamespace(default_model_configuration_id=None))

    with patch(
        "onyx.llm.factory.fetch_llm_provider_for_legacy_selection",
        return_value=provider,
    ):
        resolved = _resolve_provider_and_model(
            persona,
            "Vertex Main",
            "missing-model",
            MagicMock(),
        )

    assert resolved is None


def test_nameless_provider_lookup_rejects_ambiguous_matches() -> None:
    first_provider = SimpleNamespace(id=1)
    second_provider = SimpleNamespace(id=2)
    db_session = MagicMock()
    db_session.scalars.return_value = iter([first_provider, second_provider])

    resolved = fetch_existing_llm_provider_by_type_nameless(
        LlmProviderNames.VERTEX_AI, db_session
    )

    assert resolved is None


def test_legacy_named_provider_lookup_rejects_ambiguous_matches() -> None:
    first_provider = SimpleNamespace(id=1)
    second_provider = SimpleNamespace(id=2)
    db_session = MagicMock()
    db_session.scalar.return_value = first_provider
    db_session.scalars.return_value = iter([first_provider, second_provider])

    resolved = fetch_existing_llm_provider("Shared Provider", db_session)

    assert resolved is None


def test_legacy_named_provider_lookup_keeps_unique_match_compatible() -> None:
    provider = SimpleNamespace(id=1)
    db_session = MagicMock()
    db_session.scalars.return_value = iter([provider])

    resolved = fetch_existing_llm_provider("Unique Provider", db_session)

    assert resolved is provider


def test_legacy_duplicate_name_does_not_fall_through_to_nameless_type() -> None:
    duplicate_providers = [SimpleNamespace(id=1), SimpleNamespace(id=2)]
    nameless_provider = MagicMock()
    nameless_provider.model_configurations = [
        SimpleNamespace(name="shared-model", is_visible=True)
    ]
    db_session = MagicMock()
    db_session.scalars.side_effect = [
        iter(duplicate_providers),
        iter([nameless_provider]),
    ]
    persona = cast(Persona, SimpleNamespace(default_model_configuration_id=None))

    resolved = _resolve_provider_and_model(
        persona,
        LlmProviderNames.OPENROUTER,
        "shared-model",
        db_session,
    )

    assert resolved is None
    assert db_session.scalars.call_count == 1


@pytest.mark.parametrize(
    "selected_model_configurations",
    [
        [],
        [SimpleNamespace(name="persona-model", is_visible=False)],
    ],
    ids=["missing", "hidden"],
)
def test_provider_only_override_rejects_unavailable_persona_default_model(
    selected_model_configurations: list[SimpleNamespace],
) -> None:
    selected_provider = MagicMock()
    selected_provider.model_configurations = selected_model_configurations
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=17),
    )

    with (
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_legacy_selection",
            return_value=selected_provider,
        ),
        patch(
            "onyx.llm.factory.fetch_model_configuration_by_id",
            return_value=SimpleNamespace(name="persona-model"),
        ),
    ):
        resolved = _resolve_provider_and_model(
            persona,
            "Selected Provider",
            None,
            MagicMock(),
        )

    assert resolved is None


def test_provider_only_override_accepts_visible_persona_default_model() -> None:
    selected_provider = MagicMock()
    selected_provider.model_configurations = [
        SimpleNamespace(name="persona-model", is_visible=True)
    ]
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=17),
    )

    with (
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_legacy_selection",
            return_value=selected_provider,
        ),
        patch(
            "onyx.llm.factory.fetch_model_configuration_by_id",
            return_value=SimpleNamespace(name="persona-model"),
        ),
    ):
        resolved = _resolve_provider_and_model(
            persona,
            "Selected Provider",
            None,
            MagicMock(),
        )

    assert resolved == (selected_provider, "persona-model")


@pytest.mark.parametrize(
    "override_model_configuration",
    [
        None,
        SimpleNamespace(name="override-model", is_visible=False),
    ],
    ids=["missing", "hidden"],
)
def test_model_only_override_rejects_unavailable_model_on_persona_provider(
    override_model_configuration: SimpleNamespace | None,
) -> None:
    provider = MagicMock()
    provider.model_configurations = [
        SimpleNamespace(name="persona-default", is_visible=True),
        *(
            [override_model_configuration]
            if override_model_configuration is not None
            else []
        ),
    ]
    persona_default = SimpleNamespace(
        name="persona-default",
        llm_provider=provider,
    )
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=17),
    )

    with patch(
        "onyx.llm.factory.fetch_model_configuration_by_id",
        return_value=persona_default,
    ):
        resolved = _resolve_provider_and_model(
            persona,
            None,
            "override-model",
            MagicMock(),
        )

    assert resolved is None


def test_provider_type_keeps_same_named_providers_unambiguous() -> None:
    intended_provider = MagicMock()
    intended_provider.model_configurations = [
        SimpleNamespace(name="shared-model", is_visible=True)
    ]
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=None),
    )
    user = SimpleNamespace(id="user-1", role="admin")
    session_context = MagicMock()
    session_context.__enter__.return_value = MagicMock()
    override = LLMOverride.model_validate(
        {
            "model_provider": "Shared Provider",
            "model_provider_type": "vertex_ai",
            "model_version": "shared-model",
        }
    )

    with (
        patch(
            "onyx.llm.factory.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_model_selection",
            return_value=intended_provider,
        ),
        patch("onyx.llm.factory.fetch_user_group_ids", return_value=set()),
        patch("onyx.llm.factory.can_user_access_llm_provider", return_value=True),
        patch(
            "onyx.llm.factory.LLMProviderView.from_model",
            side_effect=lambda provider: provider,
        ),
        patch(
            "onyx.llm.factory.llm_from_provider",
            side_effect=lambda *, llm_provider, **_: llm_provider,
        ),
    ):
        resolved = get_llm_for_persona(
            persona=persona,
            user=cast(User, user),
            llm_override=override,
        )

    assert resolved is intended_provider


def test_type_only_override_executes_unique_nameless_provider() -> None:
    nameless_provider = MagicMock()
    nameless_provider.model_configurations = [
        SimpleNamespace(name="gemini-3.6-flash", is_visible=True)
    ]
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=None),
    )
    user = SimpleNamespace(id="user-1", role="admin")
    session_context = MagicMock()
    session_context.__enter__.return_value = MagicMock()

    with (
        patch(
            "onyx.llm.factory.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_model_selection",
            return_value=nameless_provider,
        ),
        patch("onyx.llm.factory.fetch_user_group_ids", return_value=set()),
        patch("onyx.llm.factory.can_user_access_llm_provider", return_value=True),
        patch(
            "onyx.llm.factory.LLMProviderView.from_model",
            side_effect=lambda provider: provider,
        ),
        patch(
            "onyx.llm.factory.llm_from_provider",
            side_effect=lambda *, llm_provider, **_: llm_provider,
        ),
    ):
        resolved = get_llm_for_persona(
            persona=persona,
            user=cast(User, user),
            llm_override=LLMOverride(
                model_provider_type=LlmProviderNames.VERTEX_AI,
                model_version="gemini-3.6-flash",
            ),
        )

    assert resolved is nameless_provider


def test_provider_id_override_executes_one_of_multiple_nameless_providers() -> None:
    selected_provider = MagicMock()
    selected_provider.id = 8
    selected_provider.name = None
    selected_provider.provider = LlmProviderNames.OPENROUTER
    selected_provider.model_configurations = [
        SimpleNamespace(name="selected/model", is_visible=True)
    ]
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=None),
    )
    user = SimpleNamespace(id="user-1", role="admin")
    session_context = MagicMock()
    session_context.__enter__.return_value = MagicMock()

    with (
        patch(
            "onyx.llm.factory.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            "onyx.llm.factory.fetch_existing_llm_provider_by_id",
            return_value=selected_provider,
            create=True,
        ) as fetch_by_id,
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_model_selection"
        ) as fetch_by_selector,
        patch("onyx.llm.factory.fetch_user_group_ids", return_value=set()),
        patch("onyx.llm.factory.can_user_access_llm_provider", return_value=True),
        patch(
            "onyx.llm.factory.LLMProviderView.from_model",
            side_effect=lambda provider: provider,
        ),
        patch(
            "onyx.llm.factory.llm_from_provider",
            side_effect=lambda *, llm_provider, **_: llm_provider,
        ),
    ):
        resolved = get_llm_for_persona(
            persona=persona,
            user=cast(User, user),
            llm_override=LLMOverride(
                model_provider_id=8,
                model_provider_type=LlmProviderNames.OPENROUTER,
                model_version="selected/model",
            ),
        )

    assert resolved is selected_provider
    fetch_by_id.assert_called_once_with(8, session_context.__enter__.return_value)
    fetch_by_selector.assert_not_called()


def test_provider_id_override_rejects_provider_type_mismatch() -> None:
    selected_provider = MagicMock()
    selected_provider.id = 8
    selected_provider.name = None
    selected_provider.provider = LlmProviderNames.ANTHROPIC
    selected_provider.model_configurations = [
        SimpleNamespace(name="selected/model", is_visible=True)
    ]

    with patch(
        "onyx.llm.factory.fetch_existing_llm_provider_by_id",
        return_value=selected_provider,
        create=True,
    ):
        resolved = _resolve_provider_and_model(
            cast(Persona, SimpleNamespace(default_model_configuration_id=None)),
            None,
            "selected/model",
            MagicMock(),
            provider_type_override=LlmProviderNames.OPENROUTER,
            provider_id_override=8,
        )

    assert resolved is None


def test_deleted_provider_id_fails_closed_without_legacy_name_fallback() -> None:
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=None),
    )
    session_context = MagicMock()
    session_context.__enter__.return_value = MagicMock()

    with (
        patch(
            "onyx.llm.factory.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch(
            "onyx.llm.factory.fetch_existing_llm_provider_by_id",
            return_value=None,
        ),
        patch(
            "onyx.llm.factory.fetch_llm_provider_for_legacy_selection"
        ) as legacy_lookup,
        patch("onyx.llm.factory.get_default_llm") as get_default_llm,
        pytest.raises(OnyxError),
    ):
        get_llm_for_persona(
            persona=persona,
            user=MagicMock(),
            llm_override=LLMOverride(
                model_provider_id=8,
                model_provider_type=LlmProviderNames.OPENROUTER,
                model_version="selected/model",
            ),
        )

    legacy_lookup.assert_not_called()
    get_default_llm.assert_not_called()


def test_synthesized_name_and_type_resolve_unique_nameless_provider() -> None:
    nameless_provider = MagicMock()
    nameless_provider.model_configurations = [
        SimpleNamespace(name="gemini-3.6-flash", is_visible=True)
    ]

    with (
        patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_name_and_type",
            return_value=None,
        ),
        patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_type_nameless",
            return_value=nameless_provider,
        ),
    ):
        resolved = fetch_llm_provider_for_model_selection(
            LlmProviderNames.VERTEX_AI,
            LlmProviderNames.VERTEX_AI,
            "gemini-3.6-flash",
            MagicMock(),
        )

    assert resolved is nameless_provider


def test_explicit_unresolvable_override_does_not_use_default() -> None:
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=None),
    )
    user = MagicMock()
    db_session = MagicMock()
    session_context = MagicMock()
    session_context.__enter__.return_value = db_session

    with (
        patch(
            "onyx.llm.factory.get_session_with_current_tenant",
            return_value=session_context,
        ),
        patch("onyx.llm.factory._resolve_provider_and_model", return_value=None),
        patch("onyx.llm.factory.get_default_llm") as get_default_llm,
        pytest.raises(OnyxError) as exc,
    ):
        get_llm_for_persona(
            persona=persona,
            user=user,
            llm_override=LLMOverride(
                model_provider=LlmProviderNames.VERTEX_AI,
                model_version="missing-model",
            ),
        )

    assert exc.value.error_code is OnyxErrorCode.INVALID_INPUT
    get_default_llm.assert_not_called()


def test_model_selection_without_persona_does_not_use_default() -> None:
    with (
        patch("onyx.llm.factory.get_default_llm") as get_default_llm,
        pytest.raises(OnyxError) as exc,
    ):
        get_llm_for_persona(
            persona=None,
            user=MagicMock(),
            llm_override=LLMOverride(
                model_provider=LlmProviderNames.VERTEX_AI,
                model_version="gemini-3.6-flash",
            ),
        )

    assert exc.value.error_code is OnyxErrorCode.INVALID_INPUT
    get_default_llm.assert_not_called()


def test_model_only_override_without_persona_default_does_not_use_default() -> None:
    persona = cast(
        Persona,
        SimpleNamespace(id=1, default_model_configuration_id=None),
    )

    with (
        patch("onyx.llm.factory.get_default_llm") as get_default_llm,
        pytest.raises(OnyxError) as exc,
    ):
        get_llm_for_persona(
            persona=persona,
            user=MagicMock(),
            llm_override=LLMOverride(model_version="gemini-3.6-flash"),
        )

    assert exc.value.error_code is OnyxErrorCode.INVALID_INPUT
    get_default_llm.assert_not_called()


def _build_provider_view(
    provider: str,
    max_input_tokens: int | None,
) -> LLMProviderView:
    return LLMProviderView(
        id=1,
        name="test-provider",
        provider=provider,
        model_configurations=[
            ModelConfigurationView(
                name="test-model",
                is_visible=True,
                max_input_tokens=max_input_tokens,
                supports_image_input=False,
            )
        ],
        api_key=None,
        api_base="http://localhost:11434",
        api_version=None,
        custom_config=None,
        is_public=True,
        is_auto_mode=False,
        groups=[],
        personas=[],
        deployment_name=None,
    )


def test_get_llm_sets_ollama_num_ctx_model_kwarg() -> None:
    with patch("onyx.llm.factory.LitellmLLM") as mock_litellm_llm:
        get_llm(
            provider=LlmProviderNames.OLLAMA_CHAT,
            model="test-model",
            deployment_name=None,
            max_input_tokens=4096,
            model_kwargs={"num_ctx": 8192},
        )

        kwargs = mock_litellm_llm.call_args.kwargs
        assert kwargs["model_kwargs"] == {"num_ctx": 8192}


def test_get_llm_does_not_set_ollama_num_ctx_for_non_ollama_provider() -> None:
    with patch("onyx.llm.factory.LitellmLLM") as mock_litellm_llm:
        get_llm(
            provider=LlmProviderNames.OPENAI,
            model="gpt-4o-mini",
            deployment_name=None,
            max_input_tokens=4096,
        )

        kwargs = mock_litellm_llm.call_args.kwargs
        assert kwargs["model_kwargs"] == {}


def test_llm_from_provider_passes_configured_ollama_num_ctx() -> None:
    provider = _build_provider_view(
        provider=LlmProviderNames.OLLAMA_CHAT,
        max_input_tokens=16384,
    )

    with patch("onyx.llm.factory.get_llm") as mock_get_llm:
        llm_from_provider(
            model_name="test-model",
            llm_provider=provider,
        )

        kwargs = mock_get_llm.call_args.kwargs
        assert kwargs["max_input_tokens"] == 16384
        assert kwargs["model_kwargs"] == {"num_ctx": 16384}


def test_llm_from_provider_omits_ollama_num_ctx_when_model_context_unknown() -> None:
    provider = _build_provider_view(
        provider=LlmProviderNames.OLLAMA_CHAT,
        max_input_tokens=None,
    )

    with (
        patch(
            "onyx.llm.factory.get_max_input_tokens_from_llm_provider",
            return_value=32000,
        ),
        patch("onyx.llm.factory.get_llm") as mock_get_llm,
    ):
        llm_from_provider(
            model_name="test-model",
            llm_provider=provider,
        )

        kwargs = mock_get_llm.call_args.kwargs
        assert kwargs["max_input_tokens"] == 32000
        assert kwargs["model_kwargs"] == {}


def test_llm_from_provider_never_sets_ollama_num_ctx_for_non_ollama_provider() -> None:
    provider = _build_provider_view(
        provider=LlmProviderNames.OPENAI,
        max_input_tokens=16384,
    )

    with patch("onyx.llm.factory.get_llm") as mock_get_llm:
        llm_from_provider(
            model_name="test-model",
            llm_provider=provider,
        )

        kwargs = mock_get_llm.call_args.kwargs
        assert kwargs["max_input_tokens"] == 16384
        assert kwargs["model_kwargs"] == {}
