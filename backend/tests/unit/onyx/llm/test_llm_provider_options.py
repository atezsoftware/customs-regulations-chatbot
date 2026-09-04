from datetime import datetime, timezone

import pytest

from onyx.llm.well_known_providers.auto_update_models import (
    LLMProviderRecommendation,
    LLMRecommendations,
)
from onyx.llm.well_known_providers.constants import (
    OPENAI_PROVIDER_NAME,
    VERTEXAI_PROVIDER_NAME,
)
from onyx.llm.well_known_providers.llm_provider_options import (
    _load_bundled_recommendations,
    get_vertexai_model_names,
    model_configurations_for_provider,
)
from onyx.llm.well_known_providers.models import SimpleKnownModel


def test_get_visible_models_dedupes_default_and_prefers_display_name() -> None:
    # The default is repeated in additional_visible_models (where it carries a
    # display name); get_visible_models must return it once, with the name.
    recommendations = LLMRecommendations(
        version="test",
        updated_at=datetime.now(timezone.utc),
        providers={
            "anthropic": LLMProviderRecommendation(
                default_model=SimpleKnownModel(name="claude-opus-4-8"),
                additional_visible_models=[
                    SimpleKnownModel(
                        name="claude-opus-4-8", display_name="Claude Opus 4.8"
                    ),
                    SimpleKnownModel(
                        name="claude-sonnet-4-6", display_name="Claude Sonnet 4.6"
                    ),
                ],
            )
        },
    )

    visible = recommendations.get_visible_models("anthropic")

    assert [(m.name, m.display_name) for m in visible] == [
        ("claude-opus-4-8", "Claude Opus 4.8"),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
    ]


def _build_recommendations(
    provider_name: str, visible_model_names: list[str]
) -> LLMRecommendations:
    return LLMRecommendations(
        version="test",
        updated_at=datetime.now(timezone.utc),
        providers={
            provider_name: LLMProviderRecommendation(
                default_model=SimpleKnownModel(name=visible_model_names[0]),
                additional_visible_models=[
                    SimpleKnownModel(name=model_name)
                    for model_name in visible_model_names[1:]
                ],
            )
        },
    )


def test_model_configurations_vertex_are_sorted_by_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.fetch_models_for_provider",
        lambda _provider_name: ["zeta-model", "alpha-model", "Beta-model"],
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.get_max_input_tokens",
        lambda _model_name, _provider_name: None,
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.model_supports_image_input",
        lambda _model_name, _provider_name: False,
    )

    recommendations = _build_recommendations(
        VERTEXAI_PROVIDER_NAME, ["gamma-model", "alpha-model"]
    )

    model_configurations = model_configurations_for_provider(
        VERTEXAI_PROVIDER_NAME, recommendations
    )

    assert [model.name for model in model_configurations] == [
        "alpha-model",
        "Beta-model",
        "gamma-model",
        "gemini-3.1-flash-lite",
        "gemini-3.8-flash",
        "zeta-model",
    ]
    assert [model.is_visible for model in model_configurations] == [
        True,
        False,
        True,
        True,
        True,
        False,
    ]


def test_vertex_catalog_pins_gemini_31_flash_lite() -> None:
    assert "gemini-3.1-flash-lite" in get_vertexai_model_names()


def test_vertex_catalog_pins_gemini_38_flash() -> None:
    assert "gemini-3.8-flash" in get_vertexai_model_names()


def test_vertex_recommendations_expose_gemini_31_flash_lite_in_chat() -> None:
    visible_models = _load_bundled_recommendations().get_visible_models(
        VERTEXAI_PROVIDER_NAME
    )

    assert ("gemini-3.1-flash-lite", "Gemini 3.1 Flash Lite") in {
        (model.name, model.display_name) for model in visible_models
    }


def test_vertex_recommendations_expose_gemini_38_flash_in_chat() -> None:
    visible_models = _load_bundled_recommendations().get_visible_models(
        VERTEXAI_PROVIDER_NAME
    )

    assert ("gemini-3.8-flash", "Gemini 3.8 Flash") in {
        (model.name, model.display_name) for model in visible_models
    }


def test_vertex_flash_lite_stays_visible_when_remote_recommendations_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.fetch_models_for_provider",
        lambda _provider_name: ["gemini-3.1-flash-lite"],
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.get_max_input_tokens",
        lambda _model_name, _provider_name: None,
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.model_supports_image_input",
        lambda _model_name, _provider_name: False,
    )
    lagging_recommendations = _build_recommendations(
        VERTEXAI_PROVIDER_NAME, ["gemini-3.1-pro-preview"]
    )

    configurations = model_configurations_for_provider(
        VERTEXAI_PROVIDER_NAME, lagging_recommendations
    )

    flash_lite = next(
        model for model in configurations if model.name == "gemini-3.1-flash-lite"
    )
    assert flash_lite.is_visible is True
    assert flash_lite.display_name == "Gemini 3.1 Flash Lite"


def test_vertex_38_flash_stays_visible_when_remote_recommendations_lag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.fetch_models_for_provider",
        lambda _provider_name: [],
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.get_max_input_tokens",
        lambda _model_name, _provider_name: None,
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.model_supports_image_input",
        lambda _model_name, _provider_name: False,
    )
    lagging_recommendations = _build_recommendations(
        VERTEXAI_PROVIDER_NAME, ["gemini-3.1-pro-preview"]
    )

    configurations = model_configurations_for_provider(
        VERTEXAI_PROVIDER_NAME, lagging_recommendations
    )

    flash = next(model for model in configurations if model.name == "gemini-3.8-flash")
    assert flash.is_visible is True
    assert flash.display_name == "Gemini 3.8 Flash"


def test_model_configurations_carry_display_name_and_dedupe_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The default is repeated in additional_visible_models (where it carries a
    # display name); the result must be deduped and carry that display name.
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.fetch_models_for_provider",
        lambda _provider_name: [],
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.get_max_input_tokens",
        lambda _model_name, _provider_name: None,
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.model_supports_image_input",
        lambda _model_name, _provider_name: False,
    )

    recommendations = LLMRecommendations(
        version="test",
        updated_at=datetime.now(timezone.utc),
        providers={
            "anthropic": LLMProviderRecommendation(
                default_model=SimpleKnownModel(name="claude-opus-4-8"),
                additional_visible_models=[
                    SimpleKnownModel(
                        name="claude-opus-4-8", display_name="Claude Opus 4.8"
                    ),
                    SimpleKnownModel(
                        name="claude-sonnet-4-6", display_name="Claude Sonnet 4.6"
                    ),
                ],
            )
        },
    )

    model_configurations = model_configurations_for_provider(
        "anthropic", recommendations
    )

    assert [m.name for m in model_configurations] == [
        "claude-opus-4-8",
        "claude-sonnet-4-6",
    ]
    by_name = {m.name: m for m in model_configurations}
    assert by_name["claude-opus-4-8"].display_name == "Claude Opus 4.8"
    assert all(m.is_visible for m in model_configurations)
    # Only the config's default model is flagged as the recommended default.
    assert by_name["claude-opus-4-8"].is_recommended_default is True
    assert by_name["claude-sonnet-4-6"].is_recommended_default is False


def test_model_configurations_non_vertex_preserve_provider_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.fetch_models_for_provider",
        lambda _provider_name: ["model-b", "model-a"],
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.get_max_input_tokens",
        lambda _model_name, _provider_name: None,
    )
    monkeypatch.setattr(
        "onyx.llm.well_known_providers.llm_provider_options.model_supports_image_input",
        lambda _model_name, _provider_name: False,
    )

    recommendations = _build_recommendations(
        OPENAI_PROVIDER_NAME, ["model-c", "model-a"]
    )

    model_configurations = model_configurations_for_provider(
        OPENAI_PROVIDER_NAME, recommendations
    )

    assert [model.name for model in model_configurations] == [
        "model-b",
        "model-a",
        "model-c",
    ]
