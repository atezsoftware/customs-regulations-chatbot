from unittest.mock import MagicMock, patch

import pytest

from onyx.db.models import SearchSettings
from onyx.indexing.contextual_settings import (
    ContextualIndexingConfigurationError,
    effective_contextual_rag_enabled,
    require_contextual_rag_llm,
)


def _settings(enabled: bool) -> SearchSettings:
    return SearchSettings(enable_contextual_rag=enabled)


@pytest.mark.parametrize(
    ("row_value", "env_value", "expected"),
    [(False, True, True), (True, False, True), (False, False, False)],
)
def test_effective_contextual_setting(
    row_value: bool, env_value: bool, expected: bool
) -> None:
    assert (
        effective_contextual_rag_enabled(
            _settings(row_value),
            env_enabled=env_value,
            multitenant=False,
        )
        is expected
    )


def test_multitenant_always_disables_contextual_indexing() -> None:
    assert not effective_contextual_rag_enabled(
        _settings(True), env_enabled=True, multitenant=True
    )


def test_required_contextual_model_is_returned() -> None:
    llm = MagicMock()
    with patch(
        "onyx.indexing.contextual_settings.get_contextual_rag_llm_for_search_settings",
        return_value=llm,
    ):
        assert (
            require_contextual_rag_llm(
                _settings(True), env_enabled=False, multitenant=False
            )
            is llm
        )


def test_required_contextual_model_missing_raises_typed_error() -> None:
    with (
        patch(
            "onyx.indexing.contextual_settings.get_contextual_rag_llm_for_search_settings",
            return_value=None,
        ),
        pytest.raises(
            ContextualIndexingConfigurationError,
            match="Select a contextualization model before indexing",
        ),
    ):
        require_contextual_rag_llm(
            _settings(True), env_enabled=False, multitenant=False
        )


def test_disabled_contextual_indexing_does_not_resolve_model() -> None:
    with patch(
        "onyx.indexing.contextual_settings.get_contextual_rag_llm_for_search_settings"
    ) as get_llm:
        assert (
            require_contextual_rag_llm(
                _settings(False), env_enabled=False, multitenant=False
            )
            is None
        )
    get_llm.assert_not_called()
