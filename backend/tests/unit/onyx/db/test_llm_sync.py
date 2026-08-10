"""Tests for LLM provider model sync functionality."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from onyx.db.enums import LLMModelFlowType
from onyx.db.llm import sync_auto_mode_models, sync_model_configurations
from onyx.llm.constants import LlmProviderNames
from onyx.server.manage.llm.models import SyncModelEntry


def _make_existing_model(name: str, flow_types: list[LLMModelFlowType]) -> MagicMock:
    model = MagicMock()
    model.id = 42
    model.name = name
    model.llm_model_flow_types = flow_types
    return model


class TestSyncModelConfigurations:
    """Tests for sync_model_configurations function."""

    def test_inserts_new_models(self) -> None:
        """Test that new models are inserted."""
        # Mock the provider with no existing models
        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = []

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            models = [
                SyncModelEntry(
                    name="gpt-4",
                    display_name="GPT-4",
                    max_input_tokens=128000,
                    supports_image_input=True,
                ),
                SyncModelEntry(
                    name="gpt-4o",
                    display_name="GPT-4o",
                    max_input_tokens=128000,
                    supports_image_input=True,
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 2  # Two new models
            assert (
                mock_session.execute.call_count == 2 * 3
            )  # 2 models * (model insert + chat insert + vision insert)
            mock_session.commit.assert_called_once()

    def test_openrouter_auto_mode_inserts_new_models_as_visible(self) -> None:
        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.provider = LlmProviderNames.OPENROUTER
        mock_provider.is_auto_mode = True
        mock_provider.model_configurations = []
        mock_session = MagicMock()

        with (
            patch(
                "onyx.db.llm.fetch_existing_llm_provider_by_id",
                return_value=mock_provider,
            ),
            patch(
                "onyx.db.llm.insert_new_model_configuration__no_commit"
            ) as insert_model,
        ):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=[
                    SyncModelEntry(
                        name="anthropic/claude-sonnet-4.5",
                        display_name="Claude Sonnet 4.5",
                    )
                ],
            )

        assert result == 1
        assert insert_model.call_args.kwargs["is_visible"] is True
        mock_session.commit.assert_called_once()

    def test_openrouter_auto_mode_reveals_existing_fetched_model(self) -> None:
        existing = _make_existing_model(
            "anthropic/claude-sonnet-4.5", [LLMModelFlowType.CHAT]
        )
        existing.is_visible = False
        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.provider = LlmProviderNames.OPENROUTER
        mock_provider.is_auto_mode = True
        mock_provider.model_configurations = [existing]
        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id",
            return_value=mock_provider,
        ):
            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=[
                    SyncModelEntry(
                        name="anthropic/claude-sonnet-4.5",
                        display_name="Claude Sonnet 4.5",
                    )
                ],
            )

        assert result == 0
        assert existing.is_visible is True
        mock_session.commit.assert_called_once()

    def test_openrouter_current_discovery_hides_stale_existing_model(self) -> None:
        current = _make_existing_model(
            "anthropic/claude-sonnet-4.5", [LLMModelFlowType.CHAT]
        )
        current.is_visible = False
        stale = _make_existing_model("stale/model", [LLMModelFlowType.CHAT])
        stale.is_visible = True
        provider = MagicMock()
        provider.id = 1
        provider.provider = LlmProviderNames.OPENROUTER
        provider.is_auto_mode = True
        provider.model_configurations = [current, stale]
        db_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id",
            return_value=provider,
        ):
            result = sync_model_configurations(
                db_session=db_session,
                provider_id=1,
                models=[
                    SyncModelEntry(
                        name="anthropic/claude-sonnet-4.5",
                        display_name="Claude Sonnet 4.5",
                    )
                ],
            )

        assert result == 0
        assert current.is_visible is True
        assert stale.is_visible is False
        db_session.commit.assert_called_once()

    def test_openrouter_discovery_replaces_stale_active_default_atomically(
        self,
    ) -> None:
        current = _make_existing_model(
            "anthropic/claude-sonnet-4.5", [LLMModelFlowType.CHAT]
        )
        current.is_visible = False
        stale_default = _make_existing_model("retired/default", [LLMModelFlowType.CHAT])
        stale_default.is_visible = True
        stale_default.llm_provider_id = 1
        provider = MagicMock()
        provider.id = 1
        provider.provider = LlmProviderNames.OPENROUTER
        provider.is_auto_mode = True
        provider.model_configurations = [current, stale_default]
        db_session = MagicMock()
        events: list[str] = []
        db_session.commit.side_effect = lambda: events.append("commit")

        def record_default_update(**kwargs: object) -> None:
            assert kwargs["provider_id"] == 1
            assert kwargs["model"] == "anthropic/claude-sonnet-4.5"
            assert kwargs["flow_type"] is LLMModelFlowType.CHAT
            events.append("default")

        with (
            patch(
                "onyx.db.llm.fetch_existing_llm_provider_by_id",
                return_value=provider,
            ),
            patch(
                "onyx.db.llm.fetch_default_llm_model",
                return_value=stale_default,
            ),
            patch(
                "onyx.db.llm._update_default_model__no_commit",
                side_effect=record_default_update,
            ),
        ):
            result = sync_model_configurations(
                db_session=db_session,
                provider_id=1,
                models=[
                    SyncModelEntry(
                        name="anthropic/claude-sonnet-4.5",
                        display_name="Claude Sonnet 4.5",
                    )
                ],
            )

        assert result == 0
        assert current.is_visible is True
        assert stale_default.is_visible is False
        assert events == ["default", "commit"]

    def test_openrouter_discovery_fails_before_hiding_only_active_default(
        self,
    ) -> None:
        stale_default = _make_existing_model("retired/default", [LLMModelFlowType.CHAT])
        stale_default.is_visible = True
        stale_default.llm_provider_id = 1
        provider = MagicMock()
        provider.id = 1
        provider.provider = LlmProviderNames.OPENROUTER
        provider.is_auto_mode = True
        provider.model_configurations = [stale_default]
        db_session = MagicMock()

        with (
            patch(
                "onyx.db.llm.fetch_existing_llm_provider_by_id",
                return_value=provider,
            ),
            patch(
                "onyx.db.llm.fetch_default_llm_model",
                return_value=stale_default,
            ),
            pytest.raises(ValueError, match="no currently available replacement"),
        ):
            sync_model_configurations(
                db_session=db_session,
                provider_id=1,
                models=[],
            )

        assert stale_default.is_visible is True
        db_session.commit.assert_not_called()

    def test_skips_existing_models(self) -> None:
        """Existing models with up-to-date flows are left untouched."""
        # Existing model already has the capabilities the source reports.
        mock_existing_model = _make_existing_model(
            "gpt-4", [LLMModelFlowType.CHAT, LLMModelFlowType.VISION]
        )

        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = [mock_existing_model]

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            models = [
                SyncModelEntry(
                    name="gpt-4",  # Existing - should be skipped
                    display_name="GPT-4",
                    max_input_tokens=128000,
                    supports_image_input=True,
                ),
                SyncModelEntry(
                    name="gpt-4o",  # New - should be inserted
                    display_name="GPT-4o",
                    max_input_tokens=128000,
                    supports_image_input=True,
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 1  # Only one new model
            assert mock_session.execute.call_count == 3

    def test_no_commit_when_no_new_models(self) -> None:
        """Test that commit is not called when nothing new or upgraded."""
        mock_existing_model = _make_existing_model(
            "gpt-4", [LLMModelFlowType.CHAT, LLMModelFlowType.VISION]
        )

        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = [mock_existing_model]

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            models = [
                SyncModelEntry(
                    name="gpt-4",  # Already exists
                    display_name="GPT-4",
                    max_input_tokens=128000,
                    supports_image_input=True,
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 0
            mock_session.commit.assert_not_called()

    def test_raises_on_missing_provider(self) -> None:
        """Test that ValueError is raised when provider not found."""
        mock_session = MagicMock()

        with patch("onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=None):
            with pytest.raises(ValueError, match="not found"):
                sync_model_configurations(
                    db_session=mock_session,
                    provider_id=999,
                    models=[SyncModelEntry(name="model", display_name="Model")],
                )

    def test_inserts_reasoning_flow_when_supports_reasoning(self) -> None:
        """Test that a REASONING flow row is created when supports_reasoning=True."""
        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = []

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            models = [
                SyncModelEntry(
                    name="deepseek-r1",
                    display_name="DeepSeek R1",
                    max_input_tokens=65536,
                    supports_image_input=True,
                    supports_reasoning=True,
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 1
            # 1 model insert + 3 flow inserts (CHAT + VISION + REASONING)
            assert mock_session.execute.call_count == 4
            mock_session.commit.assert_called_once()

    def test_handles_missing_optional_fields(self) -> None:
        """Test that optional fields default correctly."""
        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = []

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            # Model with only required fields (max_input_tokens and supports_image_input default)
            models = [
                SyncModelEntry(
                    name="model-1",
                    display_name="Model 1",
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 1
            # Verify execute was called with correct defaults
            call_args = mock_session.execute.call_args
            assert call_args is not None

    def test_upgrades_existing_model_vision_flow(self) -> None:
        """Existing model gains a VISION flow when the source newly reports it.

        Repairs rows synced before the source exposed vision (e.g. a Bifrost
        model added before vision detection resolved correctly). Returns 0 new
        models but commits the added flow.
        """
        mock_existing_model = _make_existing_model("gemini", [LLMModelFlowType.CHAT])

        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = [mock_existing_model]

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            models = [
                SyncModelEntry(
                    name="gemini",
                    display_name="Gemini",
                    supports_image_input=True,
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 0  # No new models, only an upgraded flow
            assert mock_session.execute.call_count == 1  # One VISION flow insert
            mock_session.commit.assert_called_once()

    def test_does_not_remove_flows(self) -> None:
        """Capability flags are only added, never removed: a model that already
        has VISION keeps it even if the source omits it this fetch."""
        mock_existing_model = _make_existing_model(
            "gemini", [LLMModelFlowType.CHAT, LLMModelFlowType.VISION]
        )

        mock_provider = MagicMock()
        mock_provider.id = 1
        mock_provider.model_configurations = [mock_existing_model]

        mock_session = MagicMock()

        with patch(
            "onyx.db.llm.fetch_existing_llm_provider_by_id", return_value=mock_provider
        ):
            models = [
                SyncModelEntry(
                    name="gemini",
                    display_name="Gemini",
                    supports_image_input=False,
                ),
            ]

            result = sync_model_configurations(
                db_session=mock_session,
                provider_id=1,
                models=models,
            )

            assert result == 0
            mock_session.execute.assert_not_called()
            mock_session.commit.assert_not_called()


def test_auto_mode_recommendations_do_not_reveal_stale_openrouter_model() -> None:
    existing = _make_existing_model(
        "anthropic/claude-sonnet-4.5", [LLMModelFlowType.CHAT]
    )
    existing.is_visible = False
    existing.display_name = "Claude Sonnet 4.5"
    provider = MagicMock()
    provider.id = 1
    provider.provider = LlmProviderNames.OPENROUTER
    provider.is_auto_mode = True
    db_session = MagicMock()
    db_session.scalars.return_value.all.return_value = [existing]
    recommendations = MagicMock()
    recommendations.get_default_model.return_value = None

    recommendations.get_visible_models.return_value = [
        SyncModelEntry(
            name="anthropic/claude-sonnet-4.5",
            display_name="Claude Sonnet 4.5",
        )
    ]

    changes = sync_auto_mode_models(db_session, provider, recommendations)

    assert changes == 0
    assert existing.is_visible is False
    db_session.commit.assert_called_once()


def test_auto_mode_does_not_make_stale_openrouter_model_the_default() -> None:
    stale = _make_existing_model("stale/model", [LLMModelFlowType.CHAT])
    stale.is_visible = False
    stale.display_name = "Stale Model"
    provider = MagicMock()
    provider.id = 1
    provider.provider = LlmProviderNames.OPENROUTER
    provider.is_auto_mode = True
    db_session = MagicMock()
    db_session.scalars.return_value.all.return_value = [stale]
    recommendations = MagicMock()
    recommendations.get_visible_models.return_value = [
        SyncModelEntry(name="stale/model", display_name="Stale Model")
    ]
    recommendations.get_default_model.return_value = SyncModelEntry(
        name="stale/model", display_name="Stale Model"
    )
    current_default = SimpleNamespace(llm_provider_id=1, name="current/model")

    with (
        patch("onyx.db.llm.fetch_default_llm_model", return_value=current_default),
        patch("onyx.db.llm._update_default_model__no_commit") as update_default,
    ):
        changes = sync_auto_mode_models(db_session, provider, recommendations)

    assert changes == 0
    update_default.assert_not_called()
