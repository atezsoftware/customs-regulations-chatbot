import pytest

from fs_explorer_api.agent import _normalize_indexed_text
from fs_explorer_api.server import (
    _decode_html_entities,
    _server_multi_agent_enabled,
)


def test_decode_html_entities_preserves_turkish_text() -> None:
    value = "Transit s&uuml;resinin a&#351;&#305;lmas&#305; &ldquo;otomatik&rdquo; de&#287;ildir."

    assert (
        _decode_html_entities(value)
        == "Transit süresinin aşılması “otomatik” değildir."
    )


def test_normalize_indexed_text_decodes_entities_before_model_context() -> None:
    value = "G&uuml;mr&uuml;k &amp; transit: a&#351;&#305;lma"

    assert _normalize_indexed_text(value) == "Gümrük & transit: aşılma"


def test_api_server_enables_multi_agent_by_default(monkeypatch) -> None:
    monkeypatch.delenv("FS_EXPLORER_MULTI_AGENT_ENABLED", raising=False)

    assert _server_multi_agent_enabled() is True


def test_api_server_respects_multi_agent_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_ENABLED", "false")

    assert _server_multi_agent_enabled() is False


def test_api_server_rejects_ambiguous_multi_agent_flag(monkeypatch) -> None:
    monkeypatch.setenv("FS_EXPLORER_MULTI_AGENT_ENABLED", "sometimes")

    with pytest.raises(ValueError, match="must be a boolean"):
        _server_multi_agent_enabled()
