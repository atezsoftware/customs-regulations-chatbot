from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from cryptography.exceptions import InvalidTag

from onyx.tracing.answer_graph import (
    _serialize_and_encrypt,
    _span_contents,
    load_graph_part,
    redact_graph_value,
)
from onyx.tracing.framework.span_data import GenerationSpanData
from onyx.utils.encryption import EncryptionError


def test_payload_is_encrypted_bound_to_node_and_redacted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_GRAPH_ENCRYPTION_KEY", "test-key-" * 6)
    run_id = uuid4()
    original = {
        "question": "Where is Authorization: Bearer abc123?",
        "headers": {"x-api-key": "sensitive-key"},
        "url": "https://example.test/file?X-Amz-Signature=abc",
    }
    ciphertext = _serialize_and_encrypt(
        original, run_id=run_id, node_id="a", part="input"
    )
    assert ciphertext is not None
    assert b"sensitive-key" not in ciphertext
    result = load_graph_part(ciphertext, run_id=run_id, node_id="a", part="input")
    assert result == {
        "question": "Where is Authorization: Bearer [REDACTED]?",
        "headers": {"x-api-key": "[REDACTED]"},
        "url": "https://example.test/file?X-Amz-Signature=[REDACTED]",
    }
    with pytest.raises(InvalidTag):
        load_graph_part(ciphertext, run_id=run_id, node_id="b", part="input")


def test_missing_key_never_writes_plaintext(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("ANSWER_GRAPH_ENCRYPTION_KEY", raising=False)
    monkeypatch.delenv("ENCRYPTION_KEY_SECRET", raising=False)
    with pytest.raises(EncryptionError):
        _serialize_and_encrypt(
            {"question": "sensitive"}, run_id=uuid4(), node_id="a", part="input"
        )


def test_redaction_covers_nested_secrets_and_private_key() -> None:
    assert redact_graph_value(
        {
            "nested": {
                "cookie": "session=abc",
                "text": "Basic dXNlcjpwYXNz",
            },
            "pem": "-----BEGIN PRIVATE KEY-----\nabc\n-----END PRIVATE KEY-----",
        }
    ) == {
        "nested": {
            "cookie": "[REDACTED]",
            "text": "Basic [REDACTED]",
        },
        "pem": "[REDACTED]",
    }


def test_composition_roles_are_public_without_prompts_or_other_model_configuration() -> (
    None
):
    span = MagicMock()
    span.span_data = GenerationSpanData(
        input=[{"role": "user", "content": "PRIVATE_PROMPT"}],
        model="gpt-6.1-sol",
        model_config={
            "flow": "asv3_final",
            "model_provider": "openai",
            "api_key": "PRIVATE_KEY",
            "custom_setting": "PRIVATE_CONFIG",
            "asv3_composition_profile": "flash_sol_v1",
            "asv3_composition_version": "1",
            "asv3_composition_role": "author",
            "asv3_composition_provider_id": "7",
        },
    )
    _, _, _, attributes, operation = _span_contents(span)
    assert operation == "asv3_final"
    assert attributes["asv3_composition_profile"] == "flash_sol_v1"
    assert attributes["asv3_composition_role"] == "author"
    assert attributes["asv3_composition_provider_id"] == 7
    assert attributes["asv3_composition_version"] == 1
    assert "api_key" not in attributes and "custom_setting" not in attributes
    assert "PRIVATE_" not in str(attributes)


@pytest.mark.parametrize("role", [None, [], {}, "unknown", "author\nsecret"])
def test_invalid_composition_metadata_is_not_promoted(role: object) -> None:
    span = MagicMock()
    span.span_data = GenerationSpanData(
        model="gpt-6.1-sol",
        model_config={
            "asv3_composition_profile": "sol_reference_v1",
            "asv3_composition_version": "1",
            "asv3_composition_role": role,
            "asv3_composition_provider_id": "7",
        },
    )
    assert not any(
        key.startswith("asv3_composition_") for key in _span_contents(span)[3]
    )


def test_large_payload_round_trips_through_encrypted_database_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANSWER_GRAPH_ENCRYPTION_KEY", "test-key-" * 6)
    run_id = uuid4()
    ciphertext = _serialize_and_encrypt(
        {"text": "a" * 70000}, run_id=run_id, node_id="large", part="output"
    )
    assert ciphertext is not None
    assert b"a" * 100 not in ciphertext
    assert load_graph_part(
        ciphertext, run_id=run_id, node_id="large", part="output"
    ) == {"text": "a" * 70000}
