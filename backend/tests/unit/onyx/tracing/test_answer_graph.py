from uuid import uuid4

import pytest
from cryptography.exceptions import InvalidTag

from onyx.tracing.answer_graph import (
    _serialize_and_encrypt,
    load_graph_part,
    redact_graph_value,
)
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
