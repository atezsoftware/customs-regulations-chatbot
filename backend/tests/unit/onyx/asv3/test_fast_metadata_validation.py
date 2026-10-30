"""Root validation preserves exact transport bytes, token order and failures."""

import copy
import json
from collections.abc import Callable
from typing import cast
from unittest.mock import patch

import pytest
from pydantic import JsonValue

import onyx.asv3.cheap_metadata_transport as codec
from onyx.asv3.cropped_handoff_transport import (
    canonical_cropped_handoff_payload,
    expand_cropped_handoff_transport,
)
from onyx.asv3.cropped_transport import serialize_cropped_payload
from onyx.asv3.source_metadata_transport import share_source_metadata
from tests.unit.onyx.asv3.test_cheap_metadata_transport import tokens
from tests.unit.onyx.asv3.test_cropped_context_metadata_codec import full_payload

_EXTRA = (
    "research_tasks",
    "repair_research_observations",
    "cropped_completion",
    "session_research",
)


def _payload() -> dict[str, JsonValue]:
    result = full_payload()
    result["research_tasks"] = [
        {
            "task_id": f"task-{index}",
            "status": "completed",
            "outcome": {
                "data": {
                    "conditions": [
                        {
                            "text": "A AND B; exception C and later procedure D. İĞŞ 東京\n\t  ",
                            "values": [None, True, False, 1, 1.0, "1", [], {}],
                            "literal_fields": {"layout_ref": [0, ["literal"]]},
                            "other_literal": {"value_ref": {"value_ref": True}},
                        }
                        for _ in range(3)
                    ],
                    "next_position": None,
                    "absence_proven": False,
                }
            },
        }
        for index in range(3)
    ]
    return result


def _input(form: str) -> dict[str, JsonValue]:
    result = _payload()
    if form == "shared":
        catalogue = cast(
            list[dict[str, JsonValue]], result["original_metadata_catalogue"]
        )
        rows, sources = share_source_metadata(catalogue)
        result["original_metadata_catalogue"] = cast(JsonValue, rows)
        result["original_source_metadata"] = sources
    elif form in {"v1", "v2"}:
        result = codec.encode_cheap_metadata(
            result, tokens, extra_fields=() if form == "v1" else _EXTRA
        )
    return result


@pytest.mark.parametrize("form", ["flat", "shared", "v1", "v2"])
@pytest.mark.parametrize("serializer", ["cheap", "cropped"])
def test_root_validation_keeps_exact_bytes_counter_order_and_full_roundtrip(
    form: str, serializer: str
) -> None:
    payload = _input(form)
    before = copy.deepcopy(payload)
    outputs: list[str] = []
    counter_calls: list[list[str]] = []
    for enabled in (False, True):
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            return tokens(text)

        if serializer == "cropped":
            output = serialize_cropped_payload(
                payload, counter, validate_roots_once=enabled
            )
        else:
            output = codec.serialize_cheap_payload(
                payload,
                counter,
                extra_fields=_EXTRA,
                validate_roots_once=enabled,
            )
        outputs.append(output)
        counter_calls.append(calls)
        assert payload == before
    assert outputs[0] == outputs[1]
    assert counter_calls[0] == counter_calls[1]
    expected = canonical_cropped_handoff_payload(before)
    assert expand_cropped_handoff_transport(json.loads(outputs[1])) == expected


@pytest.mark.parametrize("opt_in", [False, None, 1, "true"])
def test_only_boolean_true_avoids_duplicate_subtree_serialization(
    opt_in: object,
) -> None:
    payload = _payload()
    serialized_counts: list[int] = []
    outputs: list[dict[str, JsonValue]] = []
    original_text = codec._text
    for enabled in (False, opt_in, True):
        containers: list[object] = []

        def measured(value: object, *, compact: bool = False) -> str:
            if isinstance(value, (dict, list)):
                containers.append(value)
            return original_text(value, compact=compact)

        with patch.object(codec, "_text", measured):
            outputs.append(
                codec.encode_cheap_metadata(
                    payload,
                    tokens,
                    extra_fields=_EXTRA,
                    validate_roots_once=cast(bool, enabled),
                )
            )
        serialized_counts.append(len(containers))
    assert outputs[0] == outputs[1] == outputs[2]
    assert serialized_counts[0] == serialized_counts[1]
    assert serialized_counts[2] < serialized_counts[0] // 2


@pytest.mark.parametrize(
    "invalid",
    [
        {"nested": object()},
        {"nested": {1: "nonstring key"}},
        {"nested": [float("nan")]},
        {"nested": [float("inf")]},
        {"nested": ("tuple",)},
        {"nested": [1 << 15_000]},
    ],
)
def test_invalid_nested_data_keeps_exact_exception_type_and_message(
    invalid: object,
) -> None:
    payload = _payload()
    payload["research_tasks"] = cast(JsonValue, invalid)
    errors: list[tuple[type[BaseException], str]] = []
    for enabled in (False, True):
        with pytest.raises((TypeError, ValueError)) as caught:
            serialize_cropped_payload(payload, tokens, validate_roots_once=enabled)
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("failure_at", [1, 7, 21])
def test_token_counter_exception_occurs_after_the_same_exact_inputs(
    failure_at: int,
) -> None:
    class CounterFailure(Exception):
        pass

    calls_by_mode: list[list[str]] = []
    for enabled in (False, True):
        calls: list[str] = []

        def counter(text: str) -> int:
            calls.append(text)
            if len(calls) == failure_at:
                raise CounterFailure("counter unavailable")
            return tokens(text)

        with pytest.raises(CounterFailure, match="counter unavailable"):
            serialize_cropped_payload(_payload(), counter, validate_roots_once=enabled)
        calls_by_mode.append(calls)
    assert calls_by_mode[0] == calls_by_mode[1]


@pytest.mark.parametrize("damage", ["bool_version", "unused_value", "invalid_ref"])
def test_opt_in_keeps_decoder_rejections_for_existing_encoded_inputs(
    damage: str,
) -> None:
    payload = _input("v2")
    if damage == "bool_version":
        header = cast(dict[str, JsonValue], payload["original_metadata_codec"])
        header["version"] = True
    elif damage == "unused_value":
        values = cast(list[JsonValue], payload["original_metadata_values"])
        values.append("Unused hidden literal")
    else:
        payload["session_research"] = {"value_ref": True}
    errors = []
    for enabled in (False, True):
        with pytest.raises(ValueError) as caught:
            codec.serialize_cheap_payload(
                payload, tokens, extra_fields=_EXTRA, validate_roots_once=enabled
            )
        errors.append((type(caught.value), str(caught.value)))
    assert errors[0] == errors[1]


def test_cropped_opt_in_reaches_every_codec_candidate() -> None:
    payload = _payload()
    original = codec._encode_cheap_metadata
    enabled: list[bool] = []

    def capture(
        payload: dict[str, JsonValue],
        counter: Callable[[str], int],
        *,
        extra_fields: tuple[str, ...] = (),
        identity_refs: bool = False,
        small_containers: bool = False,
        validate_roots_once: bool = False,
    ) -> dict[str, JsonValue]:
        enabled.append(validate_roots_once)
        return original(
            payload,
            counter,
            extra_fields=extra_fields,
            identity_refs=identity_refs,
            small_containers=small_containers,
            validate_roots_once=validate_roots_once,
        )

    with patch.object(codec, "_encode_cheap_metadata", capture):
        serialize_cropped_payload(payload, tokens, validate_roots_once=True)
    assert len(enabled) >= 3 and all(value is True for value in enabled)
