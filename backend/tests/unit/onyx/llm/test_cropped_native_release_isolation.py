"""Cropped transport preserves the already deployed Experimental cache contract."""

import copy

import pytest

from onyx.tracing.framework.create import generation_span, trace
from tests.unit.onyx.llm.test_asv3_native_cache_breakpoints import (
    prompt,
    without_breakpoints,
)
from tests.unit.onyx.llm.test_asv3_response_terminal_capture import (
    REPORTED_USAGE,
    events,
    invoke,
    model,
)
from tests.unit.onyx.llm.test_tool_argument_stream import EventStream


@pytest.mark.parametrize("explicit_cache", [False, True])
def test_cropped_history_cache_preserves_original_and_zero_write_contracts(
    explicit_cache: bool,
) -> None:
    ordinary = model(capture=False, service_tier="flex")
    experimental = model(service_tier="flex")
    cropped = ordinary.with_cropped_response_terminal_capture(
        explicit_cache=explicit_cache
    )
    raw = events("completed", copy.deepcopy(REPORTED_USAGE))
    raw[-1]["response"]["service_tier"] = "priority"
    messages = prompt(3)
    _, before = invoke(experimental, EventStream(copy.deepcopy(raw)), prompt=messages)
    with (
        trace("Cropped isolated cache transport"),
        generation_span(model="gpt-6.1-sol") as span,
    ):
        response, selected = invoke(
            cropped, EventStream(copy.deepcopy(raw)), prompt=messages
        )
    _, after = invoke(
        model(service_tier="flex"), EventStream(copy.deepcopy(raw)), prompt=messages
    )
    assert not ordinary._cropped_response_terminal_capture
    assert ordinary._cropped_prompt_cache is None
    assert before == after
    assert "prompt_cache_breakpoint" in repr(before)
    content = copy.deepcopy(selected)
    if explicit_cache:
        assert "prompt_cache_breakpoint" not in repr(selected)
        assert content.pop("prompt_cache_options") == {"mode": "explicit", "ttl": "30m"}
        assert content == without_breakpoints(before)
    else:
        assert content == before
    assert response.usage is not None
    assert response.usage.service_tier == "priority"
    assert span.span_data.request_params is not None
    metadata = span.span_data.request_params
    assert (
        metadata["asv3_native_response_terminal"]
        == metadata["asv3_exp_cropped_response_terminal"]
    )
    assert metadata["service_tier"] == "priority"
    assert metadata["asv3_exp_cropped_response_terminal"][
        "requested_tool_cache_breakpoints"
    ] == (0 if explicit_cache else 3)


def test_cropped_growing_history_keeps_exact_earlier_cache_endpoints() -> None:
    cropped = model(capture=False).with_cropped_response_terminal_capture()
    _, previous = invoke(
        cropped, EventStream(events("completed", REPORTED_USAGE)), prompt=prompt(2)
    )
    _, following = invoke(
        cropped, EventStream(events("completed", REPORTED_USAGE)), prompt=prompt(3)
    )
    assert following["input"][: len(previous["input"]) - 1] == previous["input"][:-1]
    assert following["input"][-1] != previous["input"][-1]
    assert "prompt_cache_breakpoint" not in repr(following["input"][-1])
    assert "prompt_cache_options" not in following


def test_cropped_binding_does_not_override_cheap_boundary_policy() -> None:
    cheap = model(capture=False).with_asv3_exp_cheap_cache()
    both = (
        model(capture=False)
        .with_asv3_exp_cheap_cache()
        .with_cropped_response_terminal_capture()
    )
    messages = prompt(6)
    _, expected = invoke(
        cheap, EventStream(events("completed", REPORTED_USAGE)), prompt=messages
    )
    _, actual = invoke(
        both, EventStream(events("completed", REPORTED_USAGE)), prompt=messages
    )
    assert actual == expected
    assert repr(actual).count("prompt_cache_breakpoint") == 1
