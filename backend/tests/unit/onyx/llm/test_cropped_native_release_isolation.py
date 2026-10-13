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
def test_cropped_zero_markers_leave_original_experimental_wire_unchanged(
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
    assert "prompt_cache_breakpoint" not in repr(selected)
    assert "prompt_cache_breakpoint" in repr(before)
    content = copy.deepcopy(selected)
    if explicit_cache:
        assert content.pop("prompt_cache_options") == {"mode": "explicit", "ttl": "30m"}
    assert content == without_breakpoints(before)
    assert response.usage is not None
    assert response.usage.service_tier == "priority"
    assert span.span_data.request_params is not None
    metadata = span.span_data.request_params
    assert (
        metadata["asv3_native_response_terminal"]
        == metadata["asv3_exp_cropped_response_terminal"]
    )
    assert metadata["service_tier"] == "priority"
    assert (
        metadata["asv3_exp_cropped_response_terminal"][
            "requested_tool_cache_breakpoints"
        ]
        == 0
    )
