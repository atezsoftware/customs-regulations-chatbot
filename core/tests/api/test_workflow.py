"""Tests for workflow.py's per-run agent construction and isolation."""

import os

from unittest.mock import patch

from fs_explorer_api.workflow import get_run_agent, new_workflow


@patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"})
class TestNewWorkflowAgentConfig:
    """`new_workflow()` must bind model/temperature/hook per call, not via
    module-level globals — see its docstring for the concurrent-request
    race that design avoids.
    """

    def test_agent_is_constructed_with_given_config(self) -> None:
        _workflow, resource_manager = new_workflow(model="model-a", temperature=0.3)

        agent = get_run_agent(resource_manager)

        assert agent._llm.model == "model-a"
        assert agent._llm.temperature == 0.3

    def test_two_calls_do_not_leak_config_into_each_other(self) -> None:
        # Simulates two requests whose new_workflow() calls interleave:
        # under the old set_agent_llm_config()-then-get_agent() design, the
        # second call here would have overwritten the module globals the
        # first request's agent hadn't been constructed from yet. Now each
        # call constructs and registers its own agent immediately, with no
        # shared mutable state in between.
        _workflow_a, resource_manager_a = new_workflow(model="model-a")
        _workflow_b, resource_manager_b = new_workflow(model="model-b")

        agent_a = get_run_agent(resource_manager_a)
        agent_b = get_run_agent(resource_manager_b)

        assert agent_a._llm.model == "model-a"
        assert agent_b._llm.model == "model-b"
        assert agent_a is not agent_b

    def test_default_config_when_omitted(self) -> None:
        from fs_explorer_api.llm.gemini import DEFAULT_GEMINI_MODEL

        _workflow, resource_manager = new_workflow()

        agent = get_run_agent(resource_manager)

        assert agent._llm.model == DEFAULT_GEMINI_MODEL
