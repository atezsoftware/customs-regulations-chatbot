"""Admit LC calls against settled totals, including outstanding reservations."""

from onyx.legal_composite.budget import WorkflowBudget


class SettledUsageBudget(WorkflowBudget):
    def _usage_overrun_blocks_calls(self) -> bool:
        return (
            self._input > self.policy.max_input_tokens
            or self._output > self.policy.max_output_tokens
            or self._cost > self.policy.max_cost_usd
        )
