from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue

from onyx.asv3.models import RunStopped
from onyx.legal_composite.models import WorkflowPolicy

ResearchStopReason = Literal[
    "host_research_deadline", "host_call_timeout", "provider_timeout"
]
_RESEARCH_STOP_REASONS = frozenset(
    {"host_research_deadline", "host_call_timeout", "provider_timeout"}
)


class ResearchPhaseClosed(RunStopped):
    """Research ended without revoking the reserved finalization allocation."""

    def __init__(self, reason: ResearchStopReason) -> None:
        if reason not in _RESEARCH_STOP_REASONS:
            raise ValueError("Unknown research closure reason")
        self.reason = reason
        super().__init__(
            f"Research phase closed ({reason}); finalization allocation retained"
        )


class CallReservation(BaseModel):
    model_config = ConfigDict(frozen=True)

    call_id: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    input_price_per_million: float
    output_price_per_million: float
    timeout_seconds: float


class WorkflowBudget:
    """Atomically allocate estimated spend, retaining a draft and a review."""

    def __init__(
        self,
        policy: WorkflowPolicy,
        clock: Callable[[], float] = time.monotonic,
        *,
        deadline: float | None = None,
    ) -> None:
        if deadline is not None and (
            math.isnan(deadline)
            or deadline == -math.inf
            or (deadline == math.inf and math.isfinite(policy.timeout_seconds))
        ):
            raise ValueError("Invalid workflow deadline")
        self.policy = policy
        self._clock = clock
        latest_deadline = clock() + policy.timeout_seconds
        self.deadline = (
            latest_deadline if deadline is None else min(deadline, latest_deadline)
        )
        self._started = (
            self.deadline - policy.timeout_seconds
            if math.isfinite(self.deadline)
            else clock()
        )
        self._lock = threading.RLock()
        self._calls = 0
        self._input = 0
        self._output = 0
        self._cost = 0.0
        self._pending_final_calls = 2
        self._final_input = 0
        self._final_output = 0
        self._final_cost = 0.0
        self._reservations: dict[str, CallReservation] = {}
        self._settled: set[str] = set()
        self._usage_overrun = False
        self._stop_reason: str | None = None
        self._research_stop_reason: ResearchStopReason | None = None
        self._selection_reserve_seconds = 0.0

    def retain_selection_time(self, seconds: float) -> None:
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError("Invalid selection time reserve")
        with self._lock:
            if self._calls:
                raise ValueError("Selection time must be retained before generations")
            self._selection_reserve_seconds = seconds

    def begin_selection(self) -> None:
        with self._lock:
            self._selection_reserve_seconds = 0.0

    def configure_finalization(
        self,
        input_tokens_per_call: int,
        output_tokens_per_call: int,
        cost_per_call_usd: float,
    ) -> None:
        if (
            input_tokens_per_call < 0
            or output_tokens_per_call < 1
            or not math.isfinite(cost_per_call_usd)
            or cost_per_call_usd < 0
        ):
            raise ValueError("Invalid finalization reservation")
        with self._lock:
            if self._calls:
                raise ValueError("Finalization must be reserved before invoking models")
            if (
                input_tokens_per_call * 2 > self.policy.max_input_tokens
                or output_tokens_per_call * 2 > self.policy.max_output_tokens
                or cost_per_call_usd * 2 > self.policy.max_cost_usd
            ):
                raise RunStopped(
                    "The selected model cannot fit the finalization budget"
                )
            self._final_input = input_tokens_per_call
            self._final_output = output_tokens_per_call
            self._final_cost = cost_per_call_usd

    def remaining_seconds(self, finalizing: bool = False) -> float:
        retained = (
            0
            if finalizing
            else self.policy.finalization_reserve_seconds
            + self._selection_reserve_seconds
        )
        return max(0.0, self.deadline - self._clock() - retained)

    def check_active(self, finalizing: bool = False) -> None:
        self.check_response_active(finalizing)
        if self.remaining_seconds(finalizing) < 3:
            raise RunStopped("Workflow deadline reached; finalization time retained")

    def check_response_active(self, finalizing: bool = False) -> None:
        """Completed calls need a live deadline, without a new-call time reserve."""
        with self._lock:
            if self._stop_reason:
                raise RunStopped(self._stop_reason)
            if not finalizing and self._research_stop_reason is not None:
                raise ResearchPhaseClosed(self._research_stop_reason)
        if self.remaining_seconds(finalizing) <= 0:
            raise RunStopped("Workflow deadline reached; finalization time retained")

    def research_available(self) -> bool:
        with self._lock:
            pending = self._pending_final_calls
            return (
                not self._usage_overrun_blocks_calls()
                and self._stop_reason is None
                and self._research_stop_reason is None
                and self.remaining_seconds() >= 3
                and self._calls < self.policy.max_model_calls - pending
                and self._input
                < self.policy.max_input_tokens - pending * self._final_input
                and self._output
                < self.policy.max_output_tokens - pending * self._final_output
                and self._cost < self.policy.max_cost_usd - pending * self._final_cost
            )

    def _usage_overrun_blocks_calls(self) -> bool:
        # Estimates are accounting hints when no business spending ceiling is configured.
        return self._usage_overrun and math.isfinite(self.policy.max_cost_usd)

    def affordable_input_tokens(
        self,
        output_tokens: int,
        input_price_per_million: float,
        output_price_per_million: float,
        finalizing: bool = False,
    ) -> int:
        """Preview a call's input limit; request remains the atomic admission fence."""
        if output_tokens < 1 or any(
            not math.isfinite(rate) or rate < 0
            for rate in (input_price_per_million, output_price_per_million)
        ):
            raise ValueError("Invalid generation allocation")
        with self._lock:
            self.check_active(finalizing)
            if self._usage_overrun_blocks_calls():
                raise RunStopped(
                    "Provider usage exceeded the estimate; no further spend authorized"
                )
            pending = (
                max(0, self._pending_final_calls - 1)
                if finalizing
                else self._pending_final_calls
            )
            available_input = (
                self.policy.max_input_tokens - pending * self._final_input - self._input
            )
            available_cost = (
                self.policy.max_cost_usd
                - pending * self._final_cost
                - self._cost
                - output_tokens * output_price_per_million / 1_000_000
            )
            if (
                self._calls + 1 > self.policy.max_model_calls - pending
                or self._output + output_tokens
                > self.policy.max_output_tokens - pending * self._final_output
                or available_input < 0
                or available_cost < 0
            ):
                raise RunStopped(
                    "Workflow model budget exhausted; finalization allocation retained"
                )
            if (
                input_price_per_million == 0
                or available_cost
                >= available_input * input_price_per_million / 1_000_000
            ):
                return available_input
            return min(
                available_input,
                math.floor(available_cost * 1_000_000 / input_price_per_million),
            )

    def request(
        self,
        input_tokens: int,
        output_tokens: int,
        input_price_per_million: float,
        output_price_per_million: float,
        finalizing: bool = False,
    ) -> CallReservation:
        if (
            input_tokens < 0
            or output_tokens < 1
            or any(
                not math.isfinite(rate) or rate < 0
                for rate in (input_price_per_million, output_price_per_million)
            )
        ):
            raise ValueError("Invalid generation allocation")
        cost = (
            input_tokens * input_price_per_million
            + output_tokens * output_price_per_million
        ) / 1_000_000
        with self._lock:
            self.check_active(finalizing)
            if self._usage_overrun_blocks_calls():
                raise RunStopped(
                    "Provider usage exceeded the estimate; no further spend authorized"
                )
            pending = (
                max(0, self._pending_final_calls - 1)
                if finalizing
                else self._pending_final_calls
            )
            if (
                self._calls + 1 > self.policy.max_model_calls - pending
                or self._input + input_tokens
                > self.policy.max_input_tokens - pending * self._final_input
                or self._output + output_tokens
                > self.policy.max_output_tokens - pending * self._final_output
                or self._cost + cost
                > self.policy.max_cost_usd - pending * self._final_cost
            ):
                raise RunStopped(
                    "Workflow model budget exhausted; finalization allocation retained"
                )
            reservation = CallReservation(
                call_id=str(uuid4()),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=cost,
                input_price_per_million=input_price_per_million,
                output_price_per_million=output_price_per_million,
                timeout_seconds=min(
                    self.policy.max_call_seconds, self.remaining_seconds(finalizing)
                ),
            )
            self._calls += 1
            self._input += input_tokens
            self._output += output_tokens
            self._cost += cost
            if finalizing:
                self._pending_final_calls = max(0, self._pending_final_calls - 1)
            self._reservations[reservation.call_id] = reservation
            return reservation

    def stop(self, reason: str) -> None:
        with self._lock:
            self._stop_reason = reason

    def close_research(self, reason: ResearchStopReason) -> None:
        if reason not in _RESEARCH_STOP_REASONS:
            raise ValueError("Unknown research closure reason")
        with self._lock:
            self._research_stop_reason = self._research_stop_reason or reason

    def settle(
        self,
        reservation: CallReservation,
        actual_input_tokens: int,
        actual_output_tokens: int,
    ) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (actual_input_tokens, actual_output_tokens)
        ):
            raise ValueError("Invalid provider usage")
        with self._lock:
            if (
                self._reservations.get(reservation.call_id) != reservation
                or reservation.call_id in self._settled
            ):
                raise ValueError("Unknown or already settled generation")
            actual_cost = (
                actual_input_tokens * reservation.input_price_per_million
                + actual_output_tokens * reservation.output_price_per_million
            ) / 1_000_000
            self._input += actual_input_tokens - reservation.input_tokens
            self._output += actual_output_tokens - reservation.output_tokens
            self._cost = max(
                0.0, self._cost + actual_cost - reservation.estimated_cost_usd
            )
            self._settled.add(reservation.call_id)
            self._usage_overrun |= (
                actual_input_tokens > reservation.input_tokens
                or actual_output_tokens > reservation.output_tokens
                or self._cost > self.policy.max_cost_usd
            )
            if (
                self._input > self.policy.max_input_tokens
                or self._output > self.policy.max_output_tokens
                or self._cost > self.policy.max_cost_usd
            ):
                self._stop_reason = self._stop_reason or (
                    "Provider actual usage exceeded workflow capacity; no further generations allowed"
                )

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            snapshot: dict[str, JsonValue] = {
                "model_calls": self._calls,
                "input_tokens": self._input,
                "output_tokens": self._output,
                "estimated_cost_usd": round(self._cost, 8),
                "elapsed_seconds": round(self._clock() - self._started, 3),
                "pending_final_calls": self._pending_final_calls,
                "usage_overrun": self._usage_overrun,
                "estimate_overrun_blocks_calls": self._usage_overrun_blocks_calls(),
                "stop_reason": self._stop_reason,
                "unsettled_calls": len(self._reservations) - len(self._settled),
                "cost_basis": "uncached upper-rate estimate; failed calls retain their allocation",
            }
            if self._research_stop_reason is not None:
                snapshot["research_stop_reason"] = self._research_stop_reason
            return snapshot
