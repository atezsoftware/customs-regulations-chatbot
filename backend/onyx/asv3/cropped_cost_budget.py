"""Shared prospective generation budget for the isolated Cropped workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from dataclasses import dataclass
from typing import Literal, cast
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.models import RunContext, RunStopped
from onyx.llm.model_response import Usage


class CroppedResearchBudgetReached(RunStopped):
    """Research must yield its remaining allowance to final writing."""


class CroppedRepairBudgetReached(CroppedResearchBudgetReached):
    """Paid repair is closed; supported first finalization remains available."""


CroppedCostPhase = Literal["research", "repair", "finalization"]


def cropped_cache_write_partition_verified(terminal: object) -> bool:
    """Require the actual provider partition; normalized zero is not a witness."""
    if not isinstance(terminal, dict):
        return False
    diagnostic = cast(dict[str, object], terminal)
    if diagnostic.get("usage_status") != "provider_reported":
        return False
    usage = diagnostic.get("provider_reported_usage")
    if not isinstance(usage, dict):
        return False
    raw = cast(dict[str, object], usage)
    details = raw.get("input_tokens_details")
    if not isinstance(details, dict):
        return False
    partitions = cast(dict[str, object], details)
    incoming = raw.get("input_tokens")
    cached = partitions.get("cached_tokens", 0)
    written = partitions.get("cache_write_tokens")
    if any(type(value) is not int for value in (incoming, cached, written)):
        return False
    incoming, cached, written = (
        cast(int, value) for value in (incoming, cached, written)
    )
    return min(incoming, cached, written) >= 0 and cached + written <= incoming


@dataclass(frozen=True)
class CroppedCostTicket:
    call_id: str
    owner: str
    research: bool
    input_upper_tokens: int
    max_tokens: int
    reserved_usd: float
    input_rate: float
    output_rate: float
    cache_rate: float
    cache_write_rate: float | None
    bounded: bool
    price_available: bool
    output_bound_known: bool
    cache_writes_disabled: bool = False
    phase: CroppedCostPhase = "research"
    service_tier: str | None = None

    @property
    def reservation_upper_verified(self) -> bool:
        return (
            self.price_available
            and self.output_bound_known
            and (self.cache_writes_disabled or self.cache_write_rate is not None)
        )


class _CostCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    call_id: str = Field(min_length=1)
    owner: str
    research: bool
    known_cost_usd: float | None = Field(ge=0)
    retained_reserve_usd: float | None = Field(ge=0)
    input_upper_tokens: int = Field(ge=0)
    max_tokens: int = Field(gt=0)
    provider_rejected_before_generation: bool
    budget_enforced: bool
    price_available: bool
    output_bound_known: bool
    reservation_upper_verified: bool
    input_rate: float | None = Field(ge=0)
    output_rate: float | None = Field(ge=0)
    cache_read_rate: float | None = Field(ge=0)
    cache_write_rate: float | None = Field(ge=0)
    cache_writes_disabled: bool
    cache_write_contract_violated: bool
    input_reservation_rate: float | None = Field(ge=0)
    reserved_usd: float = Field(ge=0)
    usage: Usage | None
    phase: CroppedCostPhase
    service_tier: str | None
    cache_write_partition_verified: bool
    cache_write_exposure_usd: float = Field(ge=0)


class _ExternalCharge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)

    charge_id: str = Field(min_length=1)
    known_cost_usd: float = Field(ge=0)
    uncertain_upper_usd: float = Field(ge=0)


def _interrupted_reservation(ticket: CroppedCostTicket) -> dict[str, JsonValue]:
    return {
        "call_id": ticket.call_id,
        "owner": ticket.owner,
        "research": ticket.research,
        "phase": ticket.phase,
        "service_tier": ticket.service_tier,
        "cache_write_partition_verified": False,
        "cache_write_exposure_usd": 0.0,
        "known_cost_usd": None,
        "retained_reserve_usd": ticket.reserved_usd
        if ticket.reservation_upper_verified
        else None,
        "input_upper_tokens": ticket.input_upper_tokens,
        "max_tokens": ticket.max_tokens,
        "provider_rejected_before_generation": False,
        "budget_enforced": ticket.bounded,
        "price_available": ticket.price_available,
        "output_bound_known": ticket.output_bound_known,
        "reservation_upper_verified": ticket.reservation_upper_verified,
        "input_rate": ticket.input_rate if ticket.price_available else None,
        "output_rate": ticket.output_rate if ticket.price_available else None,
        "cache_read_rate": ticket.cache_rate if ticket.price_available else None,
        "cache_write_rate": ticket.cache_write_rate,
        "cache_writes_disabled": ticket.cache_writes_disabled,
        "cache_write_contract_violated": False,
        "input_reservation_rate": max(
            ticket.input_rate,
            ticket.cache_rate,
            0.0 if ticket.cache_writes_disabled else ticket.cache_write_rate or 0.0,
        )
        if ticket.price_available
        else None,
        "reserved_usd": ticket.reserved_usd,
        "usage": None,
    }


def _usage_consistent(
    usage: Usage, *, incoming_upper: int, output_upper: int, bounded: bool
) -> bool:
    return (
        usage.prompt_tokens >= 0
        and usage.cache_read_input_tokens >= 0
        and usage.cache_creation_input_tokens >= 0
        and usage.cache_read_input_tokens + usage.cache_creation_input_tokens
        <= usage.prompt_tokens
        and usage.completion_tokens >= 0
        and (not bounded or usage.completion_tokens <= output_upper)
        and usage.prompt_tokens <= incoming_upper
    )


def _usage_cost(
    usage: Usage,
    *,
    incoming: float,
    outgoing: float,
    cached: float,
    written: float | None,
) -> float | None:
    if usage.cache_creation_input_tokens and written is None:
        return None
    cost = (
        (
            usage.prompt_tokens
            - usage.cache_read_input_tokens
            - usage.cache_creation_input_tokens
        )
        * incoming
        + usage.cache_read_input_tokens * cached
        + usage.cache_creation_input_tokens * (written or 0.0)
        + usage.completion_tokens * outgoing
    )
    return (
        max(cost, usage.cost)
        if usage.cost is not None and math.isfinite(usage.cost) and usage.cost >= 0
        else cost
    )


def _validate_cost_call(row: dict[str, JsonValue]) -> _CostCall:
    call = _CostCall.model_validate(row)
    rates = (call.input_rate, call.output_rate, call.cache_read_rate)
    if call.price_available != all(rate is not None for rate in rates):
        raise ValueError("Cropped cost checkpoint pricing changed")
    if not call.price_available and any(
        rate is not None for rate in (*rates, call.cache_write_rate)
    ):
        raise ValueError("Cropped unpriced checkpoint contains assumed prices")
    if call.reservation_upper_verified != (
        call.price_available
        and call.output_bound_known
        and (call.cache_writes_disabled or call.cache_write_rate is not None)
    ):
        raise ValueError("Cropped cache-write reservation proof changed")
    if call.cache_write_contract_violated != (
        call.cache_writes_disabled
        and call.usage is not None
        and call.usage.cache_creation_input_tokens > 0
    ):
        raise ValueError("Cropped disabled cache-write contract changed")
    if call.price_available:
        input_reservation_rate = max(
            cast(float, call.input_rate),
            cast(float, call.cache_read_rate),
            0.0 if call.cache_writes_disabled else call.cache_write_rate or 0.0,
        )
        if call.input_reservation_rate != input_reservation_rate or not math.isclose(
            call.reserved_usd,
            call.input_upper_tokens * input_reservation_rate
            + call.max_tokens * cast(float, call.output_rate),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Cropped declared input reservation changed")
    elif call.input_reservation_rate is not None:
        raise ValueError("Cropped unpriced input reservation contains an assumed price")
    if call.known_cost_usd is not None:
        usage = call.usage
        if usage is None or not _usage_consistent(
            usage,
            incoming_upper=call.input_upper_tokens,
            output_upper=call.max_tokens,
            bounded=call.budget_enforced,
        ):
            raise ValueError("Cropped measured checkpoint usage changed")
        if call.price_available:
            expected = _usage_cost(
                usage,
                incoming=cast(float, call.input_rate),
                outgoing=cast(float, call.output_rate),
                cached=cast(float, call.cache_read_rate),
                written=call.cache_write_rate,
            )
        else:
            expected = usage.cost if usage.cache_creation_input_tokens == 0 else None
        if (
            expected is None
            or not math.isfinite(expected)
            or not math.isclose(
                call.known_cost_usd, expected, rel_tol=1e-12, abs_tol=1e-12
            )
        ):
            raise ValueError("Cropped measured checkpoint cache partition changed")
    exposure = (
        max(0.0, (call.cache_write_rate or 0.0) - (call.input_rate or 0.0))
        * (
            call.usage.prompt_tokens
            - call.usage.cache_read_input_tokens
            - call.usage.cache_creation_input_tokens
        )
        if call.usage is not None
        and not call.cache_write_partition_verified
        and not call.cache_writes_disabled
        and call.known_cost_usd is not None
        else 0.0
    )
    if not math.isclose(
        call.cache_write_exposure_usd, exposure, rel_tol=1e-12, abs_tol=1e-12
    ):
        raise ValueError("Cropped cache-write exposure changed")
    return call


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _rates(
    model: str, provider: str, incoming: int, service_tier: str | None = None
) -> tuple[float, float, float, float | None]:
    from onyx.asv3.cropped_pricing import model_pricing_info, token_rate

    info = model_pricing_info(model, provider)
    incoming_rate = token_rate(info, "input_cost_per_token", incoming, service_tier)
    outgoing_rate = token_rate(info, "output_cost_per_token", incoming, service_tier)
    cache_rate = token_rate(info, "cache_read_input_token_cost", incoming, service_tier)
    write_rate = token_rate(
        info, "cache_creation_input_token_cost", incoming, service_tier
    )
    if incoming_rate is None or outgoing_rate is None:
        raise RunStopped("Cropped selected-model registry pricing is unavailable")
    if cache_rate is None:
        cache_rate = incoming_rate
    suffix = f"_{service_tier}" if service_tier in {"flex", "priority"} else ""
    write_fields = [
        value
        for key, value in info.items()
        if key == "cache_creation_input_token_cost" + suffix
        or re.fullmatch(
            r"cache_creation_input_token_cost_above_\d+k_tokens" + suffix, key
        )
    ]
    if write_rate is None and any(value is not None for value in write_fields):
        raise RunStopped("Cropped selected-model cache-write pricing is unavailable")
    return incoming_rate, outgoing_rate, cache_rate, write_rate


class CroppedCostBudget:
    """Share emergency phase limits without treating the normal cost target as a cap."""

    def __init__(
        self,
        context: RunContext,
        request: str,
        *,
        research_usd: float = 10.0,
        total_llm_usd: float = 1.50,
        repair_usd: float = 10.0,
    ) -> None:
        if (
            any(
                isinstance(value, bool) or not math.isfinite(value) or value <= 0
                for value in (research_usd, total_llm_usd, repair_usd)
            )
            or repair_usd > research_usd
        ):
            raise ValueError("Invalid Cropped cost phase policy")
        self.context = context
        self.run_id = context.run_id
        self.scope_hash = _hash(context.scope)
        self.request_hash = _hash(request)
        self.research_usd = research_usd
        self.total_llm_usd = total_llm_usd
        self.repair_usd = repair_usd
        self._lock = threading.RLock()
        self._tickets: dict[str, CroppedCostTicket] = {}
        self._calls: list[dict[str, JsonValue]] = []
        self._interrupted_reservations: list[dict[str, JsonValue]] = []
        self._spent = 0.0
        self._uncertain = 0.0
        self._research_spent = 0.0
        self._finish_reason: str | None = None
        self._bounded = False
        self._unpriced: set[str] = set()
        self._external: dict[str, _ExternalCharge] = {}

    def check_context(self, context: RunContext) -> None:
        if context.run_id != self.run_id or _hash(context.scope) != self.scope_hash:
            raise ValueError("Cropped cost run or scope changed")

    def prepare(
        self,
        model_name: str,
        provider: str,
        input_tokens: int,
        requested_max_tokens: int | None,
        *,
        research: bool,
        owner: str,
        enforce: bool = True,
        cache_writes_disabled: bool = False,
        phase: CroppedCostPhase | None = None,
        service_tier: str | None = None,
    ) -> CroppedCostTicket:
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("Invalid prospective input count")
        self.check_context(self.context)
        if type(cache_writes_disabled) is not bool:
            raise ValueError("Invalid cache-write transport contract")
        phase = ("research" if research else "finalization") if phase is None else phase
        if phase not in {"research", "repair", "finalization"}:
            raise ValueError("Invalid Cropped cost phase")
        # The separate 240k gate leaves space for this framing upper estimate.
        upper = math.ceil(input_tokens * 1.10) + 4096
        if upper > 272_000:
            raise RunStopped("Cropped prospective provider input exceeds 272k")
        try:
            resolved = (
                _rates(model_name, provider, upper)
                if service_tier is None
                else _rates(model_name, provider, upper, service_tier)
            )
            input_rate, output_rate, cache_rate, *write_rates = resolved
            cache_write_rate = write_rates[0] if write_rates else None
            price_available = True
        except RunStopped as error:
            if phase != "finalization" and (enforce or self._bounded):
                exception = (
                    CroppedRepairBudgetReached
                    if phase == "repair"
                    else CroppedResearchBudgetReached
                )
                raise exception(
                    "Cropped selected pricing is unavailable; retain the unknown cost and write from recorded facts"
                ) from error
            input_rate = output_rate = cache_rate = 0.0
            cache_write_rate = None
            price_available = False
        requested = requested_max_tokens or (16384 if research else 32768)
        output_bound_known = requested_max_tokens is not None
        if not output_bound_known:
            import litellm

            try:
                info = litellm.get_model_info(
                    model=model_name, custom_llm_provider=provider
                )
                capacity = info.get("max_output_tokens")
                if type(capacity) is int and capacity > 0:
                    requested = capacity
                    output_bound_known = True
            except Exception:
                pass
        if type(requested) is not int or requested <= 0:
            raise ValueError("Invalid prospective output count")
        with self._lock:
            if self._finish_reason == "provider_violated_disabled_cache_writes":
                raise RunStopped("Cropped provider violated disabled cache writes")
            if phase != "finalization" and (
                self._finish_reason == "research_generation_allowance_reached"
                or not self._bound_known()
            ):
                exception = (
                    CroppedRepairBudgetReached
                    if phase == "repair"
                    else CroppedResearchBudgetReached
                )
                raise exception(
                    "Cropped research is closed or prior charges have no verified bound; write from recorded facts"
                )
            bounded = self._bounded or enforce
            if bounded:
                self._bounded = True
                if phase != "finalization" and (
                    not price_available
                    or not output_bound_known
                    or (cache_write_rate is None and not cache_writes_disabled)
                    or self._unpriced
                    or any(
                        not ticket.price_available
                        or not ticket.output_bound_known
                        or (
                            ticket.cache_write_rate is None
                            and not ticket.cache_writes_disabled
                        )
                        for ticket in self._tickets.values()
                    )
                ):
                    exception = (
                        CroppedRepairBudgetReached
                        if phase == "repair"
                        else CroppedResearchBudgetReached
                    )
                    raise exception(
                        "Cropped cannot enforce a budget with unknown prior charges"
                    )
            reserved = sum(ticket.reserved_usd for ticket in self._tickets.values())
            limit = self.repair_usd if phase == "repair" else self.research_usd
            available = (
                limit
                - self._spent
                - self._uncertain
                - reserved
                - sum(
                    row.known_cost_usd + row.uncertain_upper_usd
                    for row in self._external.values()
                )
            )
            input_cost = upper * max(
                input_rate,
                cache_rate,
                0.0 if cache_writes_disabled else cache_write_rate or 0.0,
            )
            affordable = (
                int(max(0, available - input_cost) / output_rate)
                if output_rate
                else requested
            )
            allowed = requested
            if phase != "finalization" and price_available:
                allowed = min(allowed, affordable)
            minimum = min(requested, 2048 if phase == "research" else 8192)
            if allowed < minimum:
                if phase == "repair":
                    raise CroppedRepairBudgetReached(
                        "Cropped repair cost limit reached; retain gaps and supported parts"
                    )
                self._finish_reason = "research_generation_allowance_reached"
                raise CroppedResearchBudgetReached(
                    "Cropped research cost limit reached; retain originals and write"
                )
            ticket = CroppedCostTicket(
                str(uuid4()),
                owner,
                research,
                upper,
                allowed,
                input_cost + allowed * output_rate,
                input_rate,
                output_rate,
                cache_rate,
                cache_write_rate,
                bounded,
                price_available,
                output_bound_known,
                cache_writes_disabled,
                phase,
                service_tier,
            )
            self._tickets[ticket.call_id] = ticket
            return ticket

    def complete(
        self,
        ticket: CroppedCostTicket,
        usage: Usage | None,
        error: BaseException | None = None,
        *,
        cache_write_partition_verified: bool = True,
    ) -> None:
        if type(cache_write_partition_verified) is not bool:
            raise ValueError("Invalid cache-write usage proof")
        with self._lock:
            if self._tickets.get(ticket.call_id) != ticket:
                raise ValueError(
                    "Cropped generation reservation changed or was completed twice"
                )
            self._tickets.pop(ticket.call_id)
            status = getattr(error, "status_code", None)
            rejected = error is not None and status in {400, 401, 403, 404, 422, 429}
            cost: float | None = None
            usage_consistent = usage is not None and _usage_consistent(
                usage,
                incoming_upper=ticket.input_upper_tokens,
                output_upper=ticket.max_tokens,
                bounded=ticket.bounded,
            )
            cache_write_contract_violated = (
                ticket.cache_writes_disabled
                and usage is not None
                and usage.cache_creation_input_tokens > 0
            )
            if (
                ticket.price_available
                and usage is not None
                and usage_consistent
                and (
                    usage.cache_creation_input_tokens == 0
                    or ticket.cache_write_rate is not None
                )
            ):
                cost = _usage_cost(
                    usage,
                    incoming=ticket.input_rate,
                    outgoing=ticket.output_rate,
                    cached=ticket.cache_rate,
                    written=ticket.cache_write_rate,
                )
            elif (
                not ticket.price_available
                and usage is not None
                and usage_consistent
                and usage.cache_creation_input_tokens == 0
                and usage.cost is not None
                and math.isfinite(usage.cost)
                and usage.cost >= 0
            ):
                cost = usage.cost
            if cost is not None:
                self._spent += cost
                if ticket.research:
                    self._research_spent += cost
            elif not rejected:
                if ticket.reservation_upper_verified:
                    self._uncertain += ticket.reserved_usd
                else:
                    self._unpriced.add(ticket.call_id)
                if self._bounded:
                    self._finish_reason = "unpriced_or_inconsistent_provider_usage"
            if cache_write_contract_violated and cost is None:
                self._unpriced.add(ticket.call_id)
            write_exposure = (
                max(0.0, (ticket.cache_write_rate or 0.0) - ticket.input_rate)
                * (
                    usage.prompt_tokens
                    - usage.cache_read_input_tokens
                    - usage.cache_creation_input_tokens
                )
                if cost is not None
                and usage is not None
                and not cache_write_partition_verified
                and not ticket.cache_writes_disabled
                else 0.0
            )
            if (
                cost is not None
                and not cache_write_partition_verified
                and not ticket.cache_writes_disabled
                and ticket.cache_write_rate is None
            ):
                self._unpriced.add(ticket.call_id)
                self._finish_reason = "unpriced_cache_write_exposure"
            self._uncertain += write_exposure
            self._calls.append(
                {
                    "call_id": ticket.call_id,
                    "owner": ticket.owner,
                    "research": ticket.research,
                    "phase": ticket.phase,
                    "service_tier": ticket.service_tier,
                    "cache_write_partition_verified": cache_write_partition_verified,
                    "cache_write_exposure_usd": write_exposure,
                    "known_cost_usd": cost,
                    "retained_reserve_usd": (
                        ticket.reserved_usd
                        if ticket.reservation_upper_verified
                        else None
                    )
                    if cost is None and not rejected
                    else 0,
                    "input_upper_tokens": ticket.input_upper_tokens,
                    "max_tokens": ticket.max_tokens,
                    "provider_rejected_before_generation": rejected,
                    "budget_enforced": ticket.bounded,
                    "price_available": ticket.price_available,
                    "output_bound_known": ticket.output_bound_known,
                    "reservation_upper_verified": ticket.reservation_upper_verified,
                    "input_rate": ticket.input_rate if ticket.price_available else None,
                    "output_rate": ticket.output_rate
                    if ticket.price_available
                    else None,
                    "cache_read_rate": ticket.cache_rate
                    if ticket.price_available
                    else None,
                    "cache_write_rate": ticket.cache_write_rate,
                    "cache_writes_disabled": ticket.cache_writes_disabled,
                    "cache_write_contract_violated": cache_write_contract_violated,
                    "input_reservation_rate": max(
                        ticket.input_rate,
                        ticket.cache_rate,
                        0.0
                        if ticket.cache_writes_disabled
                        else ticket.cache_write_rate or 0.0,
                    )
                    if ticket.price_available
                    else None,
                    "reserved_usd": ticket.reserved_usd,
                    "usage": usage.model_dump(mode="json")
                    if usage is not None
                    else None,
                }
            )
            if cache_write_contract_violated:
                self._finish_reason = "provider_violated_disabled_cache_writes"
                raise RunStopped("Cropped provider violated disabled cache writes")
            if not self.research_allowed:
                self._finish_reason = (
                    self._finish_reason or "research_generation_allowance_reached"
                )

    def _cost_bound(self) -> float:
        return (
            self._spent
            + self._uncertain
            + sum(ticket.reserved_usd for ticket in self._tickets.values())
            + sum(
                row.known_cost_usd + row.uncertain_upper_usd
                for row in self._external.values()
            )
        )

    def _bound_known(self) -> bool:
        return not self._unpriced and all(
            ticket.reservation_upper_verified for ticket in self._tickets.values()
        )

    @property
    def repair_allowed(self) -> bool:
        with self._lock:
            return self._bound_known() and self._cost_bound() < self.repair_usd

    @property
    def research_allowed(self) -> bool:
        with self._lock:
            return (
                self._finish_reason != "research_generation_allowance_reached"
                and self._bound_known()
                and self._cost_bound() < self.research_usd
            )

    def record_external_charge(
        self,
        charge_id: str,
        *,
        known_cost_usd: float,
        uncertain_upper_usd: float,
        context: RunContext,
    ) -> None:
        """Add one observed ancillary charge, preserving its separately declared uncertainty."""
        self.check_context(context)
        charge = _ExternalCharge(
            charge_id=charge_id,
            known_cost_usd=known_cost_usd,
            uncertain_upper_usd=uncertain_upper_usd,
        )
        with self._lock:
            previous = self._external.get(charge_id)
            if previous is not None and previous != charge:
                raise ValueError("Cropped external charge identity changed")
            self._external[charge_id] = charge

    def state_instructions(self) -> dict[str, JsonValue]:
        with self._lock:
            return {
                "normal_target_usd": self.total_llm_usd,
                "target_is_hard_cap": False,
                "repair_limit_usd": self.repair_usd,
                "research_limit_usd": self.research_usd,
                "recorded_cost_upper_usd": self._cost_bound()
                if self._bound_known()
                else None,
                "repair_allowed": self.repair_allowed,
                "research_allowed": self.research_allowed,
                "first_finalization_allowed": True,
                "ancillary_capture": "explicit_recorded_charges_only",
                "instruction": "Do not repair after the repair limit. At the research limit, use recorded facts and originals for supported first finalization with precise unresolved limits; missing evidence does not prove absence.",
            }

    @property
    def finish_reason(self) -> str | None:
        with self._lock:
            return self._finish_reason

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            state: dict[str, JsonValue] = {
                "calls": cast(list[JsonValue], self._calls.copy()),
                "interrupted_reservations": cast(
                    list[JsonValue],
                    [
                        *self._interrupted_reservations,
                        *[
                            _interrupted_reservation(ticket)
                            for ticket in self._tickets.values()
                        ],
                    ],
                ),
                "spent_usd": self._spent,
                "research_spent_usd": self._research_spent,
                # Interrupted calls keep their reserve across resume.
                "uncertain_usd": self._uncertain
                + sum(
                    row.reserved_usd
                    for row in self._tickets.values()
                    if row.reservation_upper_verified
                ),
                "finish_reason": self._finish_reason,
                "bounded": self._bounded,
                "unpriced_call_ids": list(
                    sorted(
                        self._unpriced
                        | {
                            ticket.call_id
                            for ticket in self._tickets.values()
                            if not ticket.reservation_upper_verified
                        }
                    )
                ),
                "external_charges": cast(
                    JsonValue,
                    [row.model_dump(mode="json") for row in self._external.values()],
                ),
            }
            return {
                "version": 5,
                "run_id": self.run_id,
                "scope_hash": self.scope_hash,
                "request_hash": self.request_hash,
                "research_usd": self.research_usd,
                "total_llm_usd": self.total_llm_usd,
                "repair_usd": self.repair_usd,
                "state": state,
                "state_hash": _hash(state),
                "price_source": "selected_model_registry",
                "pricing_policy": "phase_limits_exclusive_cache_partitions_v3",
                "excluded_services": ["unrecorded_ancillary_services"],
            }

    def restore(self, payload: dict[str, JsonValue]) -> None:
        if type(payload.get("version")) is not int:
            raise ValueError("Invalid Cropped cost checkpoint version")
        expected = self.export()
        # Only the former shipped default may upgrade; charge validation stays exact.
        legacy_repair_upgrade = (
            self.research_usd,
            self.total_llm_usd,
            self.repair_usd,
        ) == (10.0, 1.50, 10.0) and (
            payload.get("research_usd"),
            payload.get("total_llm_usd"),
            payload.get("repair_usd"),
        ) == (10.0, 1.50, 5.0)
        for field in (
            "version",
            "run_id",
            "scope_hash",
            "request_hash",
            "research_usd",
            "total_llm_usd",
            "repair_usd",
            "price_source",
            "pricing_policy",
            "excluded_services",
        ):
            if field == "repair_usd" and legacy_repair_upgrade:
                continue
            if payload.get(field) != expected[field]:
                raise ValueError("Cropped cost checkpoint ownership or policy changed")
        state = payload.get("state")
        if not isinstance(state, dict) or payload.get("state_hash") != _hash(state):
            raise ValueError("Cropped cost checkpoint changed")
        totals = [
            state.get(key)
            for key in ("spent_usd", "research_spent_usd", "uncertain_usd")
        ]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            for value in totals
        ):
            raise ValueError("Invalid Cropped cost checkpoint totals")
        calls = state.get("calls")
        interrupted = state.get("interrupted_reservations")
        reason = state.get("finish_reason")
        bounded = state.get("bounded")
        unpriced = state.get("unpriced_call_ids")
        external = state.get("external_charges")
        if (
            not isinstance(calls, list)
            or any(not isinstance(row, dict) for row in calls)
            or not isinstance(interrupted, list)
            or any(not isinstance(row, dict) for row in interrupted)
            or (reason is not None and not isinstance(reason, str))
            or type(bounded) is not bool
            or not isinstance(unpriced, list)
            or any(not isinstance(value, str) for value in unpriced)
            or not isinstance(external, list)
            or any(not isinstance(value, dict) for value in external)
        ):
            raise ValueError("Invalid Cropped cost checkpoint state")
        validated_calls = [
            _validate_cost_call(cast(dict[str, JsonValue], row)) for row in calls
        ]
        validated_interrupted = [
            _validate_cost_call(cast(dict[str, JsonValue], row)) for row in interrupted
        ]
        validated_external = [
            _ExternalCharge.model_validate(value) for value in external
        ]
        if len({row.charge_id for row in validated_external}) != len(
            validated_external
        ):
            raise ValueError("Duplicate Cropped external charge")
        if any(
            call.usage is not None
            or call.known_cost_usd is not None
            or call.provider_rejected_before_generation
            or call.retained_reserve_usd
            != (call.reserved_usd if call.reservation_upper_verified else None)
            for call in validated_interrupted
        ):
            raise ValueError("Cropped interrupted reservation proof changed")
        all_calls = [*validated_calls, *validated_interrupted]
        if any(call.cache_write_contract_violated for call in validated_calls) and (
            reason != "provider_violated_disabled_cache_writes"
        ):
            raise ValueError("Cropped cache-write contract stop was removed")
        if len({call.call_id for call in all_calls}) != len(all_calls):
            raise ValueError("Duplicate Cropped cost checkpoint call")
        if len(set(cast(list[str], unpriced))) != len(unpriced) or any(
            call.known_cost_usd is None
            and not call.provider_rejected_before_generation
            and (
                not call.reservation_upper_verified
                or call.cache_write_contract_violated
            )
            and call.call_id not in unpriced
            for call in all_calls
        ):
            raise ValueError("Cropped unpriced checkpoint charge was removed")
        if (
            not math.isclose(
                cast(float, totals[0]),
                sum(call.known_cost_usd or 0 for call in validated_calls),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                cast(float, totals[1]),
                sum(
                    call.known_cost_usd or 0
                    for call in validated_calls
                    if call.research
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            or not math.isclose(
                cast(float, totals[2]),
                sum(
                    (call.retained_reserve_usd or 0) + call.cache_write_exposure_usd
                    for call in all_calls
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "Cropped cost checkpoint totals differ from recorded calls"
            )
        with self._lock:
            if self._tickets:
                raise ValueError(
                    "Cannot restore a Cropped cost checkpoint during generation"
                )
            self._spent, self._research_spent, self._uncertain = (
                float(cast(float, value)) for value in totals
            )
            if self._research_spent > self._spent:
                raise ValueError("Cropped cost checkpoint exceeds its policy")
            self._calls = cast(list[dict[str, JsonValue]], calls.copy())
            self._interrupted_reservations = cast(
                list[dict[str, JsonValue]], interrupted.copy()
            )
            self._finish_reason = reason
            self._bounded = bounded
            self._unpriced = set(cast(list[str], unpriced))
            self._external = {row.charge_id: row for row in validated_external}
