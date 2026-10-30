"""Explicit Fast-only research handoff, without granting publication approval."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import Decision, OutcomeStatus, RunContext, ToolOutcome, ToolSpec

FAST_HANDOFF = "asv3_fast_research_handoff"
FAST_FINISH_TOOL = "finish_research"
_PUBLICATION_TOOLS = frozenset(
    {
        "submit_answer",
        "submit_partial_answer",
        "submit_retained_answer",
        "submit_retained_partial_answer",
    }
)


class FastFinding(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question_ids: list[str] = Field(min_length=1)
    statement: str = Field(min_length=1)
    evidence_numbers: list[int]


class FastResearchHandoff(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    findings: list[FastFinding]
    open_needs: list[str]


def fast_research_phase(context: RunContext, *, writing: bool = False) -> bool:
    return (
        context.services.get("asv3_fast") is True
        and not context.depth
        and not writing
        and FAST_HANDOFF not in context.services
    )


def fast_tool_definitions(
    tools: list[dict[str, JsonValue]], *, research: bool
) -> list[dict[str, JsonValue]]:
    """Change exposed actions, retaining the exact original tool definitions."""
    excluded = _PUBLICATION_TOOLS if research else {FAST_FINISH_TOOL}
    return [
        tool
        for tool in tools
        if not (
            isinstance(function := tool.get("function"), dict)
            and function.get("name") in excluded
        )
    ]


def fast_research_decision_gap(decision: Decision) -> str | None:
    if decision.answer or not decision.calls:
        return "Fast research requires an exposed action; hand off findings with finish_research instead of writing a draft"
    if (
        any(call.name == FAST_FINISH_TOOL for call in decision.calls)
        and len(decision.calls) != 1
    ):
        return "finish_research must be the only action in its native decision"
    for call in decision.calls:
        if call.name != "compose_tool_calls":
            continue
        steps = call.arguments.get("steps")
        if isinstance(steps, list) and any(
            isinstance(step, dict)
            and step.get("tool") in {*_PUBLICATION_TOOLS, FAST_FINISH_TOOL}
            for step in steps
        ):
            return "Research composition cannot publish or finish; call finish_research on its own after reading results"
    return None


def _validated_handoff(
    payload: object, ledger: EvidenceLedger, context: RunContext
) -> FastResearchHandoff:
    handoff = FastResearchHandoff.model_validate(payload)
    state = context.services.get("research_state")
    questions = getattr(state, "questions", None)
    allowed = (
        {f"q{index}" for index in range(len(questions))}
        if isinstance(questions, (list, tuple))
        else None
    )
    for finding in handoff.findings:
        if allowed is not None and set(finding.question_ids) - allowed:
            raise ValueError("Handoff findings must bind this run's questions")
        if any(
            number < 1 or ledger.get(number) is None
            for number in finding.evidence_numbers
        ):
            raise ValueError("Handoff findings require recorded original citations")
    return handoff


def restore_fast_handoff(
    previous: dict[str, JsonValue] | None,
    ledger: EvidenceLedger,
    context: RunContext,
) -> None:
    if context.services.get("asv3_fast") is not True or previous is None:
        return
    saved = previous.get(FAST_HANDOFF)
    if saved is not None:
        context.services[FAST_HANDOFF] = _validated_handoff(
            saved, ledger, context
        ).model_dump(mode="json")


def finish_research_spec(ledger: EvidenceLedger, owner: RunContext) -> ToolSpec:
    def finish(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_active()
        if (
            owner.services.get("asv3_fast") is not True
            or context.depth
            or context.run_id != owner.run_id
            or context.scope != owner.scope
        ):
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Only the Fast root may hand research to its answer model",
            )
        if FAST_HANDOFF in owner.services:
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Research was already handed to the selected answer model",
            )
        try:
            handoff = _validated_handoff(
                {key: value for key, value in args.items() if not key.startswith("_")},
                ledger,
                context,
            )
        except ValueError as error:
            return ToolOutcome(status=OutcomeStatus.INVALID, summary=str(error))
        owner.check_active()
        owner.services[FAST_HANDOFF] = handoff.model_dump(mode="json")
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Research notes retained for writing; no answer was published",
            data={"legal_approval": False, **handoff.model_dump(mode="json")},
        )

    return ToolSpec(
        name=FAST_FINISH_TOOL,
        description=(
            "Hand concise question-bound findings and precise open needs to the selected "
            "answer model without writing a prose draft. Call on its own after useful "
            "research; it publishes nothing and does not approve the legal findings."
        ),
        parameters=FastResearchHandoff.model_json_schema(),
        handler=finish,
        parallel_safe=False,
        consumes_tool_budget=False,
    )
