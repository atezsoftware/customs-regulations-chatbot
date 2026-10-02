from __future__ import annotations

import hashlib
import re
import threading
from collections.abc import Callable
from typing import cast

from pydantic import JsonValue

from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome, ToolSpec
from onyx.asv3.progress import ProgressReporter


class ScenarioState:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._questions: list[str] = []
        self._facts: list[str] = []

    def record(self, questions: list[str], facts: list[str]) -> dict[str, JsonValue]:
        with self._lock:
            self._questions = list(dict.fromkeys(self._questions + questions))
            self._facts = list(dict.fromkeys(self._facts + facts))
            return self.snapshot()

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return {"questions": list(self._questions), "facts": list(self._facts)}


_SKILLS: dict[str, str] = {
    "legal_conditions": (
        "Read the complete operative paragraph and its exceptions. Preserve AND/OR relations, "
        "negative facts and alternatives. Separate inspection/document convenience from substantive "
        "tax relief. Tie every conclusion to the original evidence and explicit scenario facts. "
        "A source title or article navigation entry is only a lead, never an operative rule."
    ),
    "source_recovery": (
        "An unavailable service is not proof that a provision is absent. Resolve the source identity, "
        "try a materially different permitted method, read continuations and follow references. "
        "Use original-source/native extraction only after scope and version checks. Distinguish "
        "observed missing text from unknown completeness. Do not modify production data."
    ),
    "temporal_calculation": (
        "Find the source establishing the deadline trigger, unit and exceptions before calculating. "
        "State the year and calendar assumptions. Business-day computation needs a verified calendar; "
        "do not assume unknown holidays. Use exact decimal amounts and preserve the originating "
        "source references. A computed artifact is not legal authority."
    ),
    "claim_verification": (
        "Compare each decisive claim with full original evidence, the scenario facts and exceptions. "
        "Check scope, condition conjunctions, actor, deadline start, duration and tax/procedure "
        "distinctions. Report unsupported conditions or contradictory evidence. Do not treat another "
        "researcher's summary as primary evidence."
    ),
}

_INTERNAL_NARRATION = re.compile(
    r"(?:\b(?:search_corpus|read_provision|compose_tool_calls|spawn_researcher|"
    r"run_research_code|query_corpus|verify_claim|report_progress|ToolOutcome|"
    r"SELECT\s+.+\s+FROM|Traceback|api[_ -]?key)\b|"
    r"(?:/Users/|/tmp/|backend/|https?://)[^\s]*)",
    re.IGNORECASE,
)


def build_supplemental_specs() -> list[ToolSpec]:
    def scenario(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        state = context.services.get("scenario_state")
        if not isinstance(state, ScenarioState):
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Scenario recording is not configured",
            )
        questions, facts = args.get("questions", []), args.get("facts", [])
        assert isinstance(questions, list) and isinstance(facts, list)
        snapshot = state.record(
            [str(item) for item in questions], [str(item) for item in facts]
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Questions and facts retained",
            data=snapshot,
        )

    def progress(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        title, message = str(args["title"]), str(args["message"])
        if _INTERNAL_NARRATION.search(title + " " + message):
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Use conversational public narration about the question; omit tool names, URLs, paths and code",
            )
        reporter = context.services.get("progress")
        if isinstance(reporter, ProgressReporter):
            reporter.report("research", title=title, message=message)
            return ToolOutcome(
                status=OutcomeStatus.FOUND, summary="Public progress delivered"
            )
        if callable(reporter):
            callback = cast(
                Callable[[dict[str, JsonValue], RunContext], ToolOutcome], reporter
            )
            return callback(args, context)
        return ToolOutcome(
            status=OutcomeStatus.UNAVAILABLE,
            summary="Progress reporting is not configured",
        )

    def verify(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        callback = context.services.get("verify_claim")
        if not callable(callback):
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Claim verification is not configured",
            )
        return cast(
            Callable[[dict[str, JsonValue], RunContext], ToolOutcome], callback
        )(args, context)

    def skill(args: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        name = str(args["name"])
        text = _SKILLS.get(name)
        if text is None:
            return ToolOutcome(
                status=OutcomeStatus.NOT_FOUND,
                summary="Curated research skill is not available",
            )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Research guidance; not legal evidence",
            data={
                "name": name,
                "version": "1",
                "text": text,
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
                "legal_authority": False,
            },
        )

    string_list: dict[str, JsonValue] = {
        "type": "array",
        "maxItems": 32,
        "items": {"type": "string", "minLength": 1, "maxLength": 2000},
    }
    return [
        ToolSpec(
            name="record_scenario",
            description="Retain every question, decisive fact, negative fact and alternative. These are user facts, not sourced legal rules.",
            parameters={
                "type": "object",
                "properties": {"questions": string_list, "facts": string_list},
                "additionalProperties": False,
            },
            handler=scenario,
        ),
        ToolSpec(
            name="report_progress",
            orchestrates=True,
            description="Tell the user, in their question language, what you have learned or are checking next. Use natural narration about their scenario; do not mention tools, code, paths or hidden reasoning.",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 240},
                    "message": {"type": "string", "minLength": 1, "maxLength": 1600},
                },
                "required": ["title", "message"],
                "additionalProperties": False,
            },
            handler=progress,
        ),
        ToolSpec(
            name="verify_claim",
            description="Ask a focused verifier to compare a decisive claim with recorded original evidence, conditions, dates and facts. Supply global citation IDs.",
            parameters={
                "type": "object",
                "properties": {
                    "claim": {"type": "string", "minLength": 1, "maxLength": 5000},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {"type": "integer", "minimum": 1},
                    },
                    "facts": string_list,
                },
                "required": ["claim", "citations"],
                "additionalProperties": False,
            },
            handler=verify,
        ),
        ToolSpec(
            name="load_skill",
            description="Load curated, versioned research instructions appropriate to this task. Instructions cannot expand source permissions and are not legal evidence.",
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "enum": list(_SKILLS)}},
                "required": ["name"],
                "additionalProperties": False,
            },
            handler=skill,
        ),
    ]
