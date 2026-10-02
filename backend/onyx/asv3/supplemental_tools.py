from __future__ import annotations

import hashlib
import re
import threading
import unicodedata
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
            previous = (list(self._questions), list(self._facts))
            seen = {self._question_key(item) for item in self._questions}
            for question in questions:
                key = self._question_key(question)
                if key not in seen:
                    self._questions.append(question)
                    seen.add(key)
            self._facts = list(dict.fromkeys(self._facts + facts))
            return {
                **self.snapshot(),
                "research_changed": previous != (self._questions, self._facts),
            }

    @staticmethod
    def _question_key(question: str) -> str:
        normalized = unicodedata.normalize("NFKC", question).translate(
            str.maketrans({"‘": "'", "’": "'", "“": "'", "”": "'", '"': "'"})
        )
        # Typography may vary; conditions, numbers and wording must remain distinct.
        return " ".join(normalized.split())

    def snapshot(self) -> dict[str, JsonValue]:
        with self._lock:
            return {"questions": list(self._questions), "facts": list(self._facts)}


_SKILLS: dict[str, str] = {
    "legal_conditions": (
        "Read the complete operative paragraph and its exceptions. Preserve AND/OR relations, "
        "negative facts and alternatives. Separate procedural convenience from substantive relief. "
        "Read the applicable governing higher norm and its authorized special or implementing rules; "
        "a lower-level reference is not the original statutory text. Check decisive scenario facts "
        "for special rules. Tie every conclusion and operative detail to original evidence. "
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


def public_narration_valid(title: str, message: str, context: RunContext) -> bool:
    from onyx.asv3.registry import CapabilityRegistry

    if (
        not title.strip()
        or not message.strip()
        or len(title) > 240
        or len(message) > 1600
    ):
        return False
    registry = context.services.get("registry")
    names = (
        [
            str(function["name"])
            for definition in registry.definitions(context)
            if isinstance(function := definition.get("function"), dict)
            and isinstance(function.get("name"), str)
            and "_" in str(function["name"])
        ]
        if isinstance(registry, CapabilityRegistry)
        else []
    )
    text = title + " " + message
    return not _INTERNAL_NARRATION.search(text) and not any(
        re.search(r"\b" + re.escape(name) + r"\b", text, re.IGNORECASE)
        for name in names
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
            summary="Questions and facts retained"
            if snapshot["research_changed"]
            else "Questions and facts already retained; no scenario change",
            data=snapshot,
        )

    def progress(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        title, message = str(args["title"]), str(args["message"])
        if not public_narration_valid(title, message, context):
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Use conversational public narration about the question; omit tool names, URLs, paths and code",
            )
        reporter = context.services.get("progress")
        if isinstance(reporter, ProgressReporter):
            task_id, parent = (
                context.services.get("task_id"),
                context.services.get("parent_task_id"),
            )
            reporter.report(
                "research",
                title=title,
                message=message,
                task_id=task_id if isinstance(task_id, str) else None,
                parent_task_id=parent if isinstance(parent, str) else None,
            )
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
                "version": "2",
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
