"""Question-only followups bound to the Cropped writer's real unresolved state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    model_validator,
)

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.models import RunContext
from onyx.asv3.outcome_map import OutcomeMap
from onyx.asv3.research_state import ResearchState


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _hash(value: object) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _require_cropped(context: RunContext) -> None:
    if (
        context.services.get("asv3_exp_cropped") is not True
        or context.services.get("asv3_exp_cheap") is True
    ):
        raise ValueError("Completion followups require the isolated Cropped workflow")


class CroppedRequiredFollowup(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    need_id: str | None = Field(default=None, pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")
    outcome_id: str | None = Field(
        default=None, pattern=r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$"
    )
    question: str = Field(min_length=1, max_length=1000)
    material_rationale: str = Field(min_length=1, max_length=1000)
    definitely_necessary: bool
    material: bool

    @model_validator(mode="after")
    def bound_question(self) -> CroppedRequiredFollowup:
        if self.need_id is None and self.outcome_id is None:
            raise ValueError(
                "A required followup must bind an existing need or outcome"
            )
        if not self.question.strip().endswith("?"):
            raise ValueError(
                "A required followup must be an explicit unresolved question"
            )
        if not self.material_rationale.strip():
            raise ValueError("A required followup needs an explicit material rationale")
        return self


def cropped_required_followups_schema() -> dict[str, JsonValue]:
    """Expose an optional terminal metadata field without importing the registry."""
    return {
        "type": "array",
        "items": cast(
            dict[str, JsonValue], CroppedRequiredFollowup.model_json_schema()
        ),
    }


def _state(context: RunContext) -> tuple[dict[str, JsonValue], dict[str, JsonValue]]:
    research = context.services.get("research_state")
    outcomes = context.services.get("outcome_map")
    scope_hash = hashlib.sha256(
        json.dumps(context.scope, sort_keys=True).encode()
    ).hexdigest()
    for state in (research, outcomes):
        if isinstance(state, (ResearchState, OutcomeMap)) and (
            state.run_id != context.run_id or state.scope_hash != scope_hash
        ):
            raise ValueError("Completion state belongs to another run or scope")
    return (
        research.export() if isinstance(research, ResearchState) else {},
        outcomes.export() if isinstance(outcomes, OutcomeMap) else {},
    )


def _rows(
    state: dict[str, JsonValue], name: str, identity: str
) -> dict[str, dict[str, JsonValue]]:
    rows = state.get(name, [])
    assert isinstance(rows, list)
    return {
        row[identity]: row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get(identity), str)
    }


@dataclass(frozen=True)
class CroppedCompletionContext:
    run_id: str
    scope_hash: str
    owner: str
    depth: int
    request: str
    facts_json: str
    research_json: str
    outcomes_json: str
    publication_gap_json: str

    @property
    def fingerprint(self) -> str:
        return _hash(self.__dict__)

    def export(self) -> dict[str, JsonValue]:
        return cast(dict[str, JsonValue], dict(self.__dict__))

    @classmethod
    def restore(cls, value: dict[str, JsonValue]) -> CroppedCompletionContext:
        fields = set(cls.__dataclass_fields__)
        if (
            set(value) != fields
            or type(value.get("depth")) is not int
            or cast(int, value["depth"]) < 0
        ):
            raise ValueError("Invalid completion snapshot fields")
        if any(not isinstance(value[name], str) for name in fields - {"depth"}):
            raise ValueError("Invalid completion snapshot field types")
        result = cls(
            run_id=cast(str, value["run_id"]),
            scope_hash=cast(str, value["scope_hash"]),
            owner=cast(str, value["owner"]),
            depth=cast(int, value["depth"]),
            request=cast(str, value["request"]),
            facts_json=cast(str, value["facts_json"]),
            research_json=cast(str, value["research_json"]),
            outcomes_json=cast(str, value["outcomes_json"]),
            publication_gap_json=cast(str, value["publication_gap_json"]),
        )
        for name, expected in (
            ("facts_json", list),
            ("research_json", dict),
            ("outcomes_json", dict),
            ("publication_gap_json", dict),
        ):
            literal = getattr(result, name)
            parsed = json.loads(literal)
            if not isinstance(parsed, expected) or _json(parsed) != literal:
                raise ValueError("Completion snapshot JSON changed")
        if any(not isinstance(fact, str) for fact in json.loads(result.facts_json)):
            raise ValueError("Completion snapshot facts must remain literal strings")
        return result

    def writer_view(self) -> dict[str, JsonValue]:
        research = json.loads(self.research_json)
        outcomes = json.loads(self.outcomes_json)
        return {
            "full_question": self.request,
            "facts": json.loads(self.facts_json),
            "completion_context_fingerprint": self.fingerprint,
            "research_needs": research.get("needs", []),
            "requested_outcomes": outcomes.get("outcomes", []),
            "outcome_resolutions": outcomes.get("resolutions", []),
            "publication_gap": json.loads(self.publication_gap_json),
            "instruction": (
                "Answer all supported parts of the full question, retaining material conditions, "
                "exceptions and unknown facts. An unavailable original is not proof that the law "
                "is absent and does not verify its legal effect. Do not present an unsupported "
                "conclusion as complete. If an existing material need or outcome definitely "
                "requires further work, report it optionally in _required_followups with its "
                "actual IDs, an explicit question ending in ?, and a material rationale. "
                "Set definitely_necessary and material explicitly; omit speculative or optional "
                "extensions and already resolved matters. Questions are unresolved navigation, "
                "not legal findings or source approval. Do not include internal costs, budgets "
                "or implementation details in the user answer or followup questions."
            ),
        }


def build_cropped_completion_context(
    context: RunContext,
    request: str,
    *,
    facts: list[str] | None = None,
    publication_gap: dict[str, JsonValue] | None = None,
) -> CroppedCompletionContext:
    _require_cropped(context)
    research, outcomes = _state(context)
    return CroppedCompletionContext(
        run_id=context.run_id,
        scope_hash=_hash(context.scope),
        owner=str(context.services.get("task_id") or "root"),
        depth=context.depth,
        request=request,
        facts_json=_json(facts or []),
        research_json=_json(research),
        outcomes_json=_json(outcomes),
        publication_gap_json=_json(publication_gap or {}),
    )


@dataclass(frozen=True)
class CroppedValidatedFollowups:
    snapshot: CroppedCompletionContext
    answer_hash: str
    items: tuple[CroppedRequiredFollowup, ...]


def _eligible(
    item: CroppedRequiredFollowup,
    research: dict[str, JsonValue],
    outcomes: dict[str, JsonValue],
) -> bool:
    needs = _rows(research, "needs", "need_id")
    requested = _rows(outcomes, "outcomes", "outcome_id")
    resolutions = _rows(outcomes, "resolutions", "outcome_id")
    if item.need_id is not None and item.need_id not in needs:
        raise ValueError("Required followup binds an unknown research need")
    if item.outcome_id is not None and item.outcome_id not in requested:
        raise ValueError("Required followup binds an unknown requested outcome")
    need = needs.get(item.need_id or "")
    outcome = requested.get(item.outcome_id or "")
    if need is not None and outcome is not None:
        need_questions, outcome_questions = (
            need.get("question_ids"),
            outcome.get("question_ids"),
        )
        assert isinstance(need_questions, list) and isinstance(outcome_questions, list)
        if not set(need_questions).intersection(outcome_questions):
            raise ValueError("Followup need and outcome belong to unrelated questions")
    if not item.definitely_necessary or not item.material:
        return False
    if need is not None and (
        need.get("material") is not True or need.get("status") == "out_of_scope"
    ):
        return False
    resolution = resolutions.get(item.outcome_id or "")
    if resolution is not None and resolution.get("status") in {
        "supported",
        "conditional",
    }:
        need_gap = str(need.get("gap", "")).strip() if need is not None else ""
        if not need_gap:
            return False
    return True


def validate_cropped_required_followups(
    raw: JsonValue,
    context: RunContext,
    request: str,
    snapshot: CroppedCompletionContext,
    *,
    answer: str,
    facts: list[str] | None = None,
    publication_gap: dict[str, JsonValue] | None = None,
) -> CroppedValidatedFollowups:
    """Validate before applying this terminal action's own state updates."""
    current = build_cropped_completion_context(
        context, request, facts=facts, publication_gap=publication_gap
    )
    if current != snapshot:
        raise ValueError(
            "Completion writer context is stale or belongs to another owner"
        )
    items = TypeAdapter(list[CroppedRequiredFollowup]).validate_python(
        [] if raw is None else raw
    )
    research, outcomes = _state(context)
    allowed = set(extract_citation_numbers(answer))
    kept: list[CroppedRequiredFollowup] = []
    for item in items:
        if not _eligible(item, research, outcomes):
            continue
        if set(extract_citation_numbers(item.question)) - allowed:
            raise ValueError(
                "Followup questions cannot add citations outside the guarded answer"
            )
        if item not in kept:
            kept.append(item)
    return CroppedValidatedFollowups(snapshot, _hash(answer), tuple(kept))


def append_cropped_required_followups(
    answer: str,
    validated: CroppedValidatedFollowups,
    context: RunContext,
    request: str,
    *,
    facts: list[str] | None = None,
) -> str:
    """Append only questions, after the unchanged legal body has passed its guard."""
    _require_cropped(context)
    snapshot = validated.snapshot
    if (
        context.run_id != snapshot.run_id
        or _hash(context.scope) != snapshot.scope_hash
        or str(context.services.get("task_id") or "root") != snapshot.owner
        or context.depth != snapshot.depth
        or request != snapshot.request
        or _json(facts or []) != snapshot.facts_json
        or _hash(answer) != validated.answer_hash
    ):
        raise ValueError("Followups cannot transfer to a changed answer or scenario")
    research, outcomes = _state(context)
    old_research = cast(dict[str, JsonValue], json.loads(snapshot.research_json))
    old_outcomes = cast(dict[str, JsonValue], json.loads(snapshot.outcomes_json))
    kept: list[CroppedRequiredFollowup] = []
    for item in validated.items:
        for identity, field, old, current in (
            (item.need_id, "needs", old_research, research),
            (item.outcome_id, "outcomes", old_outcomes, outcomes),
        ):
            if identity is None:
                continue
            key = "need_id" if field == "needs" else "outcome_id"
            before, after = (
                _rows(old, field, key).get(identity),
                _rows(current, field, key).get(identity),
            )
            if before is None or after is None:
                raise ValueError("Followup binding changed after writer validation")
            semantic_fields = (
                ("question_ids", "purpose", "completion_test", "depends_on")
                if field == "needs"
                else ("question_ids", "detail", "decisive_facts")
            )
            if any(before.get(name) != after.get(name) for name in semantic_fields):
                raise ValueError("Followup meaning changed after writer validation")
        if _eligible(item, research, outcomes):
            kept.append(item)
    if not kept:
        return answer
    if not answer.strip():
        raise ValueError("Followups cannot replace a missing user answer")
    heading = (
        "If you'd like, I can research these open questions too:"
        if context.language.lower().startswith("en")
        else "İstersen şu açık soruları da araştırabilirim:"
    )
    return (
        answer
        + "\n\n"
        + heading
        + "\n"
        + "\n".join(f"- {item.question}" for item in kept)
    )
