from __future__ import annotations

import copy
import hashlib
import re
import threading
from typing import Callable, TypedDict, cast

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome, ToolSpec
from onyx.asv3.workers import WorkerPool


class _AssignmentBinding(TypedDict, total=False):
    assignment_id: str


class QuestionResearch:
    """Separate question research and arrange immutable completed answer bodies."""

    def __init__(
        self,
        context: RunContext,
        workers: WorkerPool,
        original_questions: list[str],
        *,
        host_assembly: bool = False,
    ) -> None:
        self.context = context
        self.workers = workers
        self.original_questions = tuple(original_questions)
        self.host_assembly = host_assembly
        self.assignments: list[dict[str, JsonValue]] = []
        self.answers: list[dict[str, JsonValue]] = []
        self.answer_revisions: list[dict[str, JsonValue]] = []
        self._lock = threading.RLock()
        self.publish_guard: Callable[[str], ToolOutcome | None] | None = None
        self.repair_guard: Callable[[str], ToolOutcome | None] | None = None
        self.accepted_answer_guard: (
            Callable[[dict[str, JsonValue]], ToolOutcome | None] | None
        ) = None

    def assignment(self, task_id: str) -> dict[str, JsonValue]:
        with self._lock:
            matches = [
                item for item in self.assignments if item.get("task_id") == task_id
            ]
            if len(matches) != 1:
                raise ValueError("Unknown independent task assignment")
            return copy.deepcopy(matches[0])

    def assemble_retained_answers(self) -> ToolOutcome:
        return self.assemble_answers(
            {"order": [item["question_id"] for item in self.assignments]}, self.context
        )

    @staticmethod
    def incomplete_answer(language: str) -> str:
        return (
            "Bu alt sorunun kaynak araştırması tamamlanamadı; kesin bir sonuç "
            "verebilmek için ilgili özgün hükümler ve koşulları henüz doğrulanamadı."
            if language.startswith("tr")
            else "Research for this subquestion is incomplete; the applicable original "
            "provisions and their conditions have not yet been established."
        )

    @staticmethod
    def answer_hash(answer: str) -> str:
        return hashlib.sha256(answer.encode()).hexdigest()

    def restore(self, value: JsonValue) -> None:
        if not isinstance(value, dict) or not isinstance(value.get("answers"), list):
            return
        answers = value["answers"]
        assignments = value.get("assignments", [])
        revisions = value.get("answer_revisions", [])
        if not isinstance(assignments, list) or not all(
            isinstance(item, dict) for item in [*answers, *assignments]
        ):
            raise ValueError("Invalid independent question checkpoint")
        if not isinstance(revisions, list) or not all(
            isinstance(item, dict)
            and isinstance(item.get("question_id"), str)
            and isinstance(item.get("previous_answer"), str)
            and item.get("before_hash")
            == self.answer_hash(str(item.get("previous_answer")))
            and isinstance(item.get("after_hash"), str)
            and re.fullmatch(r"[a-f0-9]{64}", str(item["after_hash"]))
            and isinstance(item.get("gap"), str)
            and str(item["gap"]).strip()
            for item in revisions
        ):
            raise ValueError("Invalid independent answer revision journal")
        restored_assignments = (
            self._validated_questions(assignments) if assignments else []
        )
        task_ids: set[str] = set()
        for item in restored_assignments:
            task_id = item.get("task_id")
            if task_id is None:
                continue
            if not isinstance(task_id, str) or not task_id or task_id in task_ids:
                raise ValueError("Invalid independent question task identity")
            task_ids.add(task_id)
        answer_ids: set[str] = set()
        for item in answers:
            assert isinstance(item, dict)
            identifier, body = item.get("question_id"), item.get("answer")
            if (
                not isinstance(identifier, str)
                or not identifier
                or identifier in answer_ids
                or not isinstance(body, str)
                or not body.strip()
                or item.get("answer_hash", self.answer_hash(body))
                != self.answer_hash(body)
            ):
                raise ValueError("Invalid independent answer checkpoint")
            answer_ids.add(identifier)
        if (
            answers
            and restored_assignments
            and answer_ids != {item["question_id"] for item in restored_assignments}
        ):
            raise ValueError("Independent checkpoint is missing an answer body")
        latest: dict[str, str] = {}
        for revision in revisions:
            assert isinstance(revision, dict)
            identifier = str(revision["question_id"])
            if identifier not in answer_ids or (
                identifier in latest and latest[identifier] != revision["before_hash"]
            ):
                raise ValueError("Independent answer revision chain changed")
            latest[identifier] = str(revision["after_hash"])
        for item in answers:
            assert isinstance(item, dict)
            if str(item["question_id"]) in latest and latest[
                str(item["question_id"])
            ] != self.answer_hash(str(item["answer"])):
                raise ValueError("Independent answer revision does not match its body")
        with self._lock:
            self.assignments = copy.deepcopy(restored_assignments)
            self.answers = [
                {
                    **copy.deepcopy(item),
                    "answer_hash": self.answer_hash(str(item["answer"])),
                }
                for item in answers
                if isinstance(item, dict)
            ]
            self.answer_revisions = copy.deepcopy(revisions)
            self._retain()

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return copy.deepcopy(
                {
                    "assignments": self.assignments,
                    "answers": self.answers,
                    "answer_revisions": self.answer_revisions,
                }
            )

    def _retain(self) -> None:
        self.context.services["independent_answers"] = self.answers
        self.context.services["question_research_started"] = bool(self.answers)
        self.context.services["independent_evidence_numbers"] = sorted(
            {
                number
                for item in self.answers
                for number in extract_citation_numbers(str(item.get("answer", "")))
            }
        )

    def preservation_gap(self, answer: str) -> ToolOutcome | None:
        if not self.answers:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Research each material subquestion separately before publication.",
                data={"missing": "independent question results"},
            )
        if self.host_assembly:
            expected = {str(item["question_id"]): item for item in self.answers}
            order = [str(item["question_id"]) for item in self.assignments]
            if set(order) != set(expected) or answer != self._arranged_text(
                order, expected
            ):
                return ToolOutcome(
                    status=OutcomeStatus.PARTIAL,
                    summary="Publish only the complete deterministic arrangement of accepted answer bodies. No extra claims, omissions, repetitions or rewrites are permitted.",
                    data={"changed_parallel_arrangement": True},
                )
            return None
        missing = [
            item["question_id"]
            for item in self.answers
            if str(item.get("answer", "")) not in answer
        ]
        if missing:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Arrange every complete answer body without shortening or rewriting it.",
                data={"missing_question_answers": missing},
            )
        return None

    @staticmethod
    def _arranged_text(
        order: list[str], expected: dict[str, dict[str, JsonValue]]
    ) -> str:
        parts: list[str] = []
        for index, identifier in enumerate(order, 1):
            item = expected[identifier]
            title, body = item.get("answer_title"), str(item["answer"])
            parts.append(
                f"## {index}. {title}\n\n{body}"
                if isinstance(title, str) and title.strip() and len(title) <= 120
                else body
            )
        return "\n\n".join(parts)

    def _validated_questions(self, raw: JsonValue) -> list[dict[str, JsonValue]]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("Supply independent material subquestions")
        questions: list[dict[str, JsonValue]] = []
        covered: set[int] = set()
        identifiers: set[str] = set()
        assigned_outcomes: set[str] = set()
        for entry in raw:
            if not isinstance(entry, dict):
                raise ValueError("Invalid independent question")
            identifier, question = entry.get("question_id"), entry.get("question")
            title, message = entry.get("public_title"), entry.get("public_message")
            parents = entry.get("parent_question_ids")
            if (
                not isinstance(identifier, str)
                or not identifier.strip()
                or identifier in identifiers
                or not isinstance(question, str)
                or not question.strip()
                or not isinstance(title, str)
                or not title.strip()
                or not isinstance(message, str)
                or not message.strip()
                or not isinstance(parents, list)
                or not parents
                or any(
                    type(number) is not int
                    or number < 1
                    or number > len(self.original_questions)
                    for number in parents
                )
            ):
                raise ValueError(
                    "Unique question IDs and valid original question coverage required"
                )
            identifiers.add(identifier)
            covered.update(number for number in parents if type(number) is int)
            answer_title = entry.get("answer_title")
            if answer_title is not None and (
                not isinstance(answer_title, str)
                or not answer_title.strip()
                or "\n" in answer_title
                or "\r" in answer_title
            ):
                raise ValueError("An answer title must be a nonempty single line")
            if (
                self.host_assembly
                and isinstance(answer_title, str)
                and (len(answer_title) > 120 or extract_citation_numbers(answer_title))
            ):
                raise ValueError(
                    "Use a short neutral topic title without citation numbers"
                )
            outcomes = entry.get("outcome_ids")
            if outcomes is not None:
                from onyx.asv3.outcome_map import OutcomeMap

                state = self.context.services.get("outcome_map")
                if (
                    not isinstance(outcomes, list)
                    or not outcomes
                    or any(
                        not isinstance(value, str) or not value for value in outcomes
                    )
                    or len(set(outcomes)) != len(outcomes)
                ):
                    raise ValueError(
                        "Assignment outcome_ids must be unique nonempty IDs"
                    )
                if not isinstance(state, OutcomeMap):
                    raise ValueError(
                        "Assignment outcome_ids require recorded outcomes; omit the optional binding or declare _outcomes on this same action"
                    )
                known = set(state.outcome_ids())
                if self.host_assembly and assigned_outcomes.intersection(outcomes):
                    raise ValueError(
                        "Each outcome must have one owning assignment; group dependent issues"
                    )
                assigned_outcomes.update(outcomes)
                unknown = set(outcomes) - known
                if unknown:
                    raise ValueError(
                        f"Undeclared assignment outcomes for {identifier}: "
                        + ", ".join(sorted(unknown))
                    )
                if set(outcomes) - set(
                    state.outcome_ids(
                        question_ids=[
                            f"q{number - 1}"
                            for number in parents
                            if type(number) is int
                        ]
                    )
                ):
                    raise ValueError(
                        f"Assignment outcomes must belong to its original questions: {identifier}"
                    )
            questions.append(dict(entry))
        if covered != set(range(1, len(self.original_questions) + 1)):
            raise ValueError(
                "Every original question needs an independent research assignment"
            )
        return questions

    def research_questions(
        self, arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        if context.depth:
            return ToolOutcome(
                status=OutcomeStatus.DENIED, summary="Root assignment only"
            )
        if self.answers:
            if self.host_assembly:
                return self.assemble_retained_answers()
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Existing independent answers retained; do not duplicate research.",
            )
        if not self.assignments:
            try:
                questions = self._validated_questions(arguments.get("questions"))
            except ValueError as error:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Independent assignments were rejected; no researcher was started.",
                    data={
                        "validation_error": str(error),
                        "instruction": "Correct the assignment coverage and bindings, preserving the original questions and scenario. outcome_ids are optional: use only recorded outcomes belonging to those original questions, or declare them with _outcomes on this same useful action. Omit outcome_ids when not using an outcome map; do not invent IDs or repeat completed research.",
                    },
                )
            with self._lock:
                self.assignments = [{**item, "task_id": None} for item in questions]
            for item in self.assignments:
                # A worker's checkpoint cannot observe its task without its question binding.
                with self._lock:
                    binding: _AssignmentBinding = (
                        {"assignment_id": str(item["question_id"])}
                        if self.host_assembly
                        else {}
                    )
                    task_id = self.workers.spawn(
                        str(item["question"]),
                        request_context=context,
                        public_title=str(item["public_title"]),
                        public_message=str(item["public_message"]),
                        independent_question=True,
                        outcome_ids=[
                            str(value)
                            for value in cast(list[JsonValue], item["outcome_ids"])
                        ]
                        if isinstance(item.get("outcome_ids"), list)
                        else None,
                        **binding,
                    )
                    item["task_id"] = task_id
        available = {item.task_id: item for item in self.workers.results(full=True)}
        tasks = [
            str(item["task_id"])
            for item in self.assignments
            if str(item["task_id"]) in available
        ]
        for item in self.assignments:
            existing = available.get(str(item["task_id"]))
            if existing is not None and (
                not existing.independent_question or existing.task != item["question"]
            ):
                raise ValueError("Independent question task binding changed")
        results = {item.task_id: item for item in self.workers.wait_until_all(tasks)}
        answers: list[dict[str, JsonValue]] = []
        for question in self.assignments:
            task_id = question["task_id"]
            result = results.get(str(task_id))
            body = result.outcome.summary if result and result.outcome else ""
            receipt_id = (
                result.outcome.data.get("parallel_answer_receipt")
                if result and result.outcome
                else None
            )
            if self.host_assembly and not isinstance(receipt_id, str):
                body = ""
            if not body.strip():
                body = self.incomplete_answer(context.language)
            answers.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "answer_title": question.get("answer_title"),
                    "parent_question_ids": question["parent_question_ids"],
                    "task_id": task_id,
                    "status": OutcomeStatus.PARTIAL.value
                    if self.host_assembly and not isinstance(receipt_id, str)
                    else result.outcome.status.value
                    if result and result.outcome
                    else result.status.value
                    if result
                    else "interrupted",
                    "answer": body,
                    "answer_hash": self.answer_hash(body),
                    **(
                        {
                            "parallel_answer_receipt": receipt_id,
                            "host_gap": not isinstance(receipt_id, str),
                        }
                        if self.host_assembly
                        else {}
                    ),
                    "evidence_numbers": list(extract_citation_numbers(body)),
                    **(
                        {"outcome_ids": copy.deepcopy(question["outcome_ids"])}
                        if "outcome_ids" in question
                        else {}
                    ),
                }
            )
        with self._lock:
            self.answers = answers
            self._retain()
        if self.host_assembly:
            return self.assemble_retained_answers()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Independent question answers retained in full for arrangement.",
            data={"question_ids": [item["question_id"] for item in self.answers]},
        )

    def repair_question_answer(
        self, arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        if self.host_assembly:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Accepted parallel bodies are immutable; corrections require their owning researcher's source validation.",
            )
        if context.depth:
            return ToolOutcome(status=OutcomeStatus.DENIED, summary="Root repair only")
        identifier, expected = (
            arguments.get("question_id"),
            arguments.get("expected_answer_hash"),
        )
        gap = arguments.get("gap")
        edits = arguments.get("edits")
        if (
            not isinstance(gap, str)
            or not gap.strip()
            or not isinstance(edits, list)
            or not edits
        ):
            raise ValueError("A targeted repair needs its material gap and exact edits")
        with self._lock:
            target = next(
                (item for item in self.answers if item["question_id"] == identifier),
                None,
            )
            if target is None:
                raise ValueError("Repair an existing completed question answer")
            body = str(target["answer"])
            before = self.answer_hash(body)
            if expected != before:
                raise ValueError(
                    "The independent answer changed; use its current answer_hash"
                )
            changes: list[tuple[int, int, str]] = []
            for edit in edits:
                if not isinstance(edit, dict):
                    raise ValueError("Invalid targeted answer edit")
                kind, anchor, text = (
                    edit.get("kind"),
                    edit.get("target_text"),
                    edit.get("text"),
                )
                if (
                    kind not in {"replace", "insert_after"}
                    or not isinstance(anchor, str)
                    or not anchor.strip()
                    or body.count(anchor) != 1
                    or not isinstance(text, str)
                    or not text.strip()
                    or not extract_citation_numbers(text)
                ):
                    raise ValueError(
                        "Each repair needs a unique literal anchor and source-cited text"
                    )
                start = body.index(anchor)
                end = start + len(anchor)
                if kind == "replace":
                    if anchor.strip() == body.strip() or re.search(r"\n\s*\n", anchor):
                        raise ValueError(
                            "Replace only the defective clause or paragraph, not the full answer"
                        )
                    changes.append((start, end, text))
                else:
                    changes.append((end, end, "\n\n" + text))
            ordered = sorted(changes)
            for index, current in enumerate(ordered[1:], 1):
                previous = ordered[index - 1]
                if current[0] < previous[1] or current[0] == previous[0]:
                    raise ValueError("Targeted answer edits overlap")
            candidate = body
            for start, end, replacement in reversed(ordered):
                candidate = candidate[:start] + replacement + candidate[end:]
            if self.repair_guard is None:
                return ToolOutcome(
                    status=OutcomeStatus.DENIED,
                    summary="Targeted answer repair requires the source publication guard",
                )
            rejection = self.repair_guard(candidate)
            if rejection is not None:
                return rejection
            after = self.answer_hash(candidate)
            if after == before:
                raise ValueError("A targeted repair must change the identified defect")
            self.answer_revisions.append(
                {
                    "question_id": identifier,
                    "previous_answer": body,
                    "before_hash": before,
                    "after_hash": after,
                    "gap": gap,
                    "edits": copy.deepcopy(edits),
                }
            )
            target.update(
                answer=candidate,
                answer_hash=after,
                evidence_numbers=list(extract_citation_numbers(candidate)),
            )
            self.context.services.pop("assembled_answer", None)
            self._retain()
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="The identified answer defect was repaired; other text and completed answers are unchanged",
                data={"question_id": identifier, "answer_hash": after},
            )

    def assemble_answers(
        self, arguments: dict[str, JsonValue], _context: RunContext
    ) -> ToolOutcome:
        order = arguments.get("order")
        expected = {str(item["question_id"]): item for item in self.answers}
        if (
            not expected
            or not isinstance(order, list)
            or any(not isinstance(item, str) for item in order)
            or len(order) != len(expected)
            or set(order) != set(expected)
        ):
            raise ValueError("Arrange every independent question ID exactly once")
        if self.host_assembly and order != [
            item["question_id"] for item in self.assignments
        ]:
            raise ValueError(
                "Parallel arrangement must preserve its accepted assignment order"
            )
        connections = arguments.get("connections", "")
        if self.host_assembly and connections:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Host arrangement cannot add unexamined legal connections. Keep dependent outcomes in the same assignment.",
            )
        if not isinstance(connections, str):
            raise ValueError("Connections must be text")
        if connections.strip() and not any(
            item.get("evidence_numbers") for item in self.answers
        ):
            raise ValueError("Connections require recorded original evidence")
        for identifier in order:
            item = expected[str(identifier)]
            if self.host_assembly:
                if self.accepted_answer_guard is None:
                    return ToolOutcome(
                        status=OutcomeStatus.DENIED,
                        summary="Parallel arrangement requires accepted child delivery validation",
                    )
                rejection = self.accepted_answer_guard(item)
                if rejection is not None:
                    return rejection
        answer = self._arranged_text(cast(list[str], order), expected)
        if connections.strip():
            answer += "\n\n" + connections
        if self.publish_guard:
            gap = self.publish_guard(answer)
            if gap is not None:
                return gap
        self.context.services["assembled_answer"] = answer
        self.context.services["independent_partial"] = any(
            item.get("status") != OutcomeStatus.FOUND.value for item in self.answers
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND, summary="Complete answer bodies arranged"
        )

    def tool_specs(self) -> list[ToolSpec]:
        specs = [
            ToolSpec(
                name="research_questions",
                description="In the first decision, inspect the conversation and retained session originals, then separate every material semantic subquestion. Preserve alternatives and prose outcomes; map each to its original question number. Each gets the same full scenario and retained originals with a separate history, without time or execution quotas. Research only new or unresolved issues. Related questions remain separate. Assignments run in parallel; complete answers return without shortening. Give each a brief neutral answer_title for presentation.",
                parameters={
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "question_id": {"type": "string", "minLength": 1},
                                    "question": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                    "answer_title": {
                                        "type": "string",
                                        "minLength": 1,
                                        "description": "Brief neutral topic label for this answer. Do not repeat the question or scenario, and do not assert an uncited legal conclusion.",
                                    },
                                    "parent_question_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "integer", "minimum": 1},
                                    },
                                    "outcome_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "uniqueItems": True,
                                        "items": {"type": "string", "minLength": 1},
                                        "description": "Optional IDs already recorded for this assignment's original questions, or declared with _outcomes on this same action. Omit outcome_ids when not using an outcome map; never invent undeclared IDs. Preserve the full scenario and source-bound conditions.",
                                    },
                                    "public_title": {"type": "string", "minLength": 1},
                                    "public_message": {
                                        "type": "string",
                                        "minLength": 1,
                                    },
                                },
                                "required": [
                                    "question_id",
                                    "question",
                                    "parent_question_ids",
                                    "public_title",
                                    "public_message",
                                ],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["questions"],
                    "additionalProperties": False,
                },
                handler=self.research_questions,
                orchestrates=True,
                parallel_safe=False,
                consumes_tool_budget=False,
            ),
            ToolSpec(
                name="repair_question_answer",
                description="Repair only an identified material defect in one completed answer using already delivered originals. Use its current answer_hash; replace a unique exact defective clause/paragraph or insert source-cited detail after a unique exact anchor. Preserve all untargeted text and other complete answers. Do not rewrite a whole answer, remove supported detail, change its scenario, or repair only style. Research a genuinely missing source before applying this patch; no new model call is required for already supplied detail.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question_id": {"type": "string", "minLength": 1},
                        "expected_answer_hash": {
                            "type": "string",
                            "pattern": "^[a-f0-9]{64}$",
                        },
                        "gap": {
                            "type": "string",
                            "minLength": 1,
                            "description": "The actual material source-supported defect being corrected; no invented review obligation.",
                        },
                        "edits": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "kind": {
                                        "type": "string",
                                        "enum": ["replace", "insert_after"],
                                    },
                                    "target_text": {
                                        "type": "string",
                                        "minLength": 1,
                                        "description": "Unique exact text from the current answer; replacement covers at most one defective paragraph or clause.",
                                    },
                                    "text": {
                                        "type": "string",
                                        "minLength": 1,
                                        "description": "Complete supported correction or missing detail with its own recorded inline original [n] citations.",
                                    },
                                },
                                "required": ["kind", "target_text", "text"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["question_id", "expected_answer_hash", "gap", "edits"],
                    "additionalProperties": False,
                },
                handler=self.repair_question_answer,
                parallel_safe=False,
                consumes_tool_budget=False,
            ),
            ToolSpec(
                name="assemble_answers",
                description="Arrange every independent answer exactly once, preserving its complete text, operative conditions and citations verbatim. Supply question IDs in useful order and optional source-cited connections; the host inserts the full bodies. Do not summarize or rewrite them.",
                parameters={
                    "type": "object",
                    "properties": {
                        "order": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "connections": {"type": "string"},
                    },
                    "required": ["order"],
                    "additionalProperties": False,
                },
                handler=self.assemble_answers,
                parallel_safe=False,
                consumes_tool_budget=False,
            ),
        ]
        if self.host_assembly:
            description = (
                "When fresh independent research is needed, assign all material outcomes once. "
                "Group issues whose legal conclusions depend on each other in the same task; "
                "do not split a rule from its exceptions, disputed applicability or later procedure. "
                "Every task receives the exact full original request, conversation and session originals. "
                "Use concise neutral answer_title labels and scenario-specific queries. Independent tasks "
                "run concurrently and their complete validated bodies are arranged verbatim by the host, "
                "without another answer-writing model. Do not use this for greetings, clarification or "
                "a complete answer already supported by retained originals. Do not create overlapping work."
            )
            return [specs[0].model_copy(update={"description": description})]
        return specs
