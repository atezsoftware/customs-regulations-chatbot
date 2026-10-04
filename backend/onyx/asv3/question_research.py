from __future__ import annotations

import copy
import threading
from typing import Callable

from pydantic import JsonValue

from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.models import OutcomeStatus, RunContext, ToolOutcome, ToolSpec
from onyx.asv3.workers import WorkerPool


class QuestionResearch:
    """Separate question research and arrange immutable completed answer bodies."""

    def __init__(
        self,
        context: RunContext,
        workers: WorkerPool,
        original_questions: list[str],
    ) -> None:
        self.context = context
        self.workers = workers
        self.original_questions = tuple(original_questions)
        self.assignments: list[dict[str, JsonValue]] = []
        self.answers: list[dict[str, JsonValue]] = []
        self._lock = threading.RLock()
        self.publish_guard: Callable[[str], ToolOutcome | None] | None = None

    def restore(self, value: JsonValue) -> None:
        if not isinstance(value, dict) or not isinstance(value.get("answers"), list):
            return
        answers = value["answers"]
        assignments = value.get("assignments", [])
        if not isinstance(assignments, list) or not all(
            isinstance(item, dict) for item in [*answers, *assignments]
        ):
            raise ValueError("Invalid independent question checkpoint")
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
            ):
                raise ValueError("Invalid independent answer checkpoint")
            answer_ids.add(identifier)
        if (
            answers
            and restored_assignments
            and answer_ids != {item["question_id"] for item in restored_assignments}
        ):
            raise ValueError("Independent checkpoint is missing an answer body")
        with self._lock:
            self.assignments = copy.deepcopy(restored_assignments)
            self.answers = [
                copy.deepcopy(item) for item in answers if isinstance(item, dict)
            ]
            self._retain()

    def export(self) -> dict[str, JsonValue]:
        with self._lock:
            return copy.deepcopy(
                {"assignments": self.assignments, "answers": self.answers}
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

    def _validated_questions(self, raw: JsonValue) -> list[dict[str, JsonValue]]:
        if not isinstance(raw, list) or not raw:
            raise ValueError("Supply independent material subquestions")
        questions: list[dict[str, JsonValue]] = []
        covered: set[int] = set()
        identifiers: set[str] = set()
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
            return ToolOutcome(
                status=OutcomeStatus.FOUND,
                summary="Existing independent answers retained; do not duplicate research.",
            )
        if not self.assignments:
            questions = self._validated_questions(arguments.get("questions"))
            with self._lock:
                self.assignments = [{**item, "task_id": None} for item in questions]
            for item in self.assignments:
                # A worker's checkpoint cannot observe its task without its question binding.
                with self._lock:
                    task_id = self.workers.spawn(
                        str(item["question"]),
                        request_context=context,
                        public_title=str(item["public_title"]),
                        public_message=str(item["public_message"]),
                        independent_question=True,
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
            if not body.strip():
                body = (
                    "Bu alt sorunun kaynak araştırması tamamlanamadı; kesin bir sonuç "
                    "verebilmek için ilgili özgün hükümler ve koşulları henüz doğrulanamadı."
                    if context.language.startswith("tr")
                    else "Research for this subquestion is incomplete; the applicable original "
                    "provisions and their conditions have not yet been established."
                )
            answers.append(
                {
                    "question_id": question["question_id"],
                    "question": question["question"],
                    "answer_title": question.get("answer_title"),
                    "parent_question_ids": question["parent_question_ids"],
                    "task_id": task_id,
                    "status": result.outcome.status.value
                    if result and result.outcome
                    else result.status.value
                    if result
                    else "interrupted",
                    "answer": body,
                    "evidence_numbers": list(extract_citation_numbers(body)),
                }
            )
        with self._lock:
            self.answers = answers
            self._retain()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Independent question answers retained in full for arrangement.",
            data={"question_ids": [item["question_id"] for item in self.answers]},
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
        connections = arguments.get("connections", "")
        if not isinstance(connections, str):
            raise ValueError("Connections must be text")
        if connections.strip() and not any(
            item.get("evidence_numbers") for item in self.answers
        ):
            raise ValueError("Connections require recorded original evidence")
        parts: list[str] = []
        for index, identifier in enumerate(order, 1):
            item = expected[str(identifier)]
            title, body = item.get("answer_title"), str(item["answer"])
            parts.append(
                f"## {index}. {title}\n\n{body}"
                if isinstance(title, str) and title.strip() and len(title) <= 120
                else body
            )
        if connections.strip():
            parts.append(connections)
        answer = "\n\n".join(parts)
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
        return [
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
