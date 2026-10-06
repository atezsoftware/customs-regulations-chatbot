"""Owned serial Experimental coordinators inside the parallel task envelope."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, JsonValue

from onyx.asv3.authority import native_named_authority_gap
from onyx.asv3.authority_requirements import AuthorityRequirements
from onyx.asv3.citation_numbers import extract_citation_numbers
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.legal_source_reviews import LegalSourceReviews
from onyx.asv3.llm_adapter import LanguageProfile, ResearchModel
from onyx.asv3.models import (
    Decision,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.outcome_map import OutcomeCondition, OutcomeMap, RequestedOutcome
from onyx.asv3.progress import (
    ProgressReporter,
    localized_notifications,
    report_source_deliveries,
)
from onyx.asv3.registry import CapabilityRegistry, build_core_specs
from onyx.asv3.research_state import ResearchState, build_research_specs
from onyx.asv3.scenario import initial_questions
from onyx.asv3.session_research import session_research_checkpoint
from onyx.asv3.supplemental_tools import ScenarioState, public_narration_valid
from onyx.asv3.workers import WorkerPool
from onyx.chat.models import ChatMessageSimple
from onyx.configs.constants import MessageType
from onyx.llm.interfaces import LLM, LLMUserIdentity, ReasoningEffort

SERIAL_SESSION_POLICY = "experimental-serial-session-v1"
_IDENTITY_FIELDS = frozenset(
    "source_id file_id document_id regulatory_chunk_id canonical_chunk_id chunk_id "
    "canonical_identity original_document_id source_sha256 text_hash payload_sha256 "
    "publication_id publication_revision publication_revision_id revision revision_id "
    "publication_version index_name index_uuid query_index_name query_index_uuid "
    "version version_unknown validity_start validity_end effective_start effective_end "
    "read_as_of_date as_of_date regulatory_validity_start_date regulatory_validity_end_date "
    "derived_role canonical_role derived external untrusted legal_authority "
    "external_tool_name external_tool_id source_regulatory_chunk_ids "
    "regulatory_document_id source_canonical_document_id canonical_document_id "
    "canonical_source_id representation_id representation_sha256 binding_id tenant_id "
    "document_set_id document_sets document_date source_document_id source_publication_id "
    "publication_payload_sha256 projection_payload_sha256 binding_sha256 effective_date "
    "validity_start_date validity_end_date context_projection_id committed_epoch "
    "document_type regulation_id regulation_number regulation_type heading_path "
    "regulatory_heading_path original_heading_path article_no paragraph_no clause_label".split()
)


def _identity_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    result = {key: value for key, value in metadata.items() if key in _IDENTITY_FIELDS}
    for name in ("canonical_metadata", "publication", "index", "provenance", "target"):
        nested = metadata.get(name)
        if isinstance(nested, dict):
            result[name] = _identity_metadata(nested)
    return result


def _historical_identity_matches(
    saved: dict[str, JsonValue], actual: dict[str, JsonValue]
) -> bool:
    for key, value in saved.items():
        current = actual.get(key)
        if isinstance(value, dict):
            if not isinstance(current, dict) or not _historical_identity_matches(
                value, current
            ):
                return False
        elif key == "version_unknown" and value is True and current is False:
            continue
        elif value is not None and current != value:
            return False
    return True


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def serial_session_history(history: str, scenario_request: str) -> str:
    """Keep an already-final full user message; otherwise add its exact bytes."""
    for role in (MessageType.USER.value, "USER"):
        final = f"{role}: {scenario_request}"
        if history == final or history.endswith("\n" + final):
            return history
    return history + ("\n" if history else "") + "USER: " + scenario_request


class _Accepted(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer_hash: str
    model_call_id: str
    status: OutcomeStatus
    kind: Literal["answer", "partial", "clarification"]
    requires_sources: bool


class _SessionState(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1] = 1
    policy: Literal["experimental-serial-session-v1"] = SERIAL_SESSION_POLICY
    owner: str
    assignment_id: str
    scope_hash: str
    request_hash: str
    scenario_hash: str
    history_hash: str
    scenario: dict[str, JsonValue]
    outcome_map: dict[str, JsonValue]
    authority_requirements: dict[str, JsonValue]
    legal_source_reviews: dict[str, JsonValue]
    session_research: dict[str, JsonValue]
    native_coordinator_sampling: dict[str, JsonValue]
    public_profile: LanguageProfile
    accepted: _Accepted | None = None


CapabilityFactory = Callable[[RunContext], list[ToolSpec]]
Verification = Callable[[RunContext, dict[str, JsonValue]], ToolOutcome]


def accepted_serial_memory_state(
    snapshot: dict[str, JsonValue],
    assignment: dict[str, JsonValue],
    receipt: dict[str, JsonValue],
    context: RunContext,
    scenario_request: str,
    history: str,
) -> dict[str, JsonValue]:
    """Read navigation only from the exact host-sealed serial session state."""
    state = _SessionState.model_validate(snapshot.get("serial_experimental_session"))
    accepted = state.accepted
    exported = state.model_dump(mode="json")
    if (
        accepted is None
        or (
            state.owner,
            state.assignment_id,
            state.scope_hash,
            state.request_hash,
            state.scenario_hash,
            state.history_hash,
        )
        != (
            assignment.get("task_id"),
            assignment.get("question_id"),
            _digest(context.scope),
            _digest(assignment.get("question")),
            _digest(scenario_request),
            _digest(serial_session_history(history, scenario_request)),
        )
        or receipt.get("task_id") != state.owner
        or receipt.get("assignment_hash") != _digest(assignment)
        or receipt.get("source_state_hash") != _digest(exported)
        or (
            receipt.get("answer_hash"),
            receipt.get("model_call_id"),
            receipt.get("status"),
        )
        != (accepted.answer_hash, accepted.model_call_id, accepted.status.value)
        or receipt.get("answer") != snapshot.get("last_draft")
        or not isinstance(receipt.get("answer"), str)
        or hashlib.sha256(str(receipt["answer"]).encode()).hexdigest()
        != accepted.answer_hash
    ):
        raise ValueError("Serial memory is not bound to its sealed assignment")
    return exported


def merge_serial_session_memory(
    memory: dict[str, JsonValue], accepted_states: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    """Keep sealed owners' source navigation without importing their live outcomes."""
    result = copy.deepcopy(memory)
    navigation = result.get("outcome_navigation")
    combined = copy.deepcopy(navigation) if isinstance(navigation, dict) else {}
    outcome_rows = combined.setdefault("outcomes", [])
    condition_rows = combined.setdefault("conditions", [])
    gap_rows = combined.setdefault("open_gaps", [])
    if not all(
        isinstance(rows, list) for rows in (outcome_rows, condition_rows, gap_rows)
    ):
        raise ValueError("Serial memory navigation rows are invalid")
    assert (
        isinstance(outcome_rows, list)
        and isinstance(condition_rows, list)
        and isinstance(gap_rows, list)
    )
    occupied = {
        str(row[key])
        for key, rows in (
            ("outcome_id", outcome_rows),
            ("condition_id", condition_rows),
        )
        for row in rows
        if isinstance(row, dict) and key in row
    }
    requests = result.get("requests", [])
    request_rows = (
        [item for item in requests if isinstance(item, str)]
        if isinstance(requests, list)
        else []
    )
    owners: set[tuple[str, str]] = set()

    def append_unique(
        rows: list[JsonValue], row: dict[str, JsonValue], key: str
    ) -> None:
        identity = str(row[key])
        if identity in occupied:
            if row not in rows:
                raise ValueError("Serial memory qualified identity collides")
            return
        occupied.add(identity)
        rows.append(row)

    for raw in accepted_states:
        state = _SessionState.model_validate(raw)
        owner = (state.owner, state.assignment_id)
        if state.accepted is None or owner in owners:
            raise ValueError("Serial memory needs unique accepted owners")
        owners.add(owner)
        current_requests = state.session_research.get("requests", [])
        if isinstance(current_requests, list):
            request_rows.extend(
                item for item in current_requests if isinstance(item, str)
            )
        local = state.session_research.get("outcome_navigation")
        if not state.outcome_map.get("outcomes") or not isinstance(local, dict):
            continue
        ids: dict[str, str] = {}

        def qualify(kind: str, identity: str) -> str:
            return "s" + _digest([*owner, kind, identity])[:63]

        rows = local.get("outcomes", [])
        if not isinstance(rows, list):
            raise ValueError("Serial memory outcomes are invalid")
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("Serial memory outcome is invalid")
            outcome = RequestedOutcome.model_validate(
                {**row, "question_ids": ["prior"]}
            )
            if outcome.outcome_id in ids:
                raise ValueError("Serial memory outcome identity repeats")
            qualified = qualify("outcome", outcome.outcome_id)
            ids[outcome.outcome_id] = qualified
            append_unique(
                outcome_rows,
                {
                    **outcome.model_dump(mode="json", exclude={"question_ids"}),
                    "outcome_id": qualified,
                },
                "outcome_id",
            )
        rows = local.get("conditions", [])
        if not isinstance(rows, list):
            raise ValueError("Serial memory conditions are invalid")
        for row in rows:
            condition = OutcomeCondition.model_validate(row)
            if set(condition.outcome_ids) - ids.keys():
                raise ValueError("Serial memory condition has unknown local outcomes")
            append_unique(
                condition_rows,
                {
                    **condition.model_dump(mode="json"),
                    "condition_id": qualify("condition", condition.condition_id),
                    "outcome_ids": [
                        ids[identity] for identity in condition.outcome_ids
                    ],
                },
                "condition_id",
            )
        rows = local.get("open_gaps", [])
        if not isinstance(rows, list):
            raise ValueError("Serial memory source gaps are invalid")
        for row in rows:
            if (
                not isinstance(row, dict)
                or row.get("outcome_id") not in ids
                or not isinstance(row.get("gap"), str)
            ):
                raise ValueError("Serial memory source gap has no local outcome")
            gap = {"outcome_id": ids[str(row["outcome_id"])], "gap": row["gap"]}
            if gap not in gap_rows:
                gap_rows.append(gap)
        combined["notice"] = local.get(
            "notice",
            "Prior findings are navigation only; reassess the actual originals and facts.",
        )
    result["requests"] = list(dict.fromkeys(request_rows))
    if outcome_rows:
        result["outcome_navigation"] = combined
    return result


def _publication_gap(
    answer: str,
    call_id: str | None,
    context: RunContext,
    ledger: EvidenceLedger,
    requirements: AuthorityRequirements,
    reviews: LegalSourceReviews,
    *,
    requires_sources: bool,
) -> ToolOutcome | None:
    numbers = extract_citation_numbers(answer)
    unknown = sorted(set(numbers) - ledger.citation_mapping().keys())
    if unknown:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Use recorded original citation numbers.",
            data={"unknown_citations": unknown},
        )
    delivered = ledger.completely_delivered(call_id or "")
    undelivered = [n for n in numbers if n not in delivered]
    if undelivered:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Read cited originals that have not reached the model.",
            data={"undelivered_citations": undelivered},
        )
    if requires_sources and not numbers:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="A legal answer needs recorded original citations. Retrieve the operative source or disclose the precise gap using submit_partial_answer.",
            data={"missing": "original legal evidence"},
        )
    gap = requirements.publication_gap(
        answer,
        call_id,
        context,
        ledger,
        native_gap=native_named_authority_gap(
            answer, ledger, strict_reference_boundaries=True
        ),
    )
    if gap is not None:
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Read and cite the named governing original beside its actual legal assertion; a lower source's reference does not supply that original.",
            data=gap,
        )
    return reviews.publication_gap(answer, call_id or "", context, ledger)


def validate_serial_session_answer(
    *,
    snapshot: dict[str, JsonValue],
    outer_context: RunContext,
    request: str,
    scenario_request: str,
    history: str,
    ledger: EvidenceLedger,
    body: str,
    call_id: str,
    status: OutcomeStatus,
) -> tuple[dict[str, JsonValue], ToolOutcome | None]:
    """Replay a sealed serial owner's policy without a model or source acquisition."""
    state = _SessionState.model_validate(snapshot.get("serial_experimental_session"))
    full_history = serial_session_history(history, scenario_request)
    if (
        state.owner,
        state.assignment_id,
        state.scope_hash,
        state.request_hash,
        state.scenario_hash,
        state.history_hash,
    ) != (
        outer_context.services.get("task_id"),
        outer_context.services.get("assignment_id"),
        _digest(outer_context.scope),
        _digest(request),
        _digest(scenario_request),
        _digest(full_history),
    ):
        raise ValueError("Serial publication owner, assignment or request changed")
    accepted = state.accepted
    if (
        accepted is None
        or (accepted.answer_hash, accepted.model_call_id, accepted.status)
        != (
            hashlib.sha256(body.encode()).hexdigest(),
            call_id,
            status,
        )
        or snapshot.get("last_draft") != body
    ):
        raise ValueError("Serial publication body or accepted call changed")
    historical = EvidenceLedger()
    evidence = snapshot.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("Serial session evidence is missing")
    historical.restore(evidence, outer_context)
    SerialExperimentalSession._check_historical_ledger(
        historical,
        ledger,
        required_call=call_id if extract_citation_numbers(body) else None,
    )
    context = RunContext(
        run_id=outer_context.run_id,
        language=state.public_profile.language,
        scope=outer_context.scope,
        services={
            "task_id": state.owner,
            "research_profile": "experimental",
            "experimental_parallel": False,
            "native_citation_label": state.public_profile.notifications[
                "native_citation"
            ][0],
        },
        budget=outer_context.budget,
        deadline=outer_context.deadline,
        research_deadline=outer_context.research_deadline,
        cancelled=outer_context.is_cancelled,
        corpus_only=outer_context.corpus_only,
    )
    requirements = AuthorityRequirements(context, request)
    requirements.restore(state.authority_requirements, context, request)
    reviews = LegalSourceReviews(context, request)
    reviews.restore(state.legal_source_reviews, context, request, ledger)
    outcomes = OutcomeMap(
        initial_questions(request),
        context,
        factual_context=request + "\n" + full_history,
        detailed_fact_errors=True,
    )
    outcomes.restore(state.outcome_map, ledger)
    gap = None
    if accepted.kind == "clarification":
        if not public_narration_valid("Clarification", body, context):
            raise ValueError("Accepted clarification changed")
    else:
        gap = _publication_gap(
            body,
            call_id,
            context,
            ledger,
            requirements,
            reviews,
            requires_sources=accepted.requires_sources,
        )
    return state.model_dump(mode="json"), gap


class SerialExperimentalSession:
    def __init__(
        self,
        *,
        outer_context: RunContext,
        request: str,
        scenario_request: str,
        history: str,
        ledger: EvidenceLedger,
        llm: LLM,
        reasoning_effort: ReasoningEffort,
        token_counter: Callable[[str], int],
        capability_factory: CapabilityFactory,
        verify: Verification,
        user_identity: LLMUserIdentity | None = None,
        on_receipt: Callable[[ToolReceipt], None] | None = None,
        checkpoint_callback: Callable[[dict[str, JsonValue]], None] | None = None,
        progress: ProgressReporter | None = None,
        allow_external: bool = False,
        notifications: dict[str, list[str]] | None = None,
    ) -> None:
        owner, assignment = (
            outer_context.services.get("task_id"),
            outer_context.services.get("assignment_id"),
        )
        if (
            outer_context.depth < 1
            or not isinstance(owner, str)
            or not owner
            or not isinstance(assignment, str)
            or not assignment
        ):
            raise ValueError("A serial session requires its independent task owner")
        self.outer_context, self.request, self.scenario_request = (
            outer_context,
            request,
            scenario_request,
        )
        self.owner, self.assignment_id, self.ledger = owner, assignment, ledger
        self.prior_history = history
        self.history = serial_session_history(history, scenario_request)
        services: dict[str, object] = {
            "lean_native_mode": True,
            "research_profile": "experimental",
            "experimental_parallel": False,
            "independent_question_mode": False,
            "task_id": owner,
            "evidence": ledger,
        }
        for name in (
            "assistant_instructions",
            "session_research",
            "shared_reads",
            "native_citation_label",
        ):
            value = outer_context.services.get(name)
            if value is not None:
                services[name] = (
                    copy.deepcopy(value)
                    if name in {"assistant_instructions", "session_research"}
                    else value
                )
        self.context = RunContext(
            run_id=outer_context.run_id,
            language=outer_context.language,
            scope=outer_context.scope,
            services=services,
            budget=outer_context.budget,
            deadline=outer_context.deadline,
            research_deadline=outer_context.research_deadline,
            cancelled=outer_context.is_cancelled,
            corpus_only=outer_context.corpus_only,
            depth=0,
            max_depth=outer_context.max_depth,
        )
        self.progress, self._checkpoint_callback = progress, checkpoint_callback
        self._first_decision, self._standalone_answer_call = True, False
        self._allow_external, self._external_requested = (
            allow_external,
            not outer_context.corpus_only,
        )
        self._clarification: str | None = None
        self._partial: str | None = None
        self._answer_requires_sources = True
        self._accepted: _Accepted | None = None
        self._accepted_state: _SessionState | None = None
        self.profile = LanguageProfile(
            language=self.context.language,
            notifications=copy.deepcopy(notifications)
            if notifications is not None
            else localized_notifications(self.context.language),
            external_requested=self._external_requested,
        )
        questions = initial_questions(request)
        self.scenario = ScenarioState(questions, frozen=True)
        self.research = ResearchState(
            questions, self.context, require_need_bindings=False
        )
        self.outcomes = OutcomeMap(
            questions,
            self.context,
            factual_context=request + "\n" + self.history,
            detailed_fact_errors=True,
        )
        self.requirements = AuthorityRequirements(self.context, request)
        self.reviews = LegalSourceReviews(self.context, request)
        self.context.services.update(
            scenario_state=self.scenario,
            research_state=self.research,
            outcome_map=self.outcomes,
            authority_requirements=self.requirements,
            legal_source_reviews=self.reviews,
        )
        parent_scenario = outer_context.services.get("scenario_state")
        if isinstance(parent_scenario, ScenarioState):
            facts = parent_scenario.snapshot().get("facts", [])
            if isinstance(facts, list):
                self.scenario.record([], [str(fact) for fact in facts])
        if progress is not None:
            self.context.services["progress"] = progress
        self.context.services["verify_claim"] = lambda args, child: verify(child, args)
        common_specs = capability_factory(self.context)
        self.registry = CapabilityRegistry()
        self.context.services["registry"] = self.registry
        self.model = ResearchModel(
            llm,
            self.context,
            user_identity=user_identity,
            reasoning_effort=reasoning_effort,
            history=self.history,
            token_counter=token_counter,
            lean_native_mode=True,
            research_llm=None,
        )

        def researcher(
            task: str, child: RunContext, updates: Callable[[], list[str]]
        ) -> ToolOutcome:
            child.services["scenario_state"] = ScenarioState([task], frozen=True)
            child.services["search_message_history"] = [
                ChatMessageSimple(
                    message=request,
                    token_count=token_counter(request),
                    message_type=MessageType.USER,
                ),
                ChatMessageSimple(
                    message=task,
                    token_count=token_counter(task),
                    message_type=MessageType.USER,
                ),
            ]
            child_model = ResearchModel(
                llm,
                child,
                user_identity=user_identity,
                reasoning_effort=reasoning_effort,
                history=self.history,
                updates=updates,
                token_counter=token_counter,
                lean_native_mode=True,
                research_llm=None,
            )
            child_harness = Harness(
                request=task,
                context=child,
                registry=self.registry,
                decide=child_model.decide,
                evidence=ledger,
                on_receipt=on_receipt,
                progress=progress,
                report_terminal=False,
                max_workers=2,
            )
            result = child_harness.run()
            return ToolOutcome(
                status=result.status,
                summary=(result.answer or "Research incomplete")[:12000],
                data={
                    "evidence_numbers": sorted(
                        {n for row in result.receipts for n in row.evidence_ids}
                    ),
                    "questions": result.questions,
                    "facts": result.facts,
                },
            )

        self.workers = WorkerPool(self.context, researcher, progress=progress)
        for spec in [
            *common_specs,
            *self.workers.tool_specs(),
            *build_research_specs(self.research, ledger),
            *self._terminal_specs(),
        ]:
            self.registry.register(spec)
        self.model.pending_tasks = lambda: [
            task.model_dump(mode="json") for task in self.workers.list()
        ]
        self.context.services["wait_for_task_change"] = self.workers.wait_for_change
        self.harness = Harness(
            request=request,
            context=self.context,
            registry=self.registry,
            decide=self.model.decide,
            evidence=ledger,
            on_receipt=on_receipt,
            checkpoint=lambda _: self._save(),
            progress=progress,
            draft_guard=lambda answer: self.publication_gap(
                answer, self.model.last_call_id
            ),
            partial_submission=lambda: self._clarification or self._partial,
            report_terminal=False,
            on_decision=self._on_decision,
            report_started=False,
        )
        for spec in build_core_specs(self.registry, ledger, self.snapshot):
            self.registry.register(spec)
        memory = self.context.services.get("session_research")
        reused = (
            memory.get("reused_evidence_numbers", [])
            if isinstance(memory, dict)
            else []
        )
        if isinstance(reused, list):
            for number in reused:
                item = ledger.get(number) if type(number) is int else None
                if item is not None:
                    self.harness.evidence_working_set.remember(
                        number, 0, len(item.text)
                    )

    def _on_decision(self, decision: Decision) -> None:
        self._standalone_answer_call = (
            len(decision.calls) == 1 and decision.calls[0].name == "submit_answer"
        )
        for call in decision.calls:
            language = call.arguments.get("_language")
            if self.profile.language == "und" and isinstance(language, str):
                try:
                    candidate = LanguageProfile(
                        language=language,
                        notifications=localized_notifications(language),
                        requires_sources=self.profile.requires_sources,
                        external_requested=self.profile.external_requested,
                    )
                except ValueError:
                    continue
                self.profile.language, self.profile.notifications = (
                    candidate.language,
                    candidate.notifications,
                )
                self.context.language = candidate.language
            notifications = call.arguments.get("_notifications")
            if isinstance(notifications, dict):
                for phase, pair in notifications.items():
                    if (
                        phase in self.profile.notifications
                        and isinstance(pair, list)
                        and len(pair) == 2
                        and all(isinstance(word, str) and word.strip() for word in pair)
                    ):
                        title, message = map(str, pair)
                        if public_narration_valid(title, message, self.context):
                            self.profile.notifications[phase] = [
                                title[:240],
                                message[:1600],
                            ]
            if (
                self._first_decision
                and call.arguments.get("_external_requested") is True
            ):
                self._external_requested = True
        self.profile.external_requested = self._external_requested
        self.context.corpus_only = not (
            self._allow_external and self._external_requested
        )
        self.context.services["native_citation_label"] = self.profile.notifications[
            "native_citation"
        ][0]
        self._first_decision = False
        if self.progress is not None:
            report_source_deliveries(
                self.ledger,
                self.model.last_call_id,
                self.context,
                self.progress,
                self.profile.notifications["tools"],
            )

    def publication_gap(
        self, answer: str, call_id: str | None, *, requires_sources: bool = True
    ) -> ToolOutcome | None:
        return _publication_gap(
            answer,
            call_id,
            self.context,
            self.ledger,
            self.requirements,
            self.reviews,
            requires_sources=requires_sources,
        )

    def _authority_gap(
        self, answer: str, call_id: str | None
    ) -> dict[str, JsonValue] | None:
        gap = native_named_authority_gap(
            answer, self.ledger, strict_reference_boundaries=True
        )
        return self.requirements.publication_gap(
            answer, call_id, self.context, self.ledger, native_gap=gap
        )

    def _terminal_specs(self) -> list[ToolSpec]:
        def ask(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
            if child.depth:
                return ToolOutcome(
                    status=OutcomeStatus.DENIED,
                    summary="Preserve completed independent answers; retain a precise missing fact beside its conditional outcome.",
                )
            text = str(args["question"]).strip()
            if not public_narration_valid("Clarification", text, child):
                return ToolOutcome(
                    status=OutcomeStatus.INVALID,
                    summary="Ask a concrete user fact without internal details.",
                )
            self._clarification = text
            return ToolOutcome(
                status=OutcomeStatus.FOUND, summary="User clarification requested"
            )

        def submit(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
            if child.depth or not self._standalone_answer_call:
                return ToolOutcome(
                    status=OutcomeStatus.DENIED,
                    summary="Only the coordinator may submit a complete answer, on its own.",
                )
            candidate = str(args["answer"]).strip()
            if not candidate:
                return ToolOutcome(
                    status=OutcomeStatus.INVALID, summary="Supply an answer."
                )
            requires_sources = args["basis"] == "originals"
            gap = self.publication_gap(
                candidate, self.model.last_call_id, requires_sources=requires_sources
            )
            self.harness.last_draft, self.harness.publication_gap = candidate, gap
            if gap is not None:
                return gap
            self._answer_requires_sources = requires_sources
            self.context.services["submitted_answer"] = candidate
            return ToolOutcome(
                status=OutcomeStatus.FOUND, summary="Complete answer submitted"
            )

        def partial(args: dict[str, JsonValue], child: RunContext) -> ToolOutcome:
            if child.depth:
                return ToolOutcome(
                    status=OutcomeStatus.DENIED,
                    summary="Return available originals and precise gaps to the coordinator; only it can publish.",
                )
            candidate = str(args["answer"]).strip()
            gap = self.publication_gap(
                candidate,
                self.model.last_call_id,
                requires_sources=bool(extract_citation_numbers(candidate)),
            )
            if gap is None:
                authority = self._authority_gap(candidate, self.model.last_call_id)
                if authority is not None:
                    gap = ToolOutcome(
                        status=OutcomeStatus.PARTIAL,
                        summary="A precise missing-original notice cannot assert an unsupported statutory result.",
                        data=authority,
                    )
            self.harness.last_draft, self.harness.publication_gap = candidate, gap
            if gap is not None:
                return gap
            self._partial = candidate
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL,
                summary="Precise supported partial answer submitted",
            )

        return [
            ToolSpec(
                name="ask_user",
                description="Ask a missing user fact that materially changes the outcome; publish the question and end this turn. Use source tools for missing legal text. Call on its own.",
                parameters={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 1600,
                        }
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
                handler=ask,
                parallel_safe=False,
                consumes_tool_budget=False,
            ),
            ToolSpec(
                name="submit_answer",
                description="Publish a complete answer and end this turn, on its own. basis=conversation only for social dialogue with no legal claims; scenario only for supplied facts or arithmetic with no legal effects; originals for legal answers supported by fully delivered original citations. Otherwise research or ask_user.",
                parameters={
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "minLength": 1},
                        "basis": {
                            "type": "string",
                            "enum": ["conversation", "scenario", "originals"],
                        },
                    },
                    "required": ["answer", "basis"],
                    "additionalProperties": False,
                },
                handler=submit,
                parallel_safe=False,
                consumes_tool_budget=False,
            ),
            ToolSpec(
                name="submit_partial_answer",
                description="End this turn with supported parts and a precise unresolved source gap. Cite each supported legal assertion. Do not turn missing evidence into a claim that no law exists. Call on its own.",
                parameters={
                    "type": "object",
                    "properties": {"answer": {"type": "string", "minLength": 1}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
                handler=partial,
                parallel_safe=False,
                consumes_tool_budget=False,
            ),
        ]

    def _state(self) -> _SessionState:
        if self._accepted_state is not None:
            return self._accepted_state.model_copy(deep=True)
        return _SessionState(
            owner=self.owner,
            assignment_id=self.assignment_id,
            scope_hash=_digest(self.context.scope),
            request_hash=_digest(self.request),
            scenario_hash=_digest(self.scenario_request),
            history_hash=_digest(self.history),
            scenario=self.scenario.snapshot(),
            outcome_map=self.outcomes.export(),
            authority_requirements=self.requirements.export(),
            legal_source_reviews=self.reviews.export(),
            session_research=session_research_checkpoint(self.context, self.request),
            native_coordinator_sampling=self.model.native_sampling_snapshot(),
            public_profile=self.profile.model_copy(deep=True),
            accepted=self._accepted,
        )

    def snapshot(self) -> dict[str, JsonValue]:
        return {
            **self.harness.snapshot(),
            "serial_experimental_session": self._state().model_dump(mode="json"),
            "workers": self.workers.export(),
        }

    def source_state(self) -> dict[str, JsonValue]:
        return self._state().model_dump(mode="json")

    def _save(self) -> None:
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(self.snapshot())

    def restore(self, snapshot: dict[str, JsonValue]) -> None:
        state = self._validate_state(snapshot)
        historical = EvidenceLedger()
        evidence = snapshot.get("evidence")
        if not isinstance(evidence, dict):
            raise ValueError("Serial session evidence is missing")
        historical.restore(evidence, self.context)
        self._check_historical_ledger(
            historical,
            self.ledger,
            required_call=state.accepted.model_call_id
            if state.accepted is not None
            and extract_citation_numbers(str(snapshot.get("last_draft", "")))
            else None,
        )
        self.harness.evidence = historical
        try:
            self.harness.restore(snapshot)
        finally:
            self.harness.evidence = self.ledger
            self.context.services["evidence"] = self.ledger
        facts = state.scenario.get("facts")
        if (
            state.scenario.get("questions") != initial_questions(self.request)
            or not isinstance(facts, list)
            or any(not isinstance(fact, str) for fact in facts)
        ):
            raise ValueError("Serial scenario questions or facts changed")
        self.scenario.record([], cast(list[str], facts))
        self.outcomes.restore(state.outcome_map, self.ledger)
        self.requirements.restore(
            state.authority_requirements, self.context, self.request
        )
        self.reviews.restore(
            state.legal_source_reviews, self.context, self.request, self.ledger
        )
        self.context.services["session_research"] = copy.deepcopy(
            state.session_research
        )
        self.model.restore_native_sampling(
            {"native_coordinator_sampling": state.native_coordinator_sampling}
        )
        workers = snapshot.get("workers")
        if not isinstance(workers, dict):
            raise ValueError("Serial optional worker state is missing")
        self.workers.restore(workers)
        self._accepted = state.accepted
        self.profile = state.public_profile.model_copy(deep=True)
        self.context.language = self.profile.language
        self._external_requested = self.profile.external_requested
        self.context.corpus_only = not (
            self._allow_external and self._external_requested
        )
        self._first_decision = not bool(self.harness.turns)
        self.context.services["native_citation_label"] = self.profile.notifications[
            "native_citation"
        ][0]
        if state.accepted is not None:
            self._accepted_state = state.model_copy(deep=True)
            self.model.last_call_id = state.accepted.model_call_id
            self.context.services["last_model_call_id"] = state.accepted.model_call_id

    def _validate_state(self, snapshot: dict[str, JsonValue]) -> _SessionState:
        state = _SessionState.model_validate(
            snapshot.get("serial_experimental_session")
        )
        if (
            state.owner,
            state.assignment_id,
            state.scope_hash,
            state.request_hash,
            state.scenario_hash,
            state.history_hash,
        ) != (
            self.owner,
            self.assignment_id,
            _digest(self.context.scope),
            _digest(self.request),
            _digest(self.scenario_request),
            _digest(self.history),
        ):
            raise ValueError("Serial session owner, scenario, request or scope changed")
        return state

    @staticmethod
    def _check_historical_ledger(
        historical: EvidenceLedger,
        current: EvidenceLedger,
        *,
        required_call: str | None = None,
    ) -> None:
        for number in historical.citation_numbers():
            saved, actual = historical.get(number), current.get(number)
            if (
                saved is None
                or actual is None
                or (saved.source_id, saved.chunk_id, saved.text_hash, saved.text)
                != (actual.source_id, actual.chunk_id, actual.text_hash, actual.text)
            ):
                raise ValueError(
                    "Serial session original differs from the canonical ledger"
                )
            if not _historical_identity_matches(
                _identity_metadata(saved.metadata), _identity_metadata(actual.metadata)
            ):
                raise ValueError(
                    "Serial session original provenance or version changed"
                )
        exported = current.export()
        saved = historical.export()
        rows = exported["deliveries"]
        saved_rows = saved["deliveries"]
        if not isinstance(rows, list) or not isinstance(saved_rows, list):
            raise ValueError("Serial session historical delivery is invalid")
        required_calls = set(cast(list[str], saved["pinned_delivery_calls"]))
        if required_call is not None:
            required_calls.add(required_call)
        for row in saved_rows:
            if not isinstance(row, dict) or not isinstance(row.get("records"), list):
                raise ValueError("Serial session historical delivery is invalid")
            for record in cast(list[JsonValue], row["records"]):
                if not isinstance(record, dict):
                    raise ValueError(
                        "Serial session historical delivery range is invalid"
                    )
                citation, start, end = (
                    record.get(key) for key in ("citation", "start_char", "end_char")
                )
                item = historical.get(citation) if type(citation) is int else None
                if (
                    item is None
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(item.text)
                    or record.get("source_id") != item.source_id
                    or record.get("chunk_id") != item.chunk_id
                    or record.get("text_hash") != item.text_hash
                    or record.get("passage_hash")
                    != hashlib.sha256(item.text[start:end].encode()).hexdigest()
                    or record.get("complete")
                    is not (start == 0 and end == len(item.text))
                ):
                    raise ValueError("Serial session historical delivery range changed")
            if row.get("call_id") in required_calls and row not in rows:
                raise ValueError(
                    "Serial session required historical delivery is missing"
                )
        if required_calls - {
            str(row["call_id"]) for row in saved_rows if isinstance(row, dict)
        }:
            raise ValueError("Serial session required historical delivery is missing")
        pins = exported["pinned_delivery_calls"]
        if not isinstance(pins, list) or set(
            cast(list[str], saved["pinned_delivery_calls"])
        ) - set(pins):
            raise ValueError("Serial session historical delivery pin is missing")

    def run(self) -> ToolOutcome:
        try:
            if self._accepted is not None:
                body = self.harness.last_draft or ""
                gap = self.validate_accepted(
                    body, self._accepted.model_call_id, self._accepted.status
                )
                if gap is not None:
                    raise ValueError(
                        "Accepted serial publication is no longer supported"
                    )
                return ToolOutcome(
                    status=self._accepted.status,
                    summary=body,
                    data={
                        "questions": list(initial_questions(self.request)),
                        "facts": self.scenario.snapshot()["facts"],
                        "evidence_numbers": list(extract_citation_numbers(body)),
                        "serial_session_state": self.source_state(),
                    },
                )
            result = self.harness.run()
            body = self._clarification or self._partial or result.answer or ""
            if not body.strip():
                return ToolOutcome(
                    status=result.status,
                    summary="",
                    data={"questions": result.questions, "facts": result.facts},
                )
            kind: Literal["answer", "partial", "clarification"] = (
                "clarification"
                if self._clarification
                else "partial"
                if self._partial
                else "answer"
            )
            requires = (
                False
                if kind == "clarification"
                else bool(extract_citation_numbers(body))
                if kind == "partial"
                else self._answer_requires_sources
            )
            status = OutcomeStatus.PARTIAL if kind != "answer" else result.status
            self._accepted = _Accepted(
                answer_hash=hashlib.sha256(body.encode()).hexdigest(),
                model_call_id=self.model.last_call_id or "",
                status=status,
                kind=kind,
                requires_sources=requires,
            )
            self.harness.last_draft = body
            self._accepted_state = self._state()
            self._save()
            return ToolOutcome(
                status=status,
                summary=body,
                data={
                    "questions": result.questions,
                    "facts": result.facts,
                    "evidence_numbers": list(extract_citation_numbers(body)),
                    "serial_session_state": self.source_state(),
                },
            )
        finally:
            self.workers.close()

    def validate_accepted(
        self, body: str, call_id: str, status: OutcomeStatus
    ) -> ToolOutcome | None:
        _state, gap = validate_serial_session_answer(
            snapshot=self.snapshot(),
            outer_context=self.outer_context,
            request=self.request,
            scenario_request=self.scenario_request,
            history=self.prior_history,
            ledger=self.ledger,
            body=body,
            call_id=call_id,
            status=status,
        )
        return gap
