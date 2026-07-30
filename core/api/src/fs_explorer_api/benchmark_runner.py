"""
Headless agent runs and LLM-judge scoring for the benchmark system.

Deliberately NOT a refactor of `server.py`'s `_run_fresh_session` into a
shared function: that code drives the live, latency-sensitive `/ws/explore`
chat path and supports resuming an interrupted run via `runs.py`'s TTL
registry. Adding a websocket-agnostic seam there risks that fragile,
already-tested resume behavior for the sake of avoiding some duplication.
Instead, `run_agentic_session` below reuses the same underlying building
blocks (`new_workflow`, `ExplorationTrace`, `set_index_context`,
`agent.stream_final_answer`, `extract_cited_sources`) with a parallel,
websocket-free orchestration loop. See
docs/superpowers/specs/2026-07-23-agentic-benchmark-design.md for the full
design.
"""

from __future__ import annotations

import html
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Literal, cast

from .agent import (
    CHUNK_BEARING_TOOLS,
    LLMCallStats,
    RetrievalStats,
    clear_index_context,
    set_index_context,
    set_search_flags,
)
from .exploration_trace import ExplorationTrace, extract_cited_sources
from .llm import get_llm_client
from .llm.base import ChatTurn
from .llm.profile import LLMProfile, LLMRole, LLMRoleConfig, load_llm_profile
from .models import JudgmentResult
from .orchestration_prompts import (
    APPLICATION_TASK_SYSTEM_PROMPT,
    EVIDENCE_WORKER_SYSTEM_PROMPT,
    FINAL_SYNTHESIS_SYSTEM_PROMPT,
    INTEGRATION_TASK_SYSTEM_PROMPT,
    SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT,
    TASK_REVIEW_SYSTEM_PROMPT,
    build_global_planner_prompt,
    build_task_coordinator_prompt,
)
from .workflow import (
    AskHumanEvent,
    GoDeeperEvent,
    HumanAnswerEvent,
    InputEvent,
    ResearchProgressEvent,
    ToolBatchEvent,
    ToolCallEvent,
    get_run_agent,
    new_workflow,
)
from fs_explorer_shared.index_config import resolve_database_url
from fs_explorer_shared.storage import PostgresStorage

logger = logging.getLogger(__name__)

BenchmarkProfileMode = Literal["candidate_all_roles", "production_roles"]
_BENCHMARK_PROFILE_MODES = frozenset({"candidate_all_roles", "production_roles"})
_PLAN_TRACE_SCHEMA_VERSION = 1
_MAX_PLAN_TRACE_CHARS = 100_000


@dataclass
class BenchmarkRunResult:
    """Everything a benchmark item needs to persist about one agent run."""

    final_result: str
    error: str | None
    incomplete: bool
    cited_sources: list[str]
    step_path: list[str]
    stats: dict[str, Any] = field(default_factory=dict)
    plan_trace: dict[str, Any] | None = None
    role_usage: list[dict[str, Any]] = field(default_factory=list)
    cited_evidence: list[dict[str, object]] = field(default_factory=list)


def _record_tool_call(
    event: ToolCallEvent,
    *,
    step_number: int,
    trace: ExplorationTrace,
    index_storage: PostgresStorage,
) -> None:
    """Mirror `server.py`'s `_tool_call_ws_message` trace recording, no WS message built."""
    resolved_document_path: str | None = None
    if event.tool_name == "get_document":
        doc_id = event.tool_input.get("doc_id")
        if isinstance(doc_id, str) and doc_id:
            document = index_storage.get_document(doc_id=doc_id)
            if document and not document["is_deleted"]:
                resolved_document_path = str(document["absolute_path"])
    trace.record_tool_call(
        step_number=step_number,
        tool_name=event.tool_name,
        tool_input=event.tool_input,
        resolved_document_path=resolved_document_path,
    )


def _sum_call_cost(llm_calls: list[LLMCallStats]) -> tuple[str | None, str | None]:
    """Pool per-call billed costs into one run total, preferring provider-reported amounts.

    Returns `(cost_usd, cost_source)`; `cost_source` is `"estimated"` if any
    contributing call was a fallback estimate, otherwise `"provider"` if any
    call reported a cost at all, otherwise `None` (no cost data available,
    e.g. a provider that doesn't report cost).
    """
    total: Decimal | None = None
    saw_estimated = False
    saw_any = False
    for call in llm_calls:
        if call.billed_cost_usd is None:
            continue
        try:
            amount = Decimal(call.billed_cost_usd)
        except InvalidOperation:
            continue
        total = amount if total is None else total + amount
        saw_any = True
        if call.cost_source == "estimated":
            saw_estimated = True
    if not saw_any:
        return None, None
    return str(total), "estimated" if saw_estimated else "provider"


def _role_usage_breakdown(llm_calls: list[LLMCallStats]) -> list[dict[str, Any]]:
    """Aggregate heterogeneous provider calls by role, purpose, provider, and model."""

    grouped: dict[tuple[str, str, str, str], list[LLMCallStats]] = {}
    for call in llm_calls:
        key = (
            call.agent_role or "legacy",
            call.purpose,
            call.provider,
            call.model,
        )
        grouped.setdefault(key, []).append(call)

    rows: list[dict[str, Any]] = []
    for (role, purpose, provider, model), calls in grouped.items():
        cost_usd, cost_source = _sum_call_cost(calls)
        prompt_tokens = sum(call.prompt_tokens for call in calls)
        completion_tokens = sum(call.completion_tokens for call in calls)
        thinking_tokens = sum(call.thinking_tokens for call in calls)
        rows.append(
            {
                "role": role,
                "purpose": purpose,
                "provider": provider,
                "model": model,
                "calls": len(calls),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
                "total_tokens": (prompt_tokens + completion_tokens + thinking_tokens),
                "cached_input_tokens": sum(call.cached_input_tokens for call in calls),
                "cache_write_tokens": sum(call.cache_write_tokens for call in calls),
                "duration_ms": round(sum(call.duration_ms for call in calls), 3),
                "cost_usd": cost_usd,
                "cost_source": cost_source,
            }
        )
    return rows


def _resolve_profile_mode(
    *,
    profile_mode: BenchmarkProfileMode | str | None,
    provider: str | None,
    model: str | None,
) -> BenchmarkProfileMode:
    """Resolve the explicit mode while preserving the legacy optional API."""

    if profile_mode is None:
        resolved = (
            "candidate_all_roles"
            if provider is not None or model is not None
            else "production_roles"
        )
    else:
        resolved = profile_mode.strip().lower()
    if resolved not in _BENCHMARK_PROFILE_MODES:
        raise ValueError(
            "Benchmark profile_mode must be 'candidate_all_roles' or "
            "'production_roles'."
        )

    if resolved == "candidate_all_roles":
        if provider is None or model is None:
            raise ValueError(
                "Benchmark provider and model must be provided together in "
                "candidate_all_roles mode."
            )
    elif provider is not None or model is not None:
        raise ValueError(
            "Benchmark provider/model must be omitted in production_roles mode."
        )
    return cast(BenchmarkProfileMode, resolved)


def _benchmark_llm_profile(
    *,
    provider: str | None,
    model: str | None,
    profile_mode: BenchmarkProfileMode | str | None = None,
) -> LLMProfile:
    """Apply one benchmark candidate to every role without flattening reasoning.

    Benchmark rows represent one provider/model candidate.  When the
    multi-agent feature is enabled, letting role clients fall back to the
    process-wide profile would make every row run the same fixed models while
    the backend still attributes results to the requested candidate.  Snapshot
    the configured role policy and replace only provider/model so planner,
    task, worker, and final retain their independently tuned reasoning levels.
    """

    resolved_mode = _resolve_profile_mode(
        profile_mode=profile_mode,
        provider=provider,
        model=model,
    )
    configured = load_llm_profile()
    if resolved_mode == "production_roles":
        # Pass an immutable snapshot explicitly. This makes one benchmark item
        # reproducible even if process environment changes between queue ticks.
        return configured

    assert provider is not None and model is not None
    candidate_provider = provider.strip().lower()
    candidate_model = model.strip()
    if not candidate_provider or not candidate_model:
        raise ValueError("Benchmark provider and model must not be empty.")

    def candidate(role: LLMRole) -> LLMRoleConfig:
        role_config = configured.for_role(role)
        return LLMRoleConfig(
            provider=candidate_provider,
            model=candidate_model,
            reasoning_effort=role_config.reasoning_effort,
        )

    return LLMProfile(
        planner=candidate("planner"),
        task=candidate("task"),
        worker=candidate("worker"),
        final=candidate("final"),
    )


def _profile_snapshot(profile: LLMProfile) -> dict[str, dict[str, str]]:
    return {
        role: {
            "provider": profile.for_role(role).provider,
            "model": profile.for_role(role).model,
            "reasoning_effort": profile.for_role(role).reasoning_effort,
        }
        for role in cast(tuple[LLMRole, ...], ("planner", "task", "worker", "final"))
    }


def _fingerprint(value: object) -> str:
    if isinstance(value, str):
        encoded = value.encode("utf-8")
    else:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _orchestration_runtime_snapshot(
    execution: dict[str, Any] | None,
) -> dict[str, Any]:
    """Capture compact reproducibility metadata without copying source documents."""

    plan = execution.get("plan") if isinstance(execution, dict) else None
    artifacts = execution.get("task_artifacts") if isinstance(execution, dict) else None
    contract_version = (
        execution.get("contract_version") if isinstance(execution, dict) else None
    )
    if contract_version is None and isinstance(plan, dict):
        contract_version = plan.get("version")
    artifact_fields = (
        sorted(
            {
                str(field)
                for artifact in artifacts
                if isinstance(artifact, dict)
                for field in artifact
            }
        )
        if isinstance(artifacts, list)
        else []
    )
    snapshot: dict[str, Any] = {
        "contract": {
            "trace_schema_version": _PLAN_TRACE_SCHEMA_VERSION,
            "plan_contract_version": contract_version,
            "plan_fields": sorted(str(field) for field in plan)
            if isinstance(plan, dict)
            else [],
            "task_artifact_fields": artifact_fields,
        }
    }
    try:
        prompts = {
            "global_planner": build_global_planner_prompt(
                max_tasks=None,
                max_list_items=None,
            ),
            "task_coordinator": build_task_coordinator_prompt(
                max_assignments_per_wave=None,
                max_worker_rounds=None,
            ),
            "evidence_worker": EVIDENCE_WORKER_SYSTEM_PROMPT,
            "task_review": TASK_REVIEW_SYSTEM_PROMPT,
            "application_task": APPLICATION_TASK_SYSTEM_PROMPT,
            "integration_task": INTEGRATION_TASK_SYSTEM_PROMPT,
            "final_synthesis": FINAL_SYNTHESIS_SYSTEM_PROMPT,
            "scenario_final_synthesis": SCENARIO_FINAL_SYNTHESIS_SYSTEM_PROMPT,
        }
        snapshot["termination_policy"] = "model_controlled_without_count_budgets"
        snapshot["prompts"] = {
            name: {
                "sha256": _fingerprint(prompt),
                "characters": len(prompt),
                "words": len(prompt.split()),
            }
            for name, prompt in prompts.items()
        }
    except Exception as exc:
        # Observability must not make an otherwise runnable benchmark fail.
        snapshot["snapshot_error"] = " ".join(str(exc).split())[:500]
    return snapshot


def _json_safe_object(value: object) -> dict[str, Any] | None:
    if value is None:
        return None
    if callable(value):
        value = cast(Callable[[], object], value)()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        value = cast(Callable[..., object], model_dump)(mode="json")
    if not isinstance(value, dict):
        return None
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError):
        return None
    return decoded if isinstance(decoded, dict) else None


def _fallback_agent_plan_trace(agent: object) -> dict[str, Any] | None:
    """Compatibility bridge until every agent exposes ``benchmark_plan_trace``."""

    result = getattr(agent, "_multi_agent_result", None)
    if result is None:
        return None
    plan = getattr(result, "plan", None)
    plan_payload = plan.model_dump(mode="json") if hasattr(plan, "model_dump") else None
    artifacts = []
    for artifact in getattr(result, "task_artifacts", ()):
        artifacts.append(
            {
                "task_id": getattr(artifact, "task_id", None),
                "status": getattr(
                    getattr(artifact, "status", None),
                    "value",
                    getattr(artifact, "status", None),
                ),
                "covered_success_criteria": list(
                    getattr(artifact, "covered_success_criteria", ())
                ),
                "uncovered_success_criteria": list(
                    getattr(artifact, "uncovered_success_criteria", ())
                ),
                "claim_count": len(getattr(artifact, "claims", ())),
                "conflicts": list(getattr(artifact, "conflicts", ())),
                "gaps": list(getattr(artifact, "gaps", ())),
            }
        )
    return {
        "schema_version": 1,
        "plan": plan_payload,
        "task_artifacts": artifacts,
        "used_plan_fallback": bool(getattr(result, "used_plan_fallback", False)),
        "incomplete": bool(getattr(result, "incomplete", False)),
    }


def _agent_plan_trace(agent: object) -> dict[str, Any] | None:
    for attribute in ("benchmark_plan_trace", "multi_agent_trace"):
        try:
            trace = _json_safe_object(getattr(agent, attribute, None))
        except Exception:
            logger.exception("Failed to snapshot agent benchmark plan trace")
            trace = None
        if trace is not None:
            return trace
    return _fallback_agent_plan_trace(agent)


def _build_plan_trace(
    *,
    agent: object,
    profile_mode: BenchmarkProfileMode,
    profile: LLMProfile,
) -> dict[str, Any]:
    execution = _agent_plan_trace(agent)
    trace: dict[str, Any] = {
        "schema_version": _PLAN_TRACE_SCHEMA_VERSION,
        "profile_mode": profile_mode,
        "role_profile": _profile_snapshot(profile),
        "runtime": _orchestration_runtime_snapshot(execution),
        "execution": execution,
    }
    encoded = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
    if len(encoded) <= _MAX_PLAN_TRACE_CHARS:
        return trace

    execution = trace.get("execution")
    return {
        "schema_version": _PLAN_TRACE_SCHEMA_VERSION,
        "profile_mode": profile_mode,
        "role_profile": trace["role_profile"],
        "runtime": trace["runtime"],
        "execution": {
            "trace_truncated": True,
            "original_characters": len(encoded),
            "sha256": _fingerprint(execution),
        },
    }


async def run_agentic_session(
    *,
    task: str,
    index_folders: list[str],
    database_url: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    profile_mode: BenchmarkProfileMode | str | None = None,
) -> BenchmarkRunResult:
    """Drive one `FsExplorerAgent` run to completion headlessly, indexed-mode only.

    Returns the same statistics a live `/ws/explore` run's `"complete"`
    event carries, without ever touching a WebSocket. A benchmark question
    always resolves to at least one already-indexed directory (see the
    design doc) — there is no raw-filesystem fallback here.
    """
    if not index_folders:
        raise ValueError("run_agentic_session requires at least one index folder")

    resolved_profile_mode = _resolve_profile_mode(
        profile_mode=profile_mode,
        provider=provider,
        model=model,
    )
    effective_profile = _benchmark_llm_profile(
        provider=provider,
        model=model,
        profile_mode=resolved_profile_mode,
    )
    resolved_database_url = resolve_database_url(database_url)
    run_started_at = time.monotonic()
    step_number = 0
    llm_calls: list[LLMCallStats] = []
    retrieval_stats: list[RetrievalStats] = []
    # RetrievalStats.step is the agent's own step counter, which a single
    # ToolBatchAction shares across 2-3 tool calls — not fine-grained
    # enough to match this function's own per-tool_call `step_number`
    # (what `step_path` uses). Since exactly one retrieval_stats fires per
    # chunk-bearing tool call, in the same order those calls are made
    # (including within a batch — see agent.py's call_tools()), this queue
    # lets the final retrieval_steps list use the same step numbering as
    # step_path, by pairing the two lists positionally once the run ends.
    pending_retrieval_step_numbers: list[int] = []

    async def _collect_llm_call(stats: LLMCallStats) -> None:
        llm_calls.append(stats)

    def _collect_retrieval(stats: RetrievalStats) -> None:
        retrieval_stats.append(stats)

    index_storage = PostgresStorage(resolved_database_url)
    handler: Any = None
    try:
        available_index_folders = [
            folder
            for folder in index_folders
            if index_storage.get_corpus_id(folder) is not None
        ]
        if not available_index_folders:
            raise ValueError("No index found for the given folders. Index them first.")

        trace = ExplorationTrace(root_directory=available_index_folders[0])
        clear_index_context()
        set_index_context(available_index_folders, resolved_database_url)
        set_search_flags(enable_semantic=True, enable_metadata=True)

        run_workflow, resource_manager = new_workflow(
            provider=provider,
            model=model,
            temperature=temperature,
            on_llm_call=_collect_llm_call,
            on_retrieval=_collect_retrieval,
            llm_profile=effective_profile,
        )
        agent = get_run_agent(resource_manager)
        handler = run_workflow.run(
            start_event=InputEvent(
                task=task,
                folder=available_index_folders[0],
                use_index=True,
                enable_semantic=True,
                enable_metadata=True,
            )
        )

        async for event in handler.stream_events():
            if isinstance(event, ResearchProgressEvent):
                step_number = max(step_number, event.sequence)
                trace.step_path.append(
                    f"{event.sequence}. {event.kind} "
                    f"({event.task_id or event.agent_id or event.agent_role})"
                )
            elif isinstance(event, ToolCallEvent):
                step_number += 1
                if event.tool_name in CHUNK_BEARING_TOOLS:
                    pending_retrieval_step_numbers.append(step_number)
                _record_tool_call(
                    event,
                    step_number=step_number,
                    trace=trace,
                    index_storage=index_storage,
                )
            elif isinstance(event, ToolBatchEvent):
                for call in event.tool_calls:
                    step_number += 1
                    if call.tool_name in CHUNK_BEARING_TOOLS:
                        pending_retrieval_step_numbers.append(step_number)
                    _record_tool_call(
                        ToolCallEvent(
                            tool_name=call.tool_name,
                            tool_input=call.to_fn_args(),
                            reason=event.reason,
                        ),
                        step_number=step_number,
                        trace=trace,
                        index_storage=index_storage,
                    )
            elif isinstance(event, GoDeeperEvent):
                step_number += 1
                trace.record_go_deeper(
                    step_number=step_number, directory=event.directory
                )
            elif isinstance(event, AskHumanEvent):
                # No human is available in a headless benchmark run. Answer
                # with a fixed fallback so the agent proceeds autonomously
                # instead of the workflow hanging on an event nobody sends.
                step_number += 1
                trace.step_path.append(f"{step_number}. ask_human ({event.question!r})")
                handler.ctx.send_event(
                    HumanAnswerEvent(
                        response=(
                            "No human is available to answer. Use your best "
                            "judgment based on the available evidence and "
                            "provide a final answer."
                        )
                    )
                )

        result = await handler
        final_result = result.final_result or ""
        result_error = result.error

        if not result_error:
            streamed_parts: list[str] = []
            async for chunk in agent.stream_final_answer(fallback_answer=final_result):
                streamed_parts.append(chunk)
            streamed_final = html.unescape("".join(streamed_parts)).strip()
            if streamed_final:
                final_result = streamed_final

        cited_sources = extract_cited_sources(final_result) if not result_error else []
        if not result_error and agent._multi_agent_result is not None:
            cited_evidence = [
                dict(source) for source in agent._multi_agent_result.evidence_sources
            ]
        else:
            cited_evidence = (
                agent.evidence_sources_for_answer(final_result)
                if not result_error
                else []
            )
        usage = agent.token_usage
        cost_usd, cost_source = _sum_call_cost(llm_calls)
        retrieval_steps = []
        for stats in retrieval_stats:
            ws_step = (
                pending_retrieval_step_numbers.pop(0)
                if pending_retrieval_step_numbers
                else stats.step
            )
            retrieval_steps.append(
                {
                    "step": ws_step,
                    "tool_name": stats.tool_name,
                    "chunk_count": stats.chunk_count,
                    "chars": stats.chars,
                    "estimated_tokens": stats.estimated_tokens,
                    "task_id": stats.task_id,
                    "agent_id": stats.agent_id,
                    "sequence": stats.sequence,
                }
            )

        final_call = next(
            (call for call in reversed(llm_calls) if call.agent_role == "final"),
            None,
        )
        role_usage = _role_usage_breakdown(llm_calls)
        plan_trace = _build_plan_trace(
            agent=agent,
            profile_mode=resolved_profile_mode,
            profile=effective_profile,
        )
        stats = {
            "steps": step_number,
            "api_calls": usage.api_calls,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "thinking_tokens": usage.thinking_tokens,
            "total_tokens": usage.total_tokens,
            "tool_result_chars": usage.tool_result_chars,
            "context_summaries": usage.context_summaries,
            "retrieval_steps": retrieval_steps,
            "duration_ms": round((time.monotonic() - run_started_at) * 1000),
            "cost_usd": cost_usd,
            "cost_source": cost_source,
            "model": final_call.model if final_call else agent.final_model,
            "provider": final_call.provider if final_call else provider,
            "profile_mode": resolved_profile_mode,
            "plan_trace": plan_trace,
            "role_usage": role_usage,
        }

        return BenchmarkRunResult(
            final_result=final_result,
            error=result_error,
            incomplete=not result_error
            and (agent.forced_stop or agent.multi_agent_incomplete),
            cited_sources=cited_sources,
            step_path=trace.step_path,
            stats=stats,
            plan_trace=plan_trace,
            role_usage=role_usage,
            cited_evidence=cited_evidence,
        )
    finally:
        try:
            if handler is not None and not handler.is_done():
                await handler.cancel_run()
        except Exception:
            logger.exception("Failed to cancel an interrupted benchmark workflow")
        finally:
            index_storage.close()
            set_search_flags(enable_semantic=False, enable_metadata=False)
            clear_index_context()


# =============================================================================
# LLM-as-judge scoring
# =============================================================================

JUDGE_SYSTEM_PROMPT = """\
You are a strict, consistent grader for a customs-regulations research \
agent's answers. Score the CANDIDATE ANSWER against the REFERENCE ANSWER / \
EXPECTED FACTS on four dimensions, each from 1 (worst) to 5 (best). Use \
exactly these anchors — do not invent your own scale:

Correctness (does it match the reference answer/expected facts?):
  1 = contradicts the reference answer/expected facts, or fabricates a rule
  3 = partially correct with a material gap or a minor factual error
  5 = fully matches the reference answer/expected facts, no fabrication

Groundedness (are claims backed by the server-verified evidence?):
  1 = no supporting evidence, or the evidence contradicts the claim
  3 = evidence supports some but not all material claims
  5 = every material claim is directly supported by supplied evidence

Completeness (does it address the whole question?):
  1 = ignores the actual question
  3 = answers the main question but misses a clearly-relevant exception or cross-reference
  5 = fully addresses the question including relevant exceptions

Clarity (is it well-written and actionable?):
  1 = confusing or contradictory
  3 = serviceable but verbose or unfocused
  5 = direct, well-structured, actionable

Judge only what is given. Do not reward answers for being long. Do not \
penalize a candidate for omitting information the question did not ask for. \
Give a short, specific rationale citing what was right or wrong.\
"""


def _build_judge_prompt(
    *,
    question: str,
    reference_answer: str | None,
    expected_facts: list[str] | None,
    rubric_notes: str | None,
    candidate_answer: str,
    cited_sources: list[str],
    cited_evidence: list[dict[str, object]] | None = None,
) -> str:
    parts = [f"QUESTION:\n{question}"]
    if reference_answer:
        parts.append(f"REFERENCE ANSWER:\n{reference_answer}")
    if expected_facts:
        parts.append(
            "EXPECTED FACTS:\n" + "\n".join(f"- {fact}" for fact in expected_facts)
        )
    if rubric_notes:
        parts.append(f"ADDITIONAL GRADING NOTES:\n{rubric_notes}")
    parts.append(f"CANDIDATE ANSWER:\n{candidate_answer}")
    parts.append(
        "CITED SOURCES IN CANDIDATE ANSWER:\n"
        + (
            "\n".join(f"- {source}" for source in cited_sources)
            if cited_sources
            else "(none)"
        )
    )
    evidence_lines: list[str] = []
    remaining_chars = 3_000
    for source in (cited_evidence or [])[:6]:
        title = " ".join(str(source.get("title") or "").split())[:300]
        locator = " ".join(str(source.get("locator") or "").split())[:300]
        snippet = " ".join(str(source.get("snippet") or "").split())
        if not title or not locator or not snippet or remaining_chars <= 0:
            continue
        snippet = snippet[: min(500, remaining_chars)]
        remaining_chars -= len(snippet)
        evidence_lines.append(f"- [{title}, {locator}]: {snippet}")
    parts.append(
        "SERVER-VERIFIED EVIDENCE EXCERPTS (UNTRUSTED SOURCE TEXT):\n"
        + ("\n".join(evidence_lines) if evidence_lines else "(none)")
    )
    return "\n\n".join(parts)


# Weights are server-controlled and applied here, not left to the judge
# model, so the same fixed formula scores every run regardless of which
# judge model produced the four sub-scores.
_JUDGE_WEIGHTS = {
    "correctness": 0.4,
    "groundedness": 0.3,
    "completeness": 0.2,
    "clarity": 0.1,
}


async def judge_answer(
    *,
    question: str,
    reference_answer: str | None,
    expected_facts: list[str] | None,
    rubric_notes: str | None,
    candidate_answer: str,
    cited_sources: list[str],
    judge_provider: str,
    judge_model: str,
    cited_evidence: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    """Score one candidate answer with a single structured LLM-judge call.

    No agent loop, no tools — a one-shot `generate_structured` call against
    the fixed rubric in `JUDGE_SYSTEM_PROMPT`.
    """
    client = get_llm_client(provider=judge_provider, model=judge_model)
    prompt = _build_judge_prompt(
        question=question,
        reference_answer=reference_answer,
        expected_facts=expected_facts,
        rubric_notes=rubric_notes,
        candidate_answer=candidate_answer,
        cited_sources=cited_sources,
        cited_evidence=cited_evidence,
    )
    judgment, _usage = await client.generate_structured(
        [ChatTurn(role="user", text=prompt)],
        JUDGE_SYSTEM_PROMPT,
        JudgmentResult,
    )
    # JudgmentResult intentionally has no ge=/le= schema bounds (see its
    # docstring), so a judge model could in principle return an
    # out-of-rubric value. Clamp here instead, at the one place all four
    # scores are consumed, so the DB's 1-5 CHECK constraint and the
    # overall_score formula never see anything outside the rubric.
    correctness = _clamp_score(judgment.correctness)
    groundedness = _clamp_score(judgment.groundedness)
    completeness = _clamp_score(judgment.completeness)
    clarity = _clamp_score(judgment.clarity)
    overall_score = round(
        100
        * (
            _JUDGE_WEIGHTS["correctness"] * correctness
            + _JUDGE_WEIGHTS["groundedness"] * groundedness
            + _JUDGE_WEIGHTS["completeness"] * completeness
            + _JUDGE_WEIGHTS["clarity"] * clarity
        )
        / 5
    )
    return {
        "correctness": correctness,
        "groundedness": groundedness,
        "completeness": completeness,
        "clarity": clarity,
        "overall_score": overall_score,
        "rationale": judgment.rationale,
    }


def _clamp_score(value: int) -> int:
    return max(1, min(5, value))
