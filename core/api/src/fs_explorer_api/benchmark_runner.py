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
import time
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from .agent import (
    LLMCallStats,
    clear_index_context,
    set_index_context,
    set_search_flags,
)
from .exploration_trace import ExplorationTrace, extract_cited_sources
from .llm import get_llm_client
from .llm.base import ChatTurn
from .models import JudgmentResult
from .workflow import (
    AskHumanEvent,
    GoDeeperEvent,
    HumanAnswerEvent,
    InputEvent,
    ToolBatchEvent,
    ToolCallEvent,
    get_run_agent,
    new_workflow,
)
from fs_explorer_shared.index_config import resolve_database_url
from fs_explorer_shared.storage import PostgresStorage


@dataclass
class BenchmarkRunResult:
    """Everything a benchmark item needs to persist about one agent run."""

    final_result: str
    error: str | None
    incomplete: bool
    cited_sources: list[str]
    step_path: list[str]
    stats: dict[str, Any] = field(default_factory=dict)


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


async def run_agentic_session(
    *,
    task: str,
    index_folders: list[str],
    database_url: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> BenchmarkRunResult:
    """Drive one `FsExplorerAgent` run to completion headlessly, indexed-mode only.

    Returns the same statistics a live `/ws/explore` run's `"complete"`
    event carries, without ever touching a WebSocket. A benchmark question
    always resolves to at least one already-indexed directory (see the
    design doc) — there is no raw-filesystem fallback here.
    """
    if not index_folders:
        raise ValueError("run_agentic_session requires at least one index folder")

    resolved_database_url = resolve_database_url(database_url)
    run_started_at = time.monotonic()
    step_number = 0
    llm_calls: list[LLMCallStats] = []

    async def _collect_llm_call(stats: LLMCallStats) -> None:
        llm_calls.append(stats)

    index_storage = PostgresStorage(resolved_database_url)
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
            if isinstance(event, ToolCallEvent):
                step_number += 1
                _record_tool_call(
                    event,
                    step_number=step_number,
                    trace=trace,
                    index_storage=index_storage,
                )
            elif isinstance(event, ToolBatchEvent):
                for call in event.tool_calls:
                    step_number += 1
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
        usage = agent.token_usage
        cost_usd, cost_source = _sum_call_cost(llm_calls)

        return BenchmarkRunResult(
            final_result=final_result,
            error=result_error,
            incomplete=not result_error and agent.forced_stop,
            cited_sources=cited_sources,
            step_path=trace.step_path,
            stats={
                "steps": step_number,
                "api_calls": usage.api_calls,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "thinking_tokens": usage.thinking_tokens,
                "total_tokens": usage.total_tokens,
                "tool_result_chars": usage.tool_result_chars,
                "context_summaries": usage.context_summaries,
                "duration_ms": round((time.monotonic() - run_started_at) * 1000),
                "cost_usd": cost_usd,
                "cost_source": cost_source,
                "model": model,
                "provider": provider,
            },
        )
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

Groundedness (are claims backed by the cited sources?):
  1 = no citations, or citations that don't support the claim made
  3 = citations present but incomplete coverage of the claims made
  5 = every material claim is backed by a cited source

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
