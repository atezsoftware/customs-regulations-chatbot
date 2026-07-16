"""Run a small real-provider A/B benchmark against synthetic customs evidence.

The script never touches Postgres. Tool functions are replaced with fixed text,
while the configured Gemini model performs the real planning and final answer.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fs_explorer_api.agent import FsExplorerAgent, TOOLS, set_search_flags
from fs_explorer_api.models import StopAction, ToolCallAction

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


EVIDENCE = {
    "DIRECT_RULE": (
        "Gümrük Kanunu Madde 241: Transit süresi aşılırsa gecikmenin "
        "uzunluğuna göre kademeli usulsüzlük cezası uygulanabilir."
    ),
    "EXCEPTION": (
        "Genelge 2012/04: Resmî belgelerle kanıtlanan mücbir sebep halinde "
        "gecikme cezası uygulanmaz."
    ),
    "CROSS_REFERENCE": (
        "Transit Rejimi Tebliği Madde 36: Gecikme öngörülürse en yakın "
        "gümrük idaresine başvurularak süre uzatımı istenir."
    ),
}

SEARCH_RESULTS = {
    "ceza": "chunk_id=chunk_direct doc_id=doc_direct article_no=241 DIRECT_RULE",
    "mücbir": "chunk_id=chunk_exception doc_id=doc_exception EXCEPTION",
    "uzat": "chunk_id=chunk_cross doc_id=doc_cross article_no=36 CROSS_REFERENCE",
}


def semantic_search(query: str, **_: Any) -> str:
    lowered = query.lower()
    selected = [value for key, value in SEARCH_RESULTS.items() if key in lowered]
    if not selected:
        selected = list(SEARCH_RESULTS.values())
    return "\n".join(selected)


def get_document(doc_id: str, **_: Any) -> str:
    marker = {
        "doc_direct": "DIRECT_RULE",
        "doc_exception": "EXCEPTION",
        "doc_cross": "CROSS_REFERENCE",
    }.get(doc_id, "DIRECT_RULE")
    return (f"{marker}: {EVIDENCE[marker]}\n") * 120


def get_chunk_context(chunk_id: str, **_: Any) -> str:
    marker = {
        "chunk_direct": "DIRECT_RULE",
        "chunk_exception": "EXCEPTION",
        "chunk_cross": "CROSS_REFERENCE",
    }.get(chunk_id, "DIRECT_RULE")
    return (f"{marker}: {EVIDENCE[marker]}\n") * 8


SCENARIOS = [
    {
        "name": "direct_rule",
        "question": "Transit süresi aşılırsa ceza uygulanır mı?",
        "required": ["241", "ceza"],
    },
    {
        "name": "exception",
        "question": "Belgeli mücbir sebepte transit gecikme cezası uygulanır mı?",
        "required": ["mücbir", "uygulanmaz"],
    },
    {
        "name": "cross_reference",
        "question": "Gecikme halinde ceza ve süre uzatımı birlikte nasıl işler?",
        "required": ["241", "uzat", "gümrük"],
    },
]


async def run_scenario(config: dict[str, Any]) -> dict[str, Any]:
    agent = FsExplorerAgent()
    agent.configure_task(
        "An indexed customs corpus is available. Answer this question using "
        f"semantic_search and the most targeted read tool available: {config['question']}"
    )
    answer = ""
    completed = False

    for _ in range(10):
        result = await agent.take_action()
        if result is None:
            break
        action, action_type = result
        if action_type == "stop":
            fallback = (
                action.action.final_result
                if isinstance(action.action, StopAction)
                else ""
            )
            async for chunk in agent.stream_final_answer(fallback):
                answer += chunk
            completed = True
            break
        if action_type == "toolcall" and isinstance(action.action, ToolCallAction):
            agent.call_tool(action.action.tool_name, action.action.to_fn_args())
        elif action_type == "toolbatch" and hasattr(action.action, "tool_calls"):
            await agent.call_tools(
                [
                    (call.tool_name, call.to_fn_args())
                    for call in action.action.tool_calls
                ]
            )
        else:
            agent.configure_task(
                "Use indexed tools and continue the original question."
            )
            continue
        agent.configure_task(
            "Use the retrieved evidence and continue. Stop when sufficient."
        )

    lowered = answer.lower()
    accuracy = completed and all(term.lower() in lowered for term in config["required"])
    accuracy = accuracy and "## sources" in lowered and "[" in answer
    return {
        "scenario": config["name"],
        "accuracy": accuracy,
        "api_calls": agent.token_usage.api_calls,
        "input_tokens": agent.token_usage.prompt_tokens,
        "output_tokens": agent.token_usage.completion_tokens,
        "thinking_tokens": agent.token_usage.thinking_tokens,
        "total_tokens": agent.token_usage.total_tokens,
        "context_summaries": agent.token_usage.context_summaries,
    }


async def main() -> None:
    originals = dict(TOOLS)
    TOOLS["semantic_search"] = semantic_search
    TOOLS["get_document"] = get_document
    if "get_chunk_context" in TOOLS:
        TOOLS["get_chunk_context"] = get_chunk_context
    TOOLS["list_indexed_documents"] = lambda: (
        "doc_direct Gümrük Kanunu; doc_exception Genelge 2012/04; "
        "doc_cross Transit Rejimi Tebliği"
    )
    set_search_flags(enable_semantic=True, enable_metadata=True)
    try:
        results = [await run_scenario(config) for config in SCENARIOS]
    finally:
        TOOLS.clear()
        TOOLS.update(originals)
        set_search_flags(enable_semantic=False, enable_metadata=False)

    totals = {
        key: sum(int(result[key]) for result in results)
        for key in (
            "api_calls",
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
            "total_tokens",
            "context_summaries",
        )
    }
    report = {
        "label": os.getenv("BENCHMARK_LABEL", "current"),
        "model": os.getenv("FS_EXPLORER_LLM_MODEL", "default"),
        "accuracy_passed": sum(bool(result["accuracy"]) for result in results),
        "accuracy_total": len(results),
        "totals": totals,
        "scenarios": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
