"""Reuse canonical acquisition without source-kind fanout or corpus inventory."""

from __future__ import annotations

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker, build_corpus_specs
from onyx.asv3.models import RunContext, RunStopped, ToolOutcome, ToolSpec
from onyx.asv3.shared_reads import READ_TOOLS, SharedReads
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.models import ResearchPlan, SourceAction

CORPUS_TOOLS = frozenset(
    {
        "resolve_source",
        "read_source_range",
        "read_chunk",
        "read_chunk_context",
        "read_provision",
        "read_named_provision",
        "search_source_text",
        "follow_reference",
        "compare_versions",
        "search_corpus",
    }
)


def corpus_specs(broker: CorpusBroker) -> list[ToolSpec]:
    specs = [
        spec
        for spec in build_corpus_specs(
            broker, require_search_targets=True, source_identity_guidance=True
        )
        if spec.name in CORPUS_TOOLS
    ]
    named = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=True)
        if spec.name == "read_named_provision"
    )
    # This fixed two-step canonical read is not an agent/orchestration capability.
    # The run allocates nested tool slots in addition to its four acquisition workers.
    specs.append(named.model_copy(update={"orchestrates": False}))
    result = []
    for spec in specs:
        if spec.name not in READ_TOOLS:
            result.append(spec)
            continue

        def shared_handler(
            arguments: dict[str, JsonValue],
            context: RunContext,
            owned_spec: ToolSpec = spec,
        ) -> ToolOutcome:
            shared = context.services.get("shared_reads")
            return (
                shared.run(
                    owned_spec.name,
                    arguments,
                    context,
                    lambda producer: owned_spec.handler(arguments, producer),
                )
                if isinstance(shared, SharedReads)
                else owned_spec.handler(arguments, context)
            )

        result.append(spec.model_copy(update={"handler": shared_handler}))
    return result


class SupersearchAcquirer(CanonicalAcquirer):
    @staticmethod
    def _normalize(actions: list[SourceAction]) -> list[SourceAction]:
        normalized = []
        for action in actions:
            if (
                action.tool == "resolve_source"
                and not str(action.arguments.get("query", "")).strip()
            ):
                raise RunStopped(
                    "Supersearch requires a focused source title; whole-corpus inventory is unavailable"
                )
            normalized.append(action.model_copy(update={"source_kind": None}))
        return normalized

    def acquire(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        return super().acquire(self._normalize(actions), plan)

    def acquire_host_actions(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        return super().acquire_host_actions(self._normalize(actions), plan)
