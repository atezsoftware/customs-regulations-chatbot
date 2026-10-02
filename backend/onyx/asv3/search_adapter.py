"""Isolate the established search pipeline and retain its original source text."""

import json
from collections.abc import Callable

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.chat.emitter import NullEmitter
from onyx.context.search.models import SearchDocsResponse
from onyx.server.query_and_chat.placement import Placement
from onyx.tools.models import SearchToolOverrideKwargs
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def build_search_adapter(
    tool: SearchTool | None, original_query: str, broker: CorpusBroker
) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
    def search(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_active()
        if tool is None:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Internal search is not configured",
            )
        isolated = tool.fork_for_independent_context(emitter=NullEmitter())
        response = isolated.run(
            placement=Placement(turn_index=0),
            override_kwargs=SearchToolOverrideKwargs(
                starting_citation_num=1, original_query=original_query
            ),
            queries=[str(args["query"])],
            search_mode=str(args.get("mode", "hybrid")),
        )
        context.check_active()
        rich = response.rich_response
        if not isinstance(rich, SearchDocsResponse):
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Search returned no source mapping",
            )
        payload = json.loads(response.llm_facing_response)
        results = payload.get("results", []) if isinstance(payload, dict) else []
        evidence: list[EvidenceItem] = []
        docs = {(doc.document_id, doc.chunk_ind): doc for doc in rich.search_docs}
        for result in results:
            if not isinstance(result, dict):
                continue
            number, text = result.get("document"), result.get("content")
            if (
                not isinstance(number, int)
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue
            doc_id = rich.citation_mapping.get(number)
            chunk = rich.citation_chunk_mapping.get(number)
            if doc_id is None or chunk is None:
                continue
            doc = docs.get((doc_id, chunk))
            if doc is None:
                continue
            # A retrieved section may contain siblings. Bind each real canonical row,
            # never the combined section text to its center's identity.
            evidence.extend(broker.hydrate_search_evidence(doc, context))
        return ToolOutcome(
            status=OutcomeStatus.FOUND if evidence else OutcomeStatus.NOT_FOUND,
            summary="Original text from the scoped search pipeline; navigation is not evidence",
            data={"source_count": len(evidence), "query": str(args["query"])},
            evidence=evidence,
        )

    return search
