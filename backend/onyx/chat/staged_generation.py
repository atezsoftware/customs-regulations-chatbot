"""Atomic publication helpers for LLM generations that may be discarded."""

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.citation_processor import DynamicCitationProcessor
from onyx.chat.emitter import BufferedEmitter, Emitter
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    AgentResponseStart,
)


def commit_staged_llm_step(
    *,
    buffered_emitter: BufferedEmitter,
    staged_state: ChatStateContainer,
    staged_citation_processor: DynamicCitationProcessor,
    emitter: Emitter,
    state_container: ChatStateContainer,
    pre_answer_processing_time: float,
    final_documents_from_emitted_citations: bool = False,
    answer_override: str | None = None,
) -> None:
    """Publish one accepted LLM step without leaking rejected draft state."""

    staged_reasoning = staged_state.get_reasoning_tokens()
    if staged_reasoning is not None:
        state_container.set_reasoning_tokens(staged_reasoning)

    staged_answer = (
        answer_override
        if answer_override is not None
        else staged_state.get_answer_tokens()
    )
    if staged_answer is not None:
        state_container.set_answer_tokens(staged_answer)

    if staged_state.get_pre_answer_processing_time() is not None:
        state_container.set_pre_answer_processing_time(pre_answer_processing_time)

    state_container.set_citation_mapping(staged_citation_processor.citation_to_doc)
    emitted_citations = staged_state.get_emitted_citations()
    for citation_number in emitted_citations:
        state_container.add_emitted_citation(citation_number)

    accepted_final_documents = (
        [
            search_doc
            for citation_number, search_doc in (
                staged_citation_processor.get_seen_citations().items()
            )
            if citation_number in emitted_citations
        ]
        if final_documents_from_emitted_citations
        else None
    )

    emitted_answer_override = False
    for packet in buffered_emitter.get_packets():
        if answer_override is not None and isinstance(packet.obj, AgentResponseDelta):
            if emitted_answer_override:
                continue
            packet = packet.model_copy(
                update={
                    "obj": packet.obj.model_copy(update={"content": answer_override})
                }
            )
            emitted_answer_override = True
        if isinstance(packet.obj, AgentResponseStart):
            response_start_updates: dict[str, object] = {
                "pre_answer_processing_seconds": pre_answer_processing_time
            }
            if accepted_final_documents is not None:
                response_start_updates["final_documents"] = accepted_final_documents
            packet = packet.model_copy(
                update={"obj": packet.obj.model_copy(update=response_start_updates)}
            )
        emitter.emit(packet)
