"""Attach retrieval provenance to real chunks without changing their source text."""

import json
from collections.abc import Callable, Sequence

from onyx.context.search.models import InferenceChunk, InferenceSection
from onyx.context.search.utils import inference_section_from_single_chunk
from onyx.regulatory.labeling.search_models import SearchLabelEvidence
from onyx.utils.logger import setup_logger

logger = setup_logger()


def completion_source_ids(
    sections: Sequence[InferenceSection],
    evidence: dict[str, tuple[SearchLabelEvidence, ...]],
) -> tuple[str, ...]:
    present = {
        chunk.regulatory_chunk_id for section in sections for chunk in section.chunks
    }
    return tuple(
        dict.fromkeys(
            entry.source_chunk_id
            for section in sections
            for entry in evidence.get(
                section.center_chunk.regulatory_chunk_id or "", ()
            )
            if entry.source_chunk_id not in present
        )
    )[:2]


def complete_label_sources(
    sections: list[InferenceSection],
    evidence: dict[str, tuple[SearchLabelEvidence, ...]],
    *,
    limit: int,
    retrieve: Callable[[tuple[str, ...]], list[InferenceChunk]],
) -> list[InferenceSection]:
    identifiers = completion_source_ids(sections, evidence)[
        : max(0, limit - len(sections))
    ]
    if not identifiers:
        return sections
    try:
        chunks = retrieve(identifiers)
    except Exception:
        logger.warning(
            "Label source completion failed; keeping selected sources", exc_info=True
        )
        return sections
    accepted: dict[str, InferenceChunk] = {}
    for chunk in chunks:
        identifier = chunk.regulatory_chunk_id
        if identifier not in identifiers or identifier in accepted:
            continue
        quotes = [
            entry.evidence_quote
            for entries in evidence.values()
            for entry in entries
            if entry.source_chunk_id == identifier
        ]
        if quotes and all(quote in chunk.content for quote in quotes):
            accepted[identifier] = chunk
    return [
        *sections,
        *(inference_section_from_single_chunk(chunk) for chunk in accepted.values()),
    ]


def annotate_label_evidence(
    sections: Sequence[InferenceSection],
    evidence: dict[str, tuple[SearchLabelEvidence, ...]],
    taxonomy_hash: str,
) -> list[InferenceSection]:
    def annotate(chunk: InferenceChunk) -> InferenceChunk:
        entries = [
            entry
            for entry in evidence.get(chunk.regulatory_chunk_id or "", ())
            if entry.evidence_quote in chunk.content
        ]
        entries = entries[:12]
        if not entries:
            return chunk
        return chunk.model_copy(
            update={
                "metadata": {
                    **chunk.metadata,
                    "regulatory_label_ids": list(
                        dict.fromkeys(entry.label_id for entry in entries)
                    ),
                    "regulatory_label_provenance": json.dumps(
                        {
                            "taxonomy_hash": taxonomy_hash,
                            "use": "Retrieval hints only. Establish applicability from cited source text. "
                            "Missing labels do not establish absence of a rule or exception.",
                            "assignments": [
                                entry.model_dump(mode="json") for entry in entries
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            }
        )

    return [
        section.model_copy(
            update={
                "center_chunk": annotate(section.center_chunk),
                "chunks": [annotate(chunk) for chunk in section.chunks],
            }
        )
        for section in sections
    ]
