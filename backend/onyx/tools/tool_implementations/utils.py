import json

from onyx.context.search.models import InferenceSection
from onyx.context.search.utils import sandbox_filename_for_document
from onyx.utils.logger import setup_logger

logger = setup_logger()


def truncate_output(output: str, max_length: int, label: str = "output") -> str:
    """Truncate to ``max_length`` and append a footer noting how many chars were elided. ``label`` is only used in the debug log."""
    truncated = output[:max_length]
    if len(output) > max_length:
        truncated += (
            f"\n... [output truncated, {len(output) - max_length} characters omitted]"
        )
        logger.debug("Truncated %s: %s", label, truncated)
    return truncated


FILE_ASSOCIATED_GUIDANCE = (
    "Only a short excerpt from this document is shown below. The complete "
    'file is available in the sandbox as "{filename}" — prefer the Python '
    "code interpreter to read, parse, or analyze it\n\n"
    "Excerpt: {content}"
)


def convert_inference_sections_to_llm_string(
    top_sections: list[InferenceSection],
    citation_start: int = 1,
    limit: int | None = None,
    include_source_type: bool = True,
    include_link: bool = False,
    include_document_id: bool = False,
    note: str | None = None,
) -> tuple[str, dict[int, str], dict[int, int]]:
    """Convert InferenceSection objects to a JSON string for LLM.

    Returns a JSON string with document results and a citation mapping.
    """
    # Apply limit if specified
    if limit is not None:
        top_sections = top_sections[:limit]

    # A citation identifies one retrieved chunk. Reusing a number for another
    # chunk in the same document makes claim-level legal citations ambiguous.
    chunk_identity_to_citation_id: dict[tuple[str, int], int] = {}
    citation_mapping: dict[int, str] = {}
    citation_chunk_mapping: dict[int, int] = {}
    current_citation_id = citation_start

    # First pass: assign citation ids to exact chunk identities.
    for section in top_sections:
        chunk = section.center_chunk
        chunk_identity = (chunk.document_id, chunk.chunk_id)
        if chunk_identity not in chunk_identity_to_citation_id:
            chunk_identity_to_citation_id[chunk_identity] = current_citation_id
            citation_mapping[current_citation_id] = chunk.document_id
            citation_chunk_mapping[current_citation_id] = chunk.chunk_id
            current_citation_id += 1

    # Second pass: build results with citation_ids assigned per document
    results = []

    for section in top_sections:
        chunk = section.center_chunk
        document_id = chunk.document_id
        citation_id = chunk_identity_to_citation_id[(document_id, chunk.chunk_id)]

        # Combine primary and secondary owners for authors
        authors = None
        if chunk.primary_owners or chunk.secondary_owners:
            authors = []
            if chunk.primary_owners:
                authors.extend(chunk.primary_owners)
            if chunk.secondary_owners:
                authors.extend(chunk.secondary_owners)

        # Format updated_at as ISO string if available
        updated_at_str = None
        if chunk.updated_at:
            updated_at_str = chunk.updated_at.isoformat()

        # Build result dictionary in desired order, only including non-None/empty fields
        result = {
            "document": citation_id,
            "title": chunk.semantic_identifier,
        }
        if updated_at_str is not None:
            result["updated_at"] = updated_at_str
        if authors is not None:
            result["authors"] = authors  # ty: ignore[invalid-assignment]
        if include_source_type:
            result["source_type"] = chunk.source_type.value
        if include_link:
            # Get the first link from the center chunk's source_links dict
            link = None
            if chunk.source_links:
                # source_links is dict[int, str], get the first value
                link = next(iter(chunk.source_links.values()), None)
            if link:
                result["url"] = link
        if include_document_id:
            result["document_identifier"] = chunk.document_id
        if chunk.file_id is not None and chunk.regulatory_chunk_id is None:
            filename = sandbox_filename_for_document(
                chunk.semantic_identifier, chunk.file_id
            )
            result["file_name"] = filename

            result["content"] = FILE_ASSOCIATED_GUIDANCE.format(
                filename=filename, content=chunk.content
            )
        else:
            result["content"] = section.combined_content
        result_metadata = dict(chunk.metadata)
        if chunk.regulatory_chunk_id is not None:
            result_metadata["regulatory_chunk_id"] = chunk.regulatory_chunk_id
        if chunk.heading_path is not None:
            result_metadata["regulatory_heading_path"] = list(chunk.heading_path)
        if result_metadata:
            result["metadata"] = json.dumps(result_metadata, ensure_ascii=False)
        results.append(result)

    payload: dict[str, object] = {}
    payload["results"] = results
    if note:
        payload["note"] = note

    return (
        json.dumps(payload, indent=2, ensure_ascii=False),
        citation_mapping,
        citation_chunk_mapping,
    )
