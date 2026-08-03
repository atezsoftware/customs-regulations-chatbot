import json
import re

from onyx.chat.citation_processor import CitationMapping, DynamicCitationProcessor
from onyx.context.search.models import SearchDoc, SearchDocsResponse
from onyx.tools.built_in_tools import CITEABLE_TOOLS_NAMES
from onyx.tools.models import ToolResponse
from onyx.utils.logger import setup_logger

logger = setup_logger()


def canonicalize_search_tool_response_citations(
    tool_response: ToolResponse,
    existing_citation_mapping: CitationMapping,
) -> None:
    """Reuse one citation number for the same exact chunk across search calls."""

    search_response = tool_response.rich_response
    if not isinstance(search_response, SearchDocsResponse):
        return

    identity_to_existing: dict[tuple[str, int], int] = {}
    for citation_num, doc in sorted(existing_citation_mapping.items()):
        identity_to_existing.setdefault((doc.document_id, doc.chunk_ind), citation_num)

    raw_to_doc: dict[int, SearchDoc] = {}
    for citation_num, document_id in search_response.citation_mapping.items():
        chunk_ind = search_response.citation_chunk_mapping.get(citation_num)
        matching_doc = next(
            (
                doc
                for doc in search_response.search_docs
                if doc.document_id == document_id
                and (chunk_ind is None or doc.chunk_ind == chunk_ind)
            ),
            None,
        )
        if matching_doc is not None:
            raw_to_doc[citation_num] = matching_doc

    if not raw_to_doc:
        return

    occupied_numbers = set(existing_citation_mapping)
    next_available = max(occupied_numbers, default=0) + 1
    identity_to_canonical = dict(identity_to_existing)
    raw_to_canonical: dict[int, int] = {}
    canonical_mapping: dict[int, str] = {}
    canonical_chunk_mapping: dict[int, int] = {}

    for raw_num, raw_doc in sorted(raw_to_doc.items()):
        document_id = raw_doc.document_id
        chunk_ind = raw_doc.chunk_ind
        identity = (document_id, chunk_ind)
        canonical_num = identity_to_canonical.get(identity)
        if canonical_num is None:
            canonical_num = raw_num
            if canonical_num in occupied_numbers:
                while next_available in occupied_numbers:
                    next_available += 1
                canonical_num = next_available
                next_available += 1
            identity_to_canonical[identity] = canonical_num
            occupied_numbers.add(canonical_num)

        raw_to_canonical[raw_num] = canonical_num
        canonical_mapping[canonical_num] = document_id
        canonical_chunk_mapping[canonical_num] = chunk_ind

    try:
        payload = json.loads(tool_response.llm_facing_response)
    except json.JSONDecodeError:
        logger.warning("Could not canonicalize citations in non-JSON tool response")
    else:
        results = payload.get("results") if isinstance(payload, dict) else None
        if isinstance(results, list):
            for result in results:
                if not isinstance(result, dict):
                    continue
                raw_num = result.get("document")
                if isinstance(raw_num, int) and raw_num in raw_to_canonical:
                    result["document"] = raw_to_canonical[raw_num]
            tool_response.llm_facing_response = json.dumps(
                payload, indent=2, ensure_ascii=False
            )

    search_response.citation_mapping = canonical_mapping
    search_response.citation_chunk_mapping = canonical_chunk_mapping


def update_citation_processor_from_tool_response(
    tool_response: ToolResponse,
    citation_processor: DynamicCitationProcessor,
) -> None:
    """Update citation processor if this was a citeable tool with a SearchDocsResponse.

    Checks if the tool call is citeable and if the response contains a SearchDocsResponse,
    then creates a mapping from citation numbers to SearchDoc objects and updates the
    citation processor.

    Args:
        tool_response: The response from the tool execution (must have tool_call set)
        citation_processor: The DynamicCitationProcessor to update
    """
    # Early return if tool_call is not set
    if tool_response.tool_call is None:
        return

    # Update citation processor if this was a search tool
    if tool_response.tool_call.tool_name in CITEABLE_TOOLS_NAMES:
        # Check if the rich_response is a SearchDocsResponse
        if isinstance(tool_response.rich_response, SearchDocsResponse):
            search_response = tool_response.rich_response

            # Create mapping from citation number to SearchDoc
            citation_to_doc: CitationMapping = {}
            for (
                citation_num,
                doc_id,
            ) in search_response.citation_mapping.items():
                chunk_ind = search_response.citation_chunk_mapping.get(citation_num)
                # Prefer the exact retrieved chunk. Older integrations omit the
                # chunk mapping and retain the previous document-only fallback.
                matching_doc = next(
                    (
                        doc
                        for doc in search_response.search_docs
                        if doc.document_id == doc_id
                        and (chunk_ind is None or doc.chunk_ind == chunk_ind)
                    ),
                    None,
                )
                if matching_doc:
                    citation_to_doc[citation_num] = matching_doc

            # Update the citation processor
            citation_processor.update_citation_mapping(citation_to_doc)


def extract_citation_order_from_text(text: str) -> list[int]:
    """Extract citation numbers from text in order of first appearance.

    Parses citation patterns like [1], [1, 2], [[1]], 【1】 etc. and returns
    the citation numbers in the order they first appear in the text.

    Args:
        text: The text containing citations

    Returns:
        List of citation numbers in order of first appearance (no duplicates)
    """
    # Same pattern used in collapse_citations and DynamicCitationProcessor
    # Group 2 captures the number in double bracket format: [[1]], 【【1】】
    # Group 4 captures the numbers in single bracket format: [1], [1, 2]
    citation_pattern = re.compile(
        r"([\[【［]{2}(\d+)[\]】］]{2})|([\[【［]([\d]+(?: *, *\d+)*)[\]】］])"
    )
    seen: set[int] = set()
    order: list[int] = []

    for match in citation_pattern.finditer(text):
        # Group 2 is for double bracket single number, group 4 is for single bracket
        if match.group(2):
            nums_str = match.group(2)
        elif match.group(4):
            nums_str = match.group(4)
        else:
            continue

        for num_str in nums_str.split(","):
            num_str = num_str.strip()
            if num_str:
                try:
                    num = int(num_str)
                    if num not in seen:
                        seen.add(num)
                        order.append(num)
                except ValueError:
                    continue

    return order


def collapse_citations(
    answer_text: str,
    existing_citation_mapping: CitationMapping,
    new_citation_mapping: CitationMapping,
) -> tuple[str, CitationMapping]:
    """Collapse the citations in the text to use the smallest possible numbers.

    This function takes citations in the text (like [25], [30], etc.) and replaces them
    with the smallest possible numbers. It starts numbering from the next available
    integer after the existing citation mapping. If a citation refers to a document
    that already exists in the existing citation mapping (matched by document_id),
    it uses the existing citation number instead of assigning a new one.

    Args:
        answer_text: The text containing citations to collapse (e.g., "See [25] and [30]")
        existing_citation_mapping: Citations already processed/displayed. These mappings
            are preserved unchanged in the output.
        new_citation_mapping: Citations from the current text that need to be collapsed.
            The keys are the citation numbers as they appear in answer_text.

    Returns:
        A tuple of (updated_text, combined_mapping) where:
        - updated_text: The text with citations replaced with collapsed numbers
        - combined_mapping: All values from existing_citation_mapping plus the new
          mappings with their (possibly renumbered) keys
    """
    # A citation identifies a retrieved chunk, not merely its parent document.
    chunk_to_existing_citation: dict[tuple[str, int], int] = {
        (doc.document_id, doc.chunk_ind): citation_num
        for citation_num, doc in existing_citation_mapping.items()
    }

    # Determine the next available citation number
    if existing_citation_mapping:
        next_citation_num = max(existing_citation_mapping.keys()) + 1
    else:
        next_citation_num = 1

    # Build the mapping from old citation numbers (in new_citation_mapping) to new numbers
    old_to_new: dict[int, int] = {}
    additional_mappings: CitationMapping = {}

    for old_num, search_doc in new_citation_mapping.items():
        chunk_identity = (search_doc.document_id, search_doc.chunk_ind)

        # Reuse a citation only for the exact same retrieved chunk.
        if chunk_identity in chunk_to_existing_citation:
            # Use the existing citation number
            old_to_new[old_num] = chunk_to_existing_citation[chunk_identity]
        else:
            # Check if this exact chunk already received a new number.
            existing_new_num = None
            for mapped_old, mapped_new in old_to_new.items():
                if (
                    mapped_old in new_citation_mapping
                    and (
                        new_citation_mapping[mapped_old].document_id,
                        new_citation_mapping[mapped_old].chunk_ind,
                    )
                    == chunk_identity
                ):
                    existing_new_num = mapped_new
                    break

            if existing_new_num is not None:
                old_to_new[old_num] = existing_new_num
            else:
                # Assign the next available number
                old_to_new[old_num] = next_citation_num
                additional_mappings[next_citation_num] = search_doc
                next_citation_num += 1

    # Pattern to match citations like [25], [1, 2, 3], [[25]], etc.
    # Also matches unicode bracket variants: 【】, ［］
    citation_pattern = re.compile(
        r"([\[【［]{2}\d+[\]】］]{2})|([\[【［]\d+(?:, ?\d+)*[\]】］])"
    )

    def replace_citation(match: re.Match) -> str:
        """Replace citation numbers in a match with their new collapsed values."""
        citation_str = match.group()

        # Determine bracket style
        if (
            citation_str.startswith("[[")
            or citation_str.startswith("【【")
            or citation_str.startswith("［［")
        ):
            open_bracket = citation_str[:2]
            close_bracket = citation_str[-2:]
            content = citation_str[2:-2]
        else:
            open_bracket = citation_str[0]
            close_bracket = citation_str[-1]
            content = citation_str[1:-1]

        # Parse and replace citation numbers
        new_nums = []
        for num_str in content.split(","):
            num_str = num_str.strip()
            if not num_str:
                continue
            try:
                num = int(num_str)
                # Only replace if we have a mapping for this number
                if num in old_to_new:
                    new_nums.append(str(old_to_new[num]))
                else:
                    # Keep original if not in our mapping
                    new_nums.append(num_str)
            except ValueError:
                new_nums.append(num_str)

        # Reconstruct the citation with original bracket style
        new_content = ", ".join(new_nums)
        return f"{open_bracket}{new_content}{close_bracket}"

    # Replace all citations in the text
    updated_text = citation_pattern.sub(replace_citation, answer_text)

    # Build the combined mapping
    combined_mapping: CitationMapping = dict(existing_citation_mapping)
    combined_mapping.update(additional_mappings)

    return updated_text, combined_mapping
