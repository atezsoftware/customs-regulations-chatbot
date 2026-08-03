from onyx.prompts.contextual_retrieval import (
    CONTEXTUAL_RAG_PROMPT2,
    DOCUMENT_SUMMARY_PROMPT,
)


def test_contextual_prompt_preserves_source_language_and_identifiers() -> None:
    assert "Write in the chunk's language" in CONTEXTUAL_RAG_PROMPT2
    assert "Preserve exact names, acronyms" in CONTEXTUAL_RAG_PROMPT2
    assert "provision or section identifiers" in CONTEXTUAL_RAG_PROMPT2
    assert "Use only information supported by the document" in (
        CONTEXTUAL_RAG_PROMPT2
    )


def test_document_summary_is_retrieval_focused_and_grounded() -> None:
    assert "in its primary language" in DOCUMENT_SUMMARY_PROMPT
    assert "Preserve exact document titles" in DOCUMENT_SUMMARY_PROMPT
    assert "scope distinctions" in DOCUMENT_SUMMARY_PROMPT
    assert "do not invent missing coverage" in DOCUMENT_SUMMARY_PROMPT
