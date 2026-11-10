"""Identify intake-specific chunk semantics without changing legacy generations."""

from collections.abc import Sequence
from typing import cast

from onyx.connectors.models import Document
from onyx.regulatory.document_intake.models import (
    IntakeChunkingVersion,
    IntakePreparationPolicy,
    IntakeSourceContext,
)

INTAKE_CHUNKING_CODE_VERSION: IntakeChunkingVersion = "3"
SUPPORTED_INTAKE_CHUNKING_VERSIONS = frozenset({"1", "2", "3", "4", "5"})


def is_multi_cell_table_row(text: str) -> bool:
    """Identify one complete source row without treating escaped pipes as columns."""

    if len(text.splitlines()) != 1:
        return False
    row = text.strip()
    pipes: list[int] = []
    escaped = False
    for index, character in enumerate(row):
        if character == "|" and not escaped:
            pipes.append(index)
        escaped = not escaped if character == "\\" else False
    return len(pipes) >= 3 and pipes[0] == 0 and pipes[-1] == len(row) - 1


def supported_intake_chunking_version(version: str) -> IntakeChunkingVersion:
    if version not in SUPPORTED_INTAKE_CHUNKING_VERSIONS:
        raise ValueError("The intake chunking version is not supported")
    return cast(IntakeChunkingVersion, version)


def intake_chunking_version(document: Document) -> str | None:
    context = (document.doc_metadata or {}).get("regulatory_document_intake")
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("Intake chunking requires a verified source context")
    if "chunking_version" not in context:
        return None
    parsed = IntakeSourceContext.model_validate(context)
    return parsed.chunking_version


def documents_intake_chunking_version(documents: Sequence[Document]) -> str | None:
    versions = {intake_chunking_version(document) for document in documents}
    if len(versions) > 1:
        raise ValueError("Mixed intake chunking versions are not permitted")
    return next(iter(versions), None)


def intake_preparation_policy(document: Document) -> IntakePreparationPolicy | None:
    context = (document.doc_metadata or {}).get("regulatory_document_intake")
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("Intake preparation requires a verified source context")
    if (
        context.get("chunking_version") not in ("4", "5")
        and context.get("preparation_policy") is None
    ):
        return None
    return IntakeSourceContext.model_validate(context).preparation_policy


def intake_markdown_heading_retention(document: Document) -> bool:
    context = (document.doc_metadata or {}).get("regulatory_document_intake")
    return (
        IntakeSourceContext.model_validate(context).retain_markdown_headings
        if context is not None
        else False
    )


def documents_intake_preparation_policy(
    documents: Sequence[Document],
) -> IntakePreparationPolicy | None:
    policies = {intake_preparation_policy(document) for document in documents}
    if len(policies) > 1:
        raise ValueError("Mixed intake preparation policies are not permitted")
    return next(iter(policies), None)
