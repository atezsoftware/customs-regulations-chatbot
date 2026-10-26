"""Identify intake-specific chunk semantics without changing legacy generations."""

from collections.abc import Sequence

from onyx.connectors.models import Document
from onyx.regulatory.document_intake.models import IntakeSourceContext

INTAKE_CHUNKING_CODE_VERSION = "1"


def intake_chunking_version(document: Document) -> str | None:
    context = (document.doc_metadata or {}).get("regulatory_document_intake")
    if context is None:
        return None
    if not isinstance(context, dict):
        raise ValueError("Intake chunking requires a verified source context")
    if "chunking_version" not in context:
        return None
    parsed = IntakeSourceContext.model_validate(context)
    if parsed.chunking_version != INTAKE_CHUNKING_CODE_VERSION:
        raise ValueError("The intake chunking version is not supported")
    return parsed.chunking_version


def documents_intake_chunking_version(documents: Sequence[Document]) -> str | None:
    versions = {intake_chunking_version(document) for document in documents}
    if len(versions) > 1:
        raise ValueError("Mixed intake chunking versions are not permitted")
    return next(iter(versions), None)
