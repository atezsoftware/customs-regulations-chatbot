from collections.abc import Callable

from onyx.configs.constants import DocumentSource
from onyx.connectors.file.connector import LocalFileConnector
from onyx.connectors.models import Document, HierarchyNode
from onyx.file_store.staging import (
    build_tracking_raw_file_callback,
    delete_files_best_effort,
)


def load_user_file_documents(
    *,
    user_file_id: str,
    file_id: str,
    file_name: str | None,
    tenant_id: str,
    connector_factory: Callable[..., LocalFileConnector] = LocalFileConnector,
) -> tuple[list[Document], list[str]]:
    """Load a user file through the one canonical connector/metadata boundary."""

    connector = connector_factory(
        file_locations=[file_id],
        file_names=[file_name] if file_name else None,
    )
    connector.load_credentials({})
    staging_callback, staged_csv_ids = build_tracking_raw_file_callback(
        metadata={"user_file_id": user_file_id, "tenant_id": tenant_id}
    )
    connector.set_raw_file_callback(staging_callback)

    documents: list[Document] = []
    try:
        for batch in connector.load_from_state():
            documents.extend(
                [
                    document
                    for document in batch
                    if not isinstance(document, HierarchyNode)
                ]
            )
    except Exception:
        delete_files_best_effort(
            staged_csv_ids,
            context=f"user-file load-failure staging cleanup uf={user_file_id}",
        )
        raise

    for document in documents:
        document.id = user_file_id
        document.source = DocumentSource.USER_FILE
    return documents, staged_csv_ids
