"""Turn each acquired source into its own reviewable, unpublished canonical file."""

import hashlib
from collections.abc import Callable, Collection, Sequence
from io import BytesIO
from uuid import UUID, uuid5

from onyx.configs.constants import DocumentSource, FileOrigin
from onyx.connectors.models import Document, TextSection
from onyx.file_processing.original_ingestion import (
    LoadedUserFileDocuments,
    extracted_plaintext,
)
from onyx.file_store.file_store import FileStore, get_default_file_store
from onyx.regulatory.amendments.annexes.models import SourceLink
from onyx.regulatory.amendments.annexes.sources import MAX_ASSET_BYTES, MAX_TEXT_CHARS
from onyx.regulatory.document_intake.models import (
    IntakeFilePreparation,
    IntakeSourceAsset,
    IntakeSourceContext,
)
from onyx.regulatory.document_intake.text import (
    prepare_intake_text,
    require_resolved_attachment_references,
)
from onyx.utils.web_content import decode_html_bytes

ChunkFile = Callable[[UUID, str, list[Document]], bool]
StorePlaintext = Callable[[UUID, str], bool]


def build_intake_document(
    *,
    package_id: UUID,
    asset: IntakeSourceAsset,
    user_file_id: UUID,
    text: str,
    links: Sequence[SourceLink],
    package_assets: Sequence[IntakeSourceAsset] = (),
    original_file_id: str | None = None,
    parent_asset_hashes: Collection[str] | None = None,
) -> Document:
    if asset.package_id != package_id:
        raise ValueError("Source asset belongs to another intake package")
    if not text.strip():
        raise ValueError("Source asset has no prepared text to review")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Prepared source text exceeds the intake limit")
    require_resolved_attachment_references(text, asset_sha256=asset.sha256, links=links)
    text = prepare_intake_text(
        text,
        asset_sha256=asset.sha256,
        links=links,
        package_assets=package_assets,
        preserve_markdown_headings=asset.mime_type == "text/markdown",
    )
    incoming = [
        link.model_copy(deep=True)
        for link in links
        if link.target_asset_hash == asset.sha256
        and link.parent_asset_hash != asset.sha256
    ]
    linked_parents = {link.parent_asset_hash for link in incoming}
    if parent_asset_hashes is not None and not set(parent_asset_hashes).issubset(
        linked_parents
    ):
        raise ValueError("Selected intake parents require actual source links")
    parents = sorted(
        linked_parents if parent_asset_hashes is None else set(parent_asset_hashes)
    )
    by_hash = {item.sha256: item for item in package_assets}
    if any(item.package_id != package_id for item in package_assets):
        raise ValueError("Source context contains an asset from another package")
    context = IntakeSourceContext(
        package_id=package_id,
        asset_id=asset.id,
        asset_sha256=asset.sha256,
        parent_asset_hashes=parents,
        parent_asset_ids=[
            by_hash[digest].id for digest in parents if digest in by_hash
        ],
        parent_titles=[
            by_hash[digest].display_name for digest in parents if digest in by_hash
        ],
        source_links=incoming,
    )
    source_url = asset.final_url or asset.original_url
    title = asset.display_name or "source"
    return Document(
        id=str(user_file_id),
        source=DocumentSource.USER_FILE,
        semantic_identifier=title,
        title=title,
        sections=[TextSection(text=text, link=source_url)],
        metadata={},
        doc_metadata={"regulatory_document_intake": context.model_dump(mode="json")},
        file_id=original_file_id or asset.file_id,
    )


def _read_verified(store: FileStore, file_id: str, sha256: str, *, limit: int) -> bytes:
    with store.read_file(file_id) as stream:
        content = stream.read(limit + 1)
    if len(content) > limit or hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError("Prepared intake source integrity changed")
    return content


def clone_intake_original(
    *, intake_id: UUID, asset: IntakeSourceAsset, store: FileStore
) -> str:
    """Own a separate deletable original while retaining immutable source evidence."""
    raw = _read_verified(store, asset.file_id, asset.sha256, limit=MAX_ASSET_BYTES)
    if len(raw) != asset.byte_count:
        raise ValueError("Prepared intake source byte count changed")
    owned_file_id = str(uuid5(intake_id, f"original:{asset.id}"))
    if owned_file_id == asset.file_id:
        raise ValueError("Imported original must have independent storage")
    return store.save_file(
        BytesIO(raw),
        display_name=asset.display_name,
        file_origin=FileOrigin.USER_FILE,
        file_type=asset.mime_type,
        file_metadata={
            "source_package_id": str(asset.package_id),
            "source_asset_id": str(asset.id),
            "source_sha256": asset.sha256,
        },
        file_id=owned_file_id,
    )


def _canonical_source_text(asset: IntakeSourceAsset, raw: bytes, prepared: str) -> str:
    if asset.mime_type != "text/html":
        return prepared
    from bs4 import BeautifulSoup

    from onyx.file_processing.html_utils import parse_html_page_basic

    soup = BeautifulSoup(decode_html_bytes(raw, asset.mime_type), "html.parser")
    for tag in soup.find_all(
        ("nav", "footer", "script", "style", "template", "noscript")
    ):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    # Source-package inspection flattens cells; canonical input retains their grid.
    return parse_html_page_basic(str(root))


def load_verified_intake_documents(
    *,
    package_id: UUID,
    asset: IntakeSourceAsset,
    user_file_id: UUID,
    links: Sequence[SourceLink],
    package_assets: Sequence[IntakeSourceAsset] = (),
    store: FileStore | None = None,
    original_file_id: str | None = None,
    parent_asset_hashes: Collection[str] | None = None,
) -> LoadedUserFileDocuments:
    """Rebuild canonical input exclusively from hash-verified retained sources."""
    if asset.package_id != package_id:
        raise ValueError("Source asset belongs to another intake package")
    if not asset.text_file_id or not asset.text_sha256:
        raise ValueError("Source asset has no verified prepared text")
    file_store = store or get_default_file_store()
    raw = _read_verified(file_store, asset.file_id, asset.sha256, limit=MAX_ASSET_BYTES)
    if len(raw) != asset.byte_count:
        raise ValueError("Prepared intake source byte count changed")
    owned_file_id = original_file_id or asset.file_id
    if owned_file_id != asset.file_id:
        _read_verified(file_store, owned_file_id, asset.sha256, limit=MAX_ASSET_BYTES)
    prepared_text = _read_verified(
        file_store, asset.text_file_id, asset.text_sha256, limit=MAX_TEXT_CHARS * 4
    ).decode("utf-8")
    text = _canonical_source_text(asset, raw, prepared_text)
    document = build_intake_document(
        package_id=package_id,
        asset=asset,
        user_file_id=user_file_id,
        text=text,
        links=links,
        package_assets=package_assets,
        original_file_id=owned_file_id,
        parent_asset_hashes=parent_asset_hashes,
    )
    return LoadedUserFileDocuments(
        [document], file_id=owned_file_id, raw_sha256=asset.sha256
    )


def prepare_intake_file(
    *,
    package_id: UUID,
    asset: IntakeSourceAsset,
    user_file_id: UUID,
    links: Sequence[SourceLink],
    tenant_id: str,
    package_assets: Sequence[IntakeSourceAsset] = (),
    store: FileStore | None = None,
    chunk_file: ChunkFile | None = None,
    store_plaintext: StorePlaintext | None = None,
    original_file_id: str | None = None,
    parent_asset_hashes: Collection[str] | None = None,
) -> IntakeFilePreparation:
    """Prepare an admitted asset without comparing or publishing search content."""
    documents = load_verified_intake_documents(
        package_id=package_id,
        asset=asset,
        user_file_id=user_file_id,
        links=links,
        package_assets=package_assets,
        store=store,
        original_file_id=original_file_id,
        parent_asset_hashes=parent_asset_hashes,
    )
    if chunk_file is None:
        from onyx.regulatory.writer_publication import chunk_owned_file

        chunk_file = chunk_owned_file
    created = chunk_file(user_file_id, tenant_id, documents)
    if store_plaintext is None:
        from onyx.file_store.utils import store_user_file_plaintext

        store_plaintext = store_user_file_plaintext
    if not store_plaintext(user_file_id, extracted_plaintext(documents)):
        raise ValueError("Prepared intake plaintext could not be stored")
    receipt = documents.original_extraction
    return IntakeFilePreparation(
        user_file_id=user_file_id,
        source_asset_id=asset.id,
        documents_sha256=receipt.documents_sha256,
        plaintext_sha256=receipt.plaintext_sha256,
        initial_chunks_created=created,
    )
