import datetime
import json
import math
from collections.abc import Callable, Iterable
from typing import Any

from elasticsearch.helpers import BulkIndexError

from onyx.access.models import DocumentAccess
from onyx.configs.app_configs import (
    MAX_CHUNKS_PER_DOC_BATCH,
    VERIFY_CREATE_ELASTICSEARCH_INDEX_ON_INIT_MT,
)
from onyx.configs.constants import PUBLIC_DOC_PAT, OnyxRedisLocks
from onyx.connectors.cross_connector_utils.miscellaneous_utils import (
    get_experts_stores_representations,
)
from onyx.connectors.models import convert_metadata_list_of_strings_to_dict
from onyx.context.search.enums import QueryType
from onyx.context.search.models import (
    IndexFilters,
    InferenceChunk,
    InferenceChunkUncleaned,
)
from onyx.db.document import check_indexed_docs_exist
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.enums import EmbeddingPrecision
from onyx.db.models import DocumentSource
from onyx.document_index.chunk_content_enrichment import (
    cleanup_content_for_chunks,
    generate_enriched_content_for_chunk_text,
)
from onyx.document_index.elasticsearch.client import (
    ElasticsearchDocumentMissingError,
    ElasticsearchIndexClient,
    ElasticsearchUpdateError,
    SearchHit,
)
from onyx.document_index.elasticsearch.constants import ElasticsearchSearchType
from onyx.document_index.elasticsearch.schema import (
    ACCESS_CONTROL_LIST_FIELD_NAME,
    CONTENT_FIELD_NAME,
    CREATED_AT_FIELD_NAME,
    DOCUMENT_SETS_FIELD_NAME,
    GLOBAL_BOOST_FIELD_NAME,
    HIDDEN_FIELD_NAME,
    PERSONAS_FIELD_NAME,
    USER_PROJECTS_FIELD_NAME,
    VALIDITY_END_DATE_FIELD_NAME,
    VALIDITY_START_DATE_FIELD_NAME,
    DocumentChunk,
    DocumentChunkWithoutVectors,
    DocumentSchema,
    get_elasticsearch_doc_chunk_id,
)
from onyx.document_index.elasticsearch.search import (
    DocumentQuery,
    get_normalization_method_and_config,
)
from onyx.document_index.interfaces_new import (
    DocumentChunkVerificationError,
    DocumentChunkVerificationRequest,
    DocumentChunkVerificationResult,
    DocumentIndex,
    DocumentInsertionRecord,
    DocumentSectionRequest,
    IndexingMetadata,
    MetadataUpdateRequest,
    SecondaryIndexDocumentMissingError,
    TenantState,
)
from onyx.indexing.models import DocMetadataAwareIndexChunk, Document
from onyx.redis.lock_context import redis_shared_lock
from onyx.regulatory.exact_search_fields import extract_legal_exact_fields
from onyx.utils.datetime import datetime_to_utc
from onyx.utils.logger import setup_logger
from onyx.utils.text_processing import remove_invalid_unicode_chars
from shared_configs.configs import MULTI_TENANT
from shared_configs.model_server_models import Embedding

logger = setup_logger(__name__)


VERIFY_INDEX_LOCK_TTL_S = 60
VERIFY_INDEX_LOCK_BLOCKING_TIMEOUT_S = 60

# Batch size for the orphan sweep's delete-by-query terms filter — well under the
# Elasticsearch terms cap (65536) so a large mid-port purge can't build an oversized query.
_PORT_ORPHAN_DELETE_BATCH_SIZE = 1000


# Per-process cache of indices we've already verified/created/applied the
# mapping for. Used for the multi-tenant cloud codepath, which attempts to
# verify or create an index on DocumentIndex init since that deployment mode
# does not run setup on application start. This attempt can be expensive, and it
# only needs to happen at most once per process lifetime, since any changes to
# an index should always be correlated with a redeploy.
_verified_index_names_for_current_process: set[str] = set()


class ElasticsearchSchemaMigrationRequiredError(RuntimeError):
    """An existing non-empty index cannot adopt the configured mapping safely."""


_IMMUTABLE_MAPPING_KEYS = (
    "type",
    "analyzer",
    "normalizer",
    "dims",
    "similarity",
    "index_options",
)


def _mapping_properties_require_recreation(
    current_properties: dict[str, Any],
    expected_properties: dict[str, Any],
) -> bool:
    for field_name, expected_property in expected_properties.items():
        current_property = current_properties.get(field_name)
        if current_property is None:
            continue
        if not isinstance(current_property, dict) or not isinstance(
            expected_property, dict
        ):
            return True
        for immutable_key in _IMMUTABLE_MAPPING_KEYS:
            if (
                immutable_key in expected_property
                and current_property.get(immutable_key)
                != expected_property[immutable_key]
            ):
                return True
        current_subfields = current_property.get("fields", {})
        expected_subfields = expected_property.get("fields", {})
        if isinstance(current_subfields, dict) and isinstance(expected_subfields, dict):
            if _mapping_properties_require_recreation(
                current_subfields, expected_subfields
            ):
                return True
    return False


def _mapping_requires_recreation(
    current_mappings: dict[str, Any], expected_mappings: dict[str, Any]
) -> bool:
    current_properties = current_mappings.get("properties", {})
    expected_properties = expected_mappings.get("properties", {})
    if not isinstance(current_properties, dict) or not isinstance(
        expected_properties, dict
    ):
        return True
    return _mapping_properties_require_recreation(
        current_properties, expected_properties
    )


def ensure_current_schema(
    *,
    index_client: ElasticsearchIndexClient,
    expected_mappings: dict[str, Any],
    index_settings: dict[str, Any],
    database_has_indexed_documents: bool | Callable[[], bool],
) -> None:
    """Apply additive mappings or safely replace an incompatible empty index."""

    current_mappings = index_client.get_index_mapping()
    if not _mapping_requires_recreation(current_mappings, expected_mappings):
        index_client.put_mapping(expected_mappings)
        return

    postgres_has_documents = (
        database_has_indexed_documents()
        if callable(database_has_indexed_documents)
        else database_has_indexed_documents
    )
    indexed_chunk_count = index_client.count_by_query({"query": {"match_all": {}}})
    if postgres_has_documents or indexed_chunk_count != 0:
        raise ElasticsearchSchemaMigrationRequiredError(
            "The existing Elasticsearch mapping is incompatible with the configured "
            "Turkish legal mapping and indexed data may be present. Create a new "
            "search index and reindex; the existing index was left untouched."
        )

    if not index_client.delete_index():
        raise ElasticsearchSchemaMigrationRequiredError(
            "The incompatible empty Elasticsearch index disappeared during schema "
            "verification. Retry setup; no index was recreated."
        )
    index_client.create_index(mappings=expected_mappings, settings=index_settings)


def _database_has_indexed_documents() -> bool:
    with get_session_with_current_tenant() as db_session:
        return check_indexed_docs_exist(db_session)


def generate_elasticsearch_filtered_access_control_list(
    access: DocumentAccess,
) -> list[str]:
    """Generates an access control list with PUBLIC_DOC_PAT removed.

    In the Elasticsearch schema this is represented by PUBLIC_FIELD_NAME.
    """
    access_control_list = access.to_acl()
    access_control_list.discard(PUBLIC_DOC_PAT)
    return list(access_control_list)


def convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
    chunk: DocumentChunkWithoutVectors,
    score: float | None,
    highlights: dict[str, list[str]],
) -> InferenceChunkUncleaned:
    """
    Generates an inference chunk from an Elasticsearch document chunk, its score,
    and its match highlights.

    Args:
        chunk: The document chunk returned by Elasticsearch.
        score: The document chunk match score as calculated by Elasticsearch. Only
            relevant for searches like hybrid search. It is acceptable for this
            value to be None for results from other queries like ID-based
            retrieval as a match score makes no sense in those contexts.
        highlights: Maps schema property name to a list of highlighted snippets
            with match terms wrapped in tags (e.g. "something <hi>keyword</hi>
            other thing").

    Returns:
        An Onyx inference chunk representation.
    """
    return InferenceChunkUncleaned(
        chunk_id=chunk.chunk_index,
        blurb=chunk.blurb,
        # Includes extra content prepended/appended during indexing.
        content=chunk.content,
        # When we read a string and turn it into a dict the keys will be
        # strings, but in this case they need to be ints.
        source_links=(
            {int(k): v for k, v in json.loads(chunk.source_links).items()}
            if chunk.source_links
            else None
        ),
        image_file_id=chunk.image_file_id,
        # Deprecated. Fill in some reasonable default.
        section_continuation=False,
        document_id=chunk.document_id,
        source_type=DocumentSource(chunk.source_type),
        semantic_identifier=chunk.semantic_identifier,
        title=chunk.title,
        boost=chunk.global_boost,
        score=score,
        hidden=chunk.hidden,
        metadata=(
            convert_metadata_list_of_strings_to_dict(chunk.metadata_list)
            if chunk.metadata_list
            else {}
        ),
        # Extract highlighted snippets from the content field, if available. In
        # the future we may want to match on other fields too, currently we only
        # use the content field.
        match_highlights=highlights.get(CONTENT_FIELD_NAME, []),
        # TODO(andrei) Consider storing a chunk content index instead of a full
        # string when working on chunk content augmentation.
        doc_summary=chunk.doc_summary,
        # TODO(andrei) Same thing as above.
        chunk_context=chunk.chunk_context,
        updated_at=chunk.last_updated,
        primary_owners=chunk.primary_owners,
        secondary_owners=chunk.secondary_owners,
        regulatory_chunk_id=chunk.regulatory_chunk_id,
        heading_path=(
            list(chunk.heading_path) if chunk.heading_path is not None else None
        ),
        validity_start_date=(
            chunk.validity_start_date.date()
            if chunk.validity_start_date is not None
            else None
        ),
        validity_end_date=(
            chunk.validity_end_date.date()
            if chunk.validity_end_date is not None
            else None
        ),
        # TODO(andrei) Same thing as chunk_context above.
        metadata_suffix=chunk.metadata_suffix,
    )


def _date_to_utc_datetime(value: datetime.date | None) -> datetime.datetime | None:
    """Validity dates are calendar dates; Elasticsearch date fields want epoch
    seconds, so anchor them at UTC midnight."""
    if value is None:
        return None
    return datetime.datetime(
        value.year, value.month, value.day, tzinfo=datetime.timezone.utc
    )


def _date_to_epoch_seconds(value: datetime.date | None) -> int | None:
    as_datetime = _date_to_utc_datetime(value)
    return int(as_datetime.timestamp()) if as_datetime is not None else None


def _convert_onyx_chunk_to_elasticsearch_document(
    chunk: DocMetadataAwareIndexChunk,
) -> DocumentChunk:
    filtered_blurb = remove_invalid_unicode_chars(chunk.blurb)
    _title = chunk.source_document.get_title_for_document_index()
    filtered_title = remove_invalid_unicode_chars(_title) if _title else None
    filtered_content = remove_invalid_unicode_chars(
        generate_enriched_content_for_chunk_text(chunk)
    )
    # Regulatory chunks locate themselves inside their document via
    # heading_path; surfacing it in the semantic identifier makes citations
    # point at the exact article/clause instead of just the file.
    _semantic_identifier = chunk.source_document.semantic_identifier
    if chunk.heading_path:
        _semantic_identifier = (
            f"{_semantic_identifier} — {' > '.join(chunk.heading_path)}"
        )
    filtered_semantic_identifier = remove_invalid_unicode_chars(_semantic_identifier)
    filtered_metadata_suffix = remove_invalid_unicode_chars(
        chunk.metadata_suffix_keyword
    )
    _metadata_list = chunk.source_document.get_metadata_str_attributes()
    filtered_metadata_list = (
        [remove_invalid_unicode_chars(metadata) for metadata in _metadata_list]
        if _metadata_list
        else None
    )
    legal_exact_fields = extract_legal_exact_fields(
        filtered_content,
        filtered_title,
        filtered_semantic_identifier,
        filtered_metadata_suffix,
        *(chunk.heading_path or []),
        *(filtered_metadata_list or []),
    )
    return DocumentChunk(
        document_id=chunk.source_document.id,
        chunk_index=chunk.chunk_id,
        # Use get_title_for_document_index to match the logic used when creating
        # the title_embedding in the embedder. This method falls back to
        # semantic_identifier when title is None (but not empty string).
        title=filtered_title,
        title_vector=chunk.title_embedding,
        content=filtered_content,
        content_vector=chunk.embeddings.full_embedding,
        source_type=chunk.source_document.source.value,
        metadata_list=filtered_metadata_list,
        metadata_suffix=filtered_metadata_suffix,
        last_updated=chunk.source_document.doc_updated_at,
        created_at=chunk.source_document.doc_created_at,
        public=chunk.access.is_public,
        access_control_list=generate_elasticsearch_filtered_access_control_list(
            chunk.access
        ),
        hidden=chunk.hidden,
        global_boost=chunk.boost,
        semantic_identifier=filtered_semantic_identifier,
        image_file_id=chunk.image_file_id,
        # Small optimization, if this list is empty we can supply None to
        # Elasticsearch and it will not store any data at all for this field, which
        # is different from supplying an empty list.
        source_links=json.dumps(chunk.source_links) if chunk.source_links else None,
        blurb=filtered_blurb,
        doc_summary=chunk.doc_summary,
        chunk_context=chunk.chunk_context,
        # Small optimization, if this list is empty we can supply None to
        # Elasticsearch and it will not store any data at all for this field, which
        # is different from supplying an empty list.
        document_sets=list(chunk.document_sets) if chunk.document_sets else None,
        # Small optimization, if this list is empty we can supply None to
        # Elasticsearch and it will not store any data at all for this field, which
        # is different from supplying an empty list.
        user_projects=chunk.user_project or None,
        personas=chunk.personas or None,
        primary_owners=get_experts_stores_representations(
            chunk.source_document.primary_owners
        ),
        secondary_owners=get_experts_stores_representations(
            chunk.source_document.secondary_owners
        ),
        # TODO(andrei): Consider not even getting this from
        # DocMetadataAwareIndexChunk and instead using ElasticsearchDocumentIndex's
        # instance variable. One source of truth -> less chance of a very bad
        # bug in prod.
        tenant_id=TenantState(tenant_id=chunk.tenant_id, multitenant=MULTI_TENANT),
        # Store ancestor hierarchy node IDs for hierarchy-based filtering.
        ancestor_hierarchy_node_ids=chunk.ancestor_hierarchy_node_ids or None,
        regulatory_chunk_id=chunk.regulatory_chunk_id,
        heading_path=chunk.heading_path or None,
        provision_identifiers=legal_exact_fields.provision_identifiers or None,
        decision_numbers=legal_exact_fields.decision_numbers or None,
        legal_dates=legal_exact_fields.legal_dates or None,
        validity_start_date=_date_to_utc_datetime(chunk.validity_start_date),
        validity_end_date=_date_to_utc_datetime(chunk.validity_end_date),
    )


class ElasticsearchDocumentIndex(DocumentIndex):
    """Elasticsearch-specific implementation of the DocumentIndex interface.

    This class provides document indexing, retrieval, and management operations
    for an Elasticsearch search engine instance. It handles the complete lifecycle
    of document chunks within a specific Elasticsearch index/schema.

    Each kind of embedding used should correspond to a different instance of
    this class, and therefore a different index in Elasticsearch.

    If in a multitenant environment and
    VERIFY_CREATE_ELASTICSEARCH_INDEX_ON_INIT_MT, will verify and create the index
    if necessary on initialization. This is because there is no logic which runs
    on cluster restart which scans through all search settings over all tenants
    and creates the relevant indices.

    Args:
        tenant_state: The tenant state of the caller.
        index_name: The name of the index to interact with.
        embedding_dim: The dimensionality of the embeddings used for the index.
        embedding_precision: The precision of the embeddings used for the index.
    """

    def __init__(
        self,
        tenant_state: TenantState,
        index_name: str,
        embedding_dim: int,
        embedding_precision: EmbeddingPrecision,
    ) -> None:
        self._index_name: str = index_name
        self._tenant_state: TenantState = tenant_state
        self._client = ElasticsearchIndexClient(index_name=self._index_name)

        if (
            self._tenant_state.multitenant
            and VERIFY_CREATE_ELASTICSEARCH_INDEX_ON_INIT_MT
            and index_name not in _verified_index_names_for_current_process
        ):
            self.verify_and_create_index_if_necessary(
                embedding_dim=embedding_dim, embedding_precision=embedding_precision
            )
            _verified_index_names_for_current_process.add(index_name)

    def verify_and_create_index_if_necessary(
        self,
        embedding_dim: int,
        embedding_precision: EmbeddingPrecision,  # noqa: ARG002
    ) -> None:
        """Verifies and creates the index if necessary.

        In a multitenant environment, the above steps happen explicitly on
        setup.

        Args:
            embedding_dim: Vector dimensionality for the vector similarity part
                of the search.
            embedding_precision: Precision of the values of the vectors for the
                similarity part of the search.

        Raises:
            Exception: There was an error verifying or creating the index.
        """
        logger.debug(
            "[ElasticsearchDocumentIndex] Verifying and creating index %s if necessary, with embedding dimension %s.",
            self._index_name,
            embedding_dim,
        )

        with redis_shared_lock(
            lock_name=f"{OnyxRedisLocks.ELASTICSEARCH_VERIFY_INDEX_LOCK_PREFIX}:{self._index_name}",
            max_time_lock_held_s=VERIFY_INDEX_LOCK_TTL_S,
            wait_for_lock_s=VERIFY_INDEX_LOCK_BLOCKING_TIMEOUT_S,
            logger=logger,
        ):
            expected_mappings = DocumentSchema.get_document_schema(
                embedding_dim, self._tenant_state.multitenant
            )

            if not self._client.index_exists():
                index_settings = (
                    DocumentSchema.get_index_settings_based_on_environment()
                )
                self._client.create_index(
                    mappings=expected_mappings,
                    settings=index_settings,
                )
            else:
                ensure_current_schema(
                    index_client=self._client,
                    expected_mappings=expected_mappings,
                    index_settings=(
                        DocumentSchema.get_index_settings_based_on_environment()
                    ),
                    database_has_indexed_documents=_database_has_indexed_documents,
                )

    def index(
        self,
        chunks: Iterable[DocMetadataAwareIndexChunk],
        indexing_metadata: IndexingMetadata,
    ) -> list[DocumentInsertionRecord]:
        """Indexes an iterable of document chunks into the document index.

        Groups chunks by document ID and for each document, deletes existing
        chunks and indexes the new chunks in bulk.

        NOTE: It is assumed that chunks for a given document are not spread out
        over multiple index() calls.

        Args:
            chunks: Document chunks with all of the information needed for
                indexing to the document index.
            indexing_metadata: Information about chunk counts for efficient
                cleaning / updating.

        Raises:
            Exception: Failed to index some or all of the chunks for the
                specified documents.

        Returns:
            List of document IDs which map to unique documents as well as if the
                document is newly indexed or had already existed and was just
                updated.
        """
        total_chunks = sum(
            cc.new_chunk_cnt
            for cc in indexing_metadata.doc_id_to_chunk_cnt_diff.values()
        )
        logger.debug(
            "[ElasticsearchDocumentIndex] Indexing %s chunks from %s documents for index %s.",
            total_chunks,
            len(indexing_metadata.doc_id_to_chunk_cnt_diff),
            self._index_name,
        )

        document_indexing_results: list[DocumentInsertionRecord] = []
        deleted_doc_ids: set[str] = set()
        # Buffer chunks per document as they arrive from the iterable.
        # When the document ID changes flush the buffered chunks.
        current_doc_id: str | None = None
        current_chunks: list[DocMetadataAwareIndexChunk] = []

        def _flush_chunks(doc_chunks: list[DocMetadataAwareIndexChunk]) -> None:
            assert len(doc_chunks) > 0, "doc_chunks is empty"

            # Create a batch of Elasticsearch-formatted chunks for bulk insertion.
            # Since we are doing this in batches, an error occurring midway
            # can result in a state where chunks are deleted and not all the
            # new chunks have been indexed.
            chunk_batch: list[DocumentChunk] = [
                _convert_onyx_chunk_to_elasticsearch_document(chunk)
                for chunk in doc_chunks
            ]
            onyx_document: Document = doc_chunks[0].source_document
            # First delete the doc's chunks from the index. This is so that
            # there are no dangling chunks in the index, in the event that the
            # new document's content contains fewer chunks than the previous
            # content.
            # TODO(andrei): This can possibly be made more efficient by checking
            # if the chunk count has actually decreased. This assumes that
            # overlapping chunks are perfectly overwritten. If we can't
            # guarantee that then we need the code as-is.
            if onyx_document.id not in deleted_doc_ids:
                num_chunks_deleted = self.delete(
                    onyx_document.id, onyx_document.chunk_count
                )
                deleted_doc_ids.add(onyx_document.id)
                # If we see that chunks were deleted we assume the doc already
                # existed. We record the result before bulk_index_documents
                # runs. If indexing raises, this entire result list is discarded
                # by the caller's retry logic, so early recording is safe.
                document_indexing_results.append(
                    DocumentInsertionRecord(
                        document_id=onyx_document.id,
                        already_existed=num_chunks_deleted > 0,
                    )
                )
            # Now index. This will raise if a chunk of the same ID exists, which
            # we do not expect because we should have deleted all chunks.
            try:
                self._client.bulk_index_documents(
                    documents=chunk_batch,
                    tenant_state=self._tenant_state,
                )
            except BulkIndexError as e:
                # There are several reasons why this might be raised, but the
                # most likely one is if the deletion has not had enough time to
                # propagate throughout the index, in which case this would be
                # raised with some form of "version_conflict_engine_exception
                # version conflict, document already exists" messaging.
                # Refresh the index and try one more time. We do not refresh
                # after every delete because this may become expensive.
                logger.warning(
                    "Failed to bulk index documents: %s. Refreshing index and trying again.",
                    e,
                )
                self._client.refresh_index()
                self._client.bulk_index_documents(
                    documents=chunk_batch,
                    tenant_state=self._tenant_state,
                    # At this point we know for sure some docs from this batch
                    # may exist, so we don't want to fail in that case.
                    update_if_exists=True,
                )

        for chunk in chunks:
            doc_id = chunk.source_document.id
            if doc_id != current_doc_id:
                if current_chunks:
                    _flush_chunks(current_chunks)
                current_doc_id = doc_id
                current_chunks = [chunk]
            elif len(current_chunks) >= MAX_CHUNKS_PER_DOC_BATCH:
                _flush_chunks(current_chunks)
                current_chunks = [chunk]
            else:
                current_chunks.append(chunk)

        if current_chunks:
            _flush_chunks(current_chunks)

        return document_indexing_results

    def delete(
        self,
        document_id: str,
        chunk_count: int | None = None,  # noqa: ARG002
    ) -> int:
        """Deletes all chunks for a given document.

        Does nothing if the specified document ID does not exist.

        TODO(andrei): Consider implementing this method to delete on document
        chunk IDs vs querying for matching document chunks. Unclear if this is
        any better though.

        Args:
            document_id: The unique identifier for the document as represented
                in Onyx, not necessarily in the document index.
            chunk_count: The number of chunks in Elasticsearch for the document.
                Defaults to None.

        Raises:
            Exception: Failed to delete some or all of the chunks for the
                document.

        Returns:
            The number of chunks successfully deleted.
        """
        logger.debug(
            "[ElasticsearchDocumentIndex] Deleting document %s from index %s.",
            document_id,
            self._index_name,
        )
        query_body = DocumentQuery.delete_from_document_id_query(
            document_id=document_id,
            tenant_state=self._tenant_state,
        )

        return self._client.delete_by_query(query_body)

    def _verification_chunk_ids(
        self,
        request: DocumentChunkVerificationRequest,
    ) -> list[str]:
        return [
            get_elasticsearch_doc_chunk_id(
                tenant_state=self._tenant_state,
                document_id=request.document_id,
                chunk_index=chunk.chunk_index,
            )
            for chunk in request.expected_chunks
        ]

    def verify_document_chunks(
        self,
        request: DocumentChunkVerificationRequest,
    ) -> DocumentChunkVerificationResult:
        """Verify the exact current Elasticsearch projection of one document."""

        self._client.refresh_index()
        actual_count = self._client.count_by_query(
            DocumentQuery.delete_from_document_id_query(
                document_id=request.document_id,
                tenant_state=self._tenant_state,
            )
        )
        expected_count = len(request.expected_chunks)
        if actual_count != expected_count:
            raise DocumentChunkVerificationError(
                f"document chunk count mismatch: expected {expected_count}, "
                f"found {actual_count}"
            )

        expected_ids = self._verification_chunk_ids(request)
        try:
            stored_chunks = self._client.get_document_chunks(expected_ids)
        except (ElasticsearchDocumentMissingError, ElasticsearchUpdateError) as error:
            raise DocumentChunkVerificationError(str(error)) from error
        if set(stored_chunks) != set(expected_ids):
            raise DocumentChunkVerificationError(
                "document chunk ID set does not match the deterministic expectation"
            )

        for expected_chunk, chunk_id in zip(
            request.expected_chunks, expected_ids, strict=True
        ):
            stored = stored_chunks[chunk_id]
            if (
                stored.document_id != request.document_id
                or stored.chunk_index != expected_chunk.chunk_index
                or stored.regulatory_chunk_id != expected_chunk.regulatory_chunk_id
            ):
                raise DocumentChunkVerificationError(
                    f"document chunk identity mismatch for {chunk_id}"
                )
            if stored.hidden is not request.expected_hidden:
                raise DocumentChunkVerificationError(
                    f"document chunk visibility mismatch for {chunk_id}"
                )
            vector = stored.content_vector
            if (
                not isinstance(vector, list)
                or len(vector) != request.content_vector_dimension
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    for value in vector
                )
            ):
                raise DocumentChunkVerificationError(
                    f"document chunk content vector mismatch for {chunk_id}"
                )

        return DocumentChunkVerificationResult(
            document_id=request.document_id,
            chunk_count=actual_count,
            document_chunk_ids=frozenset(expected_ids),
            hidden=request.expected_hidden,
        )

    def update_document_visibility(
        self,
        request: DocumentChunkVerificationRequest,
    ) -> None:
        """Strictly update every deterministic chunk; missing chunks are fatal."""

        try:
            self._client.bulk_update_documents(
                document_chunk_ids=self._verification_chunk_ids(request),
                properties_to_update={HIDDEN_FIELD_NAME: request.expected_hidden},
                surface_document_missing=True,
            )
        except (ElasticsearchDocumentMissingError, ElasticsearchUpdateError) as error:
            raise DocumentChunkVerificationError(str(error)) from error

    def delete_port_written_chunks(self, document_ids: list[str]) -> int:
        """Delete only port-written chunks (written_by_port=true) for the given docs.

        Used by the orphan sweep to remove a doc a create-only port copy resurrected,
        without touching a legitimately re-added doc (whose forward-written chunks are
        unmarked). Dedups and batches the ids under the Elasticsearch terms cap so a large
        mid-port purge can't build an oversized terms query. Returns chunks deleted.
        """
        unique_ids = list(dict.fromkeys(document_ids))
        if not unique_ids:
            return 0
        deleted = 0
        for i in range(0, len(unique_ids), _PORT_ORPHAN_DELETE_BATCH_SIZE):
            batch = unique_ids[i : i + _PORT_ORPHAN_DELETE_BATCH_SIZE]
            query_body = DocumentQuery.delete_port_written_chunks_query(
                document_ids=batch,
                tenant_state=self._tenant_state,
            )
            deleted += self._client.delete_by_query(query_body)
        return deleted

    def update(
        self,
        update_requests: list[MetadataUpdateRequest],
        surface_document_missing: bool = False,
    ) -> None:
        """Updates some set of chunks.

        NOTE: Will raise if one of the specified document chunks do not exist.
        This may be due to a concurrent ongoing indexing operation. In that
        event callers are expected to retry after a bit once the state of the
        document index is updated.
        NOTE: Documents whose chunk count is unknown (not yet indexed) or 0
        (e.g. concurrently deleted) are skipped with a warning rather than
        raising. The indexing pipeline will write the latest metadata shortly.
        NOTE: Will no-op if an update request has no fields to update.

        TODO(andrei): Consider exploring a batch API for Elasticsearch for this
        operation.

        Args:
            update_requests: A list of update requests, each containing a list
                of document IDs and the fields to update. The field updates
                apply to all of the specified documents in each update request.

        Raises:
            Exception: Failed to update some or all of the chunks for the
                specified documents.
        """
        logger.debug(
            "[ElasticsearchDocumentIndex] Processing %s chunk requests for index %s.",
            len(update_requests),
            self._index_name,
        )
        # When surfacing, keep going past a missing-doc request so later
        # requests still update; attribute only the docs that were truly missing.
        missing_chunk_ids: list[str] = []
        missing_document_ids: set[str] = set()
        for update_request in update_requests:
            properties_to_update: dict[str, Any] = dict()
            # TODO(andrei): Nit but consider if we can use DocumentChunk here so
            # we don't have to think about passing in the appropriate types into
            # this dict.
            if update_request.access is not None:
                properties_to_update[ACCESS_CONTROL_LIST_FIELD_NAME] = (
                    generate_elasticsearch_filtered_access_control_list(
                        update_request.access
                    )
                )
            if update_request.document_sets is not None:
                properties_to_update[DOCUMENT_SETS_FIELD_NAME] = list(
                    update_request.document_sets
                )
            if update_request.boost is not None:
                properties_to_update[GLOBAL_BOOST_FIELD_NAME] = int(
                    update_request.boost
                )
            if update_request.hidden is not None:
                properties_to_update[HIDDEN_FIELD_NAME] = update_request.hidden
            if update_request.project_ids is not None:
                properties_to_update[USER_PROJECTS_FIELD_NAME] = list(
                    update_request.project_ids
                )
            if update_request.persona_ids is not None:
                properties_to_update[PERSONAS_FIELD_NAME] = list(
                    update_request.persona_ids
                )
            if update_request.created_at is not None:
                # Stored as epoch seconds
                properties_to_update[CREATED_AT_FIELD_NAME] = int(
                    datetime_to_utc(update_request.created_at).timestamp()
                )

            if not properties_to_update:
                if len(update_request.document_ids) > 1:
                    update_string = f"{len(update_request.document_ids)} documents"
                else:
                    update_string = f"document {update_request.document_ids[0]}"
                logger.warning(
                    "[ElasticsearchDocumentIndex] Tried to update %s with no specified update fields. This will be a no-op.",
                    update_string,
                )
                continue

            doc_chunk_ids_to_update: list[str] = []
            chunk_id_to_doc_id: dict[str, str] = {}
            for doc_id in update_request.document_ids:
                doc_chunk_count = update_request.doc_id_to_chunk_cnt.get(doc_id, -1)
                if doc_chunk_count < 0:
                    # The chunk count is not known. This is a benign race between
                    # doc indexing and this update step, which run concurrently
                    # when a doc is indexed. The indexing step will set the chunk
                    # count (and write the latest metadata/permissions) shortly,
                    # so skip this doc rather than failing the whole update.
                    # TODO(andrei): Fix the aforementioned race condition.
                    logger.warning(
                        "[ElasticsearchDocumentIndex] Skipping update for document %s: "
                        "its chunk count is not yet known. The document was likely just "
                        "added to the indexing pipeline and the chunk count will be "
                        "updated shortly.",
                        doc_id,
                    )
                    continue
                if doc_chunk_count == 0:
                    # A chunk count of 0 typically reflects a concurrent delete +
                    # metadata sync. There are no chunks to update, so skip this
                    # doc rather than failing the whole update.
                    logger.warning(
                        "[ElasticsearchDocumentIndex] Skipping update for document %s: "
                        "its chunk count is 0.",
                        doc_id,
                    )
                    continue

                for chunk_index in range(doc_chunk_count):
                    document_chunk_id = get_elasticsearch_doc_chunk_id(
                        tenant_state=self._tenant_state,
                        document_id=doc_id,
                        chunk_index=chunk_index,
                    )
                    doc_chunk_ids_to_update.append(document_chunk_id)
                    chunk_id_to_doc_id[document_chunk_id] = doc_id

            try:
                self._client.bulk_update_documents(
                    document_chunk_ids=doc_chunk_ids_to_update,
                    properties_to_update=properties_to_update,
                    # Normal metadata sync tolerates benign 404s (indexing race);
                    # a port surfaces them instead so deferred-sync can retry.
                    ignore_missing=not surface_document_missing,
                    surface_document_missing=surface_document_missing,
                )
            except ElasticsearchDocumentMissingError as e:
                # Only raised when surfacing; record the missing docs and keep
                # processing the remaining requests.
                missing_chunk_ids.extend(e.missing_chunk_ids)
                missing_document_ids.update(
                    chunk_id_to_doc_id[cid]
                    for cid in e.missing_chunk_ids
                    if cid in chunk_id_to_doc_id
                )

        if missing_chunk_ids:
            raise ElasticsearchDocumentMissingError(
                missing_chunk_ids, sorted(missing_document_ids)
            )

    def update_regulatory_validity(
        self,
        *,
        document_id: str,
        expected_regulatory_chunk_ids: list[str],
        previous_start_date: datetime.date | None,
        previous_end_date: datetime.date | None,
        updated_start_date: datetime.date | None,
        updated_end_date: datetime.date | None,
    ) -> None:
        """Patch validity metadata after an exact canonical projection check.

        Elasticsearch has no multi-document transaction. A failed bulk update is
        therefore compensated back to the preflight-verified uniform previous
        window before the error is surfaced to the caller.
        """

        document_chunk_ids = [
            get_elasticsearch_doc_chunk_id(
                tenant_state=self._tenant_state,
                document_id=document_id,
                chunk_index=chunk_index,
            )
            for chunk_index in range(len(expected_regulatory_chunk_ids))
        ]
        expected_chunks = {
            chunk_id: (
                document_id,
                chunk_index,
                expected_regulatory_chunk_ids[chunk_index],
            )
            for chunk_index, chunk_id in enumerate(document_chunk_ids)
        }
        previous_properties: dict[str, Any] = {
            VALIDITY_START_DATE_FIELD_NAME: _date_to_epoch_seconds(previous_start_date),
            VALIDITY_END_DATE_FIELD_NAME: _date_to_epoch_seconds(previous_end_date),
        }
        updated_properties: dict[str, Any] = {
            VALIDITY_START_DATE_FIELD_NAME: _date_to_epoch_seconds(updated_start_date),
            VALIDITY_END_DATE_FIELD_NAME: _date_to_epoch_seconds(updated_end_date),
        }

        self._client.validate_regulatory_chunk_projection(
            expected_chunks,
            validity_start_date=previous_properties[VALIDITY_START_DATE_FIELD_NAME],
            validity_end_date=previous_properties[VALIDITY_END_DATE_FIELD_NAME],
        )
        try:
            self._client.bulk_update_documents(
                document_chunk_ids=document_chunk_ids,
                properties_to_update=updated_properties,
                surface_document_missing=True,
            )
        except Exception:
            try:
                self._client.bulk_update_documents(
                    document_chunk_ids=document_chunk_ids,
                    properties_to_update=previous_properties,
                    surface_document_missing=True,
                )
            except Exception as rollback_error:
                raise ElasticsearchUpdateError(
                    "Regulatory validity update failed and its compensating "
                    f"rollback also failed for document {document_id}."
                ) from rollback_error
            raise

    def id_based_retrieval(
        self,
        chunk_requests: list[DocumentSectionRequest],
        filters: IndexFilters,
        # TODO(andrei): Remove this from the new interface at some point; we
        # should not be exposing this.
        batch_retrieval: bool = False,  # noqa: ARG002
        # TODO(andrei): Add a param for whether to retrieve hidden docs.
    ) -> list[InferenceChunk]:
        """
        TODO(andrei): Consider implementing this method to retrieve on document
        chunk IDs vs querying for matching document chunks.
        """
        logger.debug(
            "[ElasticsearchDocumentIndex] Retrieving %s chunks for index %s.",
            len(chunk_requests),
            self._index_name,
        )
        results: list[InferenceChunk] = []
        for chunk_request in chunk_requests:
            search_hits: list[SearchHit[DocumentChunkWithoutVectors]] = []
            query_body = DocumentQuery.get_from_document_id_query(
                document_id=chunk_request.document_id,
                tenant_state=self._tenant_state,
                # NOTE: Index filters includes metadata tags which were filtered
                # for invalid unicode at indexing time. In theory it would be
                # ideal to do filtering here as well, in practice we never did
                # that in the Vespa codepath and have not seen issues in
                # production, so we deliberately conform to the existing logic
                # in order to not unknowningly introduce a possible bug.
                index_filters=filters,
                include_hidden=False,
                max_chunk_size=chunk_request.max_chunk_size,
                min_chunk_index=chunk_request.min_chunk_ind,
                max_chunk_index=chunk_request.max_chunk_ind,
            )
            search_hits = self._client.search(
                body=query_body,
                normalization_method=None,
                search_type=ElasticsearchSearchType.DOC_ID_RETRIEVAL,
            )
            inference_chunks_uncleaned: list[InferenceChunkUncleaned] = [
                convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
                    search_hit.document_chunk, None, {}
                )
                for search_hit in search_hits
            ]
            inference_chunks: list[InferenceChunk] = cleanup_content_for_chunks(
                inference_chunks_uncleaned
            )
            results.extend(inference_chunks)
        return results

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        # TODO(andrei): This param is not great design, get rid of it.
        final_keywords: list[str] | None,
        query_type: QueryType,  # noqa: ARG002
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        # TODO(andrei): There is some duplicated logic in this function with
        # others in this file.
        logger.debug(
            "[ElasticsearchDocumentIndex] Hybrid retrieving %s chunks for index %s.",
            num_to_retrieve,
            self._index_name,
        )
        # TODO(andrei): This could be better, the caller should just make this
        # decision when passing in the query param. See the above comment in the
        # function signature.
        final_query = " ".join(final_keywords) if final_keywords else query
        query_body = DocumentQuery.get_hybrid_search_query(
            query_text=final_query,
            query_vector=query_embedding,
            num_hits=num_to_retrieve,
            tenant_state=self._tenant_state,
            # NOTE: Index filters includes metadata tags which were filtered
            # for invalid unicode at indexing time. In theory it would be
            # ideal to do filtering here as well, in practice we never did
            # that in the Vespa codepath and have not seen issues in
            # production, so we deliberately conform to the existing logic
            # in order to not unknowningly introduce a possible bug.
            index_filters=filters,
            include_hidden=False,
        )
        normalization_method, _ = get_normalization_method_and_config()
        search_hits: list[SearchHit[DocumentChunkWithoutVectors]] = self._client.search(
            body=query_body,
            normalization_method=normalization_method,
            search_type=ElasticsearchSearchType.HYBRID,
        )

        # Good place for a breakpoint to inspect the search hits if you have
        # "explain" enabled.
        inference_chunks_uncleaned: list[InferenceChunkUncleaned] = [
            convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
                search_hit.document_chunk, search_hit.score, search_hit.match_highlights
            )
            for search_hit in search_hits
        ]
        inference_chunks: list[InferenceChunk] = cleanup_content_for_chunks(
            inference_chunks_uncleaned
        )

        return inference_chunks

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
        include_hidden: bool = False,
        high_term_coverage: bool = False,
    ) -> list[InferenceChunk]:
        # TODO(andrei): There is some duplicated logic in this function with
        # others in this file.
        logger.debug(
            "[ElasticsearchDocumentIndex] Keyword retrieving %s chunks for index %s.",
            num_to_retrieve,
            self._index_name,
        )
        query_body = DocumentQuery.get_keyword_search_query(
            query_text=query,
            num_hits=num_to_retrieve,
            tenant_state=self._tenant_state,
            # NOTE: Index filters includes metadata tags which were filtered
            # for invalid unicode at indexing time. In theory it would be
            # ideal to do filtering here as well, in practice we never did
            # that in the Vespa codepath and have not seen issues in
            # production, so we deliberately conform to the existing logic
            # in order to not unknowningly introduce a possible bug.
            index_filters=filters,
            include_hidden=include_hidden,
            high_term_coverage=high_term_coverage,
        )
        search_hits: list[SearchHit[DocumentChunkWithoutVectors]] = self._client.search(
            body=query_body,
            normalization_method=None,
            search_type=ElasticsearchSearchType.KEYWORD,
        )

        inference_chunks_uncleaned: list[InferenceChunkUncleaned] = [
            convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
                search_hit.document_chunk, search_hit.score, search_hit.match_highlights
            )
            for search_hit in search_hits
        ]
        inference_chunks: list[InferenceChunk] = cleanup_content_for_chunks(
            inference_chunks_uncleaned
        )

        return inference_chunks

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        # TODO(andrei): There is some duplicated logic in this function with
        # others in this file.
        logger.debug(
            "[ElasticsearchDocumentIndex] Semantic retrieving %s chunks for index %s.",
            num_to_retrieve,
            self._index_name,
        )
        query_body = DocumentQuery.get_semantic_search_query(
            query_embedding=query_embedding,
            num_hits=num_to_retrieve,
            tenant_state=self._tenant_state,
            # NOTE: Index filters includes metadata tags which were filtered
            # for invalid unicode at indexing time. In theory it would be
            # ideal to do filtering here as well, in practice we never did
            # that in the Vespa codepath and have not seen issues in
            # production, so we deliberately conform to the existing logic
            # in order to not unknowningly introduce a possible bug.
            index_filters=filters,
            include_hidden=False,
        )
        search_hits: list[SearchHit[DocumentChunkWithoutVectors]] = self._client.search(
            body=query_body,
            normalization_method=None,
            search_type=ElasticsearchSearchType.SEMANTIC,
        )

        inference_chunks_uncleaned: list[InferenceChunkUncleaned] = [
            convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
                search_hit.document_chunk, search_hit.score, search_hit.match_highlights
            )
            for search_hit in search_hits
        ]
        inference_chunks: list[InferenceChunk] = cleanup_content_for_chunks(
            inference_chunks_uncleaned
        )

        return inference_chunks

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
        dirty: bool | None = None,  # noqa: ARG002
    ) -> list[InferenceChunk]:
        logger.debug(
            "[ElasticsearchDocumentIndex] Randomly retrieving %s chunks for index %s.",
            num_to_retrieve,
            self._index_name,
        )
        query_body = DocumentQuery.get_random_search_query(
            tenant_state=self._tenant_state,
            index_filters=filters,
            num_to_retrieve=num_to_retrieve,
        )
        search_hits: list[SearchHit[DocumentChunkWithoutVectors]] = self._client.search(
            body=query_body,
            normalization_method=None,
            search_type=ElasticsearchSearchType.RANDOM,
        )
        inference_chunks_uncleaned: list[InferenceChunkUncleaned] = [
            convert_retrieved_elasticsearch_chunk_to_inference_chunk_uncleaned(
                search_hit.document_chunk, search_hit.score, search_hit.match_highlights
            )
            for search_hit in search_hits
        ]
        inference_chunks: list[InferenceChunk] = cleanup_content_for_chunks(
            inference_chunks_uncleaned
        )

        return inference_chunks

    def index_raw_chunks(
        self, chunks: list[DocumentChunk], use_create_only: bool = False
    ) -> None:
        """Indexes raw document chunks into Elasticsearch.

        Used by the Vespa migration task and the reindex port. The reindex port
        passes use_create_only=True so its stale backlog snapshot can never
        overwrite a chunk a live/forward writer already owns in FUTURE (an
        existing chunk is a benign 409). The port is pure gap-fill backfill of
        PRESENT, which is always >= the port in recency, so it never needs to
        overwrite an existing chunk.
        """
        logger.debug(
            "[ElasticsearchDocumentIndex] Indexing %s raw chunks for index %s.",
            len(chunks),
            self._index_name,
        )
        # Migration path (use_create_only=False): update_if_exists overwrites,
        # since the doc may already have been indexed during the Elasticsearch
        # transition period. Port path (use_create_only=True): create-only, so
        # it never overwrites.
        self._client.bulk_index_documents(
            documents=chunks,
            tenant_state=self._tenant_state,
            update_if_exists=True,
            use_create_only=use_create_only,
        )


class ElasticsearchIndexPair(DocumentIndex):
    """Pair wrapper that fans operations out to a primary Elasticsearch index and
    an optional secondary one.

    Mirrors the previous ``ElasticsearchOldDocumentIndex`` semantics minus the
    OLD-interface translation:
      - `index` writes only to primary (a separate pipeline backfills
        secondary).
      - `delete`, `update`, `verify_and_create_index_if_necessary` fan out to
        both.
      - All retrieval goes to primary.
    """

    def __init__(
        self,
        primary: ElasticsearchDocumentIndex,
        secondary: ElasticsearchDocumentIndex | None,
        # Embedding info needed at verify-and-create time per index.
        # TODO(andrei): This is dumb, fix this.
        secondary_embedding_dim: int | None = None,
        secondary_embedding_precision: EmbeddingPrecision | None = None,
        # INSTANT reindex-port: primary is a promoted, still-backfilling index; see update().
        primary_backfill_in_progress: bool = False,
    ) -> None:
        # All three secondary fields must be set together or all None — checked
        # independently so a partially-set state surfaces here rather than
        # deferring to a less informative assertion in verify_and_create.
        secondary_set = secondary is not None
        dim_set = secondary_embedding_dim is not None
        precision_set = secondary_embedding_precision is not None
        if not (secondary_set == dim_set == precision_set):
            raise ValueError(
                "Bug: Secondary ElasticsearchDocumentIndex, secondary_embedding_dim, and "
                "secondary_embedding_precision must all be set together or all be None. Got: "
                f"secondary={secondary_set}, embedding_dim={dim_set}, "
                f"embedding_precision={precision_set}."
            )
        self._primary = primary
        self._secondary = secondary
        self._secondary_embedding_dim = secondary_embedding_dim
        self._secondary_embedding_precision = secondary_embedding_precision
        self._primary_backfill_in_progress = primary_backfill_in_progress

    def verify_and_create_index_if_necessary(
        self,
        embedding_dim: int,
        embedding_precision: EmbeddingPrecision,
    ) -> None:
        self._primary.verify_and_create_index_if_necessary(
            embedding_dim, embedding_precision
        )
        if self._secondary is not None:
            assert self._secondary_embedding_dim is not None, (
                "Bug: Secondary embedding dimension is not set."
            )
            assert self._secondary_embedding_precision is not None, (
                "Bug: Secondary embedding precision is not set."
            )
            self._secondary.verify_and_create_index_if_necessary(
                self._secondary_embedding_dim, self._secondary_embedding_precision
            )

    def index(
        self,
        chunks: Iterable[DocMetadataAwareIndexChunk],
        indexing_metadata: IndexingMetadata,
    ) -> list[DocumentInsertionRecord]:
        return self._primary.index(chunks, indexing_metadata)

    def verify_document_chunks(
        self,
        request: DocumentChunkVerificationRequest,
    ) -> DocumentChunkVerificationResult:
        # Forward writes target the primary; the secondary is populated by the
        # separate port pipeline and is not part of this publication boundary.
        return self._primary.verify_document_chunks(request)

    def update_document_visibility(
        self,
        request: DocumentChunkVerificationRequest,
    ) -> None:
        self._primary.update_document_visibility(request)
        if self._secondary is not None:
            try:
                self._secondary.update_document_visibility(request)
            except DocumentChunkVerificationError as error:
                raise SecondaryIndexDocumentMissingError(
                    [request.document_id]
                ) from error

    def delete(self, document_id: str, chunk_count: int | None = None) -> int:
        total = self._primary.delete(document_id, chunk_count)
        if self._secondary is not None:
            total += self._secondary.delete(document_id, chunk_count)
        return total

    def update(self, update_requests: list[MetadataUpdateRequest]) -> None:
        if self._primary_backfill_in_progress:
            # A doc the port hasn't copied into this now-live primary yet is silently
            # missing; surface it (typed signal, like secondary) so the caller defers
            # instead of clearing needs_sync and letting the create-only port reinstall
            # a stale, possibly-revoked ACL nothing would correct.
            try:
                self._primary.update(update_requests, surface_document_missing=True)
            except ElasticsearchDocumentMissingError as e:
                raise SecondaryIndexDocumentMissingError(e.missing_document_ids)
        else:
            self._primary.update(update_requests)
        if self._secondary is not None:
            # FUTURE may not have the doc yet (port); re-raise as a typed signal
            # carrying only the docs that were actually missing.
            try:
                self._secondary.update(update_requests, surface_document_missing=True)
            except ElasticsearchDocumentMissingError as e:
                raise SecondaryIndexDocumentMissingError(e.missing_document_ids)

    def id_based_retrieval(
        self,
        chunk_requests: list[DocumentSectionRequest],
        filters: IndexFilters,
        batch_retrieval: bool = False,
    ) -> list[InferenceChunk]:
        return self._primary.id_based_retrieval(
            chunk_requests, filters, batch_retrieval
        )

    def hybrid_retrieval(
        self,
        query: str,
        query_embedding: Embedding,
        final_keywords: list[str] | None,
        query_type: QueryType,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        return self._primary.hybrid_retrieval(
            query,
            query_embedding,
            final_keywords,
            query_type,
            filters,
            num_to_retrieve,
        )

    def keyword_retrieval(
        self,
        query: str,
        filters: IndexFilters,
        num_to_retrieve: int,
        include_hidden: bool = False,
        high_term_coverage: bool = False,
    ) -> list[InferenceChunk]:
        return self._primary.keyword_retrieval(
            query,
            filters,
            num_to_retrieve,
            include_hidden=include_hidden,
            high_term_coverage=high_term_coverage,
        )

    def semantic_retrieval(
        self,
        query_embedding: Embedding,
        filters: IndexFilters,
        num_to_retrieve: int,
    ) -> list[InferenceChunk]:
        return self._primary.semantic_retrieval(
            query_embedding, filters, num_to_retrieve
        )

    def random_retrieval(
        self,
        filters: IndexFilters,
        num_to_retrieve: int = 10,
        dirty: bool | None = None,
    ) -> list[InferenceChunk]:
        return self._primary.random_retrieval(filters, num_to_retrieve, dirty)

    @property
    def primary(self) -> ElasticsearchDocumentIndex:
        return self._primary

    @property
    def secondary(self) -> ElasticsearchDocumentIndex | None:
        return self._secondary
