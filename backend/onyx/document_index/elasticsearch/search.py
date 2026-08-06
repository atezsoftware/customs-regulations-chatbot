import random
from datetime import date, datetime, timedelta, timezone
from typing import Any, TypeAlias, TypeVar

from onyx.configs.app_configs import (
    DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S,
    ELASTICSEARCH_EXPLAIN_ENABLED,
    ELASTICSEARCH_MATCH_HIGHLIGHTS_DISABLED,
    ELASTICSEARCH_PROFILING_DISABLED,
)
from onyx.configs.constants import INDEX_SEPARATOR, DocumentSource
from onyx.context.search.models import IndexFilters, Tag, TimeRange
from onyx.document_index.elasticsearch.constants import (
    ASSUMED_DOCUMENT_AGE_DAYS,
    DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW,
    DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES,
    HYBRID_SEARCH_NORMALIZATION_METHOD,
    HYBRID_SEARCH_SUBQUERY_CONFIGURATION,
    LEGAL_DATE_EXACT_BOOST,
    LEGAL_DECISION_NUMBER_EXACT_BOOST,
    LEGAL_PROVISION_EXACT_BOOST,
    HybridSearchNormalizationMethod,
    HybridSearchSubqueryConfiguration,
)
from onyx.document_index.elasticsearch.schema import (
    ACCESS_CONTROL_LIST_FIELD_NAME,
    ANCESTOR_HIERARCHY_NODE_IDS_FIELD_NAME,
    CHUNK_INDEX_FIELD_NAME,
    CONTENT_FIELD_NAME,
    CONTENT_VECTOR_FIELD_NAME,
    CREATED_AT_FIELD_NAME,
    DECISION_NUMBERS_FIELD_NAME,
    DOCUMENT_ID_FIELD_NAME,
    DOCUMENT_SETS_FIELD_NAME,
    HEADING_PATH_FIELD_NAME,
    HIDDEN_FIELD_NAME,
    LAST_UPDATED_FIELD_NAME,
    LEGAL_DATES_FIELD_NAME,
    MAX_CHUNK_SIZE_FIELD_NAME,
    METADATA_LIST_FIELD_NAME,
    PERSONAS_FIELD_NAME,
    PROVISION_IDENTIFIERS_FIELD_NAME,
    PUBLIC_FIELD_NAME,
    REGULATORY_CHUNK_ID_FIELD_NAME,
    SOURCE_TYPE_FIELD_NAME,
    TENANT_ID_FIELD_NAME,
    TITLE_FIELD_NAME,
    TITLE_VECTOR_FIELD_NAME,
    USER_PROJECTS_FIELD_NAME,
    VALIDITY_END_DATE_FIELD_NAME,
    VALIDITY_START_DATE_FIELD_NAME,
    WRITTEN_BY_PORT_FIELD_NAME,
)
from onyx.document_index.interfaces_new import TenantState
from onyx.regulatory.exact_search_fields import extract_legal_exact_fields
from onyx.regulatory.heading_path import (
    extract_regulatory_distinctive_source_hint,
    extract_regulatory_source_hint,
    extract_single_regulatory_provision_reference,
    regulatory_provision_heading_phrases,
    regulatory_query_scope_heading_phrases,
)
from onyx.utils.datetime import datetime_to_utc

# See https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-terms-query.
MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY = 65_536

# Full-text mode is intentionally stricter than ordinary keyword search without
# requiring every word from a natural-language query to occur in one chunk.
# Elasticsearch applies the first rule for 3-5 analyzed terms and the second for
# longer queries; 1-2 term queries still require every term.
FULL_TEXT_MINIMUM_SHOULD_MATCH = "2<75% 5<60%"
_REGULATORY_EXPLICIT_SCOPE_BOOST = 32.0


_T = TypeVar("_T")
TermsQuery: TypeAlias = dict[str, dict[str, list[_T]]]
TermQuery: TypeAlias = dict[str, dict[str, dict[str, _T]]]


# TODO(andrei): Turn all magic dictionaries to pydantic models.


# Hybrid fusion combines normalized document scores from multiple retrievers.
# The number and ordering of weights should match the query clauses. The values
# of the weights should sum to 1.
def _get_hybrid_search_normalization_weights() -> list[float]:
    if (
        HYBRID_SEARCH_SUBQUERY_CONFIGURATION
        is HybridSearchSubqueryConfiguration.TITLE_VECTOR_CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD
    ):
        # Since the titles are included in the contents, the embedding matches
        # are heavily downweighted as they act as a boost rather than an
        # independent scoring component.
        search_title_vector_weight = 0.1
        search_content_vector_weight = 0.45
        # Single keyword weight for both title and content (merged from former
        # title keyword + content keyword).
        search_keyword_weight = 0.45

        # NOTE: It is critical that the order of these weights matches the order
        # of the sub-queries in the hybrid search.
        hybrid_search_normalization_weights = [
            search_title_vector_weight,
            search_content_vector_weight,
            search_keyword_weight,
        ]
    elif (
        HYBRID_SEARCH_SUBQUERY_CONFIGURATION
        is HybridSearchSubqueryConfiguration.CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD
    ):
        search_content_vector_weight = 0.5
        # Single keyword weight for both title and content (merged from former
        # title keyword + content keyword).
        search_keyword_weight = 0.5

        # NOTE: It is critical that the order of these weights matches the order
        # of the sub-queries in the hybrid search.
        hybrid_search_normalization_weights = [
            search_content_vector_weight,
            search_keyword_weight,
        ]
    else:
        raise ValueError(
            f"Bug: Unhandled hybrid search subquery configuration: {HYBRID_SEARCH_SUBQUERY_CONFIGURATION}."
        )

    assert sum(hybrid_search_normalization_weights) == 1.0, (
        "Bug: Hybrid search normalization weights do not sum to 1.0."
    )

    return hybrid_search_normalization_weights


def get_min_max_normalization_method_and_config() -> tuple[str, dict[str, Any]]:
    min_max_normalization_method = "minmax"
    config: dict[str, Any] = {
        "normalizer": "minmax",
        "weights": _get_hybrid_search_normalization_weights(),
        "implementation": "onyx_client_fusion",
    }
    return min_max_normalization_method, config


def get_zscore_normalization_method_and_config() -> tuple[str, dict[str, Any]]:
    zscore_normalization_method = "zscore"
    config: dict[str, Any] = {
        "normalizer": "zscore",
        "weights": _get_hybrid_search_normalization_weights(),
        "implementation": "onyx_client_fusion",
    }
    return zscore_normalization_method, config


def get_normalization_method_and_config() -> tuple[str, dict[str, Any]]:
    if HYBRID_SEARCH_NORMALIZATION_METHOD is HybridSearchNormalizationMethod.MIN_MAX:
        return get_min_max_normalization_method_and_config()
    elif HYBRID_SEARCH_NORMALIZATION_METHOD is HybridSearchNormalizationMethod.ZSCORE:
        return get_zscore_normalization_method_and_config()
    else:
        raise ValueError(
            f"Bug: Unhandled hybrid search normalization: {HYBRID_SEARCH_NORMALIZATION_METHOD}."
        )


class DocumentQuery:
    """
    TODO(andrei): Implement multi-phase search strategies.
    TODO(andrei): Implement document boost.
    TODO(andrei): Implement document age.
    """

    @staticmethod
    def get_from_document_id_query(
        document_id: str,
        tenant_state: TenantState,
        index_filters: IndexFilters,
        include_hidden: bool,
        max_chunk_size: int,
        min_chunk_index: int | None,
        max_chunk_index: int | None,
        get_full_document: bool = True,
    ) -> dict[str, Any]:
        """
        Returns a final search query which gets chunks from a given document ID.

        This query can be directly supplied to the Elasticsearch client.

        TODO(andrei): Currently capped at 10k results. Implement scroll/point in
        time for results so that we can return arbitrarily-many IDs.

        Args:
            document_id: Onyx document ID. Notably not an Elasticsearch document
                ID, which points to what Onyx would refer to as a chunk.
            tenant_state: Tenant state containing the tenant ID.
            index_filters: Filters for the document retrieval query.
            include_hidden: Whether to include hidden documents.
            max_chunk_size: Document chunks are categorized by the maximum
                number of tokens they can hold. This parameter specifies the
                maximum size category of document chunks to retrieve.
            min_chunk_index: The minimum chunk index to retrieve, inclusive. If
                None, no minimum chunk index will be applied.
            max_chunk_index: The maximum chunk index to retrieve, inclusive. If
                None, no maximum chunk index will be applied.
            get_full_document: Whether to get the full document body. If False,
                Elasticsearch will only return the matching document chunk IDs plus
                metadata; the source data will be omitted from the response. Use
                this for performance optimization if Elasticsearch IDs are
                sufficient. Defaults to True.

        Returns:
            A dictionary representing the final ID search query.
        """
        filter_clauses = DocumentQuery._get_search_filters(
            tenant_state=tenant_state,
            include_hidden=include_hidden,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            project_id_filter=index_filters.project_id_filter,
            persona_id_filter=index_filters.persona_id_filter,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            as_of_date=index_filters.as_of_date,
            regulatory_chunks_only=index_filters.regulatory_chunks_only,
            min_chunk_index=min_chunk_index,
            max_chunk_index=max_chunk_index,
            max_chunk_size=max_chunk_size,
            document_id=document_id,
            attached_document_ids=index_filters.attached_document_ids,
            hierarchy_node_ids=index_filters.hierarchy_node_ids,
        )
        final_get_ids_query: dict[str, Any] = {
            "query": {"bool": {"filter": filter_clauses}},
            # We include this to make sure Elasticsearch does not revert to
            # returning some number of results less than the index max allowed
            # return size.
            "size": DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW,
            # By default exclude retrieving the vector fields in order to save
            # on retrieval cost as we don't need them upstream.
            "_source": {
                "excludes": [TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME]
            },
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
        }
        if not get_full_document:
            # If we explicitly do not want the underlying document, we will only
            # retrieve IDs.
            final_get_ids_query["_source"] = False
        if not ELASTICSEARCH_PROFILING_DISABLED:
            final_get_ids_query["profile"] = True

        return final_get_ids_query

    @staticmethod
    def delete_from_document_id_query(
        document_id: str,
        tenant_state: TenantState,
    ) -> dict[str, Any]:
        """
        Returns a final search query which deletes chunks from a given document
        ID.

        This query can be directly supplied to the Elasticsearch client.

        Intended to be supplied to the Elasticsearch client's delete_by_query
        method.

        TODO(andrei): There is no limit to the number of document chunks that
        can be deleted by this query. This could get expensive. Consider
        implementing batching.

        Args:
            document_id: Onyx document ID. Notably not an Elasticsearch document
                ID, which points to what Onyx would refer to as a chunk.
            tenant_state: Tenant state containing the tenant ID.

        Returns:
            A dictionary representing the final delete query.
        """
        filter_clauses = DocumentQuery._get_search_filters(
            tenant_state=tenant_state,
            # Delete hidden docs too.
            include_hidden=True,
            access_control_list=None,
            source_types=[],
            tags=[],
            document_sets=[],
            project_id_filter=None,
            persona_id_filter=None,
            created_at_range=None,
            updated_at_range=None,
            min_chunk_index=None,
            max_chunk_index=None,
            max_chunk_size=None,
            document_id=document_id,
        )
        final_delete_query: dict[str, Any] = {
            "query": {"bool": {"filter": filter_clauses}},
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
        }
        if not ELASTICSEARCH_PROFILING_DISABLED:
            final_delete_query["profile"] = True

        return final_delete_query

    @staticmethod
    def delete_port_written_chunks_query(
        document_ids: list[str],
        tenant_state: TenantState,
    ) -> dict[str, Any]:
        """Delete-by-query matching only PORT-written chunks (written_by_port=true) of the
        given documents in this tenant. The orphan sweep uses it to remove a resurrected
        doc while leaving a legitimately re-added one (its forward-written chunks are
        unmarked) untouched — so no Postgres re-check is needed."""
        filter_clauses: list[dict[str, Any]] = [
            {"terms": {DOCUMENT_ID_FIELD_NAME: list(document_ids)}},
            {"term": {WRITTEN_BY_PORT_FIELD_NAME: {"value": True}}},
        ]
        # Single-tenant indices have no tenant_id field (added only in multitenant mode);
        # a term on the unmapped field would match zero docs. Mirror _get_search_filters.
        if tenant_state.multitenant:
            filter_clauses.append(
                {"term": {TENANT_ID_FIELD_NAME: {"value": tenant_state.tenant_id}}}
            )
        final_delete_query: dict[str, Any] = {
            "query": {"bool": {"filter": filter_clauses}},
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
        }
        if not ELASTICSEARCH_PROFILING_DISABLED:
            final_delete_query["profile"] = True

        return final_delete_query

    @staticmethod
    def get_hybrid_search_query(
        query_text: str,
        query_vector: list[float],
        num_hits: int,
        tenant_state: TenantState,
        index_filters: IndexFilters,
        include_hidden: bool,
    ) -> dict[str, Any]:
        """Returns a final hybrid search query.

        Elasticsearch 8.6 does not provide the retriever API, so both min-max
        and z-score modes return an internal fusion specification. The index
        client executes the same lexical and vector lanes independently and
        combines their normalized, weighted scores.

        TODO(andrei): There is some duplicated logic in this function with
        others in this file.

        Args:
            query_text: The text to query for.
            query_vector: The vector embedding of the text to query for.
            num_hits: The final number of hits to return.
            tenant_state: Tenant state containing the tenant ID.
            index_filters: Filters for the hybrid search query.
            include_hidden: Whether to include hidden documents.

        Returns:
            A dictionary representing the final hybrid search query.
        """
        # WARNING: Profiling does not work with hybrid search; do not add it at
        # this level. See https://github.com/elasticsearch-project/neural-search/issues/1255

        if num_hits > DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW:
            raise ValueError(
                f"Bug: num_hits ({num_hits}) is greater than the current maximum allowed "
                f"result window ({DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW})."
            )

        # TODO(andrei, yuhong): We can tune this more dynamically based on
        # num_hits.
        # Elasticsearch requires rank_window_size >= the requested size. Keep
        # the tuned 500-candidate floor while allowing larger explicit result
        # requests to remain valid.
        max_results_per_subquery = max(DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES, num_hits)

        hybrid_search_subqueries = DocumentQuery._get_hybrid_search_subqueries(
            query_text,
            query_vector,
            vector_candidates=max_results_per_subquery,
            preferred_source_hint=(
                extract_regulatory_source_hint(query_text)
                if index_filters.regulatory_chunks_only
                else None
            ),
            preferred_distinctive_source_hint=(
                extract_regulatory_distinctive_source_hint(query_text)
                if index_filters.regulatory_chunks_only
                else None
            ),
        )
        hybrid_search_filters = DocumentQuery._get_search_filters(
            tenant_state=tenant_state,
            include_hidden=include_hidden,
            # TODO(andrei): We've done no filtering for PUBLIC_DOC_PAT up to
            # now. This should not cause any issues but it can introduce
            # redundant filters in queries that may affect performance.
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            project_id_filter=index_filters.project_id_filter,
            persona_id_filter=index_filters.persona_id_filter,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            as_of_date=index_filters.as_of_date,
            regulatory_chunks_only=index_filters.regulatory_chunks_only,
            min_chunk_index=None,
            max_chunk_index=None,
            attached_document_ids=index_filters.attached_document_ids,
            hierarchy_node_ids=index_filters.hierarchy_node_ids,
            forced_document_sets=index_filters.forced_document_set,
        )

        final_hybrid_search_body: dict[str, Any] = {
            "size": num_hits,
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
            # Exclude retrieving the vector fields in order to save on
            # retrieval cost as we don't need them upstream.
            "_source": {
                "excludes": [TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME]
            },
        }

        normalizer = (
            "minmax"
            if HYBRID_SEARCH_NORMALIZATION_METHOD
            is HybridSearchNormalizationMethod.MIN_MAX
            else "zscore"
        )
        final_hybrid_search_body["_onyx_hybrid_fusion"] = {
            "subqueries": hybrid_search_subqueries,
            "weights": _get_hybrid_search_normalization_weights(),
            "filters": hybrid_search_filters,
            "rank_window_size": max_results_per_subquery,
            "normalizer": normalizer,
        }

        if not ELASTICSEARCH_MATCH_HIGHLIGHTS_DISABLED:
            final_hybrid_search_body["highlight"] = (
                DocumentQuery._get_match_highlights_configuration()
            )

        # Explain is for scoring breakdowns. Setting this significantly
        # increases query latency.
        if ELASTICSEARCH_EXPLAIN_ENABLED:
            final_hybrid_search_body["explain"] = True

        return final_hybrid_search_body

    @staticmethod
    def get_keyword_search_query(
        query_text: str,
        num_hits: int,
        tenant_state: TenantState,
        index_filters: IndexFilters,
        include_hidden: bool,
        high_term_coverage: bool = False,
    ) -> dict[str, Any]:
        """Returns a final keyword search query.

        This query can be directly supplied to the Elasticsearch client.

        TODO(andrei): There is some duplicated logic in this function with
        others in this file.

        Args:
            query_text: The text to query for.
            num_hits: The final number of hits to return.
            tenant_state: Tenant state containing the tenant ID.
            index_filters: Filters for the keyword search query.
            include_hidden: Whether to include hidden documents.
            high_term_coverage: Require high analyzed-term coverage instead of
                accepting a match on any one query term.

        Returns:
            A dictionary representing the final keyword search query.
        """
        if num_hits > DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW:
            raise ValueError(
                f"Bug: num_hits ({num_hits}) is greater than the current maximum allowed "
                f"result window ({DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW})."
            )

        keyword_search_filters = DocumentQuery._get_search_filters(
            tenant_state=tenant_state,
            include_hidden=include_hidden,
            # TODO(andrei): We've done no filtering for PUBLIC_DOC_PAT up to
            # now. This should not cause any issues but it can introduce
            # redundant filters in queries that may affect performance.
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            project_id_filter=index_filters.project_id_filter,
            persona_id_filter=index_filters.persona_id_filter,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            as_of_date=index_filters.as_of_date,
            regulatory_chunks_only=index_filters.regulatory_chunks_only,
            min_chunk_index=None,
            max_chunk_index=None,
            attached_document_ids=index_filters.attached_document_ids,
            hierarchy_node_ids=index_filters.hierarchy_node_ids,
            forced_document_sets=index_filters.forced_document_set,
        )

        provision_reference = (
            extract_single_regulatory_provision_reference(query_text)
            if index_filters.regulatory_chunks_only
            else None
        )
        keyword_search_query = (
            DocumentQuery._get_title_content_combined_keyword_search_query(
                query_text,
                search_filters=keyword_search_filters,
                high_term_coverage=high_term_coverage,
                required_heading_phrases=(
                    regulatory_provision_heading_phrases(provision_reference)
                    if provision_reference is not None
                    else None
                ),
                preferred_scope_heading_phrases=(
                    regulatory_query_scope_heading_phrases(query_text)
                    if index_filters.regulatory_chunks_only
                    else None
                ),
                preferred_source_hint=(
                    extract_regulatory_source_hint(query_text)
                    if index_filters.regulatory_chunks_only
                    else None
                ),
                preferred_distinctive_source_hint=(
                    extract_regulatory_distinctive_source_hint(query_text)
                    if index_filters.regulatory_chunks_only
                    else None
                ),
            )
        )

        final_keyword_search_query: dict[str, Any] = {
            "query": keyword_search_query,
            "size": num_hits,
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
            # Exclude retrieving the vector fields in order to save on
            # retrieval cost as we don't need them upstream.
            "_source": {
                "excludes": [TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME]
            },
        }

        if not ELASTICSEARCH_MATCH_HIGHLIGHTS_DISABLED:
            final_keyword_search_query["highlight"] = (
                DocumentQuery._get_match_highlights_configuration()
            )

        if not ELASTICSEARCH_PROFILING_DISABLED:
            final_keyword_search_query["profile"] = True

        # Explain is for scoring breakdowns. Setting this significantly
        # increases query latency.
        if ELASTICSEARCH_EXPLAIN_ENABLED:
            final_keyword_search_query["explain"] = True

        return final_keyword_search_query

    @staticmethod
    def get_semantic_search_query(
        query_embedding: list[float],
        num_hits: int,
        tenant_state: TenantState,
        index_filters: IndexFilters,
        include_hidden: bool,
    ) -> dict[str, Any]:
        """Returns a final semantic search query.

        This query can be directly supplied to the Elasticsearch client.

        TODO(andrei): There is some duplicated logic in this function with
        others in this file.

        Args:
            query_embedding: The vector embedding of the text to query for.
            num_hits: The final number of hits to return.
            tenant_state: Tenant state containing the tenant ID.
            index_filters: Filters for the semantic search query.
            include_hidden: Whether to include hidden documents.

        Returns:
            A dictionary representing the final semantic search query.
        """
        if num_hits > DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW:
            raise ValueError(
                f"Bug: num_hits ({num_hits}) is greater than the current maximum allowed "
                f"result window ({DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW})."
            )

        semantic_search_filters = DocumentQuery._get_search_filters(
            tenant_state=tenant_state,
            include_hidden=include_hidden,
            # TODO(andrei): We've done no filtering for PUBLIC_DOC_PAT up to
            # now. This should not cause any issues but it can introduce
            # redundant filters in queries that may affect performance.
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            project_id_filter=index_filters.project_id_filter,
            persona_id_filter=index_filters.persona_id_filter,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            as_of_date=index_filters.as_of_date,
            regulatory_chunks_only=index_filters.regulatory_chunks_only,
            min_chunk_index=None,
            max_chunk_index=None,
            attached_document_ids=index_filters.attached_document_ids,
            hierarchy_node_ids=index_filters.hierarchy_node_ids,
            forced_document_sets=index_filters.forced_document_set,
        )

        semantic_search_query = (
            DocumentQuery._get_content_vector_similarity_search_query(
                query_embedding,
                vector_candidates=num_hits,
                search_filters=semantic_search_filters,
            )
        )

        final_semantic_search_query: dict[str, Any] = {
            # Elasticsearch 8.6 supports approximate kNN through the top-level
            # search option. The Query DSL `knn` clause arrived in later 8.x
            # releases and is therefore intentionally not used here.
            "knn": semantic_search_query["knn"],
            "size": num_hits,
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
            # Exclude retrieving the vector fields in order to save on
            # retrieval cost as we don't need them upstream.
            "_source": {
                "excludes": [TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME]
            },
        }

        if not ELASTICSEARCH_PROFILING_DISABLED:
            final_semantic_search_query["profile"] = True

        # Explain is for scoring breakdowns. Setting this significantly
        # increases query latency.
        if ELASTICSEARCH_EXPLAIN_ENABLED:
            final_semantic_search_query["explain"] = True

        return final_semantic_search_query

    @staticmethod
    def get_random_search_query(
        tenant_state: TenantState,
        index_filters: IndexFilters,
        num_to_retrieve: int,
    ) -> dict[str, Any]:
        """Returns a final search query that gets document chunks randomly.

        Args:
            tenant_state: Tenant state containing the tenant ID.
            index_filters: Filters for the random search query.
            num_to_retrieve: Number of document chunks to retrieve.

        Returns:
            A dictionary representing the final random search query.
        """
        search_filters = DocumentQuery._get_search_filters(
            tenant_state=tenant_state,
            include_hidden=False,
            access_control_list=index_filters.access_control_list,
            source_types=index_filters.source_type or [],
            tags=index_filters.tags or [],
            document_sets=index_filters.document_set or [],
            project_id_filter=index_filters.project_id_filter,
            persona_id_filter=index_filters.persona_id_filter,
            created_at_range=index_filters.created_at_range,
            updated_at_range=index_filters.updated_at_range,
            as_of_date=index_filters.as_of_date,
            regulatory_chunks_only=index_filters.regulatory_chunks_only,
            min_chunk_index=None,
            max_chunk_index=None,
            attached_document_ids=index_filters.attached_document_ids,
            hierarchy_node_ids=index_filters.hierarchy_node_ids,
            forced_document_sets=index_filters.forced_document_set,
        )
        final_random_search_query = {
            "query": {
                "function_score": {
                    "query": {"bool": {"filter": search_filters}},
                    # See
                    # https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-function-score-query
                    "random_score": {
                        # We'll use a different seed per invocation.
                        "seed": random.randint(0, 1_000_000),
                        # Some field which has a unique value per document
                        # chunk.
                        "field": "_seq_no",
                    },
                    # Replaces whatever score was computed in the query.
                    "boost_mode": "replace",
                }
            },
            "size": num_to_retrieve,
            "timeout": f"{DEFAULT_ELASTICSEARCH_QUERY_TIMEOUT_S}s",
            # Exclude retrieving the vector fields in order to save on
            # retrieval cost as we don't need them upstream.
            "_source": {
                "excludes": [TITLE_VECTOR_FIELD_NAME, CONTENT_VECTOR_FIELD_NAME]
            },
        }
        if not ELASTICSEARCH_PROFILING_DISABLED:
            final_random_search_query["profile"] = True

        return final_random_search_query

    @staticmethod
    def _get_hybrid_search_subqueries(
        query_text: str,
        query_vector: list[float],
        # The default number of neighbors to consider for knn vector similarity
        # search. This is higher than the number of results because the scoring
        # is hybrid. For a detailed breakdown, see where the default value is
        # set.
        vector_candidates: int = DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES,
        preferred_source_hint: str | None = None,
        preferred_distinctive_source_hint: str | None = None,
    ) -> list[dict[str, Any]]:
        """Returns subqueries for hybrid search.

        Each of these subqueries are the "hybrid" component of this search. We
        search on various things and combine results.

        The return of this function is not sufficient to be directly supplied to
        the Elasticsearch client. See get_hybrid_search_query.

        Normalization is not performed here.
        The weights of each subquery are applied by the client-side fusion step.

        The exact subqueries executed depend on the
        HYBRID_SEARCH_SUBQUERY_CONFIGURATION setting.

        NOTE: Each query is independent during the search phase; there is no
        backfilling of scores for missing query components. What this means is
        that if a document was a good vector match but did not show up for
        keyword, it gets a score of 0 for the keyword component of the hybrid
        scoring. This is not as bad as just disregarding a score though as there
        is normalization applied after. So really it is "increasing" the missing
        score compared to if it was included and the range was renormalized.
        This does however mean that between docs that have high scores for say
        the vector field, the keyword scores between them are completely ignored
        unless they also showed up in the keyword query as a reasonably high
        match. TLDR, this is a bit of unique funky behavior but it seems ok.

        NOTE: Options considered and rejected:
        - minimum_should_match: Since it's hybrid search and users often provide
          semantic queries, there is often a lot of terms, and very low number
          of meaningful keywords (and a low ratio of keywords).
        - fuzziness AUTO: Typo tolerance (0/1/2 edit distance by term length).
          It's mostly for typos as the analyzer ("english" by default) already
          does some stemming and tokenization. In testing datasets, this makes
          recall slightly worse. It also is less performant so not really any
          reason to do it.

        Args:
            query_text: The text of the query to search for.
            query_vector: The vector embedding of the query to search for.
            num_candidates: The number of candidates to consider for vector
                similarity search.
        """
        # Build sub-queries for hybrid search. Order must match fusion weights.
        if (
            HYBRID_SEARCH_SUBQUERY_CONFIGURATION
            is HybridSearchSubqueryConfiguration.TITLE_VECTOR_CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD
        ):
            return [
                DocumentQuery._get_title_vector_similarity_search_query(
                    query_vector, vector_candidates
                ),
                DocumentQuery._get_content_vector_similarity_search_query(
                    query_vector, vector_candidates
                ),
                DocumentQuery._get_title_content_combined_keyword_search_query(
                    query_text,
                    preferred_source_hint=preferred_source_hint,
                    preferred_distinctive_source_hint=(
                        preferred_distinctive_source_hint
                    ),
                ),
            ]
        elif (
            HYBRID_SEARCH_SUBQUERY_CONFIGURATION
            is HybridSearchSubqueryConfiguration.CONTENT_VECTOR_TITLE_CONTENT_COMBINED_KEYWORD
        ):
            return [
                DocumentQuery._get_content_vector_similarity_search_query(
                    query_vector, vector_candidates
                ),
                DocumentQuery._get_title_content_combined_keyword_search_query(
                    query_text,
                    preferred_source_hint=preferred_source_hint,
                    preferred_distinctive_source_hint=(
                        preferred_distinctive_source_hint
                    ),
                ),
            ]
        else:
            raise ValueError(
                f"Bug: Unhandled hybrid search subquery configuration: {HYBRID_SEARCH_SUBQUERY_CONFIGURATION}"
            )

    @staticmethod
    def _get_title_vector_similarity_search_query(
        query_vector: list[float],
        vector_candidates: int = DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES,
    ) -> dict[str, Any]:
        return {
            "knn": {
                "field": TITLE_VECTOR_FIELD_NAME,
                "query_vector": query_vector,
                "k": vector_candidates,
                "num_candidates": vector_candidates,
            }
        }

    @staticmethod
    def _get_content_vector_similarity_search_query(
        query_vector: list[float],
        vector_candidates: int = DEFAULT_NUM_HYBRID_SUBQUERY_CANDIDATES,
        search_filters: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        query = {
            "knn": {
                "field": CONTENT_VECTOR_FIELD_NAME,
                "query_vector": query_vector,
                "k": vector_candidates,
                "num_candidates": vector_candidates,
            }
        }

        if search_filters is not None:
            query["knn"]["filter"] = {"bool": {"filter": search_filters}}  # ty: ignore[invalid-assignment]

        return query

    @staticmethod
    def _get_title_content_combined_keyword_search_query(
        query_text: str,
        search_filters: list[dict[str, Any]] | None = None,
        high_term_coverage: bool = False,
        required_heading_phrases: tuple[str, ...] | None = None,
        preferred_scope_heading_phrases: tuple[str, ...] | None = None,
        preferred_source_hint: str | None = None,
        preferred_distinctive_source_hint: str | None = None,
    ) -> dict[str, Any]:
        lexical_match_parameters: dict[str, Any] = {"query": query_text}
        if high_term_coverage:
            lexical_match_parameters["minimum_should_match"] = (
                FULL_TEXT_MINIMUM_SHOULD_MATCH
            )
        else:
            lexical_match_parameters["operator"] = "or"

        lexical_should_clauses: list[dict[str, Any]] = [
            {
                "match": {
                    TITLE_FIELD_NAME: {
                        **lexical_match_parameters,
                        # The title fields are strongly discounted as they are
                        # included in the content. This is only a minor boost.
                        "boost": 0.1,
                    }
                }
            },
            {
                "match_phrase": {
                    TITLE_FIELD_NAME: {
                        "query": query_text,
                        "slop": 1,
                        "boost": 0.2,
                    }
                }
            },
            {
                "match": {
                    CONTENT_FIELD_NAME: {
                        **lexical_match_parameters,
                        "boost": 1.0,
                    }
                }
            },
            {
                "match_phrase": {
                    CONTENT_FIELD_NAME: {
                        "query": query_text,
                        # Number of words allowed between phrase terms.
                        "slop": 1,
                        "boost": 1.5,
                    }
                }
            },
            {
                # Regulatory headings carry identifiers that may not be
                # repeated in the operative chunk text.
                "match": {
                    HEADING_PATH_FIELD_NAME: {
                        **lexical_match_parameters,
                        "boost": 1.2,
                    }
                }
            },
            {
                "match_phrase": {
                    HEADING_PATH_FIELD_NAME: {
                        "query": query_text,
                        "slop": 1,
                        "boost": 2.0,
                    }
                }
            },
        ]
        query = {
            "bool": {
                "should": lexical_should_clauses,
                # Ensures at least one match subquery from the query is present
                # in the document. This defaults to 1, unless a filter or must
                # clause is supplied, in which case it defaults to 0.
                "minimum_should_match": 1,
            }
        }

        legal_exact_fields = extract_legal_exact_fields(query_text)
        exact_field_boosts = (
            (
                PROVISION_IDENTIFIERS_FIELD_NAME,
                legal_exact_fields.provision_identifiers,
                LEGAL_PROVISION_EXACT_BOOST,
            ),
            (
                DECISION_NUMBERS_FIELD_NAME,
                legal_exact_fields.decision_numbers,
                LEGAL_DECISION_NUMBER_EXACT_BOOST,
            ),
            (
                LEGAL_DATES_FIELD_NAME,
                legal_exact_fields.legal_dates,
                LEGAL_DATE_EXACT_BOOST,
            ),
        )
        for field_name, values, boost in exact_field_boosts:
            if values:
                lexical_should_clauses.append(
                    {
                        "constant_score": {
                            "filter": {"terms": {field_name: values}},
                            "boost": boost,
                        }
                    }
                )

        if required_heading_phrases:
            query["bool"]["must"] = [
                {
                    "bool": {
                        "should": [
                            {
                                "match_phrase": {
                                    HEADING_PATH_FIELD_NAME: {
                                        "query": phrase,
                                        "slop": 0,
                                    }
                                }
                            }
                            for phrase in required_heading_phrases
                        ],
                        "minimum_should_match": 1,
                    }
                }
            ]
            # The structural clause establishes candidate identity. Topic and
            # source terms remain optional ranking signals so morphology or a
            # minor spelling difference cannot suppress the exact provision.
            query["bool"]["minimum_should_match"] = 0
            lexical_should_clauses.append(
                {
                    "multi_match": {
                        "query": query_text,
                        "fields": [
                            f"{TITLE_FIELD_NAME}^0.5",
                            CONTENT_FIELD_NAME,
                            f"{HEADING_PATH_FIELD_NAME}^1.2",
                        ],
                        "fuzziness": "AUTO",
                        "prefix_length": 1,
                        "boost": 0.8,
                    }
                }
            )

        ranking_should_clauses: list[dict[str, Any]] = []
        if preferred_scope_heading_phrases:
            ranking_should_clauses.append(
                {
                    "dis_max": {
                        "queries": [
                            {
                                "constant_score": {
                                    "filter": {
                                        "match_phrase": {
                                            HEADING_PATH_FIELD_NAME: {
                                                "query": phrase,
                                                "slop": 0,
                                            }
                                        }
                                    },
                                    "boost": _REGULATORY_EXPLICIT_SCOPE_BOOST,
                                }
                            }
                            for phrase in preferred_scope_heading_phrases
                        ],
                        "tie_breaker": 0.0,
                    }
                }
            )

        if preferred_source_hint:
            source_boost_queries: list[dict[str, Any]] = [
                {
                    "constant_score": {
                        "filter": {
                            "multi_match": {
                                "query": preferred_source_hint,
                                "fields": [
                                    f"{TITLE_FIELD_NAME}^2.0",
                                    f"{HEADING_PATH_FIELD_NAME}^3.0",
                                ],
                                "minimum_should_match": (
                                    FULL_TEXT_MINIMUM_SHOULD_MATCH
                                ),
                                "fuzziness": "AUTO",
                                "prefix_length": 1,
                            }
                        },
                        "boost": 8.0,
                    }
                }
            ]
            if (
                preferred_distinctive_source_hint
                and preferred_distinctive_source_hint != preferred_source_hint
            ):
                source_boost_queries.append(
                    {
                        "constant_score": {
                            "filter": {
                                "multi_match": {
                                    "query": preferred_distinctive_source_hint,
                                    "fields": [
                                        f"{TITLE_FIELD_NAME}^2.0",
                                        f"{HEADING_PATH_FIELD_NAME}^3.0",
                                    ],
                                    "minimum_should_match": "100%",
                                    "fuzziness": "AUTO",
                                    "prefix_length": 1,
                                }
                            },
                            "boost": 8.0,
                        }
                    }
                )
            ranking_should_clauses.append(
                {
                    "dis_max": {
                        "queries": source_boost_queries,
                        "tie_breaker": 0.0,
                    }
                }
            )

        if high_term_coverage:
            # A cross-fields clause permits a source/provision identifier in a
            # heading and its operative terms in content to satisfy one bounded
            # lexical match. All three fields share the same analyzer.
            lexical_should_clauses.append(
                {
                    "multi_match": {
                        "query": query_text,
                        "type": "cross_fields",
                        "fields": [
                            f"{TITLE_FIELD_NAME}^0.1",
                            CONTENT_FIELD_NAME,
                            f"{HEADING_PATH_FIELD_NAME}^1.2",
                        ],
                        "minimum_should_match": FULL_TEXT_MINIMUM_SHOULD_MATCH,
                    }
                }
            )

        if ranking_should_clauses:
            if required_heading_phrases:
                # Exact structural identity is already the candidate gate. Keep
                # every source/scope signal optional so it affects only order.
                lexical_should_clauses.extend(ranking_should_clauses)
            else:
                # A source or scope label is a ranking hint, not evidence that
                # the requested operative terms occur in this chunk. Nest the
                # lexical alternatives under a required clause so a title-only
                # source match cannot admit an otherwise unrelated chunk.
                query["bool"]["must"] = [
                    {
                        "bool": {
                            "should": lexical_should_clauses,
                            "minimum_should_match": 1,
                        }
                    }
                ]
                query["bool"]["should"] = ranking_should_clauses
                query["bool"]["minimum_should_match"] = 0

        if search_filters is not None:
            query["bool"]["filter"] = search_filters

        return query

    @staticmethod
    def _get_search_filters(
        tenant_state: TenantState,
        include_hidden: bool,
        access_control_list: list[str] | None,
        source_types: list[DocumentSource],
        tags: list[Tag],
        document_sets: list[str],
        project_id_filter: int | None,
        persona_id_filter: int | None,
        created_at_range: TimeRange | None,
        updated_at_range: TimeRange | None,
        min_chunk_index: int | None,
        max_chunk_index: int | None,
        as_of_date: date | None = None,
        regulatory_chunks_only: bool = False,
        max_chunk_size: int | None = None,
        document_id: str | None = None,
        # Assistant knowledge filters
        attached_document_ids: list[str] | None = None,
        hierarchy_node_ids: list[int] | None = None,
        # Operator-forced document-set scope (NAMES), applied as a standalone AND clause.
        forced_document_sets: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Returns filters to be passed into the "filter" key of a search query.

        The "filter" key applies a logical AND operator to its elements, so
        every subfilter must evaluate to true in order for the document to be
        retrieved. This function returns a list of such subfilters.
        See https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-bool-query.

        TODO(ENG-3874): The terms queries returned by this function can be made
        more performant for large cardinality sets by sorting the values by
        their UTF-8 byte order.

        TODO(ENG-3875): This function can take even better advantage of filter
        caching by grouping "static" filters together into one sub-clause.

        Args:
            tenant_state: Tenant state containing the tenant ID.
            include_hidden: Whether to include hidden documents.
            access_control_list: Access control list for the documents to
                retrieve. If None, there is no restriction on the documents that
                can be retrieved. If not None, only public documents can be
                retrieved, or non-public documents where at least one acl
                provided here is present in the document's acl list.
            source_types: If supplied, only documents of one of these source
                types will be retrieved.
            tags: If supplied, only documents with an entry in their metadata
                list corresponding to a tag will be retrieved.
            document_sets: If supplied, only documents with at least one
                document set ID from this list will be retrieved.
            project_id_filter: If not None, only documents with this project ID
                in user projects will be retrieved. Additive — only applied
                when a knowledge scope already exists.
            persona_id_filter: If not None, only documents whose personas array
                contains this persona ID will be retrieved. Primary — creates
                a knowledge scope on its own.
            created_at_range: Inclusive window on the document's created_at.
            updated_at_range: Inclusive window on the document's last_updated.
                See document_index/FILTER_SEMANTICS.md ("Time filtering").
            as_of_date: Regulatory temporal filter — restricts results to the
                half-open validity window (validity_start_date <= as_of_date AND
                (validity_end_date is unset OR > as_of_date)).
                Chunks without a validity window (every non-regulatory chunk)
                always pass. None disables the filter.
            regulatory_chunks_only: Require a regulatory_chunk_id, excluding
                legacy whole-file/user-file chunks from regulatory assistants.
            min_chunk_index: The minimum chunk index to retrieve, inclusive. If
                None, no minimum chunk index will be applied.
            max_chunk_index: The maximum chunk index to retrieve, inclusive. If
                None, no maximum chunk index will be applied.
            max_chunk_size: The type of chunk to retrieve, specified by the
                maximum number of tokens it can hold. If None, no filter will be
                applied for this. Defaults to None.
                NOTE: See DocumentChunk.max_chunk_size.
            document_id: The document ID to retrieve. If None, no filter will be
                applied for this. Defaults to None.
            attached_document_ids: Document IDs explicitly attached to the
                assistant. If provided along with hierarchy_node_ids, documents
                matching EITHER criteria will be retrieved (OR logic).
            hierarchy_node_ids: Hierarchy node IDs (folders/spaces) attached to
                the assistant. Matches chunks where ancestor_hierarchy_node_ids
                contains any of these values.

        Raises:
            ValueError: document_id and attached_document_ids were supplied
                together. This is not allowed because they operate on the same
                schema field, and it does not semantically make sense to use
                them together.
            ValueError: Too many of one of the collection arguments was
                supplied.

        Returns:
            A list of filters to be passed into the "filter" key of a search
                query.
        """

        def _get_acl_visibility_filter(
            access_control_list: list[str],
        ) -> dict[str, dict[str, list[TermQuery[bool] | TermsQuery[str]] | int]]:
            """Returns a filter for the access control list.

            Since this returns an isolated bool should clause, it can be cached
            in Elasticsearch independently of other clauses in _get_search_filters.

            Args:
                access_control_list: The access control list to restrict
                    documents to.

            Raises:
                ValueError: The number of access control list entries is greater
                    than MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY.

            Returns:
                A filter for the access control list.
            """
            # Logical OR operator on its elements.
            acl_visibility_filter: dict[str, dict[str, Any]] = {
                "bool": {
                    "should": [{"term": {PUBLIC_FIELD_NAME: {"value": True}}}],
                    "minimum_should_match": 1,
                }
            }
            if access_control_list:
                if len(access_control_list) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                    raise ValueError(
                        f"Too many access control list entries: {len(access_control_list)}. Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                    )
                # Use terms instead of a list of term within a should clause
                # because Lucene will optimize the filtering for large sets of
                # terms. Small sets of terms are not expected to perform any
                # differently than individual term clauses.
                acl_subclause: TermsQuery[str] = {
                    "terms": {ACCESS_CONTROL_LIST_FIELD_NAME: list(access_control_list)}
                }
                acl_visibility_filter["bool"]["should"].append(
                    acl_subclause  # ty: ignore[invalid-argument-type]
                )
            return acl_visibility_filter

        def _get_source_type_filter(
            source_types: list[DocumentSource],
        ) -> TermsQuery[str]:
            """Returns a filter for the source types.

            Since this returns an isolated terms clause, it can be cached in
            Elasticsearch independently of other clauses in _get_search_filters.

            Args:
                source_types: The source types to restrict documents to.

            Raises:
                ValueError: The number of source types is greater than
                    MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY.
                ValueError: An empty list was supplied.

            Returns:
                A filter for the source types.
            """
            if not source_types:
                raise ValueError(
                    "source_types cannot be empty if trying to create a source type filter."
                )
            if len(source_types) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                raise ValueError(
                    f"Too many source types: {len(source_types)}. Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                )
            # Use terms instead of a list of term within a should clause because
            # Lucene will optimize the filtering for large sets of terms. Small
            # sets of terms are not expected to perform any differently than
            # individual term clauses.
            return {
                "terms": {
                    SOURCE_TYPE_FIELD_NAME: [
                        source_type.value for source_type in source_types
                    ]
                }
            }

        def _get_tag_filter(tags: list[Tag]) -> TermsQuery[str]:
            """Returns a filter for the tags.

            Since this returns an isolated terms clause, it can be cached in
            Elasticsearch independently of other clauses in _get_search_filters.

            Args:
                tags: The tags to restrict documents to.

            Raises:
                ValueError: The number of tags is greater than
                    MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY.
                ValueError: An empty list was supplied.

            Returns:
                A filter for the tags.
            """
            if not tags:
                raise ValueError(
                    "tags cannot be empty if trying to create a tag filter."
                )
            if len(tags) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                raise ValueError(
                    f"Too many tags: {len(tags)}. Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                )
            # Kind of an abstraction leak, see
            # convert_metadata_dict_to_list_of_strings for why metadata list
            # entries are expected to look this way.
            tag_str_list = [
                f"{tag.tag_key}{INDEX_SEPARATOR}{tag.tag_value}" for tag in tags
            ]
            # Use terms instead of a list of term within a should clause because
            # Lucene will optimize the filtering for large sets of terms. Small
            # sets of terms are not expected to perform any differently than
            # individual term clauses.
            return {"terms": {METADATA_LIST_FIELD_NAME: tag_str_list}}

        def _get_document_set_filter(document_sets: list[str]) -> TermsQuery[str]:
            """Returns a filter for the document sets.

            Since this returns an isolated terms clause, it can be cached in
            Elasticsearch independently of other clauses in _get_search_filters.

            Args:
                document_sets: The document sets to restrict documents to.

            Raises:
                ValueError: The number of document sets is greater than
                    MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY.
                ValueError: An empty list was supplied.

            Returns:
                A filter for the document sets.
            """
            if not document_sets:
                raise ValueError(
                    "document_sets cannot be empty if trying to create a document set filter."
                )
            if len(document_sets) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                raise ValueError(
                    f"Too many document sets: {len(document_sets)}. Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                )
            # Use terms instead of a list of term within a should clause because
            # Lucene will optimize the filtering for large sets of terms. Small
            # sets of terms are not expected to perform any differently than
            # individual term clauses.
            return {"terms": {DOCUMENT_SETS_FIELD_NAME: list(document_sets)}}

        def _get_user_project_filter(project_id: int) -> TermQuery[int]:
            return {"term": {USER_PROJECTS_FIELD_NAME: {"value": project_id}}}

        def _get_persona_filter(persona_id: int) -> TermQuery[int]:
            return {"term": {PERSONAS_FIELD_NAME: {"value": persona_id}}}

        def _get_date_range_clause(
            field_name: str,
            gte: datetime | None,
            lte: datetime | None,
            include_undated: bool,
        ) -> dict[str, Any]:
            """Inclusive [gte, lte] range clause on a date field; when
            include_undated is True, documents missing the field also match.
            Isolated bool clause, so Elasticsearch can cache it independently."""
            # Convert to UTC if not already so the bounds are comparable to the
            # document data.
            range_bounds: dict[str, int | str] = {}
            if gte is not None:
                range_bounds["gte"] = int(datetime_to_utc(gte).timestamp())
            if lte is not None:
                range_bounds["lte"] = int(datetime_to_utc(lte).timestamp())
            # Elasticsearch treats numeric date range values as epoch millis
            # unless the query declares a format, even when the field mapping
            # itself uses epoch seconds.
            range_bounds["format"] = "epoch_second"

            # Logical OR operator on its elements.
            date_range_clause: dict[str, Any] = {
                "bool": {"should": [], "minimum_should_match": 1}
            }
            date_range_clause["bool"]["should"].append(
                {"range": {field_name: range_bounds}}
            )
            if include_undated:
                date_range_clause["bool"]["should"].append(
                    {"bool": {"must_not": {"exists": {"field": field_name}}}}
                )
            return date_range_clause

        def _get_document_time_filter(
            created_at_range: TimeRange | None,
            updated_at_range: TimeRange | None,
        ) -> list[dict[str, Any]]:
            """One null-tolerant clause per set range, to be AND-ed into the
            filter. created_at always keeps undated documents (over-extend);
            last_updated keeps them only for an old, open-ended lower bound, so
            recent-window queries aren't flooded by undated docs."""
            clauses: list[dict[str, Any]] = []
            if created_at_range is not None and created_at_range.has_bounds():
                clauses.append(
                    _get_date_range_clause(
                        CREATED_AT_FIELD_NAME,
                        gte=created_at_range.start,
                        lte=created_at_range.end,
                        include_undated=True,
                    )
                )
            if updated_at_range is not None and updated_at_range.has_bounds():
                include_undated = (
                    updated_at_range.start is not None
                    and updated_at_range.end is None
                    and updated_at_range.start
                    < datetime.now(timezone.utc)
                    - timedelta(days=ASSUMED_DOCUMENT_AGE_DAYS)
                )
                clauses.append(
                    _get_date_range_clause(
                        LAST_UPDATED_FIELD_NAME,
                        gte=updated_at_range.start,
                        lte=updated_at_range.end,
                        include_undated=include_undated,
                    )
                )
            return clauses

        def _get_validity_filter(as_of_date: date) -> dict[str, Any]:
            """AND of two null-tolerant clauses: the chunk must have started
            being valid by as_of_date (or never set a start — valid since
            indexing) and must not yet have stopped being valid (or never set
            an end — still valid). Non-regulatory chunks never set either
            field, so both clauses degrade to "field missing" and the chunk
            always passes."""
            as_of_epoch = int(
                datetime(
                    as_of_date.year,
                    as_of_date.month,
                    as_of_date.day,
                    tzinfo=timezone.utc,
                ).timestamp()
            )
            started_clause = {
                "bool": {
                    "should": [
                        {
                            "range": {
                                VALIDITY_START_DATE_FIELD_NAME: {
                                    "lte": as_of_epoch,
                                    "format": "epoch_second",
                                }
                            }
                        },
                        {
                            "bool": {
                                "must_not": {
                                    "exists": {"field": VALIDITY_START_DATE_FIELD_NAME}
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
            not_ended_clause = {
                "bool": {
                    "should": [
                        {
                            "range": {
                                VALIDITY_END_DATE_FIELD_NAME: {
                                    "gt": as_of_epoch,
                                    "format": "epoch_second",
                                }
                            }
                        },
                        {
                            "bool": {
                                "must_not": {
                                    "exists": {"field": VALIDITY_END_DATE_FIELD_NAME}
                                }
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                }
            }
            return {"bool": {"must": [started_clause, not_ended_clause]}}

        def _get_chunk_index_filter(
            min_chunk_index: int | None, max_chunk_index: int | None
        ) -> dict[str, Any]:
            range_clause: dict[str, Any] = {"range": {CHUNK_INDEX_FIELD_NAME: {}}}
            if min_chunk_index is not None:
                range_clause["range"][CHUNK_INDEX_FIELD_NAME]["gte"] = min_chunk_index
            if max_chunk_index is not None:
                range_clause["range"][CHUNK_INDEX_FIELD_NAME]["lte"] = max_chunk_index
            return range_clause

        def _get_attached_document_id_filter(
            doc_ids: list[str],
        ) -> TermsQuery[str]:
            """
            Returns a filter for documents explicitly attached to an assistant.

            Since this returns an isolated terms clause, it can be cached in
            Elasticsearch independently of other clauses in _get_search_filters.

            Args:
                doc_ids: The document IDs to restrict documents to.

            Raises:
                ValueError: The number of document IDs is greater than
                    MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY.
                ValueError: An empty list was supplied.

            Returns:
                A filter for the document IDs.
            """
            if not doc_ids:
                raise ValueError(
                    "doc_ids cannot be empty if trying to create a document ID filter."
                )
            if len(doc_ids) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                raise ValueError(
                    f"Too many document IDs: {len(doc_ids)}. Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                )
            # Use terms instead of a list of term within a should clause because
            # Lucene will optimize the filtering for large sets of terms. Small
            # sets of terms are not expected to perform any differently than
            # individual term clauses.
            return {"terms": {DOCUMENT_ID_FIELD_NAME: list(doc_ids)}}

        def _get_hierarchy_node_filter(
            node_ids: list[int],
        ) -> TermsQuery[int]:
            """
            Returns a filter for chunks whose ancestors include any of the given
            hierarchy nodes.

            Since this returns an isolated terms clause, it can be cached in
            Elasticsearch independently of other clauses in _get_search_filters.

            Args:
                node_ids: The hierarchy node IDs to restrict documents to.

            Raises:
                ValueError: The number of hierarchy node IDs is greater than
                    MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY.
                ValueError: An empty list was supplied.

            Returns:
                A filter for the hierarchy node IDs.
            """
            if not node_ids:
                raise ValueError(
                    "node_ids cannot be empty if trying to create a hierarchy node ID filter."
                )
            if len(node_ids) > MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY:
                raise ValueError(
                    f"Too many hierarchy node IDs: {len(node_ids)}. Max allowed: {MAX_NUM_TERMS_ALLOWED_IN_TERMS_QUERY}."
                )
            # Use terms instead of a list of term within a should clause because
            # Lucene will optimize the filtering for large sets of terms. Small
            # sets of terms are not expected to perform any differently than
            # individual term clauses.
            return {"terms": {ANCESTOR_HIERARCHY_NODE_IDS_FIELD_NAME: list(node_ids)}}

        if document_id is not None and attached_document_ids is not None:
            raise ValueError(
                "document_id and attached_document_ids cannot be used together."
            )

        filter_clauses: list[dict[str, Any]] = []

        if not include_hidden:
            filter_clauses.append({"term": {HIDDEN_FIELD_NAME: {"value": False}}})

        if access_control_list is not None:
            # If an access control list is provided, the caller can only
            # retrieve public documents, and non-public documents where at least
            # one acl provided here is present in the document's acl list. If
            # there is explicitly no list provided, we make no restrictions on
            # the documents that can be retrieved.
            filter_clauses.append(_get_acl_visibility_filter(access_control_list))

        if forced_document_sets:
            # Its own top-level AND clause (not merged into the OR-based
            # knowledge_filter below), so it INTERSECTS rather than widens; placed
            # after the ACL clause so it never loosens permissions.
            filter_clauses.append(_get_document_set_filter(forced_document_sets))

        if source_types:
            # If at least one source type is provided, the caller will only
            # retrieve documents whose source type is present in this input
            # list.
            filter_clauses.append(_get_source_type_filter(source_types))

        if tags:
            # If at least one tag is provided, the caller will only retrieve
            # documents where at least one tag provided here is present in the
            # document's metadata list.
            filter_clauses.append(_get_tag_filter(tags))

        # Knowledge scope: explicit knowledge attachments restrict what an
        # assistant can see. When none are set the assistant searches
        # everything.
        #
        # persona_id_filter is a primary trigger — a persona with user files IS
        # explicit knowledge, so it can start a knowledge scope on its own.
        #
        # project_id_filter is a primary trigger — a chat inside a project is
        # scoped to that project, so project_id_filter restricts the search to
        # the project's files on its own (project chats do not search team
        # knowledge).
        has_knowledge_scope = (
            attached_document_ids
            or hierarchy_node_ids
            or document_sets
            or persona_id_filter is not None
            or project_id_filter is not None
        )

        if has_knowledge_scope:
            # Since this returns an isolated bool should clause, it can be
            # cached in Elasticsearch independently of other clauses in
            # _get_search_filters.
            knowledge_filter: dict[str, Any] = {
                "bool": {"should": [], "minimum_should_match": 1}
            }
            if attached_document_ids:
                knowledge_filter["bool"]["should"].append(
                    _get_attached_document_id_filter(attached_document_ids)
                )
            if hierarchy_node_ids:
                knowledge_filter["bool"]["should"].append(
                    _get_hierarchy_node_filter(hierarchy_node_ids)
                )
            if document_sets:
                knowledge_filter["bool"]["should"].append(
                    _get_document_set_filter(document_sets)
                )
            if persona_id_filter is not None:
                knowledge_filter["bool"]["should"].append(
                    _get_persona_filter(persona_id_filter)
                )
            if project_id_filter is not None:
                knowledge_filter["bool"]["should"].append(
                    _get_user_project_filter(project_id_filter)
                )
            filter_clauses.append(knowledge_filter)

        if created_at_range is not None or updated_at_range is not None:
            filter_clauses.extend(
                _get_document_time_filter(created_at_range, updated_at_range)
            )

        if min_chunk_index is not None or max_chunk_index is not None:
            filter_clauses.append(
                _get_chunk_index_filter(min_chunk_index, max_chunk_index)
            )

        if as_of_date is not None:
            filter_clauses.append(_get_validity_filter(as_of_date))

        if regulatory_chunks_only:
            filter_clauses.append({"exists": {"field": REGULATORY_CHUNK_ID_FIELD_NAME}})

        if document_id is not None:
            filter_clauses.append(
                {"term": {DOCUMENT_ID_FIELD_NAME: {"value": document_id}}}
            )

        if max_chunk_size is not None:
            filter_clauses.append(
                {"term": {MAX_CHUNK_SIZE_FIELD_NAME: {"value": max_chunk_size}}}
            )

        if tenant_state.multitenant:
            filter_clauses.append(
                {"term": {TENANT_ID_FIELD_NAME: {"value": tenant_state.tenant_id}}}
            )

        return filter_clauses

    @staticmethod
    def _get_match_highlights_configuration() -> dict[str, Any]:
        """
        Gets configuration for returning match highlights for a hit.
        """
        match_highlights_configuration: dict[str, Any] = {
            "fields": {
                CONTENT_FIELD_NAME: {
                    # See https://www.elastic.co/docs/reference/elasticsearch/rest-apis/highlighting
                    "type": "unified",
                    # The length in chars of a match snippet. Somewhat
                    # arbitrarily-chosen. The Vespa codepath limited total
                    # highlights length to 400 chars. fragment_size *
                    # number_of_fragments = 400 should be good enough.
                    "fragment_size": 100,
                    # The number of snippets to return per field per document
                    # hit.
                    "number_of_fragments": 4,
                    # These tags wrap matched keywords and they match what Vespa
                    # used to return. Use them to minimize changes to our code.
                    "pre_tags": ["<hi>"],
                    "post_tags": ["</hi>"],
                }
            }
        }

        return match_highlights_configuration
