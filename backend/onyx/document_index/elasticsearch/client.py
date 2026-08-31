import copy
import json
import logging
import statistics
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from http import HTTPStatus
from typing import Any, Generic, TypeVar

from elasticsearch import (
    ApiError,
    Elasticsearch,
    NotFoundError,
)
from elasticsearch.helpers import bulk
from pydantic import BaseModel

from onyx.configs.app_configs import (
    DEFAULT_ELASTICSEARCH_CLIENT_TIMEOUT_S,
    ELASTICSEARCH_ADMIN_PASSWORD,
    ELASTICSEARCH_ADMIN_USERNAME,
    ELASTICSEARCH_CA_CERTS,
    ELASTICSEARCH_CLIENT_CERT,
    ELASTICSEARCH_CLIENT_KEY,
    ELASTICSEARCH_HOST,
    ELASTICSEARCH_REST_API_PORT,
    ELASTICSEARCH_USE_SSL,
    ELASTICSEARCH_VERIFY_CERTS,
    PIT_KEEP_ALIVE,
)
from onyx.document_index.elasticsearch.constants import (
    DEFAULT_MAX_CHUNK_SIZE,
    ElasticsearchSearchType,
)
from onyx.document_index.elasticsearch.schema import (
    CHUNK_INDEX_FIELD_NAME,
    CONTENT_VECTOR_FIELD_NAME,
    DOCUMENT_ID_FIELD_NAME,
    MAX_CHUNK_SIZE_FIELD_NAME,
    REGULATORY_CHUNK_ID_FIELD_NAME,
    TITLE_VECTOR_FIELD_NAME,
    VALIDITY_END_DATE_FIELD_NAME,
    VALIDITY_START_DATE_FIELD_NAME,
    DocumentChunk,
    DocumentChunkWithoutVectors,
    get_elasticsearch_doc_chunk_id,
)
from onyx.document_index.elasticsearch.search import (
    DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW,
)
from onyx.document_index.interfaces_new import TenantState
from onyx.server.metrics.elasticsearch_search import (
    observe_elasticsearch_search,
    record_elasticsearch_search_error,
    track_elasticsearch_search,
)
from onyx.utils.logger import setup_logger
from onyx.utils.timing import log_function_time

CLIENT_THRESHOLD_TO_LOG_SLOW_SEARCH_MS = 2000
DEFAULT_INDEX_SETTINGS_TIMEOUT_S = 15

_RETRYABLE_UPDATE_ERROR_TYPES = (
    "already_closed_exception",
    "search_phase_execution_exception",
)


logger = setup_logger(__name__)
# Set the logging level to WARNING to ignore INFO and DEBUG logs from
# elasticsearch. By default it emits INFO-level logs for every request.
# The elasticsearch-py library uses "elasticsearch" as the logger name for HTTP
# requests.
elasticsearch_logger = logging.getLogger("elasticsearch")
elasticsearch_logger.setLevel(logging.WARNING)


SchemaDocumentModel = TypeVar("SchemaDocumentModel")


class SearchHit(BaseModel, Generic[SchemaDocumentModel]):
    """Represents a hit from Elasticsearch in response to a query.

    Templated on the specific document model as defined by a schema.
    """

    model_config = {"frozen": True}

    # The document chunk source retrieved from Elasticsearch.
    document_chunk: SchemaDocumentModel
    # The match score for the document chunk as calculated by Elasticsearch. Only
    # relevant for "fuzzy searches"; this will be None for direct queries where
    # score is not relevant like direct retrieval on ID.
    score: float | None = None
    # Maps schema property name to a list of highlighted snippets with match
    # terms wrapped in tags (e.g. "something <hi>keyword</hi> other thing").
    match_highlights: dict[str, list[str]] = {}
    # Score explanation from Elasticsearch when "explain": true is set in the
    # query. Contains detailed breakdown of how the score was calculated.
    explanation: dict[str, Any] | None = None


class IndexInfo(BaseModel):
    """
    Represents information about an Elasticsearch index.
    """

    model_config = {"frozen": True}

    name: str
    health: str
    status: str
    num_primary_shards: str
    num_replica_shards: str
    docs_count: str
    docs_deleted: str
    created_at: str
    total_size: str
    primary_shards_size: str


class ElasticsearchUpdateError(Exception):
    """
    An error occurred when updating one or more Elasticsearch document chunks which
    was caught by ElasticsearchIndexClient. This exception is not exhaustive of all
    exceptions update calls can raise.
    """


class ElasticsearchIndexError(Exception):
    """
    An error occurred when indexing one or more Elasticsearch document chunks which
    was caught by ElasticsearchIndexClient. This exception is not exhaustive of all
    exceptions index calls can raise.
    """


class ElasticsearchDocumentMissingError(Exception):
    """Target chunks don't exist on an _update (404) and the caller opted to
    surface this rather than fail (reindex port: doc not in FUTURE yet)."""

    def __init__(
        self,
        missing_chunk_ids: list[str],
        missing_document_ids: list[str] | None = None,
    ) -> None:
        self.missing_chunk_ids = missing_chunk_ids
        # Only the layer that built the chunk ids knows the doc mapping; the
        # client raises with chunks only and the index layer fills doc ids in.
        self.missing_document_ids = missing_document_ids or []
        super().__init__(
            f"{len(missing_chunk_ids)} document chunk(s) missing during update."
        )


# Server-side error.type strings (not exposed as enums by elasticsearch-py; cf.
# _RETRYABLE_UPDATE_ERROR_TYPES above). Status codes use http.HTTPStatus.
_DOCUMENT_MISSING_ERROR_TYPE = "document_missing_exception"
_VERSION_CONFLICT_ERROR_TYPE = "version_conflict_engine_exception"
# Raised by a search whose PIT has expired/been deleted; we re-open and retry.
_SEARCH_CONTEXT_MISSING_ERROR_TYPE = "search_context_missing"
# Chunks per PIT-scan page. A port doc-batch is small (INDEX_BATCH_SIZE docs), so
# one page covers a batch; paging still protects against a pathological doc.
_PIT_SCAN_PAGE_SIZE = 1000


class ElasticsearchServerSideTimeout(Exception):
    """
    A server-side timeout occurred when searching an Elasticsearch index.
    """


def _summarize_bulk_errors(errors: list[dict[str, Any]]) -> str:
    """Reduce raw bulk per-item errors to (op, status, type) counts.

    error.reason / caused_by echo a preview of the offending document's field
    values; dumping them into an exception message would leak indexed content
    into logs, so only op/status/type are surfaced.
    """
    counts: Counter[tuple[str, Any, str]] = Counter()
    for error in errors:
        op, item = next(iter(error.items()), ("", {}))
        item = item if isinstance(item, dict) else {}
        err_obj = item.get("error")
        err_type = err_obj.get("type", "") if isinstance(err_obj, dict) else ""
        counts[(op, item.get("status", 0), err_type)] += 1
    return ", ".join(
        f"{count}x op={op or 'unknown'} status={status} type={err_type or 'unknown'}"
        for (op, status, err_type), count in sorted(
            counts.items(), key=lambda kv: str(kv)
        )
    )


def get_new_body_without_vectors(body: dict[str, Any]) -> dict[str, Any]:
    """Recursively replaces vectors in the body with their length.

    TODO(andrei): Do better.

    Args:
        body: The body to replace the vectors.

    Returns:
        A copy of body with vectors replaced with their length.
    """
    new_body: dict[str, Any] = {}
    for k, v in body.items():
        if k == "vector":
            new_body[k] = len(v)
        elif isinstance(v, dict):
            new_body[k] = get_new_body_without_vectors(v)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
            new_body[k] = [get_new_body_without_vectors(item) for item in v]
        else:
            new_body[k] = v
    return new_body


class ElasticsearchClient(AbstractContextManager):
    """Client for interacting with Elasticsearch for cluster-level operations.

    Args:
        host: The host of the Elasticsearch cluster.
        port: The port of the Elasticsearch cluster.
        auth: The basic-auth credentials for the Elasticsearch cluster, a tuple
            of (username, password).
        use_ssl: Whether to use SSL for the Elasticsearch cluster.
        verify_certs: Whether to verify the server certificate. Defaults to
            ELASTICSEARCH_VERIFY_CERTS.
        ca_certs: CA bundle path used to verify the server certificate.
        client_cert: Client certificate path for mutual TLS.
        client_key: Client private key path for mutual TLS.
        ssl_show_warn: Whether to show warnings for SSL certificates. Defaults
            to False.
        timeout: The timeout for the Elasticsearch cluster. Defaults to
            DEFAULT_ELASTICSEARCH_CLIENT_TIMEOUT_S.
    """

    def __init__(
        self,
        host: str = ELASTICSEARCH_HOST,
        port: int = ELASTICSEARCH_REST_API_PORT,
        auth: tuple[str, str] = (
            ELASTICSEARCH_ADMIN_USERNAME,
            ELASTICSEARCH_ADMIN_PASSWORD,
        ),
        use_ssl: bool = ELASTICSEARCH_USE_SSL,
        verify_certs: bool = ELASTICSEARCH_VERIFY_CERTS,
        ca_certs: str | None = ELASTICSEARCH_CA_CERTS,
        client_cert: str | None = ELASTICSEARCH_CLIENT_CERT,
        client_key: str | None = ELASTICSEARCH_CLIENT_KEY,
        ssl_show_warn: bool = False,
        timeout: int = DEFAULT_ELASTICSEARCH_CLIENT_TIMEOUT_S,
    ):
        logger.debug(
            "Creating Elasticsearch client with host %s, port %s and timeout "
            "%s seconds.",
            host,
            port,
            timeout,
        )
        scheme = "https" if use_ssl else "http"
        client_options: dict[str, Any] = {
            "basic_auth": auth,
            "verify_certs": verify_certs,
            "ssl_show_warn": ssl_show_warn,
            # NOTE: This timeout applies to all requests the client makes,
            # including bulk indexing. When exceeded, the client will raise a
            # ConnectionTimeout and return no useful results. The Elasticsearch
            # server will log that the client cancelled the request. To get
            # partial results from Elasticsearch, pass in a timeout parameter to
            # your request body that is less than this value.
            "request_timeout": timeout,
        }
        if ca_certs is not None:
            client_options["ca_certs"] = ca_certs
        if client_cert is not None:
            client_options["client_cert"] = client_cert
        if client_key is not None:
            client_options["client_key"] = client_key
        self._client = Elasticsearch(f"{scheme}://{host}:{port}", **client_options)

    def __exit__(self, *_: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def put_cluster_settings(self, settings: dict[str, Any]) -> bool:
        """Puts cluster settings.

        Args:
            settings: The settings to put.

        Raises:
            Exception: There was an error putting the cluster settings.

        Returns:
            True if the settings were put successfully, False otherwise.
        """
        response = self._client.cluster.put_settings(
            persistent=settings.get("persistent"),
            transient=settings.get("transient"),
        )
        if response.get("acknowledged", False):
            logger.info("Successfully put cluster settings.")
            return True
        else:
            logger.error("Failed to put cluster settings: %s.", response)
            return False

    @log_function_time(print_only=True, debug_only=True)
    def list_indices_with_info(self) -> list[IndexInfo]:
        """
        Lists the indices in the Elasticsearch cluster with information about each
        index.

        Returns:
            A list of IndexInfo objects for each index.
        """
        response = self._client.cat.indices(format="json").body
        if not isinstance(response, list):
            raise RuntimeError(
                "Elasticsearch cat indices returned a non-list response."
            )
        indices: list[IndexInfo] = []
        for raw_index_info in response:
            indices.append(
                IndexInfo(
                    name=raw_index_info.get("index", ""),
                    health=raw_index_info.get("health", ""),
                    status=raw_index_info.get("status", ""),
                    num_primary_shards=raw_index_info.get("pri", ""),
                    num_replica_shards=raw_index_info.get("rep", ""),
                    docs_count=raw_index_info.get("docs.count", ""),
                    docs_deleted=raw_index_info.get("docs.deleted", ""),
                    created_at=raw_index_info.get("creation.date.string", ""),
                    total_size=raw_index_info.get("store.size", ""),
                    primary_shards_size=raw_index_info.get("pri.store.size", ""),
                )
            )
        return indices

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def cluster_health(
        self,
        level: str = "cluster",
        index: str | None = None,
    ) -> dict[str, Any]:
        """Gets the cluster health.

        See the Elasticsearch documentation for more information on the cluster
        health API:

        Args:
            level: The level of detail. One of "cluster", "indices", "shards",
                or "awareness_attributes". Defaults to "cluster".
            index: Optionally scope the health response to a specific index.
                Defaults to None (whole cluster).

        Returns:
            The raw cluster health response.
        """
        return dict(self._client.cluster.health(index=index, level=level).body)

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def cat_shards(
        self,
        index: str | None = None,
        columns: str = "index,shard,prirep,state,unassigned.reason,unassigned.for,node",
    ) -> list[dict[str, Any]]:
        """Lists shards in the cluster.

        See the Elasticsearch documentation for more information on the cat shards
        API:

        Args:
            index: Optionally scope to a specific index. Defaults to None (all
                indices).
            columns: Comma-separated list of columns to return. Maps to the
                ``h`` query parameter.

        Returns:
            A list of dicts, one per shard, with the requested columns as keys.
        """
        response = self._client.cat.shards(format="json", h=columns, index=index).body
        if not isinstance(response, list):
            raise RuntimeError("Elasticsearch cat shards returned a non-list response.")
        return response

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def allocation_explain(
        self,
        index: str | None = None,
        shard: int | None = None,
        primary: bool | None = None,
    ) -> dict[str, Any]:
        """Explains why a shard is or is not allocated.

        With no args, Elasticsearch picks an arbitrary unassigned shard to explain.
        To scope to a specific shard, all three args must be provided together.

        See the Elasticsearch documentation for more information on the cluster
        allocation explain API:

        Args:
            index: The index name.
            shard: The shard ID.
            primary: Whether the shard is a primary (True) or replica (False).

        Returns:
            The raw allocation explanation response.
        """
        return dict(
            self._client.cluster.allocation_explain(
                index=index,
                shard=shard,
                primary=primary,
            ).body
        )

    @log_function_time(print_only=True, debug_only=True)
    def reroute_retry_failed(self) -> dict[str, Any]:
        """Triggers a cluster reroute with retry_failed=true.

        Useful when shards are stuck UNASSIGNED due to ALLOCATION_FAILED with
        max retries exceeded (default 5). This resets the failure counter and
        attempts allocation again. The cluster's own allocation_explain output
        recommends this when the ``max_retry`` decider is blocking.

        See the Elasticsearch documentation for more information on the cluster
        reroute API:

        Returns:
            The raw reroute response. Includes ``acknowledged`` and the
                post-reroute cluster state.
        """
        return dict(self._client.cluster.reroute(retry_failed=True).body)

    @log_function_time(print_only=True, debug_only=True)
    def ping(self) -> bool:
        """Pings the Elasticsearch cluster.

        Returns:
            True if Elasticsearch could be reached, False if it could not.
        """
        return self._client.ping()

    def close(self) -> None:
        """Closes the client.

        Raises:
            Exception: There was an error closing the client.
        """
        self._client.close()


class ElasticsearchIndexClient(ElasticsearchClient):
    """Client for interacting with Elasticsearch for index-level operations.

    Elasticsearch's Python module has pretty bad typing support so this client
    attempts to protect the rest of the codebase from this. As a consequence,
    most methods here return the minimum data needed for the rest of Onyx, and
    tend to rely on Exceptions to handle errors.

    TODO(andrei): This class currently assumes the structure of the database
    schema when it returns a DocumentChunk. Make the class, or at least the
    search method, templated on the structure the caller can expect.

    Args:
        index_name: The name of the index to interact with.
        host: The host of the Elasticsearch cluster.
        port: The port of the Elasticsearch cluster.
        auth: The basic-auth credentials for the Elasticsearch cluster, a tuple
            of (username, password).
        use_ssl: Whether to use SSL for the Elasticsearch cluster.
        verify_certs: Whether to verify the server certificate. Defaults to
            ELASTICSEARCH_VERIFY_CERTS.
        ca_certs: CA bundle path used to verify the server certificate.
        client_cert: Client certificate path for mutual TLS.
        client_key: Client private key path for mutual TLS.
        ssl_show_warn: Whether to show warnings for SSL certificates. Defaults
            to False.
        timeout: The timeout for the Elasticsearch cluster. Defaults to
            DEFAULT_ELASTICSEARCH_CLIENT_TIMEOUT_S.
    """

    def __init__(
        self,
        index_name: str,
        host: str = ELASTICSEARCH_HOST,
        port: int = ELASTICSEARCH_REST_API_PORT,
        auth: tuple[str, str] = (
            ELASTICSEARCH_ADMIN_USERNAME,
            ELASTICSEARCH_ADMIN_PASSWORD,
        ),
        use_ssl: bool = ELASTICSEARCH_USE_SSL,
        verify_certs: bool = ELASTICSEARCH_VERIFY_CERTS,
        ca_certs: str | None = ELASTICSEARCH_CA_CERTS,
        client_cert: str | None = ELASTICSEARCH_CLIENT_CERT,
        client_key: str | None = ELASTICSEARCH_CLIENT_KEY,
        ssl_show_warn: bool = False,
        timeout: int = DEFAULT_ELASTICSEARCH_CLIENT_TIMEOUT_S,
        emit_metrics: bool = True,
    ):
        super().__init__(
            host=host,
            port=port,
            auth=auth,
            use_ssl=use_ssl,
            verify_certs=verify_certs,
            ca_certs=ca_certs,
            client_cert=client_cert,
            client_key=client_key,
            ssl_show_warn=ssl_show_warn,
            timeout=timeout,
        )
        self._index_name = index_name
        self._emit_metrics = emit_metrics
        logger.debug(
            "Elasticsearch client created successfully for index %s.",
            self._index_name,
        )

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def create_index(self, mappings: dict[str, Any], settings: dict[str, Any]) -> None:
        """Creates the index.

        See the Elasticsearch documentation for more information on mappings and
        settings.

        Args:
            mappings: The mappings for the index to create.
            settings: The settings for the index to create.

        Raises:
            Exception: There was an error creating the index.
        """
        logger.debug("Creating index %s.", self._index_name)
        response = self._client.indices.create(
            index=self._index_name,
            mappings=mappings,
            settings=settings,
        )
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to create index {self._index_name}.")
        response_index = response.get("index", "")
        if response_index != self._index_name:
            raise RuntimeError(
                f"Elasticsearch responded with index name {response_index} when creating index "
                f"{self._index_name}."
            )
        logger.debug("Index %s created successfully.", self._index_name)

    @log_function_time(print_only=True, debug_only=True)
    def delete_index(self) -> bool:
        """Deletes the index.

        Raises:
            Exception: There was an error deleting the index.

        Returns:
            True if the index was deleted, False if it did not exist.
        """
        if not self._client.indices.exists(index=self._index_name):
            logger.warning(
                "Tried to delete index %s but it does not exist.",
                self._index_name,
            )
            return False

        logger.info("Deleting index %s.", self._index_name)
        response = self._client.indices.delete(index=self._index_name)
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to delete index {self._index_name}.")
        logger.info("Index %s deleted successfully.", self._index_name)
        return True

    @log_function_time(print_only=True, debug_only=True)
    def index_exists(self) -> bool:
        """Checks if the index exists.

        Raises:
            Exception: There was an error checking if the index exists.

        Returns:
            True if the index exists, False if it does not.
        """
        return bool(self._client.indices.exists(index=self._index_name))

    @log_function_time(print_only=True, debug_only=True)
    def get_index_mapping(self) -> dict[str, Any]:
        """Return this index's mapping definition."""

        response = self._client.indices.get_mapping(index=self._index_name)
        index_response = response.get(self._index_name)
        if not isinstance(index_response, dict):
            raise RuntimeError(
                f"Elasticsearch returned no mapping for index {self._index_name}."
            )
        mappings = index_response.get("mappings")
        if not isinstance(mappings, dict):
            raise RuntimeError(
                f"Elasticsearch returned an invalid mapping for index {self._index_name}."
            )
        return mappings

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def put_mapping(self, mappings: dict[str, Any]) -> None:
        """Updates the index mapping in an idempotent manner.

        - Existing fields with the same definition: No-op (succeeds silently).
        - New fields: Added to the index.
        - Existing fields with different types: Raises exception (requires
          reindex).

        See the Elasticsearch documentation for more information:

        Args:
            mappings: The complete mapping definition to apply. This will be
                merged with existing mappings in the index.

        Raises:
            Exception: There was an error updating the mappings, such as
                attempting to change the type of an existing field.
        """
        logger.debug("Putting mappings for index %s.", self._index_name)
        response = self._client.indices.put_mapping(
            index=self._index_name,
            **mappings,
        )
        if not response.get("acknowledged", False):
            raise RuntimeError(
                f"Failed to put the mapping update for index {self._index_name}."
            )
        logger.debug("Successfully put mappings for index %s.", self._index_name)

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def validate_index(self, expected_mappings: dict[str, Any]) -> bool:
        """Validates the index.

        Short-circuit returns False on the first mismatch. Logs the mismatch.

        See the Elasticsearch documentation for more information on the index
        mappings.

        Args:
            mappings: The expected mappings of the index to validate.

        Raises:
            Exception: There was an error validating the index.

        Returns:
            True if the index is valid, False if it is not based on the mappings
                supplied.
        """
        # Elasticsearch's documentation makes no mention of what happens when you
        # invoke client.indices.get on an index that does not exist, so we check
        # for existence explicitly just to be sure.
        exists_response = self.index_exists()
        if not exists_response:
            logger.warning(
                "Tried to validate index %s but it does not exist.",
                self._index_name,
            )
            return False
        logger.debug("Validating index %s.", self._index_name)

        get_result = self._client.indices.get(index=self._index_name)
        index_info: dict[str, Any] = get_result.get(self._index_name, {})
        if not index_info:
            raise ValueError(
                f"Bug: Elasticsearch did not return any index info for index {self._index_name}, "
                "even though it confirmed that the index exists."
            )
        index_mapping_properties: dict[str, Any] = index_info.get("mappings", {}).get(
            "properties", {}
        )
        expected_mapping_properties: dict[str, Any] = expected_mappings.get(
            "properties", {}
        )
        assert expected_mapping_properties, (
            "Bug: No properties were found in the provided expected mappings."
        )

        for property in expected_mapping_properties:
            if property not in index_mapping_properties:
                logger.warning(
                    'The field "%s" was not found in the index %s.',
                    property,
                    self._index_name,
                )
                return False

            expected_property_type = expected_mapping_properties[property].get(
                "type", ""
            )
            assert expected_property_type, (
                f'Bug: The field "{property}" in the supplied expected schema mappings has no type.'
            )

            index_property_type = index_mapping_properties[property].get("type", "")
            if expected_property_type != index_property_type:
                logger.warning(
                    'The field "%s" in the index %s has type %s '
                    "but the expected type is %s.",
                    property,
                    self._index_name,
                    index_property_type,
                    expected_property_type,
                )
                return False

        logger.debug("Index %s validated successfully.", self._index_name)
        return True

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def update_settings(
        self,
        settings: dict[str, Any],
        timeout: float = DEFAULT_INDEX_SETTINGS_TIMEOUT_S,
    ) -> None:
        """Updates the settings of the index.

        See the Elasticsearch documentation for more information on the index
        settings.

        Args:
            settings: The settings to update the index with.

        Raises:
            Exception: There was an error updating the settings of the index.
        """
        logger.debug("Updating settings of index %s.", self._index_name)
        response = self._client.indices.put_settings(
            index=self._index_name,
            settings=settings,
            timeout=f"{timeout}s",
        )
        if not response.get("acknowledged", False):
            raise RuntimeError(
                f"Failed to update settings of index {self._index_name}."
            )
        logger.debug("Settings of index %s updated successfully.", self._index_name)

    @log_function_time(print_only=True, debug_only=True)
    def get_settings(
        self,
        include_defaults: bool = False,
        flat_settings: bool = False,
        pretty: bool = False,
        human: bool = False,
        timeout: float = DEFAULT_INDEX_SETTINGS_TIMEOUT_S,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Gets the settings of the index.

        Args:
            include_defaults: Whether to include default settings which have not
                been explicitly set. Defaults to False.
            flat_settings: Whether to return settings in flat format vs nested
                dictionaries. Defaults to False.
            pretty: Whether to pretty-format the returned JSON response.
                Defaults to False.
            human: Whether to return statistics in human-readable format.
                Defaults to False.

        Returns:
            The settings of the index, and optionally the default settings. If
                include_defaults is False, the default settings will be None.

        Raises:
            Exception: There was an error getting the settings of the index.
        """
        logger.debug("Getting settings of index %s.", self._index_name)
        response = self._client.options(request_timeout=timeout).indices.get_settings(
            index=self._index_name,
            include_defaults=include_defaults,
            flat_settings=flat_settings,
            pretty=pretty,
            human=human,
        )
        return response[self._index_name]["settings"], response[self._index_name].get(
            "defaults", None
        )

    @log_function_time(print_only=True, debug_only=True)
    def open_index(self, timeout: float = DEFAULT_INDEX_SETTINGS_TIMEOUT_S) -> None:
        """Opens the index.

        Raises:
            Exception: There was an error opening the index.
        """
        logger.debug("Opening index %s.", self._index_name)
        response = self._client.indices.open(
            index=self._index_name, timeout=f"{timeout}s"
        )
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to open index {self._index_name}.")
        logger.debug("Index %s opened successfully.", self._index_name)

    @log_function_time(print_only=True, debug_only=True)
    def close_index(self, timeout: float = DEFAULT_INDEX_SETTINGS_TIMEOUT_S) -> None:
        """Closes the index.

        Raises:
            Exception: There was an error closing the index.
        """
        logger.debug("Closing index %s.", self._index_name)
        response = self._client.indices.close(
            index=self._index_name, timeout=f"{timeout}s"
        )
        if not response.get("acknowledged", False):
            raise RuntimeError(f"Failed to close index {self._index_name}.")
        logger.debug("Index %s closed successfully.", self._index_name)

    @log_function_time(
        print_only=True,
        debug_only=True,
        include_args_subset={
            "document": str,
            "tenant_state": str,
            "update_if_exists": str,
        },
    )
    def index_document(
        self,
        document: DocumentChunk,
        tenant_state: TenantState,
        update_if_exists: bool = False,
    ) -> None:
        """Indexes a document.

        Args:
            document: The document to index. In Onyx this is a chunk of a
                document, Elasticsearch simply refers to this as a document as
                well.
            tenant_state: The tenant state of the caller.
            update_if_exists: Whether to update the document if it already
                exists. If False, will raise an exception if the document
                already exists. Defaults to False.

        Raises:
            Exception: There was an error indexing the document. This includes
                the case where a document with the same ID already exists if
                update_if_exists is False.
        """
        logger.debug(
            "Trying to index document ID %s for tenant %s. update_if_exists=%s.",
            document.document_id,
            tenant_state.tenant_id,
            update_if_exists,
        )
        document_chunk_id: str = get_elasticsearch_doc_chunk_id(
            tenant_state=tenant_state,
            document_id=document.document_id,
            chunk_index=document.chunk_index,
            max_chunk_size=document.max_chunk_size,
        )
        body: dict[str, Any] = document.model_dump(exclude_none=True)
        # client.create will raise if a doc with the same ID exists.
        # client.index does not do this.
        if update_if_exists:
            result = self._client.index(
                index=self._index_name,
                id=document_chunk_id,
                document=body,
            )
        else:
            result = self._client.create(
                index=self._index_name,
                id=document_chunk_id,
                document=body,
            )
        result_id = result.get("_id", "")
        # Sanity check.
        if result_id != document_chunk_id:
            raise RuntimeError(
                f'Upon trying to index a document, Elasticsearch responded with ID "{result_id}" '
                f'instead of "{document_chunk_id}" which is the ID it was given.'
            )
        result_string: str = result.get("result", "")
        match result_string:
            # Sanity check.
            case "created":
                pass
            case "updated":
                if not update_if_exists:
                    raise RuntimeError(
                        f'The Elasticsearch client returned result "updated" for indexing document '
                        f'chunk "{document_chunk_id}". This indicates that a document chunk with '
                        "that ID already exists, which is not expected."
                    )
            case _:
                raise RuntimeError(
                    f'Unknown Elasticsearch indexing result: "{result_string}".'
                )
        logger.debug("Successfully indexed %s.", document_chunk_id)

    @log_function_time(
        print_only=True,
        debug_only=True,
        include_args_subset={
            "documents": len,
            "tenant_state": str,
            "update_if_exists": str,
            "use_create_only": str,
        },
    )
    def bulk_index_documents(
        self,
        documents: list[DocumentChunk],
        tenant_state: TenantState,
        update_if_exists: bool = False,
        use_create_only: bool = False,
    ) -> None:
        """Bulk indexes documents.

        Raises if there are any errors during the bulk index. It should be
        assumed that no documents in the batch were indexed successfully if
        there is an error.

        Retries on 429 too many requests.

        Args:
            documents: The documents to index. In Onyx this is a chunk of a
                document, Elasticsearch simply refers to this as a document as
                well.
            tenant_state: The tenant state of the caller.
            update_if_exists: Whether to update the document if it already
                exists. If False, will raise an exception if the document
                already exists. Defaults to False.
            use_create_only: When True, write each chunk with _op_type=create
                (don't overwrite if it already exists) and treat the resulting
                409 as benign. The reindex port uses this so a stale backlog
                write can never clobber a chunk a live/forward writer already
                owns in FUTURE. Default False leaves the write path unchanged.

        Raises:
            Exception: There was an error during the bulk index. This
                includes the case where a document with the same ID already
                exists if update_if_exists is False.
            BulkIndexError: There was an error during the bulk index. This is a
                known specific error type that is raised by the Elasticsearch
                library's bulk function.
            ElasticsearchIndexError: The number of successful operations reported
                by Elasticsearch does not match the number of documents.
        """
        if not documents:
            return
        logger.debug(
            "Bulk indexing %s documents for tenant %s. update_if_exists=%s "
            "use_create_only=%s.",
            len(documents),
            tenant_state.tenant_id,
            update_if_exists,
            use_create_only,
        )
        data = []
        for document in documents:
            document_chunk_id: str = get_elasticsearch_doc_chunk_id(
                tenant_state=tenant_state,
                document_id=document.document_id,
                chunk_index=document.chunk_index,
                max_chunk_size=document.max_chunk_size,
            )
            body: dict[str, Any] = document.model_dump(exclude_none=True)
            # create-only never overwrites: an existing chunk (a live/forward
            # writer already owns it) comes back as a benign 409.
            if use_create_only:
                op_type = "create"
            else:
                op_type = "index" if update_if_exists else "create"
            data_for_document: dict[str, Any] = {
                "_index": self._index_name,
                "_id": document_chunk_id,
                "_op_type": op_type,
                "_source": body,
            }
            data.append(data_for_document)

        if use_create_only:
            # a chunk that already exists is owned by a live/forward writer, so
            # the port yields with a benign 409 instead of failing the batch
            successes, errors = bulk(
                self._client,
                data,
                max_retries=3,
                raise_on_error=False,
                raise_on_exception=True,
            )
            if not isinstance(errors, list):
                raise ElasticsearchIndexError(
                    "Elasticsearch bulk helper returned malformed error details."
                )
            benign_conflicts = self._benign_create_conflict_count(errors)
        else:
            # any error fails the batch (the caller may refresh-retry
            # on the BulkIndexError that bulk raises)
            successes, _ = bulk(
                self._client,
                data,
                max_retries=3,
                raise_on_error=True,
                raise_on_exception=True,
            )
            benign_conflicts = 0

        if successes + benign_conflicts != len(documents):
            raise ElasticsearchIndexError(
                f"Bulk index for index {self._index_name}: successful operations ({successes}) "
                f"plus benign version conflicts ({benign_conflicts}) does not match the number "
                f"of documents ({len(documents)})."
            )
        logger.debug(
            "Successfully bulk indexed %s documents (%s benign version conflicts).",
            len(documents),
            benign_conflicts,
        )

    def _benign_create_conflict_count(self, errors: list[dict[str, Any]]) -> int:
        """Count benign 409s from create-only writes (the chunk already exists,
        so a live/forward writer owns it and the port yields); raise
        ElasticsearchIndexError on any other error.

        elasticsearch-py exposes no typed model for bulk per-item errors (bulk() ->
        Any, BulkIndexError.errors -> List[Any]); they are raw {op_type: {...}}
        dicts, so we read the fields directly. A create-conflict is keyed under
        "create" (the op_type) and reports status 409 / version_conflict.
        """
        benign = 0
        fatal: list[dict[str, Any]] = []
        for error in errors:
            item = error.get("create") or {}
            err_type = (item.get("error") or {}).get("type", "")
            if (
                item.get("status") == HTTPStatus.CONFLICT
                and err_type == _VERSION_CONFLICT_ERROR_TYPE
            ):
                benign += 1
            else:
                fatal.append(error)
        if fatal:
            raise ElasticsearchIndexError(
                f"Failed to bulk index documents for index {self._index_name}. "
                f"{len(fatal)} fatal error(s) occurred: {_summarize_bulk_errors(fatal)}"
            )
        return benign

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def delete_document(self, document_chunk_id: str) -> bool:
        """Deletes a document.

        Args:
            document_chunk_id: The Elasticsearch ID of the document chunk to
                delete.

        Raises:
            Exception: There was an error deleting the document.

        Returns:
            True if the document was deleted, False if it was not found.
        """
        try:
            logger.debug(
                "Trying to delete document chunk %s from index %s.",
                document_chunk_id,
                self._index_name,
            )
            result = self._client.delete(index=self._index_name, id=document_chunk_id)
        except ApiError as e:
            if e.status_code == 404:
                logger.debug(
                    "Document chunk %s not found in index %s.",
                    document_chunk_id,
                    self._index_name,
                )
                return False
            else:
                raise e

        result_string: str = result.get("result", "")
        match result_string:
            case "deleted":
                logger.debug(
                    "Successfully deleted document chunk %s from index %s.",
                    document_chunk_id,
                    self._index_name,
                )
                return True
            case "not_found":
                logger.debug(
                    "Document chunk %s not found in index %s.",
                    document_chunk_id,
                    self._index_name,
                )
                return False
            case _:
                raise RuntimeError(
                    f'Unknown Elasticsearch deletion result: "{result_string}".'
                )

    @log_function_time(print_only=True, debug_only=True)
    def delete_by_query(
        self,
        query_body: dict[str, Any],
        refresh: bool = False,
        max_docs: int | None = None,
    ) -> int:
        """Deletes documents by a query.

        Args:
            query_body: The body of the query to delete documents by.
            refresh: Refresh the affected shards once the delete completes, so an
                immediate follow-up count/search sees the deletions (they are
                otherwise not visible until the next auto-refresh).
            max_docs: Delete at most this many matching docs, then return. Bounds a
                single call so it can't run past the client's HTTP timeout on a huge
                match set; the caller re-runs until the match set is empty.

        Raises:
            Exception: There was an error deleting the documents.

        Returns:
            The number of documents deleted.
        """
        logger.debug(
            "Trying to delete documents by query for index %s.",
            self._index_name,
        )
        # elasticsearch-py can move keyword parameters into a supplied body.
        # Keep this reusable query object immutable for follow-up operations.
        params: dict[str, Any] = {
            "index": self._index_name,
            "body": copy.deepcopy(query_body),
        }
        if refresh:
            params["refresh"] = True
        if max_docs is not None:
            params["max_docs"] = max_docs
        result = self._client.delete_by_query(**params)
        if result.get("timed_out", False):
            raise RuntimeError(
                f"Delete by query timed out for index {self._index_name}."
            )
        if len(result.get("failures", [])) > 0:
            raise RuntimeError(
                f"Failed to delete some or all of the documents for index {self._index_name}."
            )

        num_deleted = result.get("deleted", 0)
        num_processed = result.get("total", 0)
        if num_deleted != num_processed:
            raise RuntimeError(
                f"Failed to delete some or all of the documents for index {self._index_name}. "
                f"{num_deleted} documents were deleted out of {num_processed} documents that were "
                "processed."
            )

        logger.debug(
            "Successfully deleted %s documents by query for index %s.",
            num_deleted,
            self._index_name,
        )
        return num_deleted

    def count_by_query(self, query_body: dict[str, Any]) -> int:
        """Counts documents matching a query for this index (the _count API).

        Used as reclaim's deletion gate (count == 0 means the slice drained), so it
        fails closed: a partial count from shard failures under-reports and could
        falsely green-light deletion, so raise instead of trusting it.
        """
        result = self._client.count(
            index=self._index_name,
            query=query_body.get("query"),
        )
        shards = result.get("_shards", {})
        if shards.get("failed", 0):
            raise RuntimeError(
                f"Count for index {self._index_name} hit shard failures ({shards}); "
                "refusing a partial count as a deletion gate."
            )
        return int(result["count"])

    @log_function_time(
        print_only=True,
        debug_only=True,
        include_args_subset={
            "document_chunk_id": str,
            "properties_to_update": lambda x: x.keys(),
        },
    )
    def update_document(
        self,
        document_chunk_id: str,
        properties_to_update: dict[str, Any],
        ignore_missing: bool = False,
    ) -> None:
        """Updates an Elasticsearch document chunk's properties.

        Args:
            document_chunk_id: The Elasticsearch ID of the document chunk to
                update.
            properties_to_update: The properties of the document to update. Each
                property should exist in the schema.
            ignore_missing: If True, silently return instead of raising when the
                document chunk does not exist (Elasticsearch responds with a 404).
                Defaults to False.

        Raises:
            Exception: There was an error updating the document.
        """
        logger.debug(
            "Trying to update document chunk %s for index %s.",
            document_chunk_id,
            self._index_name,
        )
        try:
            result = self._client.update(
                index=self._index_name,
                id=document_chunk_id,
                doc=properties_to_update,
                source=False,
            )
        except ApiError as e:
            if ignore_missing and e.status_code == 404:
                logger.debug(
                    "Document chunk %s not found in index %s; ignoring as requested.",
                    document_chunk_id,
                    self._index_name,
                )
                return
            raise
        result_id = result.get("_id", "")
        # Sanity check.
        if result_id != document_chunk_id:
            raise RuntimeError(
                f'Upon trying to update a document, Elasticsearch responded with ID "{result_id}" '
                f'instead of "{document_chunk_id}" which is the ID it was given.'
            )
        result_string: str = result.get("result", "")
        match result_string:
            # Sanity check.
            case "updated":
                logger.debug(
                    "Successfully updated document chunk %s for index %s.",
                    document_chunk_id,
                    self._index_name,
                )
                return
            case "noop":
                logger.warning(
                    'Elasticsearch reported a no-op when trying to update document with ID "%s".',
                    document_chunk_id,
                )
                return
            case _:
                raise RuntimeError(
                    f'The Elasticsearch client returned result "{result_string}" for updating '
                    f'document chunk "{document_chunk_id}". This is unexpected.'
                )

    @log_function_time(
        print_only=True,
        debug_only=True,
        include_args_subset={
            "document_chunk_ids": len,
            "properties_to_update": lambda x: x.keys(),
        },
    )
    def bulk_update_documents(
        self,
        document_chunk_ids: list[str],
        properties_to_update: dict[str, Any],
        ignore_missing: bool = False,
        surface_document_missing: bool = False,
    ) -> None:
        """Bulk updates Elasticsearch document chunks' properties.

        The ``properties_to_update`` is applied to all the document chunks with
        the given IDs.

        Args:
            document_chunk_ids: The Elasticsearch IDs of the document chunks to
                update.
            properties_to_update: The properties of the document to update. Each
                property should exist in the schema.
            ignore_missing: If True, document chunks that do not exist
                (Elasticsearch reports a 404 ``document_missing_exception``) are
                skipped instead of being treated as fatal errors. Defaults to
                False.
            surface_document_missing: When True and the only fatal errors are 404
                document_missing, raise ElasticsearchDocumentMissingError instead of
                ElasticsearchUpdateError (FUTURE write during a reindex port).
                Takes precedence over ``ignore_missing``.

        Raises:
            Exception: There was an error during the bulk update.
            BulkIndexError: There was an error during the bulk update. This is a
                known specific error type that is raised by the Elasticsearch
                library's bulk function.
            ElasticsearchUpdateError: The number of successful operations reported
                by Elasticsearch does not match the number of document chunks to
                update, or there was at least one other kind of fatal error for
                a particular document chunk.
            ElasticsearchDocumentMissingError: ``surface_document_missing`` was set
                and the only fatal errors were 404 document_missing.
        """
        if not document_chunk_ids:
            return
        logger.debug(
            "Bulk updating %s document chunks for index %s.",
            len(document_chunk_ids),
            self._index_name,
        )
        data = []
        for document_chunk_id in document_chunk_ids:
            data.append(
                {
                    "_index": self._index_name,
                    "_id": document_chunk_id,
                    "_op_type": "update",
                    "doc": properties_to_update,
                }
            )
        # max_retries is the number of times to retry a request if we get a 429.
        # We do not raise on error (the default behavior of ``bulk`` is to
        # raise) because we want to attempt to retry certain failed chunks in
        # this function. Raising on exception indicates something went wrong
        # with the entire batch, which we do not consider retryable in this
        # function.
        successes, errors = bulk(
            self._client,
            data,
            max_retries=3,
            raise_on_error=False,
            raise_on_exception=True,
        )
        if not isinstance(errors, list):
            raise ElasticsearchUpdateError(
                "Elasticsearch bulk helper returned malformed error details."
            )

        ignored_missing_count = 0
        missing_chunk_ids: list[str] = []
        if errors:
            retryable_ids = []
            fatal_errors = []
            for error in errors:
                # error is {"update": {...}} since we only issue updates in this
                # function.
                info = error.get("update")
                if info is None:
                    raise ElasticsearchUpdateError(
                        "Elasticsearch returned a malformed error."
                    )
                status = info.get("status", 0)
                err_obj = info.get("error", {})
                err_type = err_obj.get("type", "") if isinstance(err_obj, dict) else ""

                if (
                    (ignore_missing or surface_document_missing)
                    and status == HTTPStatus.NOT_FOUND
                    and err_type == _DOCUMENT_MISSING_ERROR_TYPE
                ):
                    if surface_document_missing:
                        # doc not in this index yet; surface instead of failing
                        # (FUTURE write during a reindex port)
                        missing_chunk_id = info.get("_id", "")
                        if not missing_chunk_id:
                            raise ElasticsearchUpdateError(
                                "Elasticsearch returned a document_missing error when trying to bulk "
                                f"update document chunks for index {self._index_name}. Error: {error}. "
                                "The error did not contain an ID however.",
                            )
                        missing_chunk_ids.append(missing_chunk_id)
                    else:
                        # ignore_missing: skip silently (benign indexing race)
                        logger.debug(
                            "Document chunk %s not found in index %s during bulk update; "
                            "ignoring as requested.",
                            info.get("_id", ""),
                            self._index_name,
                        )
                        ignored_missing_count += 1
                elif status >= 500 and err_type in _RETRYABLE_UPDATE_ERROR_TYPES:
                    # We have seen a bug in Elasticsearch version 3.4.0 when using
                    # the knn plugin and when derived_source is enabled (the
                    # default), when Elasticsearch is under load sometimes updates
                    # fail transiently with these errors. This is retryable, and
                    # we do so once here. This should be fixed in Elasticsearch
                    # 3.6.0. See
                    # https://github.com/elasticsearch-project/k-NN/issues/3191
                    logger.warning(
                        "Elasticsearch returned a retryable error when trying to bulk update "
                        "document chunks for index %s. Error: %s. Retrying once.",
                        self._index_name,
                        error,
                    )
                    retryable_id = info.get("_id", "")
                    if not retryable_id:
                        raise ElasticsearchUpdateError(
                            "Elasticsearch returned a retryable error when trying to bulk update "
                            f"document chunks for index {self._index_name}. Error: {error}. The "
                            "error did not contain an ID however.",
                        )
                    retryable_ids.append(retryable_id)
                else:
                    fatal_errors.append(error)

            if fatal_errors:
                raise ElasticsearchUpdateError(
                    f"Failed to bulk update document chunks for index {self._index_name}. "
                    f"{len(fatal_errors)} fatal error(s) occurred: "
                    f"{_summarize_bulk_errors(fatal_errors)}"
                )

            data = []
            for document_chunk_id in retryable_ids:
                data.append(
                    {
                        "_index": self._index_name,
                        "_id": document_chunk_id,
                        "_op_type": "update",
                        "doc": properties_to_update,
                    }
                )
            # max_retries is the number of times to retry a request if we get a
            # 429.
            # Explicitly raise on error and exception, we will no longer attempt
            # retries.
            new_successes, _ = bulk(
                self._client,
                data,
                max_retries=3,
                raise_on_error=True,
                raise_on_exception=True,
            )
            if new_successes != len(retryable_ids):
                raise ElasticsearchUpdateError(
                    "Elasticsearch reported no errors during the second bulk update but the number of "
                    f"successful operations ({new_successes}) does not match the number of "
                    f"document chunks retried ({len(retryable_ids)})."
                )
            successes += new_successes

        # ignored-missing are subtracted from the expected total; surfaced-
        # missing are reported separately and not counted as successes.
        expected_successes = len(document_chunk_ids) - ignored_missing_count
        if successes + len(missing_chunk_ids) != expected_successes:
            raise ElasticsearchUpdateError(
                f"Elasticsearch reported no errors during bulk update but the number of successful "
                f"operations ({successes}) plus missing ({len(missing_chunk_ids)}) does not match "
                f"the number of document chunks ({expected_successes})."
            )
        if missing_chunk_ids:
            raise ElasticsearchDocumentMissingError(missing_chunk_ids)
        logger.debug(
            "Successfully bulk updated %s document chunks.", len(document_chunk_ids)
        )

    def validate_regulatory_chunk_projection(
        self,
        expected_chunks: dict[str, tuple[str, int, str]],
        *,
        validity_start_date: int | None,
        validity_end_date: int | None,
    ) -> None:
        """Verify exact canonical identities and old dates before a bulk patch.

        ``expected_chunks`` maps the deterministic Elasticsearch document id to
        ``(document_id, projection_chunk_index, regulatory_chunk_id)``. Only
        the small identity/date fields are fetched; vectors and content never
        cross the wire.
        """

        if not expected_chunks:
            return
        expected_counts_by_document_id: dict[str, int] = {}
        for document_id, _, _ in expected_chunks.values():
            expected_counts_by_document_id[document_id] = (
                expected_counts_by_document_id.get(document_id, 0) + 1
            )
        for document_id, expected_count in expected_counts_by_document_id.items():
            actual_count = self.count_by_query(
                {
                    "query": {
                        "term": {
                            DOCUMENT_ID_FIELD_NAME: {"value": document_id},
                        }
                    }
                }
            )
            if actual_count != expected_count:
                raise ElasticsearchUpdateError(
                    "Regulatory projection chunk count mismatch for document "
                    f"{document_id}: expected {expected_count}, found {actual_count}."
                )
        source_fields = [
            DOCUMENT_ID_FIELD_NAME,
            CHUNK_INDEX_FIELD_NAME,
            REGULATORY_CHUNK_ID_FIELD_NAME,
            VALIDITY_START_DATE_FIELD_NAME,
            VALIDITY_END_DATE_FIELD_NAME,
        ]
        response = self._client.mget(
            index=self._index_name,
            docs=[
                {"_id": chunk_id, "_source": source_fields}
                for chunk_id in expected_chunks
            ],
        )
        documents = response.get("docs")
        if not isinstance(documents, list):
            raise ElasticsearchUpdateError(
                "Elasticsearch returned a malformed regulatory projection preflight."
            )

        seen: set[str] = set()
        missing: list[str] = []
        mismatches: list[str] = []
        for document in documents:
            if not isinstance(document, dict):
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned a malformed document in regulatory "
                    "projection preflight."
                )
            chunk_id = document.get("_id")
            if not isinstance(chunk_id, str) or chunk_id not in expected_chunks:
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned an unexpected chunk in regulatory "
                    "projection preflight."
                )
            seen.add(chunk_id)
            if not document.get("found", False):
                missing.append(chunk_id)
                continue
            source = document.get("_source")
            if not isinstance(source, dict):
                mismatches.append(chunk_id)
                continue
            expected_document_id, expected_index, expected_regulatory_id = (
                expected_chunks[chunk_id]
            )
            if (
                source.get(DOCUMENT_ID_FIELD_NAME) != expected_document_id
                or source.get(CHUNK_INDEX_FIELD_NAME) != expected_index
                or source.get(REGULATORY_CHUNK_ID_FIELD_NAME) != expected_regulatory_id
                or source.get(VALIDITY_START_DATE_FIELD_NAME) != validity_start_date
                or source.get(VALIDITY_END_DATE_FIELD_NAME) != validity_end_date
            ):
                mismatches.append(chunk_id)

        missing.extend(sorted(set(expected_chunks) - seen))
        if missing:
            raise ElasticsearchDocumentMissingError(sorted(set(missing)))
        if mismatches:
            preview = ", ".join(sorted(set(mismatches))[:5])
            raise ElasticsearchUpdateError(
                "Regulatory projection identity/date mismatch for "
                f"{len(set(mismatches))} chunk(s): {preview}"
            )

    def get_document_chunk_identities(
        self,
        document_id: str,
    ) -> dict[str, tuple[int, str]]:
        """Fetch every stored identity for one document without vector payloads."""

        raw_response = self._client.search(
            index=self._index_name,
            size=DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW,
            query={
                "term": {
                    DOCUMENT_ID_FIELD_NAME: {"value": document_id},
                }
            },
            source_includes=[CHUNK_INDEX_FIELD_NAME, REGULATORY_CHUNK_ID_FIELD_NAME],
        )
        response = (
            raw_response if isinstance(raw_response, dict) else dict(raw_response.body)
        )
        hits_container = response.get("hits")
        hits = hits_container.get("hits") if isinstance(hits_container, dict) else None
        if not isinstance(hits, list):
            raise ElasticsearchUpdateError(
                "Elasticsearch returned malformed regulatory chunk identities."
            )
        if len(hits) >= DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW:
            raise ElasticsearchUpdateError(
                "Regulatory identity preflight exceeded the safe result window."
            )

        identities: dict[str, tuple[int, str]] = {}
        for hit in hits:
            if not isinstance(hit, dict):
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned a malformed regulatory identity."
                )
            chunk_id = hit.get("_id")
            source = hit.get("_source")
            if not isinstance(chunk_id, str) or not isinstance(source, dict):
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned an incomplete regulatory identity."
                )
            chunk_index = source.get(CHUNK_INDEX_FIELD_NAME)
            regulatory_chunk_id = source.get(REGULATORY_CHUNK_ID_FIELD_NAME)
            if not isinstance(chunk_index, int) or not isinstance(
                regulatory_chunk_id, str
            ):
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned an invalid regulatory identity."
                )
            identities[chunk_id] = (chunk_index, regulatory_chunk_id)
        return identities

    def get_document_chunks(
        self,
        document_chunk_ids: list[str],
    ) -> dict[str, DocumentChunk]:
        """Fetch an exact set of stored chunks, including their vectors."""

        if not document_chunk_ids:
            return {}
        if len(set(document_chunk_ids)) != len(document_chunk_ids):
            raise ValueError("document_chunk_ids must be unique")
        expected_ids = set(document_chunk_ids)
        response = self._client.mget(
            index=self._index_name,
            docs=[
                {
                    "_id": chunk_id,
                    "_source": {
                        "includes": [
                            "*",
                            CONTENT_VECTOR_FIELD_NAME,
                            TITLE_VECTOR_FIELD_NAME,
                        ]
                    },
                }
                for chunk_id in document_chunk_ids
            ],
        )
        documents = response.get("docs")
        if not isinstance(documents, list):
            raise ElasticsearchUpdateError(
                "Elasticsearch returned a malformed document verification response."
            )

        chunks: dict[str, DocumentChunk] = {}
        missing: list[str] = []
        for document in documents:
            if not isinstance(document, dict):
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned a malformed document during verification."
                )
            chunk_id = document.get("_id")
            if not isinstance(chunk_id, str) or chunk_id not in expected_ids:
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned an unexpected chunk during verification."
                )
            if chunk_id in chunks or chunk_id in missing:
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned a duplicate chunk during verification."
                )
            if not document.get("found", False):
                missing.append(chunk_id)
                continue
            source = document.get("_source")
            if not isinstance(source, dict):
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned a chunk without a source during verification."
                )
            try:
                chunks[chunk_id] = DocumentChunk.model_validate(source)
            except ValueError as error:
                raise ElasticsearchUpdateError(
                    "Elasticsearch returned an invalid chunk during verification."
                ) from error

        missing.extend(sorted(expected_ids - set(chunks) - set(missing)))
        if missing:
            raise ElasticsearchDocumentMissingError(sorted(missing))
        if set(chunks) != expected_ids:
            raise ElasticsearchUpdateError(
                "Elasticsearch document verification returned a different ID set."
            )
        return chunks

    @log_function_time(print_only=True, debug_only=True, include_args=True)
    def get_document(self, document_chunk_id: str) -> DocumentChunk:
        """Gets an Elasticsearch document chunk.

        Will raise an exception if the document chunk is not found.

        Args:
            document_chunk_id: The Elasticsearch ID of the document chunk to get.

        Raises:
            Exception: There was an error getting the document. This includes
                the case where the document is not found.

        Returns:
            The document chunk.
        """
        logger.debug(
            "Trying to get document chunk %s from index %s.",
            document_chunk_id,
            self._index_name,
        )
        result = self._client.get(
            index=self._index_name,
            id=document_chunk_id,
            source_includes=[
                "*",
                CONTENT_VECTOR_FIELD_NAME,
                TITLE_VECTOR_FIELD_NAME,
            ],
        )
        found_result: bool = result.get("found", False)
        if not found_result:
            raise RuntimeError(
                f'Document chunk with ID "{document_chunk_id}" was not found.'
            )

        document_chunk_source: dict[str, Any] | None = result.get("_source")
        if not document_chunk_source:
            raise RuntimeError(
                f'Document chunk with ID "{document_chunk_id}" has no data.'
            )

        logger.debug(
            "Successfully got document chunk %s from index %s.",
            document_chunk_id,
            self._index_name,
        )
        return DocumentChunk.model_validate(document_chunk_source)

    @log_function_time(print_only=True, debug_only=True)
    def search(
        self,
        body: dict[str, Any],
        normalization_method: str | None,
        search_type: ElasticsearchSearchType = ElasticsearchSearchType.UNKNOWN,
    ) -> list[SearchHit[DocumentChunkWithoutVectors]]:
        """Searches the index.

        NOTE: Does not return vector fields. In order to take advantage of
        performance benefits, the search body should exclude the schema's vector
        fields.

        TODO(andrei): Ideally we could check that every field in the body is
        present in the index, to avoid a class of runtime bugs that could easily
        be caught during development. Or change the function signature to accept
        a predefined pydantic model of allowed fields.

        Args:
            body: The body of the search request. See the Elasticsearch
                documentation for more information on search request bodies.
            normalization_method: The score normalization method used by the
                hybrid query. This is retained for diagnostics; the query body
                contains the client-side fusion specification.
            search_type: Label for Prometheus metrics. Does not affect search
                behavior.

        Raises:
            Exception: There was an error searching the index.

        Returns:
            List of search hits that match the search request.
        """
        logger.debug(
            "Trying to search index %s with normalization method %s.",
            self._index_name,
            normalization_method,
        )
        result: dict[str, Any]
        ctx = self._get_emit_metrics_context_manager(search_type)
        with ctx:
            try:
                t0 = time.perf_counter()
                raw_result = (
                    self._search_hybrid_fusion(body)
                    if "_onyx_hybrid_fusion" in body
                    else self._client.search(
                        index=self._index_name,
                        **self._search_kwargs_from_body(body),
                    )
                )
                result = (
                    raw_result
                    if isinstance(raw_result, dict)
                    else dict(raw_result.body)
                )
                client_duration_s = time.perf_counter() - t0
                hits, time_took, timed_out, phase_took, profile = (
                    self._get_hits_and_profile_from_search_result(result)
                )
                # Inside the try/except so that server-side timeouts (which
                # raise inside this helper) land in
                # record_elasticsearch_search_error and never reach
                # observe_elasticsearch_search — keeping the latency histograms
                # clean of timed-out queries.
                self._log_search_result_perf(
                    time_took=time_took,
                    timed_out=timed_out,
                    phase_took=phase_took,
                    profile=profile,
                    body=body,
                    normalization_method=normalization_method,
                    raise_on_timeout=True,
                )
                if self._emit_metrics:
                    observe_elasticsearch_search(
                        search_type, client_duration_s, time_took
                    )
            except Exception as e:
                if self._emit_metrics:
                    record_elasticsearch_search_error(search_type, e)
                raise

        search_hits: list[SearchHit[DocumentChunkWithoutVectors]] = []
        for hit in hits:
            document_chunk_source: dict[str, Any] | None = hit.get("_source")
            if not document_chunk_source:
                raise RuntimeError(
                    f'Document chunk with ID "{hit.get("_id", "")}" has no data.'
                )
            document_chunk_score = hit.get("_score", None)
            match_highlights: dict[str, list[str]] = hit.get("highlight", {})
            explanation: dict[str, Any] | None = hit.get("_explanation", None)
            search_hit = SearchHit[DocumentChunkWithoutVectors](
                document_chunk=DocumentChunkWithoutVectors.model_validate(
                    document_chunk_source
                ),
                score=document_chunk_score,
                match_highlights=match_highlights,
                explanation=explanation,
            )
            search_hits.append(search_hit)
        logger.debug(
            "Successfully searched index %s and got %s hits.",
            self._index_name,
            len(search_hits),
        )
        return search_hits

    def _search_hybrid_fusion(self, body: dict[str, Any]) -> dict[str, Any]:
        """Fuse independent query lanes without requiring the retriever API."""
        specification = body["_onyx_hybrid_fusion"]
        subqueries: list[dict[str, Any]] = specification["subqueries"]
        weights: list[float] = specification["weights"]
        filters: list[dict[str, Any]] = specification["filters"]
        rank_window_size: int = specification["rank_window_size"]
        normalizer: str = specification["normalizer"]

        merged_hits: dict[str, dict[str, Any]] = {}
        merged_scores: Counter[str] = Counter()
        total_took = 0
        timed_out = False

        for subquery, weight in zip(subqueries, weights, strict=True):
            if "knn" in subquery:
                knn_query = dict(subquery["knn"])
                knn_query["filter"] = {"bool": {"filter": filters}}
                request_body: dict[str, Any] = {"knn": knn_query}
            else:
                request_body = {
                    "query": {"bool": {"must": [subquery], "filter": filters}}
                }

            request_body.update(
                {
                    "size": rank_window_size,
                    "timeout": body["timeout"],
                    "_source": body["_source"],
                }
            )
            if "highlight" in body:
                request_body["highlight"] = body["highlight"]
            if "explain" in body:
                request_body["explain"] = body["explain"]

            raw_response = self._client.search(
                index=self._index_name,
                **self._search_kwargs_from_body(request_body),
            )
            response = (
                raw_response
                if isinstance(raw_response, dict)
                else dict(raw_response.body)
            )
            total_took += int(response.get("took", 0))
            timed_out = timed_out or bool(response.get("timed_out", False))
            hits: list[dict[str, Any]] = response.get("hits", {}).get("hits", [])
            scores = [float(hit.get("_score") or 0.0) for hit in hits]
            normalized_scores = self._normalize_hybrid_scores(scores, normalizer)

            for hit, normalized_score in zip(hits, normalized_scores, strict=True):
                hit_id = str(hit["_id"])
                merged_scores[hit_id] += weight * normalized_score
                previous = merged_hits.get(hit_id)
                if previous is None:
                    merged_hits[hit_id] = dict(hit)
                elif hit.get("highlight"):
                    previous["highlight"] = hit["highlight"]

        for hit_id, hit in merged_hits.items():
            hit["_score"] = merged_scores[hit_id]
        ranked_hits = sorted(
            merged_hits.values(), key=lambda hit: float(hit["_score"]), reverse=True
        )[: int(body["size"])]
        return {
            "took": total_took,
            "timed_out": timed_out,
            "hits": {"hits": ranked_hits},
        }

    @staticmethod
    def _normalize_hybrid_scores(scores: list[float], normalizer: str) -> list[float]:
        if not scores:
            return []
        if normalizer == "minmax":
            minimum = min(scores)
            score_range = max(scores) - minimum
            return (
                [(score - minimum) / score_range for score in scores]
                if score_range > 0
                else [1.0] * len(scores)
            )
        if normalizer == "zscore":
            mean = statistics.fmean(scores)
            standard_deviation = statistics.pstdev(scores) if len(scores) > 1 else 0.0
            return (
                [(score - mean) / standard_deviation for score in scores]
                if standard_deviation > 0
                else [1.0] * len(scores)
            )
        raise ValueError(f"Unsupported hybrid score normalizer: {normalizer}")

    @staticmethod
    def _search_kwargs_from_body(body: dict[str, Any]) -> dict[str, Any]:
        """Translate raw REST field names to elasticsearch-py 8 parameters."""
        search_kwargs = dict(body)
        if "_source" in search_kwargs:
            search_kwargs["source"] = search_kwargs.pop("_source")
        if "from" in search_kwargs:
            search_kwargs["from_"] = search_kwargs.pop("from")
        return search_kwargs

    @log_function_time(print_only=True, debug_only=True)
    def search_for_document_ids(
        self,
        body: dict[str, Any],
        search_type: ElasticsearchSearchType = ElasticsearchSearchType.UNKNOWN,
    ) -> list[str]:
        """Searches the index and returns only document chunk IDs.

        In order to take advantage of the performance benefits of only returning
        IDs, the body should have a key, value pair of "_source": False.
        Otherwise, Elasticsearch will return the entire document body and this
        method's performance will be the same as the search method's.

        TODO(andrei): Ideally we could check that every field in the body is
        present in the index, to avoid a class of runtime bugs that could easily
        be caught during development.

        Args:
            body: The body of the search request. See the Elasticsearch
                documentation for more information on search request bodies.
                TODO(andrei): Make this a more deep interface; callers shouldn't
                need to know to set _source: False for example.
            search_type: Label for Prometheus metrics. Does not affect search
                behavior.

        Raises:
            Exception: There was an error searching the index.

        Returns:
            List of document chunk IDs that match the search request.
        """
        logger.debug(
            "Trying to search for document chunk IDs in index %s.",
            self._index_name,
        )
        if "_source" not in body or body["_source"] is not False:
            logger.warning(
                "The body of the search request for document chunk IDs is missing the key, "
                'value pair of "_source": False. This query will therefore be inefficient.'
            )

        ctx = self._get_emit_metrics_context_manager(search_type)
        with ctx:
            try:
                t0 = time.perf_counter()
                raw_result = self._client.search(
                    index=self._index_name,
                    **self._search_kwargs_from_body(body),
                )
                result = (
                    raw_result
                    if isinstance(raw_result, dict)
                    else dict(raw_result.body)
                )
                client_duration_s = time.perf_counter() - t0
                hits, time_took, timed_out, phase_took, profile = (
                    self._get_hits_and_profile_from_search_result(result)
                )
                # Inside the try/except so that server-side timeouts (which
                # raise inside this helper) land in
                # record_elasticsearch_search_error and never reach
                # observe_elasticsearch_search — keeping the latency histograms
                # clean of timed-out queries.
                self._log_search_result_perf(
                    time_took=time_took,
                    timed_out=timed_out,
                    phase_took=phase_took,
                    profile=profile,
                    body=body,
                    raise_on_timeout=True,
                )
                if self._emit_metrics:
                    observe_elasticsearch_search(
                        search_type, client_duration_s, time_took
                    )
            except Exception as e:
                if self._emit_metrics:
                    record_elasticsearch_search_error(search_type, e)
                raise

        # TODO(andrei): Implement scroll/point in time for results so that we
        # can return arbitrarily-many IDs.
        if len(hits) == DEFAULT_ELASTICSEARCH_MAX_RESULT_WINDOW:
            logger.warning(
                "The search request for document chunk IDs returned the maximum number of "
                "results. It is extremely likely that there are more hits in Elasticsearch than the "
                "returned results."
            )

        # Extract only the _id field from each hit.
        document_chunk_ids: list[str] = []
        for hit in hits:
            document_chunk_id = hit.get("_id")
            if not document_chunk_id:
                raise RuntimeError(
                    "Received a hit from Elasticsearch but the _id field is missing."
                )
            document_chunk_ids.append(document_chunk_id)
        logger.debug(
            "Successfully searched for document chunk IDs in index %s and got %s hits.",
            self._index_name,
            len(document_chunk_ids),
        )
        return document_chunk_ids

    def open_pit(self, keep_alive: str = PIT_KEEP_ALIVE) -> str:
        """Opens a point-in-time (PIT) over this index for a consistent scan.

        The PIT pins the index across searches so concurrent writes don't shift
        the result set. The caller passes the returned id into
        fetch_chunks_for_doc_ids and releases it with close_pit when done.

        Args:
            keep_alive: How long the PIT lives between uses; each search extends
                the lease.

        Raises:
            RuntimeError: Elasticsearch returned no pit_id.

        Returns:
            The point-in-time id.
        """
        response = self._client.open_point_in_time(
            index=self._index_name, keep_alive=keep_alive
        )
        pit_id = response.get("id")
        if not pit_id:
            raise RuntimeError(
                f"open_point_in_time returned no id for index {self._index_name}."
            )
        return pit_id

    def close_pit(self, pit_id: str) -> None:
        """Releases a PIT. Best-effort — a leaked PIT self-expires after keep_alive.

        Args:
            pit_id: The point-in-time id to delete.
        """
        try:
            self._client.close_point_in_time(id=pit_id)
        except NotFoundError:
            pass

    def fetch_chunks_for_doc_ids(
        self,
        pit_id: str,
        doc_ids: list[str],
        search_after: list[object] | None = None,
        page_size: int = _PIT_SCAN_PAGE_SIZE,
        keep_alive: str = PIT_KEEP_ALIVE,
    ) -> tuple[list[DocumentChunkWithoutVectors], list[object] | None, str]:
        """Fetches one page of regular chunks for a batch of documents from a PIT.

        Filters to regular chunks (max_chunk_size == DEFAULT_MAX_CHUNK_SIZE),
        sorts by (document_id, chunk_index), and pages with search_after.
        Vectors are excluded — the port re-embeds. If the PIT expired the scan
        re-opens it and retries once.

        Args:
            pit_id: The point-in-time id from open_pit.
            doc_ids: The document ids whose chunks to fetch.
            search_after: The sort cursor from the previous page; None for the
                first page.
            page_size: Max chunks per page.
            keep_alive: PIT lease extension applied on each search.

        Raises:
            ElasticsearchServerSideTimeout: The search timed out server-side; the
                caller should retry the batch.
            Exception: There was an error searching the index.

        Returns:
            A tuple of (chunks, next_search_after, pit_id_in_use). next_search_after
            is None once the batch is exhausted; pit_id_in_use reflects the new PIT
            when the scan re-opened, so the caller passes it forward.
        """
        if not doc_ids:
            return [], None, pit_id

        # Background scans intentionally skip the user-search metrics/pipeline that
        # search() applies; we still detect a server-side timeout below so a
        # truncated page is never mistaken for the end of the scan.
        try:
            result = self._client.search(
                **self._search_kwargs_from_body(
                    self._pit_scan_body(
                        pit_id, doc_ids, search_after, page_size, keep_alive
                    )
                )
            )
        except NotFoundError as e:
            if not self._is_pit_expired(e):
                raise
            logger.debug(
                "PIT %s expired mid-scan for index %s; reopening.",
                pit_id,
                self._index_name,
            )
            pit_id = self.open_pit(keep_alive)
            result = self._client.search(
                **self._search_kwargs_from_body(
                    self._pit_scan_body(
                        pit_id, doc_ids, search_after, page_size, keep_alive
                    )
                )
            )

        if result.get("timed_out"):
            # A timed-out page returns partial hits; treating it as a short page
            # would silently end the scan early, so fail and let the caller retry.
            raise ElasticsearchServerSideTimeout(
                f"PIT scan of index {self._index_name} timed out server-side."
            )

        hits: list[dict[str, Any]] = result.get("hits", {}).get("hits", [])
        chunks: list[DocumentChunkWithoutVectors] = []
        last_sort: list[object] | None = None
        for hit in hits:
            source = hit.get("_source")
            if not source:
                raise RuntimeError(
                    f'Document chunk with ID "{hit.get("_id", "")}" has no data.'
                )
            chunks.append(DocumentChunkWithoutVectors.model_validate(source))
            last_sort = hit.get("sort")

        # A short page means the batch is exhausted; a full page means resume from
        # the last hit's sort values on the next call.
        next_search_after = last_sort if len(hits) == page_size else None
        return chunks, next_search_after, pit_id

    def iter_chunks_for_doc_ids(
        self,
        doc_ids: list[str],
        page_size: int = _PIT_SCAN_PAGE_SIZE,
        keep_alive: str = PIT_KEEP_ALIVE,
    ) -> Iterator[list[DocumentChunkWithoutVectors]]:
        """Scans regular chunks for a batch of documents, one page at a time.

        Owns the whole PIT lifecycle: opens it, pages with search_after, re-opens
        transparently on expiry, and always closes it (even if the consumer
        raises). The preferred entry point so callers can't leak a PIT.

        Args:
            doc_ids: The document ids whose chunks to scan.
            page_size: Max chunks per page.
            keep_alive: PIT lease extension applied on each search.

        Yields:
            One page (list) of chunks at a time.
        """
        if not doc_ids:
            return
        pit_id = self.open_pit(keep_alive)
        try:
            search_after: list[object] | None = None
            while True:
                chunks, search_after, pit_id = self.fetch_chunks_for_doc_ids(
                    pit_id,
                    doc_ids,
                    search_after=search_after,
                    page_size=page_size,
                    keep_alive=keep_alive,
                )
                if chunks:
                    yield chunks
                if search_after is None:
                    return
        finally:
            self.close_pit(pit_id)

    def _pit_scan_body(
        self,
        pit_id: str,
        doc_ids: list[str],
        search_after: list[object] | None,
        page_size: int,
        keep_alive: str,
    ) -> dict[str, Any]:
        """Builds the PIT search body for one page.

        No index= is sent — the PIT pins the index; keep_alive in the pit block
        extends the lease on every page.
        """
        body: dict[str, Any] = {
            "pit": {"id": pit_id, "keep_alive": keep_alive},
            "size": page_size,
            "_source": {
                "excludes": [CONTENT_VECTOR_FIELD_NAME, TITLE_VECTOR_FIELD_NAME]
            },
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {DOCUMENT_ID_FIELD_NAME: doc_ids}},
                        # Elasticsearch holds no large/mini chunks today, so this
                        # matches everything; kept as a guard if that changes
                        {"term": {MAX_CHUNK_SIZE_FIELD_NAME: DEFAULT_MAX_CHUNK_SIZE}},
                    ]
                }
            },
            "sort": [
                {DOCUMENT_ID_FIELD_NAME: "asc"},
                {CHUNK_INDEX_FIELD_NAME: "asc"},
            ],
        }
        if search_after is not None:
            body["search_after"] = search_after
        return body

    @staticmethod
    def _is_pit_expired(error: NotFoundError) -> bool:
        """True if the 404 is an expired/deleted PIT (search_context_missing).

        The type can be nested under root_cause, so match the stringified body.
        """
        return _SEARCH_CONTEXT_MISSING_ERROR_TYPE in str(
            getattr(error, "info", "")
        ) or _SEARCH_CONTEXT_MISSING_ERROR_TYPE in str(error)

    @log_function_time(print_only=True, debug_only=True)
    def refresh_index(self) -> None:
        """Refreshes the index to make recent changes searchable.

        In Elasticsearch, documents are not immediately searchable after indexing.
        This method forces a refresh to make them available for search.

        Raises:
            Exception: There was an error refreshing the index.
        """
        self._client.indices.refresh(index=self._index_name)

    def _get_hits_and_profile_from_search_result(
        self, result: dict[str, Any]
    ) -> tuple[list[Any], int | None, bool | None, dict[str, Any], dict[str, Any]]:
        """Extracts the hits and profiling information from a search result.

        Args:
            result: The search result to extract the hits from.

        Raises:
            Exception: There was an error extracting the hits from the search
                result.

        Returns:
            A tuple containing the hits from the search result, the time taken
                to execute the search in milliseconds, whether the search timed
                out, the time taken to execute each phase of the search, and the
                profile.
        """
        time_took: int | None = result.get("took")
        timed_out: bool | None = result.get("timed_out")
        phase_took: dict[str, Any] = result.get("phase_took", {})
        profile: dict[str, Any] = result.get("profile", {})

        hits_first_layer: dict[str, Any] = result.get("hits", {})
        if not hits_first_layer:
            raise RuntimeError(
                f"Hits field missing from response when trying to search index {self._index_name}."
            )
        hits_second_layer: list[Any] = hits_first_layer.get("hits", [])

        return hits_second_layer, time_took, timed_out, phase_took, profile

    def _log_search_result_perf(
        self,
        time_took: int | None,
        timed_out: bool | None,
        phase_took: dict[str, Any],
        profile: dict[str, Any],
        body: dict[str, Any],
        normalization_method: str | None = None,
        raise_on_timeout: bool = False,
    ) -> None:
        """Logs the performance of a search result.

        Args:
            time_took: The time taken to execute the search in milliseconds.
            timed_out: Whether the search timed out.
            phase_took: The time taken to execute each phase of the search.
            profile: The profile for the search.
            body: The body of the search request for logging.
            normalization_method: The score normalization method used for the
                search, if any, for logging. Defaults to None.
            raise_on_timeout: Whether to raise an exception if the search timed
                out. Note that the result may still contain useful partial
                results. Defaults to False.

        Raises:
            Exception: If raise_on_timeout is True and the search timed out.
        """
        if time_took and time_took > CLIENT_THRESHOLD_TO_LOG_SLOW_SEARCH_MS:
            logger.warning(
                "Elasticsearch client warning: Search for index %s took %s milliseconds.\n"
                "Body: %s\n"
                "Normalization method: %s\n"
                "Phase took: %s\n"
                "Profile: %s\n",
                self._index_name,
                time_took,
                get_new_body_without_vectors(body),
                normalization_method,
                phase_took,
                json.dumps(profile, indent=2),
            )
        if timed_out:
            error_str = f"Elasticsearch client error: Search timed out for index {self._index_name}."
            logger.error(error_str)
            if raise_on_timeout:
                raise ElasticsearchServerSideTimeout(error_str)

    def _get_emit_metrics_context_manager(
        self, search_type: ElasticsearchSearchType
    ) -> AbstractContextManager[None]:
        """
        Returns the Elasticsearch search tracking context manager (which bumps the
        attempt counter and the in-flight gauge) if emit_metrics is True,
        otherwise returns a null context manager.
        """
        return (
            track_elasticsearch_search(search_type)
            if self._emit_metrics
            else nullcontext()
        )


def wait_for_elasticsearch_with_timeout(
    wait_interval_s: int = 5,
    wait_limit_s: int = 60,
    client: ElasticsearchClient | None = None,
) -> bool:
    """Waits for Elasticsearch to become ready subject to a timeout.

    Will create a new dummy client if no client is provided. Will close this
    client at the end of the function. Will not close the client if it was
    supplied.

    Args:
        wait_interval_s: The interval in seconds to wait between checks.
            Defaults to 5.
        wait_limit_s: The total timeout in seconds to wait for Elasticsearch to
            become ready. Defaults to 60.
        client: The Elasticsearch client to use for pinging. If None, a new dummy
            client will be created. Defaults to None.

    Returns:
        True if Elasticsearch is ready, False otherwise.
    """
    with nullcontext(client) if client else ElasticsearchClient() as client:
        time_start = time.monotonic()
        while True:
            if client.ping():
                logger.info("[Elasticsearch] Readiness probe succeeded. Continuing...")
                return True
            time_elapsed = time.monotonic() - time_start
            if time_elapsed > wait_limit_s:
                logger.info(
                    "[Elasticsearch] Readiness probe did not succeed within the timeout "
                    "(%s seconds).",
                    wait_limit_s,
                )
                return False
            logger.info(
                "[Elasticsearch] Readiness probe ongoing. elapsed=%s timeout=%s",
                format(time_elapsed, ".1f"),
                format(wait_limit_s, ".1f"),
            )
            time.sleep(wait_interval_s)
