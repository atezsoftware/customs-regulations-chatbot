"""Source-kind confinement layered over the unchanged canonical broker."""

from collections.abc import Callable, Generator
from datetime import date
from typing import cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, RunContext, ToolOutcome
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.legal_composite_sources import (
    SourceKind,
    SourceLaneCatalogue,
    revalidate_source_classification,
    source_scope_sha256,
)


def source_lane_filters(
    original: IndexFilters, source_ids: tuple[UUID, ...]
) -> IndexFilters:
    if not source_ids:
        raise CorpusScopeUnavailable("The source-kind lane has no authorized sources.")
    # Existing knowledge selectors form an OR; only these captured IDs may remain.
    return original.model_copy(
        deep=True,
        update={
            "document_set": None,
            "attached_document_ids": [str(identifier) for identifier in source_ids],
            "hierarchy_node_ids": None,
            "project_id_filter": None,
            "persona_id_filter": None,
        },
    )


class SourceLaneBroker(CorpusBroker):
    def __init__(
        self,
        base: CorpusBroker,
        catalogue: SourceLaneCatalogue,
        kind: SourceKind,
    ) -> None:
        if (
            catalogue.user_id != base.user.id
            or catalogue.scope_sha256 != source_scope_sha256(base.user, base.filters)
        ):
            raise PermissionError(
                "Source lane inventory differs from the authorized run scope."
            )
        source_ids = catalogue.source_ids(kind)
        filters = source_lane_filters(base.filters, source_ids)
        super().__init__(
            base.user,
            filters,
            query_indexes=base.query_indexes,
            file_store=base.file_store,
            vision_llm=base.vision_llm,
        )
        self.kind = kind
        self.catalogue = catalogue
        self.original_filters = base.filters.model_copy(deep=True)
        self._lane_scope_sha256 = source_scope_sha256(base.user, self.filters)
        self.classifications = {
            row.source_id: row
            for row in catalogue.records
            if row.source_id in source_ids
        }

    def guard_search_adapter(
        self, adapter: Callable[[dict[str, JsonValue], RunContext], ToolOutcome]
    ) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
        def search(arguments: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
            context.check_active()
            if self._lane_scope_sha256 != source_scope_sha256(self.user, self.filters):
                raise PermissionError(
                    "The captured source lane search filters changed."
                )
            outcome = adapter(arguments, context)
            return outcome.model_copy(
                update={"data": {**outcome.data, "source_lane": self.lane_provenance()}}
            )

        return search

    def lane_provenance(self) -> dict[str, JsonValue]:
        return {
            **self.catalogue.provenance(),
            "source_kind": self.kind.value,
            "lane_source_count": len(self.classifications),
            "lane_uncertain_source_count": sum(
                row.uncertain for row in self.classifications.values()
            ),
        }

    def _require_lane_source(self, source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        recorded = self.classifications.get(UUID(source_id))
        if recorded is None or not recorded.admits(self.kind):
            raise PermissionError("Source is outside the captured source-kind lane.")
        with get_session_with_current_tenant() as session:
            return revalidate_source_classification(
                session,
                user=self.user,
                filters=self.original_filters,
                recorded=recorded,
                check_active=context.check_active,
            )

    def _check_original_kind(
        self, source_id: str, _metadata: dict[str, object]
    ) -> None:
        recorded = self.classifications.get(UUID(source_id))
        if recorded is None:
            raise PermissionError("Original is outside the captured source-kind lane.")
        if not recorded.admits(self.kind):
            raise CorpusScopeUnavailable(
                "Canonical original differs from the requested source-kind lane."
            )

    def source(self, source_id: str, context: RunContext) -> CorpusSource:
        return self._require_lane_source(source_id, context)

    def sources(
        self, query: str, context: RunContext, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[CorpusSource], bool]:
        sources, more = super().sources(query, context, offset=offset, limit=limit)
        for source in sources:
            self._require_lane_source(str(source.id), context)
        return sources, more

    def shared_read_fence(
        self, source_id: str, context: RunContext
    ) -> str | ToolOutcome:
        self._require_lane_source(source_id, context)
        return super().shared_read_fence(source_id, context)

    def page(
        self,
        source_id: str,
        context: RunContext,
        *,
        start: int = 0,
        limit: int = 50,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        self._require_lane_source(source_id, context)
        source, chunks, more = super().page(
            source_id,
            context,
            start=start,
            limit=limit,
            as_of=as_of,
            historical_inventory=historical_inventory,
        )
        for chunk in chunks:
            self._check_original_kind(source_id, chunk.metadata)
        return source, chunks, more

    def chunk(
        self, source_id: str, chunk_id: str, context: RunContext
    ) -> tuple[CorpusSource, CorpusChunk | None]:
        source, chunk = super().chunk(source_id, chunk_id, context)
        if chunk is not None:
            self._check_original_kind(source_id, chunk.metadata)
        return source, chunk

    def sibling_page(
        self,
        source: CorpusSource,
        ids: tuple[str, ...],
        context: RunContext,
    ) -> Generator[CorpusChunk, None, None]:
        self._require_lane_source(str(source.id), context)
        stream = super().sibling_page(source, ids, context)
        try:
            for chunk in stream:
                self._check_original_kind(str(source.id), chunk.metadata)
                yield chunk
        finally:
            stream.close()

    def revalidate_evidence(
        self, items: list[EvidenceItem], context: RunContext
    ) -> None:
        for source_id in dict.fromkeys(item.source_id for item in items):
            self._require_lane_source(source_id, context)
        super().revalidate_evidence(items, context)
        for item in items:
            canonical = item.metadata.get("canonical_metadata")
            metadata = {
                **(canonical if isinstance(canonical, dict) else {}),
                **item.metadata,
            }
            self._check_original_kind(item.source_id, cast(dict[str, object], metadata))

    def _check_search_originals(
        self, result: dict[tuple[str, int], list[EvidenceItem]]
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        for group in result.values():
            for item in group:
                canonical = item.metadata.get("canonical_metadata")
                metadata = {
                    **(canonical if isinstance(canonical, dict) else {}),
                    **item.metadata,
                }
                self._check_original_kind(
                    item.source_id, cast(dict[str, object], metadata)
                )
        return result

    def hydrate_search_centers(
        self, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        for source_id in dict.fromkeys(doc.document_id for doc in docs):
            self._require_lane_source(source_id, context)
        return self._check_search_originals(
            super().hydrate_search_centers(docs, context)
        )

    def hydrate_search_results(
        self, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        for source_id in dict.fromkeys(doc.document_id for doc in docs):
            self._require_lane_source(source_id, context)
        return self._check_search_originals(
            super().hydrate_search_results(docs, context)
        )


def build_lane_broker(
    base: CorpusBroker, catalogue: SourceLaneCatalogue, kind: SourceKind
) -> SourceLaneBroker:
    return SourceLaneBroker(base, catalogue, kind)
