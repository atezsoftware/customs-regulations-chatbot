"""Source-kind confinement layered over the unchanged canonical broker."""

from collections.abc import Callable, Generator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextvars import ContextVar
from datetime import date
from threading import Lock, Semaphore
from typing import cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext, ToolOutcome
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable, CorpusSource
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.legal_composite_preparation import (
    revalidate_prepared_source_classification,
    revalidate_prepared_sources,
)
from onyx.db.legal_composite_sources import (
    MAX_OPENING_BATCH_SOURCES,
    SourceClassification,
    SourceKind,
    SourceLaneCatalogue,
    classify_candidate_sources,
    revalidate_source_classification,
    source_scope_sha256,
)
from onyx.legal_composite.acquisition import CanonicalEvidenceStage
from onyx.legal_composite.shared_work import SharedCanonicalCenters
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel


class CandidateSourceClassifier:
    """Run-local candidate opening singleflight, with immutable authorization scope."""

    def __init__(self, base: CorpusBroker, context: RunContext) -> None:
        self.user = base.user
        self.filters = base.filters.model_copy(deep=True)
        self.scope_sha256 = source_scope_sha256(self.user, self.filters)
        self.context = context
        self.source_kinds: dict[str, SourceKind] = {}
        self.records: dict[UUID, SourceClassification] = {}
        self._pending: dict[UUID, Future[SourceClassification | None]] = {}
        self._lock = Lock()
        self._slots = Semaphore(4)

    def classify_sources(
        self, source_ids: tuple[str, ...]
    ) -> dict[str, SourceClassification]:
        self.context.check_research_active()
        if self.scope_sha256 != source_scope_sha256(self.user, self.filters):
            raise PermissionError("Candidate classification scope changed.")
        identifiers: list[UUID] = []
        for value in dict.fromkeys(source_ids):
            try:
                identifiers.append(UUID(value))
            except ValueError:
                continue
        owned: list[UUID] = []
        with self._lock:
            futures: dict[UUID, Future[SourceClassification | None]] = {}
            for identifier in identifiers:
                future = self._pending.get(identifier)
                if future is None:
                    future = Future()
                    self._pending[identifier] = future
                    owned.append(identifier)
                futures[identifier] = future
        try:
            for start in range(0, len(owned), MAX_OPENING_BATCH_SOURCES):
                batch = tuple(owned[start : start + MAX_OPENING_BATCH_SOURCES])
                while not self._slots.acquire(timeout=0.05):
                    self.context.check_research_active()
                try:
                    self.context.check_research_active()
                    with get_session_with_current_tenant() as session:
                        records = classify_candidate_sources(
                            session,
                            user=self.user,
                            filters=self.filters,
                            source_ids=batch,
                            check_active=self.context.check_research_active,
                        )
                    if self.scope_sha256 != source_scope_sha256(
                        self.user, self.filters
                    ):
                        raise PermissionError("Candidate classification scope changed.")
                    with self._lock:
                        self.records.update(records)
                        self.source_kinds.update(
                            {
                                str(identifier): record.kind
                                for identifier, record in records.items()
                            }
                        )
                    for identifier in batch:
                        futures[identifier].set_result(records.get(identifier))
                finally:
                    self._slots.release()
        except BaseException as error:
            for identifier in owned:
                if not futures[identifier].done():
                    futures[identifier].set_exception(error)
            raise
        result: dict[str, SourceClassification] = {}
        for identifier, future in futures.items():
            while True:
                self.context.check_research_active()
                try:
                    record = future.result(timeout=0.05)
                    break
                except FutureTimeout:
                    if future.done():
                        raise
                    continue
            if record is not None:
                result[str(identifier)] = record
        return result

    def provenance(
        self, records: dict[UUID, SourceClassification] | None = None
    ) -> dict[str, JsonValue]:
        records = self.snapshot() if records is None else records
        return {
            "scope_sha256": self.scope_sha256,
            "source_count": len(records),
            "uncertain_source_count": sum(row.uncertain for row in records.values()),
            "inventory_complete": False,
            "candidate_classification_only": True,
            "classification_is_full_source_proof": False,
            "limitations": [
                "Only query candidates have been classified; corpus absence is unverified."
            ],
        }

    def snapshot(self) -> dict[UUID, SourceClassification]:
        with self._lock:
            return dict(self.records)


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
        catalogue: SourceLaneCatalogue | CandidateSourceClassifier,
        kind: SourceKind,
        shared_centers: SharedCanonicalCenters | None = None,
    ) -> None:
        if (
            catalogue.user_id
            if isinstance(catalogue, SourceLaneCatalogue)
            else catalogue.user.id
        ) != base.user.id or catalogue.scope_sha256 != source_scope_sha256(
            base.user, base.filters
        ):
            raise PermissionError(
                "Source lane inventory differs from the authorized run scope."
            )
        self.candidate_classifier = (
            catalogue if isinstance(catalogue, CandidateSourceClassifier) else None
        )
        source_ids = (
            catalogue.source_ids(kind)
            if isinstance(catalogue, SourceLaneCatalogue)
            else ()
        )
        filters = (
            source_lane_filters(base.filters, source_ids)
            if self.candidate_classifier is None
            else base.filters.model_copy(deep=True)
        )
        super().__init__(
            base.user,
            filters,
            query_indexes=base.query_indexes,
            file_store=base.file_store,
            vision_llm=base.vision_llm,
        )
        self.kind = kind
        self.shared_centers = shared_centers
        self.catalogue = catalogue
        self.original_filters = base.filters.model_copy(deep=True)
        self._lane_scope_sha256 = source_scope_sha256(base.user, self.filters)
        source_id_set = frozenset(source_ids)
        self.classifications = (
            {
                row.source_id: row
                for row in catalogue.records
                if row.source_id in source_id_set
            }
            if isinstance(catalogue, SourceLaneCatalogue)
            else catalogue.records
        )
        self._search_receipts: ContextVar[list[dict[str, JsonValue]] | None] = (
            ContextVar("legal_composite_candidate_receipts", default=None)
        )
        self._receipt_lock = Lock()

    def record_search(self, receipt: dict[str, JsonValue]) -> None:
        with self._receipt_lock:
            receipts = self._search_receipts.get()
            if receipts is not None:
                receipts.append(receipt)

    def _check_lane_scope(self) -> None:
        if self._lane_scope_sha256 != source_scope_sha256(self.user, self.filters):
            raise PermissionError("The captured source lane search filters changed.")

    def guard_search_adapter(
        self, adapter: Callable[[dict[str, JsonValue], RunContext], ToolOutcome]
    ) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
        def search(arguments: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
            context.check_active()
            self._check_lane_scope()
            receipts: list[dict[str, JsonValue]] = []
            token = self._search_receipts.set(receipts)
            try:
                outcome = adapter(arguments, context)
                self._check_lane_scope()
            finally:
                self._search_receipts.reset(token)
            incomplete = any(receipt.get("incomplete") is True for receipt in receipts)
            return outcome.model_copy(
                update={
                    "status": OutcomeStatus.PARTIAL
                    if incomplete
                    and outcome.status in {OutcomeStatus.FOUND, OutcomeStatus.NOT_FOUND}
                    else outcome.status,
                    "data": {
                        **outcome.data,
                        "source_lane": self.lane_provenance(),
                        **({"candidate_search": receipts} if receipts else {}),
                    },
                }
            )

        return search

    def lane_provenance(self) -> dict[str, JsonValue]:
        classifications = (
            self.candidate_classifier.snapshot()
            if self.candidate_classifier
            else self.classifications
        )
        return {
            **(
                self.candidate_classifier.provenance(classifications)
                if self.candidate_classifier
                else self.catalogue.provenance()
            ),
            "source_kind": self.kind.value,
            "lane_source_count": sum(
                row.admits(self.kind) for row in classifications.values()
            ),
            "lane_uncertain_source_count": sum(
                row.uncertain and row.admits(self.kind)
                for row in classifications.values()
            ),
        }

    def _require_lane_source(self, source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        self._check_lane_scope()
        if self.candidate_classifier is not None:
            self.candidate_classifier.classify_sources((source_id,))
        recorded = self.classifications.get(UUID(source_id))
        if recorded is None or not recorded.admits(self.kind):
            raise PermissionError("Source is outside the captured source-kind lane.")
        with get_session_with_current_tenant() as session:
            validate = (
                revalidate_prepared_source_classification
                if recorded.prepared
                else revalidate_source_classification
            )
            return validate(
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

    def _admitted_search_docs(
        self, docs: list[SearchDoc], context: RunContext
    ) -> list[SearchDoc]:
        context.check_active()
        self._check_lane_scope()
        identifiers = {UUID(doc.document_id) for doc in docs}
        records = [
            self.classifications[key]
            for key in identifiers
            if key in self.classifications
        ]
        if any(not record.prepared for record in records):
            for identifier in identifiers:
                self._require_lane_source(str(identifier), context)
            return docs
        with get_session_with_current_tenant() as session:
            current = revalidate_prepared_sources(
                session,
                user=self.user,
                filters=self.original_filters,
                recorded=[row for row in records if row.admits(self.kind)],
                check_active=context.check_active,
            )
        missing = identifiers - current.keys()
        if missing:
            self.record_search(
                {
                    "incomplete": True,
                    "unavailable_source_ids": sorted(str(key) for key in missing),
                }
            )
        return [doc for doc in docs if UUID(doc.document_id) in current]

    def source(self, source_id: str, context: RunContext) -> CorpusSource:
        return self._require_lane_source(source_id, context)

    def sources(
        self, query: str, context: RunContext, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[CorpusSource], bool]:
        sources, more = super().sources(query, context, offset=offset, limit=limit)
        if self.candidate_classifier is not None:
            classified = self.candidate_classifier.classify_sources(
                tuple(str(source.id) for source in sources)
            )
            sources = [
                source
                for source in sources
                if (record := classified.get(str(source.id))) is not None
                and record.admits(self.kind)
            ]
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
        if self.candidate_classifier is not None:
            return self._hydrate_candidate_sources(docs, context, centers_only=True)
        docs = self._admitted_search_docs(docs, context)

        def hydrate_verified_group(
            originals: list[SearchDoc], child: RunContext
        ) -> dict[tuple[str, int], list[EvidenceItem]]:
            child.check_research_active()
            result = super(SourceLaneBroker, self).hydrate_search_centers(
                originals, child
            )
            admitted = {
                (doc.document_id, doc.chunk_ind)
                for doc in self._admitted_search_docs(originals, child)
            }
            verified = self._check_search_originals(
                {key: value for key, value in result.items() if key in admitted}
            )
            self._stage_search_originals(verified, child)
            return verified

        if self.shared_centers is not None:
            result = self.shared_centers.read(
                docs,
                context,
                hydrate_verified_group,
                {
                    str(key): value.prepared_revision
                    for key, value in self.classifications.items()
                },
            )
            admitted = {
                (doc.document_id, doc.chunk_ind)
                for doc in self._admitted_search_docs(docs, context)
            }
            verified = self._check_search_originals(
                {key: value for key, value in result.items() if key in admitted}
            )
            self._stage_search_originals(verified, context)
            return verified
        grouped: dict[str, list[SearchDoc]] = {}
        for doc in docs:
            grouped.setdefault(doc.document_id, []).append(doc)
        groups = run_functions_tuples_in_parallel(
            [
                (hydrate_verified_group, (originals, context))
                for originals in grouped.values()
            ],
            max_workers=4,
        )
        retained: dict[tuple[str, int], list[EvidenceItem]] = {}
        for result in groups:
            retained.update(result)
        return retained

    @staticmethod
    def _stage_search_originals(
        result: dict[tuple[str, int], list[EvidenceItem]], context: RunContext
    ) -> None:
        stage = context.services.get("legal_composite_original_stage")
        if isinstance(stage, CanonicalEvidenceStage):
            stage.retain([item for group in result.values() for item in group])

    def hydrate_search_results(
        self, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        if self.candidate_classifier is not None:
            return self._hydrate_candidate_sources(docs, context, centers_only=False)
        for source_id in dict.fromkeys(doc.document_id for doc in docs):
            self._require_lane_source(source_id, context)
        return self._check_search_originals(
            super().hydrate_search_results(docs, context)
        )

    def _hydrate_candidate_sources(
        self, docs: list[SearchDoc], context: RunContext, *, centers_only: bool
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        grouped: dict[str, list[SearchDoc]] = {}
        for doc in docs:
            grouped.setdefault(doc.document_id, []).append(doc)

        def hydrate(
            source_id: str, originals: list[SearchDoc]
        ) -> dict[tuple[str, int], list[EvidenceItem]]:
            context.check_active()
            self._check_lane_scope()
            try:
                self._require_lane_source(source_id, context)
                parent = super(SourceLaneBroker, self)
                result = (
                    parent.hydrate_search_centers(originals, context)
                    if centers_only
                    else parent.hydrate_search_results(originals, context)
                )
                return self._check_search_originals(result)
            except (CorpusScopeUnavailable, PermissionError):
                context.check_active()
                self.record_search(
                    {
                        "incomplete": True,
                        "unavailable_source_ids": [source_id],
                        "full_original_proof_unavailable": True,
                        "corpus_absence_verified": False,
                    }
                )
                return {}

        results = run_functions_tuples_in_parallel(
            [
                (hydrate, (source_id, originals))
                for source_id, originals in grouped.items()
            ],
            max_workers=4,
        )
        retained: dict[tuple[str, int], list[EvidenceItem]] = {}
        for result in results:
            retained.update(result)
        return retained


def build_lane_broker(
    base: CorpusBroker,
    catalogue: SourceLaneCatalogue | CandidateSourceClassifier,
    kind: SourceKind,
    shared_centers: SharedCanonicalCenters | None = None,
) -> SourceLaneBroker:
    return SourceLaneBroker(base, catalogue, kind, shared_centers)
