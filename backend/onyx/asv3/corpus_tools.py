"""Canonical corpus tools with one shared, least-privilege source broker."""

import copy
import difflib
import json
import re
import time
from collections.abc import Callable, Generator, Iterator
from datetime import date
from hashlib import sha256
from itertools import islice
from threading import BoundedSemaphore, Event, RLock
from typing import cast
from uuid import UUID

from pydantic import JsonValue
from sqlalchemy.exc import SQLAlchemyError

from onyx.asv3.authority import known_statute_source_ids
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.focused_source_target import focused_source_target
from onyx.asv3.legal_source_navigation import (
    ProvisionNavigationAnchor,
    derive_provision_navigation_anchor,
    match_related_source_name,
)
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
    compact_evidence_metadata,
)
from onyx.asv3.parallel_execution import capability_slot, parallel_execution_enabled
from onyx.asv3.shared_reads import SharedReads
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_candidate_inventory import current_asv3_source_inventory_scope
from onyx.db.asv3_corpus import (
    CorpusChunk,
    CorpusScopeUnavailable,
    CorpusSource,
    find_related_sources,
    find_sources,
    iter_source_chunks_by_ids,
    read_search_source_closures,
    read_source_chunks,
    require_source,
    resolve_source_query_index,
    source_diagnostic,
    source_provision_position,
    source_sibling_ids,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import User
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.file_store.file_store import FileStore
from onyx.llm.interfaces import LLM
from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_regulatory_provision_references,
    parse_regulatory_article_heading,
)
from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS
from onyx.tracing.answer_graph import graph_step
from onyx.utils.threadpool_concurrency import run_functions_tuples_in_parallel
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

MAX_SCAN_CHUNKS = 2000
MAX_RESPONSE_CHARS = 64_000
MAX_SCAN_BYTES = 2 * 1024 * 1024


def json_value(value: object) -> JsonValue:
    return cast(JsonValue, json.loads(json.dumps(value, default=str)))


def schema(
    properties: dict[str, JsonValue], required: list[str]
) -> dict[str, JsonValue]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


SOURCE_FIELD: dict[str, JsonValue] = {
    "type": "string",
    "format": "uuid",
    "description": "Canonical authorized source_id returned by resolve_source.",
}


class ChunkScan:
    """Paged source traversal retaining only the current page and continuation."""

    def __init__(
        self,
        broker: "CorpusBroker",
        source_id: str,
        context: RunContext,
        *,
        start: int = 0,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> None:
        self.broker = broker
        self.source_id = source_id
        self.context = context
        self.source = broker.source(source_id, context)
        self.next_position = start
        self.truncated = True
        self.as_of = as_of
        self.historical_inventory = historical_inventory
        self.scanned_chunks = 0
        self.scanned_bytes = 0

    def __iter__(self) -> Iterator[CorpusChunk]:
        while self.scanned_chunks < MAX_SCAN_CHUNKS:
            self.source, page, more = self.broker.page(
                self.source_id,
                self.context,
                start=self.next_position,
                limit=min(100, MAX_SCAN_CHUNKS - self.scanned_chunks),
                as_of=self.as_of,
                historical_inventory=self.historical_inventory,
            )
            for chunk in page:
                self.context.check_active()
                size = len(chunk.text.encode("utf-8"))
                if self.scanned_bytes + size > MAX_SCAN_BYTES:
                    self.next_position = chunk.position
                    return
                self.scanned_bytes += size
                self.scanned_chunks += 1
                self.next_position = chunk.position + 1
                yield chunk
            if not more:
                self.truncated = False
                return
            if not page:
                return
            # Discard this page before fetching the next one.
            del page


class EvidenceSelection:
    """Retain complete selected chunks within the existing response text budget."""

    def __init__(self, source: CorpusSource) -> None:
        self.source = source
        self.items: list[EvidenceItem] = []
        self.characters = 0
        self.clipped = False
        self.next_position: int | None = None

    def add(self, chunk: CorpusChunk) -> None:
        if self.clipped or self.characters + len(chunk.text) > MAX_RESPONSE_CHARS:
            if not self.clipped:
                self.next_position = chunk.position
            self.clipped = True
            return
        self.items.append(evidence_for_chunk(self.source, chunk))
        self.characters += len(chunk.text)


class CorpusBroker:
    def __init__(
        self,
        user: User,
        filters: IndexFilters,
        *,
        query_indexes: dict[UUID, PublicationIndexSnapshot] | None = None,
        search_adapter: Callable[[dict[str, JsonValue], RunContext], ToolOutcome]
        | None = None,
        file_store: FileStore | None = None,
        vision_llm: LLM | None = None,
        allow_numbered_title_fallback: bool = False,
    ) -> None:
        self.user = user
        self.filters = filters.model_copy(deep=True)
        self.query_indexes = dict(query_indexes or {})
        self.search_adapter = search_adapter
        self.file_store = file_store
        self.vision_llm = vision_llm
        self.allow_numbered_title_fallback = allow_numbered_title_fallback
        self._index_lock = RLock()
        self._related_sources_lock = RLock()
        self._related_sources_slots = BoundedSemaphore(2)
        self._related_sources_flights: dict[
            tuple[str, ProvisionNavigationAnchor, int], Event
        ] = {}
        self._related_sources: dict[
            tuple[str, ProvisionNavigationAnchor, int], dict[str, JsonValue]
        ] = {}

    def source(self, source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        with get_session_with_current_tenant() as session:
            return require_source(
                session, user=self.user, filters=self.filters, source_id=UUID(source_id)
            )

    def shared_read_fence(
        self, source_id: str, context: RunContext
    ) -> str | ToolOutcome:
        """Authorize every subscriber before sharing its exact source acquisition."""
        context.check_active()
        try:
            with get_session_with_current_tenant() as session:
                source = require_source(
                    session,
                    user=self.user,
                    filters=self.filters,
                    source_id=UUID(source_id),
                )
                inventory = current_asv3_source_inventory_scope()
                captured = None
                if inventory is not None:
                    with inventory.lock:
                        captured = inventory.query_indexes.get(source.id)
                with self._index_lock:
                    previous = self.query_indexes.get(source.id)
                    if captured is not None:
                        if previous is not None and not previous.matches_temporal_index(
                            captured
                        ):
                            raise CorpusScopeUnavailable(
                                "Source index differs from captured authority."
                            )
                        self.query_indexes[source.id] = captured
                    if source.id not in self.query_indexes:
                        resolved = resolve_source_query_index(session, source.id)
                        if resolved is not None:
                            self.query_indexes[source.id] = resolved
                    snapshot = self.query_indexes.get(source.id)
            context.check_active()
            return sha256(
                json.dumps(
                    {
                        "run_id": context.run_id,
                        "tenant": CURRENT_TENANT_ID_CONTEXTVAR.get(),
                        "user_id": str(self.user.id),
                        "scope": context.scope,
                        "filters": self.filters.model_dump(mode="json"),
                        "source_id": str(source.id),
                        "source_name": source.name,
                        "file_id": source.file_id,
                        "query_index": snapshot.model_dump(mode="json")
                        if snapshot is not None
                        else None,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        except PermissionError:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Requested source/date is outside this run's authorized scope.",
            )
        except CorpusScopeUnavailable as error:
            return ToolOutcome(status=OutcomeStatus.UNAVAILABLE, summary=str(error))

    def sources(
        self, query: str, context: RunContext, *, offset: int = 0, limit: int = 50
    ) -> tuple[list[CorpusSource], bool]:
        context.check_active()
        with get_session_with_current_tenant() as session:
            return find_sources(
                session,
                user=self.user,
                filters=self.filters,
                query=query,
                offset=offset,
                limit=limit,
                allow_numbered_title_fallback=self.allow_numbered_title_fallback,
                allow_reversed_numbered_title_fallback=(
                    parallel_execution_enabled(context)
                ),
            )

    def related_catalog_sources(
        self,
        query_variants: tuple[str, ...],
        context: RunContext,
        *,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[list[CorpusSource], bool]:
        context.check_active()
        with get_session_with_current_tenant() as session:
            return find_related_sources(
                session,
                user=self.user,
                filters=self.filters,
                query_variants=query_variants,
                offset=offset,
                limit=limit,
            )

    def related_sources_for_provision(
        self,
        source: CorpusSource,
        evidence: list[EvidenceItem],
        target: tuple[str, str | None],
        context: RunContext,
        *,
        offset: int = 0,
    ) -> dict[str, JsonValue] | None:
        """Locate explicit title relationships without acquiring or approving their text."""
        anchor = derive_provision_navigation_anchor(
            str(source.id), evidence, target[0], target[1]
        )
        if anchor is None:
            return None
        context.check_active()
        key = (self.filters.model_dump_json(), anchor, offset)

        def lookup() -> dict[str, JsonValue]:
            query = " ".join(
                part
                for part in (
                    anchor.instrument_name,
                    anchor.qualifier,
                    anchor.article_no,
                )
                if part
            )
            query_variants = (query,)
            if anchor.instrument_number:
                query_variants += (
                    " ".join(
                        part
                        for part in (
                            anchor.instrument_number,
                            "sayılı Kanun",
                            anchor.qualifier,
                            anchor.article_no,
                        )
                        if part
                    ),
                )
            status = "available"
            try:
                sources, more = self.related_catalog_sources(
                    query_variants, context, offset=offset, limit=50
                )
            except PermissionError:
                sources, more, status = [], False, "denied"
            except (CorpusScopeUnavailable, SQLAlchemyError):
                sources, more, status = [], False, "unavailable"
            candidates: list[JsonValue] = []
            for candidate in sources:
                role = match_related_source_name(anchor, candidate.name)
                if candidate.id != source.id and role is not None:
                    candidates.append(
                        {
                            "source_id": str(candidate.id),
                            "name": candidate.name,
                            "candidate_role": role,
                        }
                    )
            return {
                "anchor_source_id": anchor.source_id,
                "instrument_name": anchor.instrument_name,
                "instrument_number": anchor.instrument_number,
                "article_no": anchor.article_no,
                "qualifier": anchor.qualifier,
                "navigation_only": True,
                "absence_proven": False,
                "status": status,
                "query": query,
                "query_variants": list(query_variants),
                "offset": offset,
                "next_offset": offset + 50 if more else None,
                "has_more": more,
                "candidates": candidates,
                "instruction": "Source titles are reading leads, not holdings or proof of legal effect. Read a material candidate's actual operative text and dates before applying it; an empty page does not establish absence.",
            }

        if parallel_execution_enabled(context):
            return self._parallel_related_sources(key, context, lookup)
        # One broker belongs to one captured user/scope. Sibling workers share its pages.
        with self._related_sources_lock:
            if key not in self._related_sources:
                self._related_sources[key] = lookup()
            return copy.deepcopy(self._related_sources[key])

    def _parallel_related_sources(
        self,
        key: tuple[str, ProvisionNavigationAnchor, int],
        context: RunContext,
        lookup: Callable[[], dict[str, JsonValue]],
    ) -> dict[str, JsonValue]:
        while True:
            context.check_active()
            with self._related_sources_lock:
                cached = self._related_sources.get(key)
                if cached is not None:
                    return copy.deepcopy(cached)
                flight = self._related_sources_flights.get(key)
                if flight is None:
                    flight = Event()
                    self._related_sources_flights[key] = flight
                    break
            while not flight.wait(0.05):
                context.check_active()

        acquired = False
        try:
            while not self._related_sources_slots.acquire(timeout=0.05):
                context.check_active()
            acquired = True
            context.check_active()
            record = lookup()
            context.check_active()
            with self._related_sources_lock:
                self._related_sources[key] = record
            return copy.deepcopy(record)
        finally:
            if acquired:
                self._related_sources_slots.release()
            with self._related_sources_lock:
                self._related_sources_flights.pop(key, None)
                flight.set()

    def related_sources_for_evidence(
        self, item: EvidenceItem, context: RunContext
    ) -> dict[str, JsonValue] | None:
        """Accept a retained genuine original; acquisition and delivery checks stay upstream."""
        anchor = derive_provision_navigation_anchor(item.source_id, [item])
        if anchor is None:
            return None
        assert item.search_doc is not None
        source = CorpusSource(
            UUID(item.source_id),
            item.search_doc.semantic_identifier,
            item.search_doc.file_id or "",
        )
        return self.related_sources_for_provision(
            source, [item], (anchor.article_no, anchor.qualifier), context
        )

    def related_source_navigation(self) -> list[dict[str, JsonValue]]:
        """Keep captured reading leads available after a native tool turn is compacted."""
        scope = self.filters.model_dump_json()
        with self._related_sources_lock:
            return copy.deepcopy(
                [
                    record
                    for (key, _, _), record in self._related_sources.items()
                    if key == scope
                    and (
                        record["candidates"]
                        or record["has_more"]
                        or record["status"] != "available"
                    )
                ]
            )

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
        context.check_active()
        identifier = UUID(source_id)
        filters = (
            self.filters.model_copy(update={"as_of_date": as_of})
            if as_of is not None
            else self.filters
        )
        if (
            self.filters.as_of_date is not None
            and as_of is not None
            and as_of != self.filters.as_of_date
        ):
            raise PermissionError(
                "The requested date differs from the captured run scope."
            )
        with get_session_with_current_tenant() as session:
            require_source(
                session, user=self.user, filters=filters, source_id=identifier
            )
            with self._index_lock:
                if identifier not in self.query_indexes:
                    snapshot = resolve_source_query_index(session, identifier)
                    if snapshot is not None:
                        self.query_indexes[identifier] = snapshot
            return read_source_chunks(
                session,
                user=self.user,
                filters=filters,
                source_id=identifier,
                start=start,
                limit=limit,
                query_indexes=self.query_indexes,
                historical_inventory=historical_inventory,
            )

    def scan(
        self,
        source_id: str,
        context: RunContext,
        *,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> tuple[CorpusSource, list[CorpusChunk], bool]:
        stream = self.iter_chunks(
            source_id, context, as_of=as_of, historical_inventory=historical_inventory
        )
        chunks = list(stream)
        return stream.source, chunks, stream.truncated

    def iter_chunks(
        self,
        source_id: str,
        context: RunContext,
        *,
        start: int = 0,
        as_of: date | None = None,
        historical_inventory: bool = False,
    ) -> ChunkScan:
        return ChunkScan(
            self,
            source_id,
            context,
            start=start,
            as_of=as_of,
            historical_inventory=historical_inventory,
        )

    def chunk(
        self, source_id: str, chunk_id: str, context: RunContext
    ) -> tuple[CorpusSource, CorpusChunk | None]:
        """Resolve an exact canonical locator before hydrating its text."""
        source = self.source(source_id, context)
        with get_session_with_current_tenant() as session:
            with self._index_lock:
                if source.id not in self.query_indexes:
                    snapshot = resolve_source_query_index(session, source.id)
                    if snapshot is not None:
                        self.query_indexes[source.id] = snapshot
            chunks = list(
                iter_source_chunks_by_ids(
                    session,
                    user=self.user,
                    filters=self.filters,
                    source_id=source.id,
                    chunk_ids=(chunk_id,),
                    index=self.query_indexes.get(source.id),
                    check_active=context.check_active,
                )
            )
        return source, next((chunk for chunk in chunks if chunk.id == chunk_id), None)

    def chunk_siblings(
        self, seed: CorpusChunk, context: RunContext
    ) -> tuple[CorpusSource, tuple[str, ...]]:
        source = self.source(str(seed.source_id), context)
        with get_session_with_current_tenant() as session:
            ids = source_sibling_ids(
                session,
                user=self.user,
                filters=self.filters,
                seed=seed,
                index=self.query_indexes.get(source.id),
            )
        return source, ids

    def sibling_page(
        self,
        source: CorpusSource,
        ids: tuple[str, ...],
        context: RunContext,
    ) -> Generator[CorpusChunk, None, None]:
        with get_session_with_current_tenant() as session:
            yield from iter_source_chunks_by_ids(
                session,
                user=self.user,
                filters=self.filters,
                source_id=source.id,
                chunk_ids=ids,
                index=self.query_indexes.get(source.id),
                check_active=context.check_active,
            )

    def provision_start(
        self, source_id: str, article: str, qualifier: str | None, context: RunContext
    ) -> int | None:
        source = self.source(source_id, context)
        with get_session_with_current_tenant() as session:
            with self._index_lock:
                if source.id not in self.query_indexes:
                    snapshot = resolve_source_query_index(session, source.id)
                    if snapshot is not None:
                        self.query_indexes[source.id] = snapshot
            return source_provision_position(
                session,
                user=self.user,
                filters=self.filters,
                source_id=source.id,
                article=article,
                qualifier=qualifier,
                index=self.query_indexes.get(source.id),
            )

    def revalidate_evidence(
        self, items: list[EvidenceItem], context: RunContext
    ) -> None:
        """Recheck authorization and immutable text before resume/final delivery."""
        canonical: dict[tuple[str, date | None], list[EvidenceItem]] = {}
        native: list[EvidenceItem] = []
        for item in items:
            context.check_active()
            if item.chunk_id is not None:
                requested = item.metadata.get("read_as_of_date")
                as_of = (
                    date.fromisoformat(str(requested))
                    if requested
                    else self.filters.as_of_date
                )
                if (
                    self.filters.as_of_date is not None
                    and as_of != self.filters.as_of_date
                ):
                    raise PermissionError(
                        "The retained evidence date differs from the captured run scope."
                    )
                canonical.setdefault((item.source_id, as_of), []).append(item)
            elif item.metadata.get("source_sha256"):
                native.append(item)
            else:
                raise CorpusScopeUnavailable("Evidence has no immutable source proof.")

        def revalidate_source(
            source_id: str, as_of: date | None, group: list[EvidenceItem]
        ) -> None:
            scoped = self.filters.model_copy(update={"as_of_date": as_of})
            retained: dict[str, list[EvidenceItem]] = {}
            for item in group:
                assert item.chunk_id is not None
                retained.setdefault(item.chunk_id, []).append(item)
            with graph_step(
                "asv3.canonical_revalidation.source",
                {
                    "source_id": source_id,
                    "as_of_date": str(as_of or "current"),
                    "retained_chunk_count": len(retained),
                },
            ) as step:
                with get_session_with_current_tenant() as session:
                    source = require_source(
                        session,
                        user=self.user,
                        filters=scoped,
                        source_id=UUID(source_id),
                    )
                    with self._index_lock:
                        if source.id not in self.query_indexes:
                            snapshot = resolve_source_query_index(session, source.id)
                            if snapshot is not None:
                                self.query_indexes[source.id] = snapshot
                    originals = iter_source_chunks_by_ids(
                        session,
                        user=self.user,
                        filters=scoped,
                        source_id=source.id,
                        chunk_ids=tuple(retained),
                        index=self.query_indexes.get(source.id),
                        check_active=context.check_active,
                    )
                    seen: set[str] = set()
                    try:
                        for current in originals:
                            if current.id in seen or current.id not in retained:
                                raise CorpusScopeUnavailable(
                                    "Retained evidence has ambiguous canonical bindings."
                                )
                            digest = sha256(current.text.encode("utf-8")).hexdigest()
                            for item in retained[current.id]:
                                if (
                                    current.text != item.text
                                    or digest != item.text_hash
                                ):
                                    raise CorpusScopeUnavailable(
                                        "Evidence changed or is outside the captured legal snapshot."
                                    )
                                if (
                                    item.search_doc is not None
                                    and current.projection_ordinal
                                    != item.search_doc.chunk_ind
                                ):
                                    raise CorpusScopeUnavailable(
                                        "Retained citation ordinal changed in its source snapshot."
                                    )
                            seen.add(current.id)
                    finally:
                        originals.close()
                    if seen != retained.keys():
                        raise CorpusScopeUnavailable(
                            "Retained evidence chunk no longer exists in its source."
                        )
                step.output_value = {"verified_chunk_count": len(seen)}

        run_functions_tuples_in_parallel(
            [
                (revalidate_source, (source_id, as_of, group))
                for (source_id, as_of), group in canonical.items()
            ],
            max_workers=4,
        )
        for item in native:
            from onyx.asv3.source_tools import read_verified_original, source_slot

            context.check_active()
            self.source(item.source_id, context)
            with source_slot(context, research=False):
                content, _, _ = read_verified_original(self, item.source_id, context)
                if sha256(content).hexdigest() != item.metadata["source_sha256"]:
                    raise CorpusScopeUnavailable(
                        "Original source changed after extraction."
                    )
                del content

    def attach_native_citation(
        self, item: EvidenceItem, number: int, message_id: int, context: RunContext
    ) -> EvidenceItem:
        """Attach a saved-run preview without inventing a canonical chunk."""
        if number < 1 or message_id < 1 or item.chunk_id is not None:
            raise ValueError("Invalid native citation identity")
        if not item.metadata.get("source_sha256") or not item.metadata.get("derived"):
            raise ValueError("Native evidence has no verified original-file proof")
        source = self.source(item.source_id, context)
        attached = item.model_copy(deep=True)
        label = context.services.get("native_citation_label")
        locator = item.metadata.get("locator")
        location = json.dumps(locator, ensure_ascii=False) if locator else ""
        title = source.name
        if isinstance(label, str) and label.strip():
            title += " · " + label.strip()[:120]
        if location:
            title += " · " + location[:240]
        attached.search_doc = SearchDoc(
            document_id=str(source.id),
            chunk_ind=-number,
            semantic_identifier=title,
            link=None,
            blurb=item.text[:240],
            source_type=DocumentSource.FILE,
            boost=0,
            hidden=False,
            metadata={
                "asv3_native_locator": json.dumps(
                    {
                        "source_sha256": item.metadata["source_sha256"],
                        "locator": item.metadata.get("locator"),
                        "text_hash": item.text_hash,
                        "derived": True,
                    },
                    ensure_ascii=False,
                ),
                "asv3_citation_preview_url": f"/api/asv3/citation/{message_id}/{number}",
            },
            match_highlights=[],
            file_id=source.file_id,
        )
        return attached

    def hydrate_search_evidence(
        self, doc: SearchDoc, context: RunContext
    ) -> list[EvidenceItem]:
        return self.hydrate_search_results([doc], context).get(
            (doc.document_id, doc.chunk_ind), []
        )

    def hydrate_search_centers(
        self, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        """Read exact search originals; further context is a research decision."""
        grouped: dict[str, list[SearchDoc]] = {}
        for doc in docs:
            if not isinstance(doc.metadata.get("regulatory_chunk_id"), str):
                raise CorpusScopeUnavailable(
                    "Retrieved section has no canonical center identity."
                )
            grouped.setdefault(doc.document_id, []).append(doc)

        def hydrate_source(
            source_id: str, centers: list[SearchDoc]
        ) -> dict[tuple[str, int], list[EvidenceItem]]:
            hydrated: dict[tuple[str, int], list[EvidenceItem]] = {}
            context.check_active()
            center_ids = tuple(
                dict.fromkeys(
                    str(doc.metadata["regulatory_chunk_id"]) for doc in centers
                )
            )
            with graph_step(
                "asv3.canonical_hydration.centers",
                {"source_id": source_id, "center_ids": center_ids},
                summary="Read exact retrieved originals without implicit expansion",
            ) as step:
                with get_session_with_current_tenant() as session:
                    source = require_source(
                        session,
                        user=self.user,
                        filters=self.filters,
                        source_id=UUID(source_id),
                    )
                    scope = current_asv3_source_inventory_scope()
                    captured_index = None
                    if scope is not None:
                        with scope.lock:
                            captured_index = scope.query_indexes.get(source.id)
                    with self._index_lock:
                        if captured_index is not None:
                            previous = self.query_indexes.get(source.id)
                            if (
                                previous is not None
                                and not previous.matches_temporal_index(captured_index)
                            ):
                                raise CorpusScopeUnavailable(
                                    "Retrieved source index differs from captured authority."
                                )
                            self.query_indexes[source.id] = captured_index
                        if source.id not in self.query_indexes:
                            snapshot = resolve_source_query_index(session, source.id)
                            if snapshot is not None:
                                self.query_indexes[source.id] = snapshot
                    originals = iter_source_chunks_by_ids(
                        session,
                        user=self.user,
                        filters=self.filters,
                        source_id=source.id,
                        chunk_ids=center_ids,
                        index=self.query_indexes.get(source.id),
                        check_active=context.check_active,
                    )
                    by_id: dict[str, CorpusChunk] = {}
                    try:
                        for chunk in originals:
                            if chunk.id in by_id:
                                raise CorpusScopeUnavailable(
                                    "Retrieved original has ambiguous canonical bindings."
                                )
                            by_id[chunk.id] = chunk
                    finally:
                        originals.close()
                context.check_active()
                for doc in centers:
                    center_id = str(doc.metadata["regulatory_chunk_id"])
                    chunk = by_id.get(center_id)
                    if chunk is None or chunk.projection_ordinal != doc.chunk_ind:
                        hydrated[(doc.document_id, doc.chunk_ind)] = []
                        continue
                    item = evidence_for_chunk(source, chunk)
                    item.metadata.update(
                        retrieval_method="established_search_exact_original",
                        article_closure_complete=False,
                        section_context="not_inferred",
                        additional_context="harness_controlled",
                        follow_context_tool="read_provision",
                        retrieved_projection_ordinal=doc.chunk_ind,
                        retrieved_center=True,
                    )
                    hydrated[(doc.document_id, doc.chunk_ind)] = [item]
                step.output_value = {
                    "requested_center_count": len(center_ids),
                    "hydrated_chunk_count": len(by_id),
                    "hydrated_characters": sum(
                        len(chunk.text) for chunk in by_id.values()
                    ),
                    "context_expanded": False,
                }
            return hydrated

        # Each independent source owns its DB session; captured tenant/read scopes travel with it.
        results = cast(
            list[dict[tuple[str, int], list[EvidenceItem]]],
            run_functions_tuples_in_parallel(
                [
                    (hydrate_source, (source_id, centers))
                    for source_id, centers in grouped.items()
                ],
                max_workers=4,
            ),
        )
        hydrated: dict[tuple[str, int], list[EvidenceItem]] = {}
        for result in results:
            hydrated.update(result)
        return hydrated

    def hydrate_search_results(
        self, docs: list[SearchDoc], context: RunContext
    ) -> dict[tuple[str, int], list[EvidenceItem]]:
        """Share one bounded structural plan per source within this search call."""
        grouped: dict[str, list[SearchDoc]] = {}
        for doc in docs:
            if not isinstance(doc.metadata.get("regulatory_chunk_id"), str):
                raise CorpusScopeUnavailable(
                    "Retrieved section has no canonical center identity."
                )
            grouped.setdefault(doc.document_id, []).append(doc)
        hydrated: dict[tuple[str, int], list[EvidenceItem]] = {}
        for source_id, centers in grouped.items():
            context.check_active()
            started = time.monotonic()
            center_ids = tuple(
                dict.fromkeys(
                    str(doc.metadata["regulatory_chunk_id"]) for doc in centers
                )
            )
            with graph_step(
                "asv3.canonical_hydration.source",
                {
                    "source_id": source_id,
                    "center_ids": center_ids,
                    "as_of_date": str(self.filters.as_of_date or "current"),
                },
                summary="Read original local provision closures",
            ) as step:
                with get_session_with_current_tenant() as session:
                    source = require_source(
                        session,
                        user=self.user,
                        filters=self.filters,
                        source_id=UUID(source_id),
                    )
                    with self._index_lock:
                        if source.id not in self.query_indexes:
                            snapshot = resolve_source_query_index(session, source.id)
                            if snapshot is not None:
                                self.query_indexes[source.id] = snapshot
                    result = read_search_source_closures(
                        session,
                        user=self.user,
                        filters=self.filters,
                        source_id=source.id,
                        center_ids=center_ids,
                        index=self.query_indexes.get(source.id),
                        check_active=context.check_active,
                        max_chars=MAX_RESPONSE_CHARS * len(center_ids),
                    )
                by_id = {chunk.id: chunk for chunk in result.chunks}
                originals = {
                    chunk.id: evidence_for_chunk(result.source, chunk)
                    for chunk in result.chunks
                }
                for doc in centers:
                    center_id = str(doc.metadata["regulatory_chunk_id"])
                    if center_id not in by_id:
                        hydrated[(doc.document_id, doc.chunk_ind)] = []
                        continue
                    items = []
                    for member in result.members[center_id]:
                        chunk = by_id.get(member)
                        if chunk is None:
                            continue
                        item = originals[member].model_copy(
                            update={
                                "metadata": {
                                    **originals[member].metadata,
                                    "retrieval_method": "established_search_canonical_hydration",
                                    "section_context": "not_inferred",
                                    "additional_context": "canonical_local_provision_closure",
                                    "article_closure_complete": result.complete[
                                        center_id
                                    ],
                                    "article_closure_continuation": list(
                                        result.continuation[center_id][:100]
                                    ),
                                    "article_closure_remaining_count": len(
                                        result.continuation[center_id]
                                    ),
                                    "source_outline_truncated": result.outline_truncated,
                                    "follow_context_tool": "read_provision",
                                    "retrieved_projection_ordinal": doc.chunk_ind,
                                    "retrieved_center": member == center_id,
                                }
                            }
                        )
                        items.append(item)
                    hydrated[(doc.document_id, doc.chunk_ind)] = items
                step.output_value = {
                    "elapsed_seconds": time.monotonic() - started,
                    "outline_rows": result.outline_rows,
                    "outline_truncated": result.outline_truncated,
                    "hydrated_chunk_count": len(result.chunks),
                    "hydrated_characters": sum(
                        len(chunk.text) for chunk in result.chunks
                    ),
                    "centers": [
                        {
                            "center_id": center,
                            "complete": result.complete[center],
                            "remaining_count": len(result.continuation[center]),
                        }
                        for center in center_ids
                    ],
                }
        return hydrated


def evidence_for_chunk(source: CorpusSource, chunk: CorpusChunk) -> EvidenceItem:
    canonical_metadata = compact_evidence_metadata(chunk.metadata)
    title = canonical_metadata.get("title")
    if not isinstance(title, str) or not title.strip():
        canonical_metadata["title"] = source.name
    metadata = {
        "regulatory_chunk_id": chunk.id,
        "regulatory_heading_path": list(chunk.heading_path),
        "regulatory_validity_start_date": chunk.validity_start.isoformat()
        if chunk.validity_start
        else "",
        "regulatory_validity_end_date": chunk.validity_end.isoformat()
        if chunk.validity_end
        else "",
    }
    links = chunk.metadata.get("source_links")
    link = (links.get("0") or links.get(0)) if isinstance(links, dict) else None
    doc = SearchDoc(
        document_id=str(source.id),
        chunk_ind=chunk.projection_ordinal,
        semantic_identifier=source.name,
        link=link if isinstance(link, str) else None,
        blurb=chunk.text[:240],
        source_type=DocumentSource.FILE,
        boost=0,
        hidden=False,
        metadata=metadata,
        match_highlights=[],
        file_id=source.file_id,
    )
    return EvidenceItem(
        source_id=str(source.id),
        chunk_id=chunk.id,
        text=chunk.text,
        search_doc=doc,
        metadata={
            "position": chunk.position,
            "heading_path": list(chunk.heading_path),
            "validity_start": chunk.validity_start.isoformat()
            if chunk.validity_start
            else None,
            "validity_end": chunk.validity_end.isoformat()
            if chunk.validity_end
            else None,
            "version_unknown": chunk.validity_start is None,
            "canonical_metadata": canonical_metadata,
            "read_as_of_date": cast(JsonValue, chunk.metadata.get("read_as_of_date")),
        },
    )


def bounded_evidence(
    source: CorpusSource, chunks: list[CorpusChunk]
) -> tuple[list[EvidenceItem], bool]:
    items = []
    size = 0
    for chunk in chunks:
        if size + len(chunk.text) > MAX_RESPONSE_CHARS:
            return items, True
        items.append(evidence_for_chunk(source, chunk))
        size += len(chunk.text)
    return items, False


def article_identity(chunk: CorpusChunk) -> tuple[str, str | None] | None:
    # Clause headings can refer to other laws; only structural headings own a chunk.
    for heading in reversed(chunk.heading_path):
        parsed = parse_regulatory_article_heading(heading)
        if parsed is not None:
            return parsed.article_no, parsed.qualifier
    article = chunk.metadata.get("article_no")
    if isinstance(article, str):
        references = extract_regulatory_provision_references("MADDE " + article)
        if references:
            return references[0].article_no, references[0].qualifier
    return None


def article_references(value: str) -> tuple[RegulatoryProvisionReference, ...]:
    text = value.strip()
    for qualifier in ("GEÇİCİ", "GECICI", "MÜKERRER", "MUKERRER"):
        if text.upper().startswith(qualifier + " "):
            text = qualifier + " MADDE " + text[len(qualifier) :].strip()
            break
    else:
        if not extract_regulatory_provision_references(text):
            text = "MADDE " + text
    return extract_regulatory_provision_references(text)


def guarded(
    handler: Callable[[dict[str, JsonValue], RunContext], ToolOutcome],
) -> Callable[[dict[str, JsonValue], RunContext], ToolOutcome]:
    def run(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        try:
            return handler(args, context)
        except PermissionError:
            return ToolOutcome(
                status=OutcomeStatus.DENIED,
                summary="Requested source/date is outside this run's authorized scope.",
            )
        except CorpusScopeUnavailable as error:
            return ToolOutcome(status=OutcomeStatus.UNAVAILABLE, summary=str(error))
        except (ValueError, TypeError) as error:
            return ToolOutcome(status=OutcomeStatus.INVALID, summary=str(error)[:240])

    return run


def build_corpus_specs(
    broker: CorpusBroker,
    *,
    require_search_targets: bool = False,
    source_identity_guidance: bool = False,
    named_provision_reads: bool = False,
) -> list[ToolSpec]:
    def resolve(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        sources, more = broker.sources(
            str(args.get("query", "")),
            context,
            offset=int(cast(int, args.get("offset", 0))),
            limit=int(cast(int, args.get("limit", 20))),
        )
        status = (
            OutcomeStatus.AMBIGUOUS
            if len(sources) > 1
            else OutcomeStatus.FOUND
            if sources
            else OutcomeStatus.NOT_FOUND
        )
        data: dict[str, JsonValue] = {
            "sources": [
                {"source_id": str(source.id), "name": source.name} for source in sources
            ],
            "has_more": more,
            "next_offset": int(cast(int, args.get("offset", 0)))
            + int(cast(int, args.get("limit", 20))),
            "absence_proven": False,
        }
        if (
            source_identity_guidance
            and parallel_execution_enabled(context)
            and status is OutcomeStatus.NOT_FOUND
            and not more
        ):
            data["lookup_diagnostic"] = {
                "code": "source_identity_no_match",
                "query_preserved": True,
                "instruction": (
                    "No source identity candidate matched these title/name terms; this does "
                    "not establish absence of source text or a legal effect. Reuse a supplied "
                    "source_id, resolve one source's own title/name, or choose content search. "
                    "Preserve the scenario date/scope and verify validity in originals."
                ),
            }
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL if more else status,
            summary="Authorized source candidates; choose an exact source_id before reading.",
            data=data,
        )

    def read_range(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source, chunks, more = broker.page(
            str(args["source_id"]),
            context,
            start=int(cast(int, args.get("start", 0))),
            limit=int(cast(int, args.get("limit", 30))),
        )
        evidence, clipped = bounded_evidence(source, chunks)
        next_position = (
            chunks[len(evidence) - 1].position + 1
            if evidence
            else int(cast(int, args.get("start", 0)))
        )
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if more or clipped
            else OutcomeStatus.FOUND
            if evidence
            else OutcomeStatus.NOT_FOUND,
            summary="Canonical ordered source range; follow next_position when partial.",
            data={
                "source_name": source.name,
                "has_more": more or clipped,
                "next_position": next_position,
            },
            evidence=evidence,
        )

    def read_chunk(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source, chunk = broker.chunk(
            str(args["source_id"]), str(args["chunk_id"]), context
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND if chunk else OutcomeStatus.NOT_FOUND,
            summary="Exact authorized canonical chunk; surrounding conditions may require context.",
            evidence=[evidence_for_chunk(source, chunk)] if chunk else [],
        )

    def chunk_context(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source, seed = broker.chunk(
            str(args["source_id"]), str(args["chunk_id"]), context
        )
        if seed is None:
            return ToolOutcome(
                status=OutcomeStatus.NOT_FOUND,
                summary="The anchor is not visible in this source snapshot.",
            )
        source, ids = broker.chunk_siblings(seed, context)
        offset = int(str(args.get("offset", 0)))
        limit = int(str(args.get("limit", 100)))
        selected_ids = ids[offset : offset + limit]
        selection = EvidenceSelection(source)
        stream = broker.sibling_page(source, selected_ids, context)
        try:
            for chunk in stream:
                selection.add(chunk)
                if selection.clipped:
                    break
        finally:
            stream.close()
        delivered_ids = {item.chunk_id for item in selection.items}
        # Advance only over delivered originals. Co-positioned chunks remain reachable.
        consumed = 0
        for chunk_id in selected_ids:
            if chunk_id not in delivered_ids:
                break
            consumed += 1
        next_offset = offset + consumed
        has_more = next_offset < len(ids)
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if has_more
            else OutcomeStatus.FOUND
            if selection.items
            else OutcomeStatus.NOT_FOUND,
            summary="All immediate-parent sibling IDs selected. Originals are delivered in pages; continue at next_offset when has_more. A sibling family is not proof of whole-article coverage.",
            data={
                "seed_chunk_id": seed.id,
                "parent_heading": list(seed.heading_path[:-1]),
                "parent_known": bool(seed.heading_path),
                "sibling_count": len(ids),
                "offset": offset,
                "next_offset": next_offset,
                "has_more": has_more,
                "evidence_truncated": selection.clipped,
                "article_closure_complete": False,
                "sibling_selection_complete": bool(seed.heading_path),
                "selected_chunk_ids": list(selected_ids),
                "delivered_chunk_ids": [item.chunk_id for item in selection.items],
                "oversized_chunk_id": selected_ids[consumed]
                if selection.clipped and consumed < len(selected_ids)
                else None,
                "oversized_chunk_instruction": "Read the exact chunk using read_chunk, then continue at the next sibling offset."
                if selection.clipped and consumed == 0
                else None,
            },
            evidence=selection.items,
        )

    def provision(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        requested = article_references(str(args["article"]))
        if len(requested) != 1:
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Use one article identity including GEÇİCİ/MÜKERRER when applicable.",
            )
        target = requested[0].article_no, requested[0].qualifier
        start = int(cast(int, args.get("start", 0)))
        if "start" not in args:
            lookup = getattr(broker, "provision_start", None)
            anchor = (
                lookup(str(args["source_id"]), target[0], target[1], context)
                if callable(lookup)
                else None
            )
            if anchor is not None:
                start = anchor
        stream = broker.iter_chunks(str(args["source_id"]), context, start=start)
        selection = EvidenceSelection(stream.source)
        identity = None
        paragraph = args.get("paragraph")
        clause = args.get("clause")
        subunit_verified = False
        reached = False
        for chunk in stream:
            own = article_identity(chunk)
            if reached and own is not None and own != target:
                stream.truncated = False
                stream.next_position = chunk.position
                break
            if own is not None:
                identity = own
            if identity == target:
                reached = True
                selection.add(chunk)
                inherited_paragraph = chunk.metadata.get("paragraph_no")
                if inherited_paragraph is None:
                    for heading in reversed(chunk.heading_path):
                        marker = re.match(r"^\s*(?:\((\d+)\)|(\d+)\.)\s", heading)
                        if marker:
                            inherited_paragraph = marker[1] or marker[2]
                            break
                subunit_verified = subunit_verified or (
                    (paragraph is None or str(inherited_paragraph) == str(paragraph))
                    and (
                        clause is None
                        or str(chunk.metadata.get("clause_label")) == str(clause)
                    )
                )
        evidence, clipped = selection.items, selection.clipped
        more = stream.truncated
        partial = (
            more or clipped or bool((paragraph or clause) and not subunit_verified)
        )
        navigation = getattr(broker, "related_sources_for_provision", None)
        related = (
            navigation(stream.source, evidence, target, context)
            if callable(navigation)
            else None
        )
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if partial
            else OutcomeStatus.FOUND
            if evidence
            else OutcomeStatus.NOT_FOUND,
            summary="Article context and continuation; requested subunit verification is explicit.",
            data={
                "article": str(args["article"]),
                "subunit_verified": subunit_verified,
                "scan_truncated": more,
                "evidence_truncated": clipped,
                "next_position": stream.next_position,
                "evidence_next_position": selection.next_position,
                "absence_proven": False,
                **(
                    {"related_source_candidates": related}
                    if related is not None
                    else {}
                ),
            },
            evidence=evidence,
        )

    def named_provision(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        ledger = context.services.get("evidence")
        known = (
            known_statute_source_ids(str(args["source_name"]), ledger)
            if isinstance(ledger, EvidenceLedger)
            else []
        )
        with capability_slot("resolve_source", context):
            if len(known) == 1:
                sources, more = [broker.source(known[0], context)], False
            else:
                sources, more = broker.sources(
                    str(args["source_name"]), context, limit=20
                )
        if len(sources) != 1 or more:
            return ToolOutcome(
                status=OutcomeStatus.AMBIGUOUS
                if sources or more
                else OutcomeStatus.NOT_FOUND,
                summary="Source identity remains unresolved; choose an exact source_id or refine the source title. This is not evidence of absent law.",
                data={
                    "sources": [
                        {"source_id": str(source.id), "name": source.name}
                        for source in sources
                    ],
                    "has_more": more,
                    "absence_proven": False,
                },
            )
        source = sources[0]
        canonical_args = {
            key: value
            for key, value in args.items()
            if key in {"article", "start", "paragraph", "clause"}
        }
        canonical_args["source_id"] = str(source.id)

        def acquire(producer: RunContext) -> ToolOutcome:
            with capability_slot("read_provision", producer):
                return provision(canonical_args, producer)

        shared = context.services.get("shared_reads")
        result = (
            shared.run(
                "read_provision",
                canonical_args,
                context,
                acquire,
            )
            if isinstance(shared, SharedReads)
            else acquire(context)
        )
        return result.model_copy(
            update={
                "data": {
                    **result.data,
                    "source_id": str(source.id),
                    "source_name": source.name,
                }
            }
        )

    def text_search(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        pattern = str(args["pattern"])
        if not pattern or len(pattern) > 512:
            raise ValueError("Pattern must contain 1..512 characters.")
        mode = str(args.get("mode", "literal"))
        matches: list[JsonValue] = []
        stream = broker.iter_chunks(
            str(args["source_id"]), context, start=int(cast(int, args.get("start", 0)))
        )
        selection = EvidenceSelection(stream.source)
        match_chunks = 0
        match_overflow = False
        compiled = None
        if mode == "regex":
            import regex

            compiled = regex.compile(pattern, flags=regex.IGNORECASE)
        for chunk in stream:
            context.check_active()
            if mode == "regex" and compiled is not None:
                try:
                    spans = [
                        (match.start(), match.end())
                        for match in islice(
                            compiled.finditer(chunk.text, timeout=0.02), 21
                        )
                    ]
                except TimeoutError:
                    return ToolOutcome(
                        status=OutcomeStatus.UNAVAILABLE,
                        summary="Regex execution exceeded its per-chunk deadline; use a simpler pattern.",
                    )
            else:
                # Unicode case-insensitive regex preserves original offsets unlike casefold indices.
                import re

                spans = [
                    (match.start(), match.end())
                    for match in islice(
                        re.finditer(
                            re.escape(pattern), chunk.text, flags=re.IGNORECASE
                        ),
                        21,
                    )
                ]
            if spans:
                match_overflow = match_overflow or len(spans) > 20
                match_chunks += 1
                selection.add(chunk)
                matches.extend(
                    {
                        "chunk_id": chunk.id,
                        "start": start,
                        "end": end,
                        "text": chunk.text[start:end],
                    }
                    for start, end in spans[:20]
                )
            if match_chunks >= 30:
                match_overflow = True
                break
        evidence, clipped = selection.items, selection.clipped
        more = stream.truncated or match_overflow or len(matches) > 200
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if more or clipped
            else OutcomeStatus.FOUND
            if evidence
            else OutcomeStatus.NOT_FOUND,
            summary="Original-text matches and canonical surrounding chunks.",
            data={
                "matches": matches[:200],
                "scan_truncated": more,
                "next_position": stream.next_position,
                "evidence_truncated": clipped,
                "evidence_next_position": selection.next_position,
                "absence_proven": False,
            },
            evidence=evidence,
        )

    def query(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        if args.get("operation", "inventory") == "inventory":
            return resolve(args, context)
        stream = broker.iter_chunks(
            str(args["source_id"]), context, start=int(cast(int, args.get("start", 0)))
        )
        headings = [
            {
                "position": chunk.position,
                "chunk_id": chunk.id,
                "heading_path": list(chunk.heading_path),
                "article": list(identity)
                if (identity := article_identity(chunk))
                else None,
                "validity_start": chunk.validity_start.isoformat()
                if chunk.validity_start
                else None,
                "validity_end": chunk.validity_end.isoformat()
                if chunk.validity_end
                else None,
            }
            for chunk in stream
        ]
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL if stream.truncated else OutcomeStatus.FOUND,
            summary="Scoped structural inventory; unknown validity remains unknown.",
            data={
                "source_name": stream.source.name,
                "headings": headings,
                "has_more": stream.truncated,
                "next_position": stream.next_position,
            },
        )

    def diagnose(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        context.check_active()
        with get_session_with_current_tenant() as session:
            result = source_diagnostic(
                session,
                user=broker.user,
                filters=broker.filters,
                source_id=UUID(str(args["source_id"])),
            )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Observed canonical metadata; index presence was not probed and absence is not proven.",
            data=cast(dict[str, JsonValue], json_value(result)),
        )

    def reference(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        identifier = str(args["chunk_id"])
        source, chunk = broker.chunk(str(args["source_id"]), identifier, context)
        if chunk is None:
            return ToolOutcome(
                status=OutcomeStatus.NOT_FOUND,
                summary="Reference origin is not in the visible source snapshot.",
            )
        refs = extract_regulatory_provision_references(chunk.text)
        if not args.get("target_source_id") or not args.get("article"):
            return ToolOutcome(
                status=OutcomeStatus.AMBIGUOUS,
                summary="Choose the referenced instrument and article; references do not imply the same source.",
                data={
                    "references": [
                        {"article": ref.article_no, "qualifier": ref.qualifier}
                        for ref in refs
                    ]
                },
                evidence=[evidence_for_chunk(source, chunk)],
            )
        requested = article_references(str(args["article"]))
        if not requested or requested[0] not in refs:
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="The chosen article is not an explicit reference in the origin text.",
            )
        if args.get("depth", 0) and int(cast(int, args["depth"])) > 4:
            return ToolOutcome(
                status=OutcomeStatus.INVALID, summary="Reference depth limit exceeded."
            )
        result = provision(
            {"source_id": args["target_source_id"], "article": args["article"]}, context
        )
        result.data["origin_chunk_id"] = identifier
        return result

    def compare(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        old_date, new_date = (
            date.fromisoformat(str(args["old_date"])),
            date.fromisoformat(str(args["new_date"])),
        )
        requested = (
            article_references(str(args["article"])) if args.get("article") else ()
        )
        if args.get("article") and len(requested) != 1:
            raise ValueError("Compare one exact article identity.")
        target = (
            (requested[0].article_no, requested[0].qualifier) if requested else None
        )
        start = int(cast(int, args.get("start", 0)))
        end = cast(int | None, args.get("end"))
        if end is not None and end <= start:
            raise ValueError("The exclusive range end must follow start.")
        if target is not None and (start or end is not None):
            raise ValueError("Choose either an article or a position range.")

        def collect(as_of: date) -> tuple[EvidenceSelection, ChunkScan, bool]:
            stream = broker.iter_chunks(
                str(args["source_id"]), context, start=start, as_of=as_of
            )
            selection = EvidenceSelection(stream.source)
            identity = None
            unknown = False
            for chunk in stream:
                if end is not None and chunk.position >= end:
                    stream.truncated = False
                    stream.next_position = end
                    break
                own = article_identity(chunk)
                if own is not None:
                    identity = own
                if target is None or identity == target:
                    selection.add(chunk)
                    unknown = unknown or chunk.validity_start is None
            return selection, stream, unknown

        old, old_scan, old_unknown = collect(old_date)
        new, new_scan, new_unknown = collect(new_date)
        left, right = old.items, new.items
        diff = "\n".join(
            difflib.unified_diff(
                "\n".join(item.text for item in left).splitlines(),
                "\n".join(item.text for item in right).splitlines(),
                fromfile=str(old_date),
                tofile=str(new_date),
            )
        )
        unknown = old_unknown or new_unknown
        partial = (
            old_scan.truncated
            or new_scan.truncated
            or old.clipped
            or new.clipped
            or len(diff) > MAX_RESPONSE_CHARS
        )
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if partial
            else OutcomeStatus.VERSION_UNKNOWN
            if unknown
            else OutcomeStatus.FOUND,
            summary="Dated source comparison; unbounded/unknown source dates do not prove a historical version.",
            data={
                "diff": diff[:MAX_RESPONSE_CHARS],
                "old_date": str(old_date),
                "new_date": str(new_date),
                "version_unknown": unknown,
                "old_next_position": old.next_position
                if old.clipped
                else old_scan.next_position,
                "new_next_position": new.next_position
                if new.clipped
                else new_scan.next_position,
                "article": args.get("article"),
                "start": start,
                "end": end,
            },
            evidence=left + right,
        )

    def search(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        if named_provision_reads:
            target = focused_source_target(args)
            if target is not None:
                source_name, article = target
                outcome = named_provision(
                    {"source_name": source_name, "article": article}, context
                )
                return outcome.model_copy(
                    update={
                        "summary": outcome.summary
                        + " No corpus-wide search was performed; related authorities and unresolved interactions require their own focused investigation.",
                        "data": {
                            **outcome.data,
                            "acquisition_method": "canonical_provision_read"
                            if outcome.evidence
                            else "canonical_source_resolution",
                            "search_performed": False,
                            "coverage_item": args.get("coverage_item"),
                            "evidence_target": args.get("evidence_target"),
                        },
                    }
                )
        if broker.search_adapter is None:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Indexed corpus search is not configured; use source resolution and canonical text tools.",
            )
        if named_provision_reads:
            with capability_slot("search_corpus", context):
                return broker.search_adapter(args, context)
        return broker.search_adapter(args, context)

    specs = [
        ToolSpec(
            name="resolve_source",
            description=(
                "Resolve one authorized source by its own title/name or identifying number; "
                "this does not search provision text or establish current validity. Reuse a "
                "supplied source_id. Multiple candidates stay ambiguous; use read_provision "
                "or content search for provisions and legal effects."
                if source_identity_guidance
                else "Resolve authorized canonical source titles; multiple candidates stay ambiguous."
            ),
            parameters=schema(
                {
                    "query": (
                        {
                            "type": "string",
                            "description": (
                                "Terms identifying one source's own title/name and number when "
                                "known. Provision/content and version-validity questions belong "
                                "to their research tools; retain scenario date/scope."
                            ),
                        }
                        if source_identity_guidance
                        else {"type": "string"}
                    ),
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["query"],
            ),
            handler=guarded(resolve),
        ),
        ToolSpec(
            name="read_source_range",
            description="Read bounded ordered canonical chunks of an authorized source; continue at next_position.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "start": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["source_id"],
            ),
            handler=guarded(read_range),
        ),
        ToolSpec(
            name="read_chunk",
            description="Read one exact canonical chunk using an already discovered source/chunk anchor.",
            parameters=schema(
                {"source_id": SOURCE_FIELD, "chunk_id": {"type": "string"}},
                ["source_id", "chunk_id"],
            ),
            handler=guarded(read_chunk),
        ),
        ToolSpec(
            name="read_chunk_context",
            description="Select every chunk with the exact same immediate heading parent as the anchor. Read their originals in pages, following next_offset while has_more. No before/after window or sibling-count cutoff. Wider parent or referenced-source reading remains your research decision.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "chunk_id": {"type": "string"},
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                ["source_id", "chunk_id"],
            ),
            handler=guarded(chunk_context),
        ),
        ToolSpec(
            name="read_provision",
            description="Read an identified article including its clauses and prerequisite context, stopping at its structural boundary. Source ID is mandatory; GEÇİCİ/MÜKERRER identities remain distinct. For an already located article, start at its known heading/first position; use evidence_next_position to continue a clipped article.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "article": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "paragraph": {"type": "string"},
                    "clause": {"type": "string"},
                },
                ["source_id", "article"],
            ),
            handler=guarded(provision),
        ),
        ToolSpec(
            name="search_source_text",
            description="Literal or timeout-bounded regex search in an authorized dated source, with original offsets.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "pattern": {"type": "string", "minLength": 1, "maxLength": 512},
                    "mode": {"type": "string", "enum": ["literal", "regex"]},
                    "start": {"type": "integer", "minimum": 0},
                },
                ["source_id", "pattern"],
            ),
            handler=guarded(text_search),
        ),
        ToolSpec(
            name="query_corpus",
            description=(
                "Read scoped source names or headings; never arbitrary SQL. inventory.query "
                "uses resolve_source's same source-name lookup, not semantic content search. "
                "Omit query for scoped inventory; headings uses source_id. Results are navigation."
                if source_identity_guidance
                else "Read scoped inventory or source headings; never arbitrary SQL."
            ),
            parameters=schema(
                {
                    "operation": {"type": "string", "enum": ["inventory", "headings"]},
                    "start": {"type": "integer", "minimum": 0},
                    "query": (
                        {
                            "type": "string",
                            "description": (
                                "Optional source-title/name terms for inventory, using the same "
                                "identity matching as resolve_source. An omitted or empty query "
                                "pages scoped source names. Use content search for operative text."
                            ),
                        }
                        if source_identity_guidance
                        else {"type": "string"}
                    ),
                    "source_id": SOURCE_FIELD,
                    "offset": {"type": "integer", "minimum": 0},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                [],
            ),
            handler=guarded(query),
        ),
        ToolSpec(
            name="follow_reference",
            description="Resolve an explicit reference from a visible origin chunk; choose target source/article before reading.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "chunk_id": {"type": "string"},
                    "target_source_id": SOURCE_FIELD,
                    "article": {"type": "string"},
                    "depth": {"type": "integer", "minimum": 0, "maximum": 4},
                },
                ["source_id", "chunk_id"],
            ),
            handler=guarded(reference),
        ),
        ToolSpec(
            name="diagnose_source",
            description="Inspect observed DB/source/publication metadata without equating tool failure with corpus absence.",
            parameters=schema({"source_id": SOURCE_FIELD}, ["source_id"]),
            handler=guarded(diagnose),
        ),
        ToolSpec(
            name="compare_versions",
            description="Compare two explicitly dated legal source snapshots; unknown boundaries stay unknown.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "old_date": {"type": "string", "format": "date"},
                    "new_date": {"type": "string", "format": "date"},
                    "article": {"type": "string"},
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 1},
                },
                ["source_id", "old_date", "new_date"],
            ),
            handler=guarded(compare),
        ),
        ToolSpec(
            name="search_corpus",
            description="Search the authorized PC corpus with the established retrieval, label and citation pipeline. Choose hybrid for semantic plus lexical retrieval, keyword for BM25 lexical ranking, or full_text for high analyzed-term coverage (not a literal substring guarantee). Source anchors and evidence targets are model-written navigation hints, not evidence."
            + (
                " An explicit single-source/article evidence target uses canonical provision reading; discover_related_sources keeps corpus discovery open for related authorities."
                if named_provision_reads
                else ""
            ),
            parameters=schema(
                {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": REGULATORY_MAX_SEARCH_QUERY_CHARS,
                        "description": "Natural-language terms or a focused phrase. This pipeline does not interpret Boolean or quoted exact-match syntax. Retain known source/instrument qualifiers in each independent query; a bare article number searches all sources. For a known source_id and article use read_provision. Use independent focused queries for distinct alternatives.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["hybrid", "keyword", "full_text"],
                    },
                    "coverage_item": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The unresolved scenario issue this focused query will answer.",
                    },
                    "evidence_target": {
                        "type": "string",
                        "minLength": 1,
                        "description": "The original rule, condition or procedure sought; an investigation target, not an assumed answer.",
                    },
                    "expand_query": {
                        "type": "boolean",
                        "description": "True when automatic semantic/lexical variants are useful. Otherwise execute your selected query directly; independent explicit searches may run together.",
                    },
                    "source_anchors": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    **(
                        {
                            "discover_related_sources": {
                                "type": "boolean",
                                "description": "True when seeking other instruments, amendments or decisions concerning a known provision, rather than that provision's own original text.",
                            }
                        }
                        if named_provision_reads
                        else {}
                    ),
                },
                ["query", "mode", "coverage_item", "evidence_target"]
                if require_search_targets
                else ["query", "mode"],
            ),
            handler=guarded(search),
            orchestrates=named_provision_reads,
        ),
    ]
    if named_provision_reads:
        specs.append(
            ToolSpec(
                name="read_named_provision",
                description="Resolve one source by its own title/name and read its identified article in the same action. Use for a known instrument and article when source_id is not yet supplied, instead of a corpus-wide article-number search. Ambiguous source identities remain unresolved. Original clauses, continuation and related-source navigation are preserved.",
                parameters=schema(
                    {
                        "source_name": {"type": "string", "minLength": 1},
                        "article": {"type": "string", "minLength": 1},
                        "start": {"type": "integer", "minimum": 0},
                        "paragraph": {"type": "string"},
                        "clause": {"type": "string"},
                    },
                    ["source_name", "article"],
                ),
                handler=guarded(named_provision),
                orchestrates=True,
            )
        )
    return specs
