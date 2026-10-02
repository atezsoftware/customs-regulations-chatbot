"""Canonical corpus tools with one shared, least-privilege source broker."""

import difflib
import json
from collections.abc import Callable
from datetime import date
from itertools import islice
from threading import RLock
from typing import cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, SearchDoc
from onyx.db.asv3_corpus import (
    CorpusChunk,
    CorpusScopeUnavailable,
    CorpusSource,
    find_sources,
    read_source_chunks,
    require_source,
    resolve_source_query_index,
    source_chunk_position,
    source_diagnostic,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.models import User
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.file_store.file_store import FileStore
from onyx.llm.interfaces import LLM
from onyx.regulatory.heading_path import (
    RegulatoryProvisionReference,
    extract_regulatory_provision_references,
)

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
    ) -> None:
        self.user = user
        self.filters = filters.model_copy(deep=True)
        self.query_indexes = dict(query_indexes or {})
        self.search_adapter = search_adapter
        self.file_store = file_store
        self.vision_llm = vision_llm
        self._index_lock = RLock()

    def source(self, source_id: str, context: RunContext) -> CorpusSource:
        context.check_active()
        with get_session_with_current_tenant() as session:
            return require_source(
                session, user=self.user, filters=self.filters, source_id=UUID(source_id)
            )

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
        chunks: list[CorpusChunk] = []
        byte_count = 0
        start = 0
        more = False
        source = self.source(source_id, context)
        for _ in range(MAX_SCAN_CHUNKS // 100):
            source, page, more = self.page(
                source_id,
                context,
                start=start,
                limit=100,
                as_of=as_of,
                historical_inventory=historical_inventory,
            )
            for chunk in page:
                byte_count += len(chunk.text.encode("utf-8"))
                if byte_count > MAX_SCAN_BYTES:
                    return source, chunks, True
                chunks.append(chunk)
            if not more or not page:
                return source, chunks, more
            start = page[-1].position + 1
        return source, chunks, more

    def revalidate_evidence(
        self, items: list[EvidenceItem], context: RunContext
    ) -> None:
        """Recheck authorization and immutable text before resume/final delivery."""
        for item in items:
            context.check_active()
            source = self.source(item.source_id, context)
            if item.chunk_id is not None:
                with get_session_with_current_tenant() as session:
                    with self._index_lock:
                        if source.id not in self.query_indexes:
                            snapshot = resolve_source_query_index(session, source.id)
                            if snapshot is not None:
                                self.query_indexes[source.id] = snapshot
                    position = source_chunk_position(
                        session,
                        source_id=source.id,
                        chunk_id=item.chunk_id,
                        index=self.query_indexes.get(source.id),
                        as_of=date.fromisoformat(str(item.metadata["read_as_of_date"]))
                        if item.metadata.get("read_as_of_date")
                        else self.filters.as_of_date,
                    )
                if position is None:
                    raise CorpusScopeUnavailable(
                        "Retained evidence chunk no longer exists in its source."
                    )
                requested = item.metadata.get("read_as_of_date")
                as_of = date.fromisoformat(str(requested)) if requested else None
                _, chunks, _ = self.page(
                    item.source_id, context, start=position, limit=100, as_of=as_of
                )
                current = next(
                    (chunk for chunk in chunks if chunk.id == item.chunk_id), None
                )
                if current is None or current.text != item.text:
                    raise CorpusScopeUnavailable(
                        "Evidence changed or is outside the captured legal snapshot."
                    )
            elif item.metadata.get("source_sha256"):
                from hashlib import sha256

                from onyx.asv3.source_tools import read_verified_original

                content, _, _ = read_verified_original(self, item.source_id, context)
                if sha256(content).hexdigest() != item.metadata["source_sha256"]:
                    raise CorpusScopeUnavailable(
                        "Original source changed after extraction."
                    )
            else:
                raise CorpusScopeUnavailable("Evidence has no immutable source proof.")

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
        """Hydrate an actual retrieved center; never invent section membership."""
        source = self.source(doc.document_id, context)
        chunk_id = doc.metadata.get("regulatory_chunk_id")
        if not isinstance(chunk_id, str):
            raise CorpusScopeUnavailable(
                "Retrieved section has no canonical center identity."
            )
        with get_session_with_current_tenant() as session:
            with self._index_lock:
                if source.id not in self.query_indexes:
                    snapshot = resolve_source_query_index(session, source.id)
                    if snapshot is not None:
                        self.query_indexes[source.id] = snapshot
            position = source_chunk_position(
                session,
                source_id=source.id,
                chunk_id=chunk_id,
                index=self.query_indexes.get(source.id),
                as_of=self.filters.as_of_date,
            )
        if position is None:
            raise CorpusScopeUnavailable(
                "Retrieved center is not in the captured source snapshot."
            )
        _, chunks, _ = self.page(str(source.id), context, start=position, limit=100)
        center = next((chunk for chunk in chunks if chunk.id == chunk_id), None)
        if center is None:
            raise CorpusScopeUnavailable(
                "Retrieved center cannot be independently hydrated."
            )
        identity = article_identity(center)
        closure, incomplete = [center], False
        if identity is not None:
            _, all_chunks, incomplete = self.scan(str(source.id), context)
            closure = []
            current_identity = None
            for chunk in all_chunks:
                own = article_identity(chunk)
                if own is not None:
                    current_identity = own
                if current_identity == identity:
                    closure.append(chunk)
            if not any(chunk.id == center.id for chunk in closure):
                closure.insert(0, center)
        items, clipped = bounded_evidence(source, closure)
        if not any(item.chunk_id == center.id for item in items):
            items.insert(0, evidence_for_chunk(source, center))
        for item in items:
            item.metadata.update(
                {
                    "retrieval_method": "established_search_canonical_hydration",
                    "section_context": "not_inferred",
                    "additional_context": "canonical_article_closure"
                    if identity
                    else "center_only",
                    "article_closure_complete": not incomplete and not clipped
                    if identity
                    else False,
                    "follow_context_tool": "read_provision",
                    "retrieved_projection_ordinal": doc.chunk_ind,
                    "retrieved_center": item.chunk_id == center.id,
                }
            )
        return items


def evidence_for_chunk(source: CorpusSource, chunk: CorpusChunk) -> EvidenceItem:
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
            "canonical_metadata": json_value(chunk.metadata),
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
    references = extract_regulatory_provision_references(" ".join(chunk.heading_path))
    if references:
        return references[-1].article_no, references[-1].qualifier
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


def build_corpus_specs(broker: CorpusBroker) -> list[ToolSpec]:
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
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL if more else status,
            summary="Authorized source candidates; choose an exact source_id before reading.",
            data={
                "sources": [
                    {"source_id": str(source.id), "name": source.name}
                    for source in sources
                ],
                "has_more": more,
                "next_offset": int(cast(int, args.get("offset", 0)))
                + int(cast(int, args.get("limit", 20))),
                "absence_proven": False,
            },
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

    def provision(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        requested = article_references(str(args["article"]))
        if len(requested) != 1:
            return ToolOutcome(
                status=OutcomeStatus.INVALID,
                summary="Use one article identity including GEÇİCİ/MÜKERRER when applicable.",
            )
        target = requested[0].article_no, requested[0].qualifier
        source, chunks, more = broker.scan(str(args["source_id"]), context)
        selected = []
        identity = None
        for chunk in chunks:
            own = article_identity(chunk)
            if own is not None:
                identity = own
            if identity == target:
                selected.append(chunk)
        evidence, clipped = bounded_evidence(source, selected)
        paragraph = args.get("paragraph")
        clause = args.get("clause")
        subunit_verified = any(
            (
                paragraph is None
                or str(chunk.metadata.get("paragraph_no")) == str(paragraph)
            )
            and (
                clause is None or str(chunk.metadata.get("clause_label")) == str(clause)
            )
            for chunk in selected
        )
        partial = (
            more or clipped or bool((paragraph or clause) and not subunit_verified)
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
                "absence_proven": False,
            },
            evidence=evidence,
        )

    def text_search(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        source, chunks, more = broker.scan(str(args["source_id"]), context)
        pattern = str(args["pattern"])
        if not pattern or len(pattern) > 512:
            raise ValueError("Pattern must contain 1..512 characters.")
        mode = str(args.get("mode", "literal"))
        matches: list[JsonValue] = []
        selected = []
        compiled = None
        if mode == "regex":
            import regex

            compiled = regex.compile(pattern, flags=regex.IGNORECASE)
        for chunk in chunks:
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
                more = more or len(spans) > 20
                selected.append(chunk)
                matches.extend(
                    {
                        "chunk_id": chunk.id,
                        "start": start,
                        "end": end,
                        "text": chunk.text[start:end],
                    }
                    for start, end in spans[:20]
                )
            if len(selected) >= 30:
                more = True
                break
        evidence, clipped = bounded_evidence(source, selected)
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
                "absence_proven": False,
            },
            evidence=evidence,
        )

    def query(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        if args.get("operation", "inventory") == "inventory":
            return resolve(args, context)
        source, chunks, more = broker.scan(str(args["source_id"]), context)
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
            for chunk in chunks
        ]
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL if more else OutcomeStatus.FOUND,
            summary="Scoped structural inventory; unknown validity remains unknown.",
            data={"source_name": source.name, "headings": headings, "has_more": more},
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
        source, chunks, more = broker.scan(str(args["source_id"]), context)
        identifier = str(args["chunk_id"])
        chunk = next((item for item in chunks if item.id == identifier), None)
        if chunk is None:
            return ToolOutcome(
                status=OutcomeStatus.PARTIAL if more else OutcomeStatus.NOT_FOUND,
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
        source, old, old_more = broker.scan(
            str(args["source_id"]), context, as_of=old_date
        )
        _, new, new_more = broker.scan(str(args["source_id"]), context, as_of=new_date)
        left, left_clipped = bounded_evidence(source, old)
        right, right_clipped = bounded_evidence(source, new)
        diff = "\n".join(
            difflib.unified_diff(
                "\n".join(item.text for item in left).splitlines(),
                "\n".join(item.text for item in right).splitlines(),
                fromfile=str(old_date),
                tofile=str(new_date),
            )
        )
        unknown = any(item.validity_start is None for item in old + new)
        partial = (
            old_more
            or new_more
            or left_clipped
            or right_clipped
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
            },
            evidence=left + right,
        )

    def search(args: dict[str, JsonValue], context: RunContext) -> ToolOutcome:
        if broker.search_adapter is None:
            return ToolOutcome(
                status=OutcomeStatus.UNAVAILABLE,
                summary="Indexed corpus search is not configured; use source resolution and canonical text tools.",
            )
        return broker.search_adapter(args, context)

    return [
        ToolSpec(
            name="resolve_source",
            description="Resolve authorized canonical source titles; multiple candidates stay ambiguous.",
            parameters=schema(
                {
                    "query": {"type": "string"},
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
            name="read_provision",
            description="Read an identified article including continuation and prerequisite context. Source ID is mandatory; GEÇİCİ/MÜKERRER identities remain distinct.",
            parameters=schema(
                {
                    "source_id": SOURCE_FIELD,
                    "article": {"type": "string"},
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
                    "mode": {"enum": ["literal", "regex"]},
                },
                ["source_id", "pattern"],
            ),
            handler=guarded(text_search),
        ),
        ToolSpec(
            name="query_corpus",
            description="Read scoped inventory or source headings; never arbitrary SQL.",
            parameters=schema(
                {
                    "operation": {"enum": ["inventory", "headings"]},
                    "query": {"type": "string"},
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
                },
                ["source_id", "old_date", "new_date"],
            ),
            handler=guarded(compare),
        ),
        ToolSpec(
            name="search_corpus",
            description="Use existing scoped index search; choose hybrid/keyword/full_text according to evidence need.",
            parameters=schema(
                {
                    "query": {"type": "string"},
                    "mode": {"enum": ["hybrid", "keyword", "full_text"]},
                },
                ["query"],
            ),
            handler=guarded(search),
        ),
    ]
