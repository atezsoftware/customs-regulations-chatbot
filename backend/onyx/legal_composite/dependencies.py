"""Acquire source-triggered authority relationships without another model planner."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from hashlib import sha256
from typing import Literal, cast

from pydantic import JsonValue

from onyx.asv3.authority import _named_native_references, explicit_reference_leads
from onyx.asv3.corpus_tools import CorpusBroker, guarded
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.legal_source_navigation import (
    ProvisionNavigationAnchor,
    derive_provision_navigation_anchor,
    match_related_source_name,
)
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    RunStopped,
    ToolOutcome,
    ToolSpec,
    model_evidence_metadata,
)
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.legal_composite_sources import SourceKind, classify_source
from onyx.legal_composite.acquisition import CanonicalAcquirer, InvalidSourceAction
from onyx.legal_composite.models import (
    AnswerReview,
    AuthorityDependency,
    DependencyOrigin,
    DraftAnswer,
    MaterialDependencyRequest,
    ResearchPlan,
    SourceAction,
    SourceRequirement,
)
from onyx.regulatory.heading_path import (
    RegulatoryArticleHeading,
    parse_regulatory_article_heading,
)
from onyx.tracing.answer_graph import graph_step


def _normalized_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return re.sub(r"\bkanun(?:u|un|unun)?$", "kanun", " ".join(folded.split()))


def _generic_normalized_name(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    folded = re.sub(r"\bkanun(?:u|un|unun)?$", "kanun", " ".join(folded.split()))
    for pattern, replacement in (
        (r"\byonetmeli(?:k|gi)$", "yonetmelik"),
        (r"\bteblig(?:i)?$", "teblig"),
        (r"\bgenelge(?:si)?$", "genelge"),
        (r"\bkararname(?:si)?$", "kararname"),
        (r"\bkarar(?:i)?$", "karar"),
        (r"\b(?:antlasma|anlasma)(?:si)?$", "antlasma"),
        (r"\bsozlesme(?:si)?$", "sozlesme"),
        (r"\banayasa(?:si)?$", "anayasa"),
    ):
        folded = re.sub(pattern, replacement, folded)
    return folded


def _body_alias(value: str) -> str:
    folded = unicodedata.normalize("NFKD", value.casefold().replace("ı", "i"))
    return " ".join(
        "".join(char for char in folded if not unicodedata.combining(char)).split()
    )


_GENERIC_FORMAL_REFERENCE = re.compile(
    r"(?<!\w)(?:(?P<number>\d{1,7}(?:/\d{1,7})?)\s+say[ıi]l[ıi]\s+)?"
    r"(?P<name>(?:(?:[A-ZÇĞİÖŞÜ][\w’'-]*|ve|ile|hakkında|ilişkin|dair)\s+){1,12}"
    r"(?i:Yönetmeliği|Yönetmelik|Tebliği|Tebliğ|Genelgesi|Genelge|Kararnamesi|Kararname|"
    r"Kararı|Karar|Antlaşması|Antlaşma|Anlaşması|Anlaşma|Sözleşmesi|Sözleşme|Anayasası|Anayasa))"
    r"(?i:(?:['’]?(?:inin|ının|unun|ünün|nin|nın|nun|nün|in|ın|un|ün))?)(?!\w)"
)


def _generic_titles(text: str) -> list[tuple[int, int, str, str | None]]:
    titles: list[tuple[int, int, str, str | None]] = []
    for match in _GENERIC_FORMAL_REFERENCE.finditer(text):
        name = match["name"]
        if _generic_normalized_name(name.split()[0]) in {
            "bu",
            "isbu",
            "ilgili",
            "anilan",
            "sozkonusu",
            "the",
            "this",
            "that",
        }:
            continue
        titles.append((match.start(), match.end(), name, match["number"]))
    return titles


def _generic_kind(name: str) -> SourceKind | None:
    return {
        "yonetmelik": SourceKind.REGULATION,
        "teblig": SourceKind.COMMUNIQUE,
        "genelge": SourceKind.CIRCULAR,
        "kararname": SourceKind.PRESIDENTIAL_DECREE,
        "karar": SourceKind.EXECUTIVE_DECISION,
        "antlasma": SourceKind.TREATY,
        "sozlesme": SourceKind.TREATY,
        "anayasa": SourceKind.CONSTITUTION,
    }.get(_generic_normalized_name(name).rsplit(" ", 1)[-1])


def _generic_reference_leads(
    item: EvidenceItem, citation: int
) -> list[dict[str, JsonValue]]:
    leads: list[dict[str, JsonValue]] = []
    titles = _generic_titles(item.text)
    for index, (start, end, name, number) in enumerate(titles):
        next_start = titles[index + 1][0] if index + 1 < len(titles) else len(item.text)
        # A title, its attached number and its syntactically connected locator stay local.
        local = item.text[start : min(next_start, end + 300)]
        alias = _body_alias(
            re.sub(r"^\d{1,7}(?:/\d{1,7})?\s+say[ıi]l[ıi]\s+", "", item.text[start:end])
        )
        for reference, found in _named_native_references(
            local,
            {alias: {number} if number else set()},
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
            unicode_ordinals=True,
        ):
            if found != alias or reference.article is None:
                continue
            leads.append(
                {
                    "formal_name": name,
                    "instrument_number": number,
                    "article": reference.article,
                    "qualifier": reference.qualifier,
                    "origin_citations": [citation],
                }
            )
    return leads


def _literal_names(text: str) -> dict[str, str]:
    # Only literal formal names supply aliases; inherited labels and filenames cannot.
    pattern = r"(?<!\w)((?:[A-ZÇĞİÖŞÜ][\w’'-]*\s+){1,12}(?i:Kanunu|Kanun|Law|Act|Statute))(?i:(?:['’]?(?:nun|nın|nin|nün|un|ın|in|ün))?)(?!\w)"
    aliases: dict[str, str] = {}
    for match in re.finditer(pattern, text):
        terms = match[1].split()
        while terms and _normalized_name(terms[0]) in {
            "bu",
            "isbu",
            "ilgili",
            "anilan",
            "sozkonusu",
            "the",
            "this",
            "that",
            "said",
        }:
            terms.pop(0)
        if len(terms) >= 2:
            name = " ".join(terms)
            aliases[_normalized_name(name)] = name
    return aliases


def _entity_key(
    name: str, number: str | None, article: str, qualifier: str | None
) -> str:
    identity = [number or _normalized_name(name), article, qualifier]
    return sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


class DependencyExpander:
    def __init__(
        self,
        *,
        broker: CorpusBroker,
        acquirer: CanonicalAcquirer,
        ledger: EvidenceLedger,
        context: RunContext,
        source_kinds: dict[str, SourceKind],
    ) -> None:
        self.broker = broker
        self.acquirer = acquirer
        self.ledger = ledger
        self.context = context
        self.source_kinds = source_kinds
        self.verified_kinds = dict(source_kinds)
        self.edges: dict[str, AuthorityDependency] = {}
        self.receipts: list[dict[str, JsonValue]] = []
        self._candidate_edges: dict[str, set[str]] = {}
        self._expanded: dict[str, str] = {}
        acquirer.host_registry.register(
            ToolSpec(
                name="dependency_related_sources",
                description="Host-only scoped navigation from a read governing original.",
                parameters={
                    "type": "object",
                    "properties": {
                        "edge_id": {"type": "string"},
                        "offset": {"type": "integer", "minimum": 0},
                    },
                    "required": ["edge_id"],
                    "additionalProperties": False,
                },
                handler=guarded(self._related),
            )
        )

    def _navigation_item(self, item: EvidenceItem) -> EvidenceItem:
        # Verified opening identity can correct a label in a navigation-only copy.
        if (
            self.verified_kinds.get(
                item.source_id, self.source_kinds.get(item.source_id)
            )
            != SourceKind.STATUTE
        ):
            return item
        copied = item.model_copy(deep=True)
        canonical = copied.metadata.get("canonical_metadata")
        copied.metadata["document_type"] = "statute"
        if isinstance(canonical, dict):
            canonical["document_type"] = "statute"
        return copied

    def _anchor(self, item: EvidenceItem) -> ProvisionNavigationAnchor | None:
        if (
            self.verified_kinds.get(
                item.source_id, self.source_kinds.get(item.source_id)
            )
            != SourceKind.STATUTE
        ):
            return None
        return derive_provision_navigation_anchor(
            item.source_id, [self._navigation_item(item)]
        )

    def _collect(
        self,
        plan: ResearchPlan,
        frontier: set[int] | None = None,
        need_bindings: dict[int, set[str]] | None = None,
    ) -> None:
        selected_numbers = (
            set(self.ledger.citation_numbers()) if frontier is None else frontier
        )
        records = cast(
            list[dict[str, JsonValue]],
            json.loads(self.ledger.serialize_records(selected_numbers, max_chars=None)),
        )
        leads = explicit_reference_leads(
            self.ledger, records, syntactic_reference_binding=True
        )
        for citation in selected_numbers:
            item = self.ledger.get(citation)
            if (
                item is None
                or item.search_doc is None
                or item.search_doc.document_id != item.source_id
                or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
                or any(
                    model_evidence_metadata(item.metadata).get(flag)
                    for flag in ("derived", "external", "untrusted", "truncated")
                )
            ):
                continue
            aliases = _literal_names(item.text)
            for reference, name in _named_native_references(
                item.text,
                {alias: set() for alias in aliases},
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
                unicode_ordinals=True,
            ):
                if reference.article is not None and name in aliases:
                    leads.append(
                        {
                            "formal_name": aliases[name],
                            "instrument_number": reference.number or None,
                            "article": reference.article,
                            "qualifier": reference.qualifier,
                            "origin_citations": [citation],
                        }
                    )
        for number in self.ledger.citation_numbers():
            if number not in selected_numbers:
                continue
            item = self.ledger.get(number)
            assert item is not None
            anchor = self._anchor(item)
            if anchor is not None:
                leads.append(
                    {
                        "formal_name": anchor.instrument_name,
                        "instrument_number": anchor.instrument_number,
                        "article": anchor.article_no,
                        "qualifier": anchor.qualifier,
                        "origin_citations": [number],
                    }
                )
        known_needs = {need.need_id for need in plan.needs}
        for lead in leads:
            name, article = lead.get("formal_name"), lead.get("article")
            if not isinstance(name, str) or not isinstance(article, str):
                continue
            number = lead.get("instrument_number")
            qualifier = lead.get("qualifier")
            number = number if isinstance(number, str) else None
            qualifier = qualifier if isinstance(qualifier, str) else None
            key = _entity_key(name, number, article, qualifier)
            edge = self._matching_edge(name, number, article, qualifier)
            if edge is None:
                edge = AuthorityDependency(
                    edge_id=key,
                    need_ids=[],
                    instrument_name=name,
                    instrument_number=number,
                    article=article,
                    qualifier=qualifier,
                    origins=[],
                )
                self.edges[key] = edge
            elif edge.instrument_number is None and number:
                edge.instrument_number = number
            origins = lead.get("origin_citations", [])
            if not isinstance(origins, list):
                continue
            for citation in origins:
                if type(citation) is not int:
                    continue
                item = self.ledger.get(citation)
                if item is None:
                    continue
                if citation not in {origin.citation for origin in edge.origins}:
                    edge.origins.append(
                        DependencyOrigin(
                            citation=citation,
                            source_id=item.source_id,
                            chunk_id=item.chunk_id,
                            text_hash=item.text_hash,
                        )
                    )
                bindings = (
                    need_bindings.get(citation, set(item.question_ids))
                    if need_bindings is not None
                    else set(item.question_ids)
                ) & known_needs
                edge.need_ids = sorted(set(edge.need_ids) | (bindings or known_needs))
        self._bind_governing()

    def _bind_governing(self) -> None:
        for number in self.ledger.citation_numbers():
            item = self.ledger.get(number)
            assert item is not None
            anchor = self._anchor(item)
            if anchor is None:
                continue
            edge = self._matching_edge(
                anchor.instrument_name,
                anchor.instrument_number,
                anchor.article_no,
                anchor.qualifier,
            )
            if edge is not None and number not in edge.governing_citations:
                if edge.instrument_number is None:
                    edge.instrument_number = anchor.instrument_number
                edge.governing_citations.append(number)
                self.acquirer._bind_originals([number], edge.need_ids)

    def _matching_edge(
        self, name: str, number: str | None, article: str, qualifier: str | None
    ) -> AuthorityDependency | None:
        matches = [
            edge
            for edge in self.edges.values()
            if edge.article == article
            and edge.qualifier == qualifier
            and not (
                edge.instrument_number and number and edge.instrument_number != number
            )
            and (
                (edge.instrument_number and number and edge.instrument_number == number)
                or _normalized_name(edge.instrument_name) == _normalized_name(name)
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def _related(
        self, arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        edge = self.edges[str(arguments["edge_id"])]
        citation = (
            edge.governing_citations[0]
            if edge.governing_citations
            else edge.origins[0].citation
        )
        item = self.ledger.get(citation)
        assert item is not None
        # The relation is a navigation lead from an already read original. Live
        # source access is checked without rereading its complete canonical text.
        self.broker.source(item.source_id, context)
        anchor = ProvisionNavigationAnchor(
            item.source_id,
            edge.instrument_name,
            edge.instrument_number,
            edge.article,
            edge.qualifier,
        )
        query = " ".join(
            part
            for part in (anchor.instrument_name, anchor.qualifier, anchor.article_no)
            if part
        )
        variants = (query,)
        if anchor.instrument_number:
            variants += (
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
        offset = int(cast(int, arguments.get("offset", 0)))
        sources, more = self.broker.related_catalog_sources(
            variants, context, offset=offset, limit=50
        )
        candidates: list[JsonValue] = []
        for candidate in sources:
            role = match_related_source_name(anchor, candidate.name)
            if str(candidate.id) != item.source_id and role is not None:
                candidates.append(
                    {
                        "source_id": str(candidate.id),
                        "name": candidate.name,
                        "candidate_role": role,
                    }
                )
        result: dict[str, JsonValue] = {
            "edge_id": edge.edge_id,
            "query_variants": list(variants),
            "navigation_only": True,
            "absence_proven": False,
            "has_more": more,
            "next_offset": offset + 50 if more else None,
            "candidates": candidates,
        }
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Scoped title relationships are uncitable reading leads, never holdings or absence proof.",
            data=result,
        )

    def _wave(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        if not actions:
            return []
        try:
            receipts = self.acquirer.acquire_host_actions(actions, plan)
        except InvalidSourceAction:
            raise
        except RunStopped:
            self.context.check_active()
            receipts = self.acquirer.last_receipts
            for edge in self.edges.values():
                edge.discovery_gaps.append(
                    "Dependency acquisition stopped before closure."
                )
        self.receipts.extend(receipts)
        return receipts

    def _kind(self, source_id: str) -> SourceKind:
        return self.source_kinds.get(source_id, SourceKind.UNKNOWN)

    def expand(
        self,
        plan: ResearchPlan,
        *,
        frontier: set[int] | None = None,
        need_bindings: dict[int, set[str]] | None = None,
    ) -> list[AuthorityDependency]:
        self._collect(plan, frontier, need_bindings)
        fresh = [
            edge
            for edge in self.edges.values()
            if self._expanded.get(edge.edge_id) != self._edge_state(edge)
        ]
        for edge in fresh:
            edge.discovery_gaps = []
            edge.incomplete_source_ids = []
        with graph_step(
            "legal_composite.authority_dependencies", {"edge_count": len(fresh)}
        ) as step:
            self._expand(fresh, plan)
            self._expanded.update(
                {edge.edge_id: self._edge_state(edge) for edge in fresh}
            )
            step.output_value = {
                "edge_count": len(self.edges),
                "governing_original_count": sum(
                    len(edge.governing_citations) for edge in self.edges.values()
                ),
                "candidate_original_count": sum(
                    len(edge.candidate_citations) for edge in self.edges.values()
                ),
            }
        return list(self.edges.values())

    @staticmethod
    def _edge_state(edge: AuthorityDependency) -> str:
        return json.dumps(
            [
                edge.need_ids,
                [origin.model_dump() for origin in edge.origins],
                edge.governing_citations,
            ],
            sort_keys=True,
        )

    def _expand(self, edges: list[AuthorityDependency], plan: ResearchPlan) -> None:
        """Discover indexed passages in one batch, then read only missing provisions.

        A search hit is not a request to traverse the candidate's whole file.
        Missing scope/disposition passages remain visible for focused research tools.
        """
        from onyx.tools.constants import REGULATORY_MAX_SEARCH_QUERY_CHARS

        actions: list[SourceAction] = []
        query_edges: dict[str, set[str]] = {}
        resolution_edges: dict[str, set[str]] = {}
        for edge in edges:
            identity = " ".join(
                part
                for part in (edge.instrument_name, edge.qualifier, edge.article)
                if part
            )
            queries = [identity]
            if edge.instrument_number:
                queries.append(
                    " ".join(
                        part
                        for part in (
                            edge.instrument_number,
                            edge.qualifier,
                            edge.article,
                        )
                        if part
                    )
                )
            for query in dict.fromkeys(queries):
                query = query[:REGULATORY_MAX_SEARCH_QUERY_CHARS]
                query_edges.setdefault(query, set()).add(edge.edge_id)
                actions.append(
                    SourceAction(
                        need_ids=edge.need_ids,
                        tool="search_corpus",
                        arguments={
                            "query": query,
                            "mode": "hybrid",
                            "expand_query": False,
                            "discover_related_sources": True,
                            "coverage_item": ", ".join(edge.need_ids),
                            "evidence_target": "Original provisions and related sources addressing this observed instrument and provision; preserve material conditions, scope and contrary effects",
                        },
                    )
                )
            actions.append(
                SourceAction(
                    need_ids=edge.need_ids,
                    tool="dependency_related_sources",
                    arguments={"edge_id": edge.edge_id},
                )
            )
            if not edge.governing_citations:
                query = edge.instrument_name
                resolution_edges.setdefault(query, set()).add(edge.edge_id)
                actions.append(
                    SourceAction(
                        need_ids=edge.need_ids,
                        tool="resolve_source",
                        arguments={"query": query, "limit": 100},
                    )
                )
        results = self._wave(actions, plan)
        provision_reads: list[SourceAction] = []
        for receipt in results:
            arguments = receipt.get("host_arguments", {})
            data = receipt.get("data", {})
            if not isinstance(arguments, dict) or not isinstance(data, dict):
                continue
            query = str(arguments.get("query", ""))
            if receipt.get("tool") == "resolve_source":
                keys = resolution_edges.get(query, set())
                for candidate in (
                    data.get("sources", [])
                    if isinstance(data.get("sources"), list)
                    else []
                ):
                    sid = (
                        candidate.get("source_id")
                        if isinstance(candidate, dict)
                        else None
                    )
                    if not isinstance(sid, str):
                        continue
                    for key in keys:
                        edge = self.edges[key]
                        provision_reads.append(
                            SourceAction(
                                need_ids=edge.need_ids,
                                tool="read_provision",
                                arguments={
                                    "source_id": sid,
                                    "article": " ".join(
                                        part
                                        for part in (edge.qualifier, edge.article)
                                        if part
                                    ),
                                },
                            )
                        )
                continue
            key = arguments.get("edge_id")
            keys = (
                {key}
                if isinstance(key, str) and key in self.edges
                else query_edges.get(query, set())
            )
            for key in keys:
                edge = self.edges[key]
                if receipt.get("status") in {"denied", "unavailable", "truncated"}:
                    edge.discovery_gaps.append(
                        "Related-source discovery did not complete; use a focused search to resolve this gap."
                    )
                searches = data.get("candidate_search", [])
                for part in searches if isinstance(searches, list) else []:
                    if not isinstance(part, dict) or part.get("incomplete") is not True:
                        continue
                    if part.get("candidate_window_saturated") and not (
                        part.get("unavailable_source_ids")
                        or part.get("full_original_proof_unavailable")
                    ):
                        edge.discovery_limits.append(
                            "Search returned a bounded candidate window; it does not prove corpus absence."
                        )
                    else:
                        edge.discovery_gaps.append(
                            "Related-source search coverage remains incomplete."
                        )
                if data.get("has_more"):
                    edge.discovery_limits.append(
                        "Additional navigation results exist; the first page does not prove corpus absence."
                    )
                candidates = data.get("candidates", [])
                for candidate in candidates if isinstance(candidates, list) else []:
                    sid = (
                        candidate.get("source_id")
                        if isinstance(candidate, dict)
                        else None
                    )
                    if isinstance(sid, str):
                        self._candidate_edges.setdefault(sid, set()).add(key)
                for citation in cast(list[int], receipt.get("citations", [])):
                    item = self.ledger.get(citation)
                    if item is not None:
                        self._candidate_edges.setdefault(item.source_id, set()).add(key)
        # Tool continuations return to the researcher; there is no host paging loop.
        self._wave(provision_reads, plan)
        self.synchronize()

    def synchronize(self) -> list[AuthorityDependency]:
        """Attach later focused reads without launching another discovery wave."""
        self._bind_governing()
        for edge in self.edges.values():
            governing_ids = {
                item.source_id
                for number in edge.governing_citations
                if (item := self.ledger.get(number)) is not None
            }
            candidate_ids = {
                sid
                for sid, keys in self._candidate_edges.items()
                if edge.edge_id in keys and sid not in governing_ids
            }
            edge.candidate_source_ids = sorted(candidate_ids)
            edge.candidate_citations = [
                number
                for number in self.ledger.citation_numbers()
                if (item := self.ledger.get(number)) is not None
                and item.source_id in candidate_ids
            ]
            acquired = {
                item.source_id
                for number in edge.candidate_citations
                if (item := self.ledger.get(number)) is not None
            }
            edge.incomplete_source_ids = sorted(candidate_ids - acquired)
            edge.judicial_source_ids = sorted(
                sid
                for sid in candidate_ids
                if self._kind(sid) == SourceKind.JUDICIAL_DECISION
                or self._original_kind(sid) == SourceKind.JUDICIAL_DECISION
            )
            edge.discovery_limits = list(dict.fromkeys(edge.discovery_limits))
            edge.discovery_gaps = list(dict.fromkeys(edge.discovery_gaps))
            self.acquirer._bind_originals(edge.candidate_citations, edge.need_ids)
        return list(self.edges.values())

    def _original_kind(self, source_id: str) -> SourceKind:
        if self._kind(source_id) != SourceKind.UNKNOWN:
            return self._kind(source_id)
        originals = [
            item
            for number in self.ledger.citation_numbers()
            if (item := self.ledger.get(number)) is not None
            and item.source_id == source_id
        ]
        originals.sort(key=lambda item: cast(int, item.metadata.get("position", -1)))
        opening_read = any(
            receipt.get("tool") == "read_source_range"
            and isinstance(arguments := receipt.get("host_arguments"), dict)
            and arguments.get("source_id") == source_id
            and arguments.get("start", 0) == 0
            and bool(receipt.get("citations"))
            for receipt in self.receipts
        )
        if not originals or not opening_read:
            return SourceKind.UNKNOWN
        item = originals[0]
        if item.search_doc is None:
            return SourceKind.UNKNOWN
        from uuid import UUID

        source = CorpusSource(
            UUID(source_id),
            item.search_doc.semantic_identifier,
            item.search_doc.file_id or "",
        )
        return classify_source(
            source, (), opening_texts=tuple(row.text for row in originals[:3])
        ).kind


HostDependencyStage = Literal["discovery", "provisions"]


class CompositeDependencyExpander(DependencyExpander):
    """Legal Composite material discovery, isolated from legacy workflow behavior."""

    def __init__(
        self,
        *,
        broker: CorpusBroker,
        acquirer: CanonicalAcquirer,
        ledger: EvidenceLedger,
        context: RunContext,
        source_kinds: dict[str, SourceKind],
        host_stage_admission: Callable[
            [list[SourceAction], ResearchPlan, HostDependencyStage], bool
        ]
        | None = None,
    ) -> None:
        super().__init__(
            broker=broker,
            acquirer=acquirer,
            ledger=ledger,
            context=context,
            source_kinds=source_kinds,
        )
        self.host_stage_admission = host_stage_admission
        self._active_edges: set[str] = set()
        self._deferred_edges: set[str] = set()

    def _defer_stage(self, stage: HostDependencyStage, reason: str) -> None:
        self._deferred_edges.update(self._active_edges)
        for key in self._active_edges:
            edge = self.edges[key]
            edge.discovery_gaps.append(
                f"Material dependency {stage} stage was deferred before closure."
            )
        self.receipts.append(
            {
                "status": "dependency_stage_deferred",
                "stage": stage,
                "reason": reason,
                "edge_ids": sorted(self._active_edges),
                "navigation_only": True,
                "absence_proven": False,
            }
        )

    def _wave(
        self, actions: list[SourceAction], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        if self.host_stage_admission is None:
            return super()._wave(actions, plan)
        if not actions:
            return []
        stage: HostDependencyStage = (
            "discovery"
            if any(action.tool != "read_provision" for action in actions)
            else "provisions"
        )
        if not self.host_stage_admission(actions, plan, stage):
            self._defer_stage(stage, "stage_admission")
            return []
        try:
            receipts = self.acquirer.acquire_host_actions(actions, plan)
        except InvalidSourceAction:
            raise
        except RunStopped:
            self.context.check_active()
            receipts = self.acquirer.last_receipts
            self._defer_stage(stage, "acquisition_stopped")
        self.receipts.extend(receipts)
        return receipts

    def _anchor(self, item: EvidenceItem) -> ProvisionNavigationAnchor | None:
        kind = self.verified_kinds.get(
            item.source_id, self.source_kinds.get(item.source_id)
        )
        if kind == SourceKind.STATUTE:
            return derive_provision_navigation_anchor(
                item.source_id, [self._navigation_item(item)]
            )
        if kind not in {
            SourceKind.CONSTITUTION,
            SourceKind.REGULATION,
            SourceKind.COMMUNIQUE,
            SourceKind.CIRCULAR,
            SourceKind.PRESIDENTIAL_DECREE,
            SourceKind.EXECUTIVE_DECISION,
            SourceKind.TREATY,
            SourceKind.UNKNOWN,
        }:
            return None
        doc = item.search_doc
        metadata = model_evidence_metadata(item.metadata)
        if (
            doc is None
            or doc.document_id != item.source_id
            or doc.metadata.get("regulatory_chunk_id") != item.chunk_id
            or sha256(item.text.encode()).hexdigest() != item.text_hash
            or any(
                metadata.get(flag)
                for flag in ("derived", "external", "untrusted", "truncated")
            )
        ):
            return None
        headings = metadata.get("heading_path")
        if not isinstance(headings, list) or not headings:
            return None
        articles: set[RegulatoryArticleHeading] = set()
        for heading in headings:
            if isinstance(heading, str):
                parsed = parse_regulatory_article_heading(heading)
                if parsed is not None:
                    articles.add(parsed)
        body_articles = {
            parsed
            for line in item.text.splitlines()
            if (parsed := parse_regulatory_article_heading(line.strip().lstrip("# ")))
            is not None
        }
        if len(articles) != 1 or not articles <= body_articles:
            return None
        identities: set[tuple[str, str | None]] = set()
        for title in (headings[0], metadata.get("title")):
            if not isinstance(title, str) or not title.strip():
                continue
            matches = _generic_titles(title)
            if len(matches) != 1 or matches[0][0] != 0 or matches[0][1] != len(title):
                return None
            _, _, name, number = matches[0]
            observed_kind = _generic_kind(name)
            if observed_kind is None or (
                kind != SourceKind.UNKNOWN and observed_kind != kind
            ):
                return None
            identities.add((name, number))
        if len(identities) != 1:
            return None
        name, number = next(iter(identities))
        if kind == SourceKind.UNKNOWN:
            first_line = next(
                (
                    line.strip().lstrip("# ")
                    for line in item.text.splitlines()
                    if line.strip()
                ),
                "",
            )
            opening = _generic_titles(first_line)
            if (
                len(opening) != 1
                or opening[0][0] != 0
                or opening[0][1] != len(first_line)
                or _generic_normalized_name(opening[0][2])
                != _generic_normalized_name(name)
                or opening[0][3] != number
                or _generic_kind(name) == SourceKind.EXECUTIVE_DECISION
            ):
                return None
        article = next(iter(articles))
        return ProvisionNavigationAnchor(
            item.source_id, name, number, article.article_no, article.qualifier
        )

    def _validated_material_leads(
        self,
        plan: ResearchPlan,
        frontier: set[int] | None = None,
        _need_bindings: dict[int, set[str]] | None = None,
        material_targets: list[MaterialDependencyRequest] | None = None,
    ) -> list[dict[str, JsonValue]]:
        selected_numbers = (
            set(self.ledger.citation_numbers()) if frontier is None else frontier
        )
        if material_targets is not None:
            requested_origins = {target.origin_citation for target in material_targets}
            if not requested_origins <= selected_numbers:
                raise InvalidSourceAction(
                    "Material dependency origin was not delivered"
                )
            for citation in requested_origins:
                item = self.ledger.get(citation)
                canonical = (
                    item.metadata.get("canonical_metadata")
                    if item is not None
                    else None
                )
                if (
                    item is None
                    or item.search_doc is None
                    or item.search_doc.document_id != item.source_id
                    or item.search_doc.metadata.get("regulatory_chunk_id")
                    != item.chunk_id
                    or sha256(item.text.encode()).hexdigest() != item.text_hash
                    or item.metadata.get("truncated")
                    or (isinstance(canonical, dict) and canonical.get("truncated"))
                    or any(
                        model_evidence_metadata(item.metadata).get(flag)
                        for flag in ("derived", "external", "untrusted", "truncated")
                    )
                ):
                    raise InvalidSourceAction(
                        "Material dependency origin is not canonical"
                    )
            selected_numbers = requested_origins
        records = cast(
            list[dict[str, JsonValue]],
            json.loads(self.ledger.serialize_records(selected_numbers, max_chars=None)),
        )
        leads = explicit_reference_leads(
            self.ledger, records, syntactic_reference_binding=True
        )
        for citation in selected_numbers:
            item = self.ledger.get(citation)
            if (
                item is None
                or item.search_doc is None
                or item.search_doc.document_id != item.source_id
                or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
                or any(
                    model_evidence_metadata(item.metadata).get(flag)
                    for flag in ("derived", "external", "untrusted", "truncated")
                )
            ):
                continue
            aliases = _literal_names(item.text)
            leads.extend(_generic_reference_leads(item, citation))
            for reference, name in _named_native_references(
                item.text,
                {alias: set() for alias in aliases},
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
                unicode_ordinals=True,
            ):
                if reference.article is not None and name in aliases:
                    leads.append(
                        {
                            "formal_name": aliases[name],
                            "instrument_number": reference.number or None,
                            "article": reference.article,
                            "qualifier": reference.qualifier,
                            "origin_citations": [citation],
                        }
                    )
        for number in self.ledger.citation_numbers():
            if number not in selected_numbers:
                continue
            item = self.ledger.get(number)
            assert item is not None
            anchor = self._anchor(item)
            if anchor is not None:
                leads.append(
                    {
                        "formal_name": anchor.instrument_name,
                        "instrument_number": anchor.instrument_number,
                        "article": anchor.article_no,
                        "qualifier": anchor.qualifier,
                        "origin_citations": [number],
                    }
                )
        known_needs = {need.need_id for need in plan.needs}
        if material_targets is not None:
            for target in material_targets:
                if (
                    not target.reason.strip()
                    or len(target.need_ids) != len(set(target.need_ids))
                    or not set(target.need_ids) <= known_needs
                    or not any(
                        self._matches_material_target(lead, target) for lead in leads
                    )
                ):
                    raise InvalidSourceAction(
                        "Material dependency does not match an observed original reference"
                    )
        return leads

    def _collect_material(
        self,
        plan: ResearchPlan,
        frontier: set[int] | None = None,
        need_bindings: dict[int, set[str]] | None = None,
        material_targets: list[MaterialDependencyRequest] | None = None,
    ) -> set[str]:
        leads = self._validated_material_leads(
            plan, frontier, need_bindings, material_targets
        )
        known_needs = {need.need_id for need in plan.needs}
        collected: set[str] = set()
        for lead in leads:
            targets = (
                [
                    target
                    for target in material_targets
                    if self._matches_material_target(lead, target)
                ]
                if material_targets is not None
                else None
            )
            if targets is not None and not targets:
                continue
            name, article = lead.get("formal_name"), lead.get("article")
            if not isinstance(name, str) or not isinstance(article, str):
                continue
            number = lead.get("instrument_number")
            qualifier = lead.get("qualifier")
            number = number if isinstance(number, str) else None
            qualifier = qualifier if isinstance(qualifier, str) else None
            key = _entity_key(name, number, article, qualifier)
            edge = self._matching_edge(name, number, article, qualifier)
            if edge is None:
                edge = AuthorityDependency(
                    edge_id=key,
                    need_ids=[],
                    instrument_name=name,
                    instrument_number=number,
                    article=article,
                    qualifier=qualifier,
                    origins=[],
                )
                self.edges[key] = edge
            elif edge.instrument_number is None and number:
                edge.instrument_number = number
            collected.add(edge.edge_id)
            origins = lead.get("origin_citations", [])
            if not isinstance(origins, list):
                continue
            for citation in origins:
                if type(citation) is not int:
                    continue
                matching_targets = (
                    [target for target in targets if target.origin_citation == citation]
                    if targets is not None
                    else None
                )
                if matching_targets is not None and not matching_targets:
                    continue
                item = self.ledger.get(citation)
                if item is None:
                    continue
                if citation not in {origin.citation for origin in edge.origins}:
                    edge.origins.append(
                        DependencyOrigin(
                            citation=citation,
                            source_id=item.source_id,
                            chunk_id=item.chunk_id,
                            text_hash=item.text_hash,
                        )
                    )
                bindings = (
                    {
                        need_id
                        for target in matching_targets
                        for need_id in target.need_ids
                    }
                    if matching_targets is not None
                    else (
                        need_bindings.get(citation, set(item.question_ids))
                        if need_bindings is not None
                        else set(item.question_ids)
                    )
                ) & known_needs
                edge.need_ids = sorted(set(edge.need_ids) | (bindings or known_needs))
        self._bind_governing()
        return collected

    @staticmethod
    def _matches_material_target(
        lead: dict[str, JsonValue], target: MaterialDependencyRequest
    ) -> bool:
        name = lead.get("formal_name")
        article = lead.get("article")
        origins = lead.get("origin_citations")
        if (
            not isinstance(name, str)
            or not isinstance(article, str)
            or not isinstance(origins, list)
        ):
            return False
        qualifier = lead.get("qualifier")
        observed_article = " ".join(
            part for part in (qualifier, article) if isinstance(part, str) and part
        )
        return (
            target.origin_citation in origins
            and _generic_normalized_name(target.instrument_name)
            == _generic_normalized_name(name)
            and " ".join(target.article.casefold().split())
            == " ".join(observed_article.casefold().split())
            and (
                target.instrument_number is None
                or target.instrument_number == lead.get("instrument_number")
            )
        )

    def _matching_edge(
        self, name: str, number: str | None, article: str, qualifier: str | None
    ) -> AuthorityDependency | None:
        matches = [
            edge
            for edge in self.edges.values()
            if edge.article == article
            and edge.qualifier == qualifier
            and not (
                edge.instrument_number and number and edge.instrument_number != number
            )
            and (
                (edge.instrument_number and number and edge.instrument_number == number)
                or _generic_normalized_name(edge.instrument_name)
                == _generic_normalized_name(name)
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def _related(
        self, arguments: dict[str, JsonValue], context: RunContext
    ) -> ToolOutcome:
        edge = self.edges[str(arguments["edge_id"])]
        citation = (
            edge.governing_citations[0]
            if edge.governing_citations
            else edge.origins[0].citation
        )
        item = self.ledger.get(citation)
        assert item is not None
        # The relation is a navigation lead from an already read original. Live
        # source access is checked without rereading its complete canonical text.
        self.broker.source(item.source_id, context)
        anchor = ProvisionNavigationAnchor(
            item.source_id,
            edge.instrument_name,
            edge.instrument_number,
            edge.article,
            edge.qualifier,
        )
        query = " ".join(
            part
            for part in (anchor.instrument_name, anchor.qualifier, anchor.article_no)
            if part
        )
        variants = (query,)
        if anchor.instrument_number:
            variants += (
                " ".join(
                    part
                    for part in (
                        anchor.instrument_number,
                        "sayılı",
                        edge.instrument_name,
                        anchor.qualifier,
                        anchor.article_no,
                    )
                    if part
                ),
            )
        offset = int(cast(int, arguments.get("offset", 0)))
        sources, more = self.broker.related_catalog_sources(
            variants, context, offset=offset, limit=50
        )
        candidates: list[JsonValue] = []
        for candidate in sources:
            role = match_related_source_name(anchor, candidate.name)
            if str(candidate.id) != item.source_id and role is not None:
                candidates.append(
                    {
                        "source_id": str(candidate.id),
                        "name": candidate.name,
                        "candidate_role": role,
                    }
                )
        result: dict[str, JsonValue] = {
            "edge_id": edge.edge_id,
            "query_variants": list(variants),
            "navigation_only": True,
            "absence_proven": False,
            "has_more": more,
            "next_offset": offset + 50 if more else None,
            "candidates": candidates,
        }
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Scoped title relationships are uncitable reading leads, never holdings or absence proof.",
            data=result,
        )

    def validate_material(
        self,
        plan: ResearchPlan,
        *,
        frontier: set[int],
        material_targets: list[MaterialDependencyRequest],
        need_bindings: dict[int, set[str]] | None = None,
    ) -> None:
        """Validate proposed relations without registering or binding originals."""

        self._validated_material_leads(plan, frontier, need_bindings, material_targets)

    def register_material(
        self,
        plan: ResearchPlan,
        *,
        frontier: set[int],
        material_targets: list[MaterialDependencyRequest],
        need_bindings: dict[int, set[str]] | None = None,
    ) -> list[AuthorityDependency]:
        """Keep validated unexamined relations visible without acquiring new sources."""
        selected = self._collect_material(
            plan, frontier, need_bindings, material_targets
        )
        for edge in self.edges.values():
            if edge.edge_id in selected and self._expanded.get(
                edge.edge_id
            ) != self._edge_state(edge):
                edge.discovery_gaps = list(
                    dict.fromkeys(
                        [
                            *edge.discovery_gaps,
                            "Material dependency acquisition was deferred before examination.",
                        ]
                    )
                )
        return self.synchronize()

    def expand(
        self,
        plan: ResearchPlan,
        *,
        frontier: set[int] | None = None,
        need_bindings: dict[int, set[str]] | None = None,
        material_targets: list[MaterialDependencyRequest] | None = None,
    ) -> list[AuthorityDependency]:
        if material_targets == []:
            return self.synchronize()
        selected = self._collect_material(
            plan, frontier, need_bindings, material_targets
        )
        fresh = [
            edge
            for edge in self.edges.values()
            if self._expanded.get(edge.edge_id) != self._edge_state(edge)
            and (material_targets is None or edge.edge_id in selected)
        ]
        for edge in fresh:
            edge.discovery_gaps = []
            edge.incomplete_source_ids = []
        with graph_step(
            "legal_composite.authority_dependencies", {"edge_count": len(fresh)}
        ) as step:
            self._active_edges = {edge.edge_id for edge in fresh}
            self._deferred_edges = set()
            try:
                self._expand(fresh, plan)
            finally:
                self._active_edges = set()
            self._expanded.update(
                {
                    edge.edge_id: self._edge_state(edge)
                    for edge in fresh
                    if edge.edge_id not in self._deferred_edges
                }
            )
            step.output_value = {
                "edge_count": len(self.edges),
                "governing_original_count": sum(
                    len(edge.governing_citations) for edge in self.edges.values()
                ),
                "candidate_original_count": sum(
                    len(edge.candidate_citations) for edge in self.edges.values()
                ),
            }
        return list(self.edges.values())


def material_dependency_gaps(
    edges: list[AuthorityDependency],
    ledger: EvidenceLedger,
    delivered: set[int],
    requirements: list[SourceRequirement] | None = None,
) -> dict[str, list[str]]:
    """Check material relation delivery and identity without deciding applicability."""
    from onyx.asv3.judicial_sections import canonical_disposition_witness

    rows = {
        number: item for number in delivered if (item := ledger.get(number)) is not None
    }

    def citation_faults(number: int) -> list[str]:
        item = ledger.get(number)
        faults: list[str] = []
        if item is None:
            faults.append("citation_missing")
        elif (
            item.search_doc is None
            or item.search_doc.document_id != item.source_id
            or item.search_doc.metadata.get("regulatory_chunk_id") != item.chunk_id
            or sha256(item.text.encode()).hexdigest() != item.text_hash
            or any(
                model_evidence_metadata(item.metadata).get(flag)
                for flag in ("derived", "external", "untrusted", "truncated")
            )
        ):
            faults.append("citation_noncanonical")
        if number not in delivered:
            faults.append("citation_undelivered")
        return [f"{fault}@{number}" for fault in faults]

    result: dict[str, list[str]] = {}
    for edge in edges:
        gaps: list[str] = []
        for origin in edge.origins:
            item = ledger.get(origin.citation)
            if item is None or (item.source_id, item.chunk_id, item.text_hash) != (
                origin.source_id,
                origin.chunk_id,
                origin.text_hash,
            ):
                gaps.append(f"origin_binding_mismatch@{origin.citation}")
            gaps.extend(citation_faults(origin.citation))
        if not edge.governing_citations:
            gaps.append("governing_original_unread")
        for number in set(edge.governing_citations + edge.candidate_citations):
            gaps.extend(citation_faults(number))
        readable = {
            number: rows[number]
            for number in edge.governing_citations + edge.candidate_citations
            if number in rows and not citation_faults(number)
        }
        acquired_sources = {item.source_id for item in readable.values()}
        for source_id in edge.candidate_source_ids:
            if source_id not in acquired_sources:
                gaps.append(f"candidate_source_unread@{source_id}")
        for source_id in edge.incomplete_source_ids:
            gaps.append(f"candidate_original_missing@{source_id}")
        if edge.discovery_gaps:
            gaps.append("discovery_gap_unresolved")
        for source_id in edge.judicial_source_ids:
            context = [
                item.model_copy(deep=True)
                for number, item in rows.items()
                if item.source_id == source_id and not citation_faults(number)
            ]
            for item in context:
                item.metadata["heading_path"] = []
            operative = False
            for number, item in readable.items():
                if item.source_id != source_id:
                    continue
                body = item.model_copy(deep=True)
                body.metadata["heading_path"] = []
                quotations = [
                    support.quotation
                    for requirement in requirements or []
                    if requirement.need_id in edge.need_ids
                    for support in requirement.supports
                    if support.citation == number
                    and support.quotation.strip()
                    and support.quotation in item.text
                ]
                spans = [
                    (item.text.index(quote), item.text.index(quote) + len(quote))
                    for quote in quotations
                ]
                cursor = 0
                for line in item.text.splitlines(keepends=True):
                    if line.strip():
                        spans.append((cursor, cursor + len(line)))
                    cursor += len(line)
                if any(
                    canonical_disposition_witness(
                        body, start, end, source_context=context
                    )
                    for start, end in spans
                ):
                    operative = True
                    break
            if not operative:
                gaps.append(f"judicial_operative_body_unread@{source_id}")
        result[edge.edge_id] = list(dict.fromkeys(gaps))
    return result


def assess_dependencies(
    edges: list[AuthorityDependency],
    plan: ResearchPlan,
    draft: DraftAnswer,
    review: AnswerReview,
    ledger: EvidenceLedger,
    delivered: set[int],
) -> tuple[bool, bool, list[str]]:
    """Require closed original witnesses or an exact disclosed material source gap."""
    from onyx.asv3.citation_numbers import extract_citation_numbers
    from onyx.asv3.judicial_sections import canonical_disposition_witness
    from onyx.asv3.models import model_evidence_metadata

    if not edges:
        return True, True, []
    assessments = review.dependency_assessments or []
    edge_map = {edge.edge_id: edge for edge in edges}
    ids = [row.edge_id for row in assessments]
    if len(ids) != len(set(ids)) or set(ids) != set(edge_map):
        return (
            False,
            False,
            [
                "Every host-discovered authority dependency needs exactly one original-grounded assessment."
            ],
        )
    safe = complete = True
    gaps: list[str] = []
    cited = set(extract_citation_numbers(draft.answer))
    originals = [
        item for number in delivered if (item := ledger.get(number)) is not None
    ]
    need_reviews = {row.need_id: row for row in review.needs}
    for row in assessments:
        edge = edge_map[row.edge_id]
        valid = (
            len(row.need_ids) == len(set(row.need_ids))
            and set(row.need_ids) == set(edge.need_ids)
            and set(row.need_ids) <= {need.need_id for need in plan.needs}
            and bool(row.explanation.strip())
            and bool(row.scope_and_date.strip())
        )
        for origin in edge.origins:
            item = ledger.get(origin.citation)
            if item is None or (item.source_id, item.chunk_id, item.text_hash) != (
                origin.source_id,
                origin.chunk_id,
                origin.text_hash,
            ):
                valid = False
        if row.temporal_status == "conditional":
            if (
                not row.conditional_excerpt
                or not row.conditional_excerpt.strip()
                or row.conditional_excerpt not in draft.answer
                or any(
                    need_reviews.get(need) is None
                    or need_reviews[need].status not in {"conditional", "unresolved"}
                    for need in edge.need_ids
                )
            ):
                valid = False
        elif row.temporal_status == "unresolved" and row.status != "unresolved":
            valid = False
        allowed = set(
            edge.governing_citations
            + edge.candidate_citations
            + [origin.citation for origin in edge.origins]
        )
        witnesses: set[int] = set()
        operative_sources: set[str] = set()
        for witness in row.witnesses:
            item = ledger.get(witness.citation)
            if (
                item is None
                or item.search_doc is None
                or witness.citation not in delivered
                or witness.citation not in allowed
                or not witness.quotation.strip()
                or witness.quotation not in item.text
                or any(
                    model_evidence_metadata(item.metadata).get(flag)
                    for flag in ("derived", "external", "untrusted", "truncated")
                )
            ):
                valid = False
                continue
            witnesses.add(witness.citation)
            if (
                witness.role == "operative"
                and item.source_id in edge.judicial_source_ids
            ):
                start = item.text.index(witness.quotation)
                # Section identity follows only delivered, contiguous original body text.
                body = item.model_copy(deep=True)
                body.metadata["heading_path"] = []
                context = [
                    original.model_copy(deep=True)
                    for original in originals
                    if original.source_id == item.source_id
                ]
                for original in context:
                    original.metadata["heading_path"] = []
                if canonical_disposition_witness(
                    body, start, start + len(witness.quotation), source_context=context
                ):
                    operative_sources.add(item.source_id)
                else:
                    valid = False
        if row.status == "unresolved":
            complete = False
            if (
                not row.gap_disclosure
                or not row.gap_disclosure.strip()
                or row.gap_disclosure not in draft.answer
                or any(
                    need not in draft.unresolved_need_ids
                    or need_reviews.get(need) is None
                    or need_reviews[need].status != "unresolved"
                    for need in edge.need_ids
                )
            ):
                valid = False
        else:
            source_witnesses = {
                item.source_id
                for citation in witnesses
                if (item := ledger.get(citation)) is not None
            }
            candidate_sources = {
                item.source_id
                for citation in edge.candidate_citations
                if (item := ledger.get(citation)) is not None
            }
            if (
                not set(edge.governing_citations) & witnesses
                or not candidate_sources <= source_witnesses
                or not set(edge.judicial_source_ids) <= operative_sources
                or edge.incomplete_source_ids
                or edge.discovery_gaps
            ):
                valid = False
            if row.status == "examined_applicable" and not witnesses <= cited:
                valid = False
        if not valid:
            safe = complete = False
            gaps.append(
                f"Authority dependency {edge.edge_id}: missing, stale, unread or non-operative original witness, or undisclosed gap."
            )
    return complete and safe, safe, gaps


def dependency_required_citations(
    edges: list[AuthorityDependency], ledger: EvidenceLedger
) -> list[int]:
    """Keep governing originals and actual disposition boundaries through relevance packing."""
    from onyx.asv3.judicial_sections import canonical_disposition_witness

    numbers = {citation for edge in edges for citation in edge.governing_citations}
    judicial_ids = {sid for edge in edges for sid in edge.judicial_source_ids}
    rows = {
        number: item
        for number in ledger.citation_numbers()
        if (item := ledger.get(number)) is not None
    }
    for edge in edges:
        by_source: dict[str, list[int]] = {}
        for number in edge.candidate_citations:
            if number in rows:
                by_source.setdefault(rows[number].source_id, []).append(number)
        for source_numbers in by_source.values():
            numbers.add(
                min(
                    source_numbers,
                    key=lambda number: cast(
                        int, rows[number].metadata.get("position", -1)
                    ),
                )
            )
    originals = [item.model_copy(deep=True) for item in rows.values()]
    for item in originals:
        item.metadata["heading_path"] = []
    for number, recorded in rows.items():
        if recorded.source_id not in judicial_ids:
            continue
        item = recorded.model_copy(deep=True)
        item.metadata["heading_path"] = []
        cursor = 0
        disposition = False
        for line in item.text.splitlines(keepends=True):
            if line.strip() and canonical_disposition_witness(
                item, cursor, cursor + len(line), source_context=originals
            ):
                disposition = True
            cursor += len(line)
        if disposition:
            numbers.add(number)
            # Preserve the immediately preceding body boundary, independently of labels.
            position = item.metadata.get("position")
            for previous, context in rows.items():
                if (
                    type(position) is int
                    and context.source_id == item.source_id
                    and context.metadata.get("position") == position - 1
                ):
                    numbers.add(previous)
    return sorted(numbers)
