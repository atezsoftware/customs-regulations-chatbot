"""Verify only discovered source identities before following their own law anchors."""

from __future__ import annotations

import re
import unicodedata
from typing import cast
from uuid import UUID

from pydantic import JsonValue

from onyx.asv3.corpus_tools import CorpusBroker
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, RunContext, model_evidence_metadata
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.legal_composite_sources import SourceKind, classify_source
from onyx.legal_composite.acquisition import CanonicalAcquirer
from onyx.legal_composite.dependencies import DependencyExpander
from onyx.legal_composite.models import AuthorityDependency, ResearchPlan, SourceAction
from onyx.regulatory.heading_path import parse_regulatory_article_heading
from onyx.regulatory.source_identity import named_law_number


def _source_id(arguments: JsonValue) -> str | None:
    value = arguments.get("source_id") if isinstance(arguments, dict) else None
    return value if isinstance(value, str) else None


def _identity_words(value: str) -> set[str]:
    folded = unicodedata.normalize(
        "NFKD", value.casefold().translate(str.maketrans("ışçğöü", "iscgou"))
    )
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    folded = folded.replace("_", " ")
    folded = re.sub(r"^\d{2,7}\s+sayili\s+", "", folded)
    return {
        "kanun" if word == "kanunu" else word
        for word in re.findall(r"[a-z0-9]+", folded)
    }


def _canonical_opening_kind(
    source: CorpusSource, openings: list[EvidenceItem]
) -> SourceKind:
    """A source's omitted body title may remain in its canonical opening heading."""
    if not openings or not any(
        type(item.metadata.get("position")) is int
        and item.metadata.get("position") == 0
        for item in openings
    ):
        return SourceKind.UNKNOWN
    for item in openings:
        doc = item.search_doc
        metadata = model_evidence_metadata(item.metadata)
        if (
            item.source_id != str(source.id)
            or item.chunk_id is None
            or doc is None
            or doc.document_id != str(source.id)
            or doc.metadata.get("regulatory_chunk_id") != item.chunk_id
            or any(
                metadata.get(flag) or item.metadata.get(flag)
                for flag in ("derived", "external", "untrusted", "truncated")
            )
        ):
            return SourceKind.UNKNOWN
    body_kind = classify_source(
        source, (), opening_texts=tuple(item.text for item in openings)
    ).kind
    if body_kind != SourceKind.UNKNOWN:
        return body_kind
    roots: set[str] = set()
    for item in openings:
        headings = item.metadata.get("heading_path")
        assert item.search_doc is not None
        if (
            not isinstance(headings, list)
            or not headings
            or not isinstance(headings[0], str)
            or item.search_doc.metadata.get("regulatory_heading_path") != headings
        ):
            return SourceKind.UNKNOWN
        roots.add(headings[0])
    if len(roots) != 1:
        return SourceKind.UNKNOWN
    root = next(iter(roots))
    filename = re.sub(
        r"\.(?:md|docx?|pdf|txt|html?)$",
        "",
        source.name.replace("\\", "/").rsplit("/", 1)[-1],
        flags=re.I,
    )
    # Filenames corroborate an original root; filenames and labels never supply it.
    words = _identity_words(root)
    if len(words) < 2 or not words <= _identity_words(filename):
        return SourceKind.UNKNOWN
    root_number, file_number = named_law_number(root), named_law_number(source.name)
    if root_number and file_number and root_number != file_number:
        return SourceKind.UNKNOWN
    identity = classify_source(source, (), opening_texts=(root,))
    return identity.kind if not identity.uncertain else SourceKind.UNKNOWN


class SupersearchDependencyExpander(DependencyExpander):
    def __init__(
        self,
        *,
        broker: CorpusBroker,
        acquirer: CanonicalAcquirer,
        ledger: EvidenceLedger,
        context: RunContext,
        source_kinds: dict[str, SourceKind],
    ) -> None:
        super().__init__(
            broker=broker,
            acquirer=acquirer,
            ledger=ledger,
            context=context,
            source_kinds=source_kinds,
        )
        self._identified_sources: set[str] = set()
        self._incomplete_reads: set[str] = set()
        self._incomplete_needs: set[str] = set()
        self._closed_provisions: set[tuple[str, str]] = set()

    def expand(
        self,
        plan: ResearchPlan,
        *,
        frontier: set[int] | None = None,
        need_bindings: dict[int, set[str]] | None = None,
    ) -> list[AuthorityDependency]:
        material_frontier = (
            set(self.ledger.citation_numbers()) if frontier is None else set(frontier)
        )
        for receipt in self.acquirer.last_receipts:
            key = self._provision_key(receipt)
            if (
                key is not None
                and self._starts_provision(receipt)
                and self._terminal_provision(receipt)
            ):
                self._closed_provisions.add(key)
        self._identify_sources(plan)
        return super().expand(
            plan, frontier=material_frontier, need_bindings=need_bindings
        )

    def _source_openings(self, source_id: str) -> list[EvidenceItem]:
        numbers: set[int] = set()
        for receipt in self.receipts:
            arguments, citations = (
                receipt.get("host_arguments"),
                receipt.get("citations"),
            )
            if (
                receipt.get("tool") == "read_source_range"
                and isinstance(arguments, dict)
                and arguments.get("source_id") == source_id
                and arguments.get("start", 0) == 0
                and isinstance(citations, list)
            ):
                numbers.update(number for number in citations if type(number) is int)
        return sorted(
            [
                item
                for number in numbers
                if (item := self.ledger.get(number)) is not None
            ],
            key=lambda item: cast(int, item.metadata.get("position", -1)),
        )

    def _identify_sources(self, plan: ResearchPlan) -> None:
        observed = {
            item.source_id: item
            for number in self.ledger.citation_numbers()
            if (item := self.ledger.get(number)) is not None
            and item.search_doc is not None
        }
        fresh = set(observed) - self._identified_sources
        actions = [
            SourceAction(
                need_ids=observed[source_id].question_ids
                or [need.need_id for need in plan.needs],
                tool="read_source_range",
                arguments={"source_id": source_id, "start": 0, "limit": 3},
            )
            for source_id in sorted(fresh)
            if not self._source_openings(source_id)
        ]
        self._wave(actions, plan)
        for source_id in fresh:
            openings = self._source_openings(source_id)
            example = observed[source_id]
            assert example.search_doc is not None
            kind = _canonical_opening_kind(
                CorpusSource(
                    UUID(source_id),
                    example.search_doc.semantic_identifier,
                    example.search_doc.file_id or "",
                ),
                openings,
            )
            self.source_kinds[source_id] = self.verified_kinds[source_id] = kind
            self._identified_sources.add(source_id)

    def _open_ambiguous_governing_sources(
        self,
        receipts: list[dict[str, JsonValue]],
        edges: list[AuthorityDependency],
        plan: ResearchPlan,
    ) -> None:
        candidates: dict[str, set[str]] = {}
        for receipt in receipts:
            arguments, data = receipt.get("host_arguments"), receipt.get("data")
            if (
                receipt.get("tool") != "read_named_provision"
                or receipt.get("status") != "ambiguous"
                or not isinstance(arguments, dict)
                or not isinstance(data, dict)
                or not isinstance(data.get("sources"), list)
            ):
                continue
            for edge in edges:
                if arguments.get("source_name") != edge.instrument_name:
                    continue
                for candidate in cast(list[JsonValue], data["sources"]):
                    source_id = _source_id(candidate)
                    name = (
                        candidate.get("name") if isinstance(candidate, dict) else None
                    )
                    if source_id is None or not isinstance(name, str):
                        continue
                    filename = re.sub(
                        r"\.(?:md|docx?|pdf|txt|html?)$",
                        "",
                        name.replace("\\", "/").rsplit("/", 1)[-1],
                        flags=re.I,
                    )
                    file_number = named_law_number(name)
                    if _identity_words(filename) != _identity_words(
                        edge.instrument_name
                    ) or (
                        edge.instrument_number
                        and file_number
                        and edge.instrument_number != file_number
                    ):
                        continue
                    candidates.setdefault(source_id, set()).update(edge.need_ids)
        self._wave(
            [
                SourceAction(
                    need_ids=sorted(needs),
                    tool="read_source_range",
                    arguments={"source_id": source_id, "start": 0, "limit": 3},
                )
                for source_id, needs in sorted(candidates.items())
                if source_id not in self._identified_sources
                and not self._source_openings(source_id)
            ],
            plan,
        )

    def _read_own_governing(
        self, edges: list[AuthorityDependency], plan: ResearchPlan
    ) -> None:
        self._bind_governing()
        actions: list[SourceAction] = []
        for edge in edges:
            article = " ".join(part for part in (edge.qualifier, edge.article) if part)
            incomplete = any(
                model_evidence_metadata(item.metadata).get("article_closure_complete")
                is not True
                and (item.source_id, article) not in self._closed_provisions
                for number in edge.governing_citations
                if (item := self.ledger.get(number)) is not None
            )
            if edge.governing_citations and not incomplete:
                continue
            candidates = []
            for source_id in sorted(self._identified_sources):
                if self.verified_kinds.get(source_id) != SourceKind.STATUTE:
                    continue
                roots: set[str] = set()
                for item in self._source_openings(source_id):
                    headings = item.metadata.get("heading_path")
                    if (
                        isinstance(headings, list)
                        and headings
                        and isinstance(headings[0], str)
                    ):
                        roots.add(headings[0])
                if len(roots) != 1:
                    continue
                root = next(iter(roots))
                if _identity_words(root) != _identity_words(edge.instrument_name):
                    continue
                number = named_law_number(root)
                if edge.instrument_number and number != edge.instrument_number:
                    continue
                candidates.append(source_id)
            # Equally authenticated instruments remain unresolved, never first-hit wins.
            if len(candidates) == 1:
                actions.append(
                    SourceAction(
                        need_ids=edge.need_ids,
                        tool="read_provision",
                        arguments={
                            "source_id": candidates[0],
                            "article": article,
                        },
                    )
                )
        self._continue_provisions(self._wave(actions, plan), plan)
        self._bind_governing()

    @staticmethod
    def _provision_key(receipt: dict[str, JsonValue]) -> tuple[str, str] | None:
        arguments, data = receipt.get("host_arguments"), receipt.get("data")
        source_id = _source_id(arguments) or _source_id(data)
        article = arguments.get("article") if isinstance(arguments, dict) else None
        if (
            receipt.get("tool") in {"read_provision", "read_named_provision"}
            and source_id is not None
            and isinstance(article, str)
        ):
            return source_id, article
        return None

    @staticmethod
    def _starts_provision(receipt: dict[str, JsonValue]) -> bool:
        arguments = receipt.get("host_arguments")
        start = arguments.get("start") if isinstance(arguments, dict) else None
        return start is None or (type(start) is int and start == 0)

    @staticmethod
    def _terminal_provision(receipt: dict[str, JsonValue]) -> bool:
        data = receipt.get("data")
        return (
            receipt.get("status") == "found"
            and bool(receipt.get("citations"))
            and not (
                isinstance(data, dict)
                and any(
                    data.get(flag)
                    for flag in ("has_more", "scan_truncated", "evidence_truncated")
                )
            )
        )

    def _continue_provisions(
        self, receipts: list[dict[str, JsonValue]], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        all_receipts = list(receipts)
        eligible = {
            key
            for receipt in receipts
            if (key := self._provision_key(receipt)) is not None
            and self._starts_provision(receipt)
            and receipt.get("citations")
        }
        expected: dict[tuple[str, str], int] = {}
        while receipts:
            actions: list[SourceAction] = []
            for receipt in receipts:
                arguments, data = receipt.get("host_arguments"), receipt.get("data")
                key = self._provision_key(receipt)
                source_id = _source_id(arguments) or _source_id(data)
                start = arguments.get("start") if isinstance(arguments, dict) else None
                prefix_known = key in eligible and (
                    self._starts_provision(receipt)
                    or (type(start) is int and expected.get(key) == start)
                )
                if (
                    key is not None
                    and prefix_known
                    and self._terminal_provision(receipt)
                ):
                    self._closed_provisions.add(key)
                elif key is not None and receipt.get("status") != "partial":
                    self._incomplete_reads.add(key[0])
                    self._incomplete_needs.update(
                        cast(list[str], receipt.get("need_ids", []))
                    )
                if (
                    receipt.get("status") != "partial"
                    or not isinstance(arguments, dict)
                    or not isinstance(data, dict)
                ):
                    continue
                source_id = arguments.get("source_id") or data.get("source_id")
                position = (
                    data.get("evidence_next_position")
                    if data.get("evidence_truncated")
                    else data.get("next_position")
                )
                if (
                    isinstance(source_id, str)
                    and isinstance(arguments.get("article"), str)
                    and prefix_known
                    and type(position) is int
                    and position > cast(int, arguments.get("start", -1))
                ):
                    assert key is not None
                    expected[key] = position
                    actions.append(
                        SourceAction(
                            need_ids=cast(list[str], receipt["need_ids"]),
                            tool="read_provision",
                            arguments={
                                "source_id": source_id,
                                "article": arguments["article"],
                                "start": position,
                            },
                        )
                    )
                else:
                    if isinstance(source_id, str):
                        self._incomplete_reads.add(source_id)
                    self._incomplete_needs.update(
                        cast(list[str], receipt.get("need_ids", []))
                    )
            receipts = self._wave(actions, plan)
            all_receipts.extend(receipts)
        return all_receipts

    def _continue_focused(
        self, receipts: list[dict[str, JsonValue]], plan: ResearchPlan
    ) -> list[dict[str, JsonValue]]:
        result = list(receipts)
        while receipts:
            actions = []
            for receipt in receipts:
                arguments, data = receipt.get("host_arguments"), receipt.get("data")
                source_id = _source_id(arguments)
                if not isinstance(arguments, dict) or source_id is None:
                    continue
                if receipt.get("status") == "partial" and isinstance(data, dict):
                    position = (
                        data.get("evidence_next_position")
                        if data.get("evidence_truncated")
                        else data.get("next_position")
                    )
                    if type(position) is int and position > cast(
                        int, arguments.get("start", -1)
                    ):
                        actions.append(
                            SourceAction(
                                need_ids=cast(list[str], receipt["need_ids"]),
                                tool="search_source_text",
                                arguments={**arguments, "start": position},
                            )
                        )
                    else:
                        self._incomplete_reads.add(source_id)
                elif receipt.get("status") not in {"found", "not_found"}:
                    self._incomplete_reads.add(source_id)
            receipts = self._wave(actions, plan)
            result.extend(receipts)
        return result

    def _candidate_article(self, citation: int) -> str | None:
        item = self.ledger.get(citation)
        if item is None:
            return None
        metadata = model_evidence_metadata(item.metadata)
        headings = metadata.get("heading_path")
        if isinstance(headings, list):
            for heading in headings:
                parsed = (
                    parse_regulatory_article_heading(heading)
                    if isinstance(heading, str)
                    else None
                )
                if parsed is not None:
                    return " ".join(
                        part for part in (parsed.qualifier, parsed.article_no) if part
                    )
        return (
            str(metadata["article_no"])
            if metadata.get("article_no") is not None
            else None
        )

    def _expand(self, edges: list[AuthorityDependency], plan: ResearchPlan) -> None:
        governing = self._wave(
            [
                SourceAction(
                    need_ids=edge.need_ids,
                    tool="read_named_provision",
                    arguments={
                        "source_name": edge.instrument_name,
                        "article": " ".join(
                            part for part in (edge.qualifier, edge.article) if part
                        ),
                    },
                )
                for edge in edges
                if not edge.governing_citations
            ],
            plan,
        )
        self._continue_provisions(governing, plan)
        self._open_ambiguous_governing_sources(governing, edges, plan)
        self._identify_sources(plan)
        self._read_own_governing(edges, plan)
        actions: list[SourceAction] = []
        query_edges: dict[str, str] = {}
        related_edges: set[str] = set()
        for edge in edges:
            if edge.governing_citations:
                related_edges.add(edge.edge_id)
                actions.append(
                    SourceAction(
                        need_ids=edge.need_ids,
                        tool="dependency_related_sources",
                        arguments={"edge_id": edge.edge_id},
                    )
                )
            query = " ".join(
                part
                for part in (
                    edge.instrument_number or edge.instrument_name,
                    edge.qualifier,
                    edge.article,
                    "iptal sınırlama yürürlük",
                )
                if part
            )
            query_edges[query] = edge.edge_id
            actions.append(
                SourceAction(
                    need_ids=edge.need_ids,
                    tool="search_corpus",
                    arguments={
                        "query": query,
                        "mode": "full_text",
                        "expand_query": False,
                        "coverage_item": ", ".join(edge.need_ids),
                        "evidence_target": "Original limiting or contrary authority for this identified provision",
                    },
                )
            )
        results = self._wave(actions, plan)
        pages = results
        while pages:
            continuation = []
            for receipt in pages:
                arguments, data = receipt.get("host_arguments"), receipt.get("data")
                offset = data.get("next_offset") if isinstance(data, dict) else None
                if (
                    receipt.get("tool") == "dependency_related_sources"
                    and isinstance(arguments, dict)
                    and isinstance(data, dict)
                    and data.get("has_more") is True
                    and type(offset) is int
                    and offset > cast(int, arguments.get("offset", -1))
                ):
                    continuation.append(
                        SourceAction(
                            need_ids=cast(list[str], receipt["need_ids"]),
                            tool="dependency_related_sources",
                            arguments={**arguments, "offset": offset},
                        )
                    )
            pages = self._wave(continuation, plan)
            results.extend(pages)
        # Search may be the first successful acquisition of a governing instrument.
        self._identify_sources(plan)
        self._read_own_governing(edges, plan)
        late_related = self._wave(
            [
                SourceAction(
                    need_ids=edge.need_ids,
                    tool="dependency_related_sources",
                    arguments={"edge_id": edge.edge_id},
                )
                for edge in edges
                if edge.governing_citations and edge.edge_id not in related_edges
            ],
            plan,
        )
        while late_related:
            results.extend(late_related)
            continuation = []
            for receipt in late_related:
                arguments, data = receipt.get("host_arguments"), receipt.get("data")
                offset = data.get("next_offset") if isinstance(data, dict) else None
                if (
                    isinstance(arguments, dict)
                    and isinstance(data, dict)
                    and data.get("has_more") is True
                    and type(offset) is int
                    and offset > cast(int, arguments.get("offset", -1))
                ):
                    continuation.append(
                        SourceAction(
                            need_ids=cast(list[str], receipt["need_ids"]),
                            tool="dependency_related_sources",
                            arguments={**arguments, "offset": offset},
                        )
                    )
            late_related = self._wave(continuation, plan)
        governing_sources = {
            edge.edge_id: {
                item.source_id
                for number in edge.governing_citations
                if (item := self.ledger.get(number)) is not None
            }
            for edge in edges
        }
        candidate_edges: dict[str, set[str]] = {}
        candidate_numbers: dict[str, set[int]] = {}
        title_only: set[str] = set()
        for receipt in results:
            arguments, data = receipt.get("host_arguments"), receipt.get("data")
            if not isinstance(arguments, dict):
                continue
            edge_id = arguments.get("edge_id") or query_edges.get(
                str(arguments.get("query", ""))
            )
            if not isinstance(edge_id, str) or edge_id not in self.edges:
                continue
            edge = self.edges[edge_id]
            own_sources = governing_sources.get(edge_id, set())
            if receipt.get("status") in {
                "error",
                "denied",
                "unavailable",
                "truncated",
                "cancelled",
                "partial",
            }:
                edge.discovery_gaps.append(
                    "Related authority discovery did not complete; absence is not established."
                )
            for number in cast(list[int], receipt.get("citations", [])):
                item = self.ledger.get(number)
                own_article = " ".join(
                    part for part in (edge.qualifier, edge.article) if part
                )
                duplicate_governing = (
                    item is not None
                    and item.source_id in own_sources
                    and self._candidate_article(number) == own_article
                    and (
                        model_evidence_metadata(item.metadata).get(
                            "article_closure_complete"
                        )
                        is True
                        or (item.source_id, own_article) in self._closed_provisions
                    )
                )
                if item is not None and not duplicate_governing:
                    candidate_edges.setdefault(item.source_id, set()).add(edge_id)
                    candidate_numbers.setdefault(item.source_id, set()).add(number)
            candidates = data.get("candidates") if isinstance(data, dict) else None
            if isinstance(candidates, list):
                for candidate in candidates:
                    source_id = _source_id(candidate)
                    if (
                        isinstance(candidate, dict)
                        and source_id is not None
                        and source_id not in own_sources
                    ):
                        candidate_edges.setdefault(source_id, set()).add(edge_id)
                        title_only.add(source_id)
        material_numbers = {
            source_id: set(numbers) for source_id, numbers in candidate_numbers.items()
        }
        # Title leads get their short original identity, then a focused passage
        # lookup. A generic regulation/statute hit never causes a full-file read.
        openings = self._wave(
            [
                SourceAction(
                    need_ids=sorted(
                        {
                            need
                            for edge_id in candidate_edges[source_id]
                            for need in self.edges[edge_id].need_ids
                        }
                    ),
                    tool="read_source_range",
                    arguments={"source_id": source_id, "start": 0, "limit": 3},
                )
                for source_id in sorted(title_only)
                if source_id not in self._identified_sources
            ],
            plan,
        )
        for receipt in openings:
            source_id = _source_id(receipt.get("host_arguments"))
            if source_id is not None:
                candidate_numbers.setdefault(source_id, set()).update(
                    cast(list[int], receipt.get("citations", []))
                )
        self._identify_sources(plan)
        focused = self._continue_focused(
            self._wave(
                [
                    SourceAction(
                        need_ids=self.edges[edge_id].need_ids,
                        tool="search_source_text",
                        arguments={
                            "source_id": source_id,
                            "pattern": self.edges[edge_id].instrument_number
                            or self.edges[edge_id].instrument_name,
                            "mode": "literal",
                        },
                    )
                    for source_id in sorted(title_only)
                    for edge_id in sorted(candidate_edges[source_id])
                    if self.verified_kinds.get(source_id)
                    != SourceKind.JUDICIAL_DECISION
                ],
                plan,
            ),
            plan,
        )
        for receipt in focused:
            source_id = _source_id(receipt.get("host_arguments"))
            if source_id is not None:
                candidate_numbers.setdefault(source_id, set()).update(
                    cast(list[int], receipt.get("citations", []))
                )
                material_numbers.setdefault(source_id, set()).update(
                    cast(list[int], receipt.get("citations", []))
                )
        article_actions = []
        for source_id, numbers in material_numbers.items():
            if self.verified_kinds.get(source_id) == SourceKind.JUDICIAL_DECISION:
                continue
            for number in numbers:
                item = self.ledger.get(number)
                article = self._candidate_article(number)
                if (
                    item is not None
                    and model_evidence_metadata(item.metadata).get(
                        "article_closure_complete"
                    )
                    is not True
                    and article is not None
                ):
                    article_actions.append(
                        SourceAction(
                            need_ids=sorted(
                                {
                                    need
                                    for edge_id in candidate_edges[source_id]
                                    for need in self.edges[edge_id].need_ids
                                }
                            ),
                            tool="read_provision",
                            arguments={"source_id": source_id, "article": article},
                        )
                    )
        article_receipts = self._continue_provisions(
            self._wave(article_actions, plan), plan
        )
        closed_articles = set(self._closed_provisions)
        for receipt in article_receipts:
            arguments = receipt.get("host_arguments")
            source_id = _source_id(arguments)
            if isinstance(arguments, dict) and source_id is not None:
                candidate_numbers.setdefault(source_id, set()).update(
                    cast(list[int], receipt.get("citations", []))
                )
        pending = {
            source_id: 0
            for source_id in candidate_edges
            if self.verified_kinds.get(source_id) == SourceKind.JUDICIAL_DECISION
        }
        complete_judicial: set[str] = set()
        while pending:
            body_receipts = self._wave(
                [
                    SourceAction(
                        need_ids=sorted(
                            {
                                need
                                for edge_id in candidate_edges[source_id]
                                for need in self.edges[edge_id].need_ids
                            }
                        ),
                        tool="read_source_range",
                        arguments={
                            "source_id": source_id,
                            "start": position,
                            "limit": 100,
                        },
                    )
                    for source_id, position in pending.items()
                ],
                plan,
            )
            next_pending = {}
            for receipt in body_receipts:
                arguments, data = receipt.get("host_arguments"), receipt.get("data")
                source_id = _source_id(arguments)
                if not isinstance(arguments, dict) or source_id is None:
                    continue
                candidate_numbers.setdefault(source_id, set()).update(
                    cast(list[int], receipt.get("citations", []))
                )
                position = data.get("next_position") if isinstance(data, dict) else None
                if (
                    isinstance(data, dict)
                    and data.get("has_more") is True
                    and type(position) is int
                    and position > pending[source_id]
                ):
                    next_pending[source_id] = position
                elif receipt.get("status") != "found":
                    for edge_id in candidate_edges[source_id]:
                        self.edges[edge_id].incomplete_source_ids.append(source_id)
                else:
                    complete_judicial.add(source_id)
            if not body_receipts:
                break
            pending = next_pending
        self._bind_governing()
        for source_id, edge_ids in candidate_edges.items():
            numbers = candidate_numbers.get(source_id, set())
            for edge_id in edge_ids:
                edge = self.edges[edge_id]
                edge.candidate_citations = sorted(
                    set(edge.candidate_citations) | numbers
                )
                if self.verified_kinds.get(source_id) == SourceKind.JUDICIAL_DECISION:
                    edge.judicial_source_ids = sorted(
                        set(edge.judicial_source_ids) | {source_id}
                    )
                incomplete_body = (
                    self.verified_kinds.get(source_id) == SourceKind.JUDICIAL_DECISION
                    and source_id not in complete_judicial
                )
                incomplete_articles = self.verified_kinds.get(
                    source_id
                ) != SourceKind.JUDICIAL_DECISION and (
                    not material_numbers.get(source_id)
                    or any(
                        model_evidence_metadata(item.metadata).get(
                            "article_closure_complete"
                        )
                        is not True
                        and (source_id, self._candidate_article(number))
                        not in closed_articles
                        for number in material_numbers.get(source_id, set())
                        if (item := self.ledger.get(number)) is not None
                    )
                )
                if (
                    not numbers
                    or incomplete_body
                    or incomplete_articles
                    or source_id in self._incomplete_reads
                ):
                    edge.incomplete_source_ids = sorted(
                        set(edge.incomplete_source_ids) | {source_id}
                    )
        for edge in edges:
            if not edge.governing_citations:
                edge.discovery_gaps.append(
                    "Own governing original could not be obtained."
                )
            article = " ".join(part for part in (edge.qualifier, edge.article) if part)
            if set(edge.need_ids) & self._incomplete_needs or any(
                item.source_id in self._incomplete_reads
                or (
                    model_evidence_metadata(item.metadata).get(
                        "article_closure_complete"
                    )
                    is not True
                    and (item.source_id, article) not in self._closed_provisions
                )
                for number in edge.governing_citations
                if (item := self.ledger.get(number)) is not None
            ):
                edge.discovery_gaps.append(
                    "A required original continuation did not complete; its unread legal interaction remains a source gap."
                )
            edge.discovery_gaps = list(dict.fromkeys(edge.discovery_gaps))
