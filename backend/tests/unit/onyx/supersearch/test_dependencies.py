"""Material-frontier closure never turns unrelated regulations into full-file reads."""

import sys
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    OutcomeStatus,
    RunContext,
    SharedBudget,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.legal_composite_sources import SourceKind
from onyx.legal_composite.models import AuthorityDependency, WorkflowPolicy
from onyx.supersearch.acquisition import SupersearchAcquirer
from onyx.supersearch.dependencies import (
    SupersearchDependencyExpander,
    _canonical_opening_kind,
)
from tests.unit.onyx.supersearch.test_engine import plan


def canonical_law_chunk(
    source: CorpusSource, article: str | None = None
) -> CorpusChunk:
    root = "4458 SAYILI GÜMRÜK KANUNU"
    return CorpusChunk(
        "original-" + (article or "opening"),
        source.id,
        f"**MADDE {article}-** Özgün madde metni."
        if article
        else "######## Kabul Tarihi: 27.10.1999 -\n*04.11.1999 tarihli, 23866 sayılı R.G. ile yayımlanmıştır*",
        int(article) if article else 0,
        int(article) if article else 0,
        (root, "MADDE " + article) if article else (root,),
        {
            "document_type": "kanun",
            "title": source.name,
            "read_as_of_date": "2026-10-08",
            "document_date": "1999-10-27",
            "legal_dates": ["1999-10-27", "1999-11-04"],
            "article_closure_complete": article is not None,
        },
        None,
        None,
        "active",
    )


def test_corpus_opening_omitted_body_title_binds_all_governing_originals() -> None:
    source = CorpusSource(
        uuid4(),
        "İhracat Mevzuatı/Kanun, Kararname, Yönetmelik, Tebliğ, Genelge/gumruk_kanunu_.md",
        "canonical-law",
    )
    articles = {
        article: canonical_law_chunk(source, article)
        for article in ("142", "143", "167")
    }
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    originals = [evidence_for_chunk(source, chunk) for chunk in articles.values()]
    for item in originals:
        item.question_ids = ["clock"]
    ledger.add(originals, context)
    calls: list[str] = []

    def opening(args, _context):
        calls.append("read_source_range")
        assert args == {"source_id": str(source.id), "start": 0, "limit": 3}
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="First three canonical original chunks; source continues",
            data={"has_more": True, "next_position": 3},
            evidence=[
                evidence_for_chunk(source, canonical_law_chunk(source)),
                evidence_for_chunk(source, canonical_law_chunk(source, "1")),
                evidence_for_chunk(source, canonical_law_chunk(source, "2")),
            ],
        )

    def search(_args, _context):
        calls.append("search_corpus")
        # Matching the governing file never makes it an unrelated candidate.
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical matching governing article",
            evidence=[evidence_for_chunk(source, articles["143"])],
        )

    broker = Mock()
    broker.chunk.side_effect = lambda _source_id, chunk_id, _context: (
        source,
        next(chunk for chunk in articles.values() if chunk.id == chunk_id),
    )
    broker.related_catalog_sources.return_value = [], False
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                handler=handler,
            )
            for name, handler in (
                ("read_source_range", opening),
                ("search_corpus", search),
            )
        ]
    )
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    edges = expander.expand(plan(), frontier={1, 2, 3})
    assert expander.verified_kinds[str(source.id)] == SourceKind.STATUTE
    assert {edge.article: edge.governing_citations for edge in edges} == {
        "142": [1],
        "143": [2],
        "167": [3],
    }
    assert all(edge.instrument_number == "4458" for edge in edges)
    assert all(not edge.discovery_gaps for edge in edges)
    assert {edge.article: edge.candidate_citations for edge in edges} == {
        "142": [2],
        "143": [],
        "167": [2],
    }
    assert calls.count("read_source_range") == 1
    assert calls.count("search_corpus") == 3
    assert broker.related_catalog_sources.call_count == 3
    for index, original in enumerate(originals, 1):
        actual = ledger.get(index)
        assert actual is not None
        assert (actual.text, actual.text_hash, actual.metadata) == (
            original.text,
            original.text_hash,
            original.metadata,
        )
        assert actual.metadata["read_as_of_date"] == "2026-10-08"
        assert actual.metadata["validity_start"] is None
        assert actual.metadata["version_unknown"] is True


@pytest.mark.parametrize("discover_late", [False, True])
def test_ambiguous_primary_recovers_from_opening_or_late_search(
    discover_late: bool,
) -> None:
    law = CorpusSource(uuid4(), "İhracat Mevzuatı/gumruk_kanunu_.md", "law")
    circular = CorpusSource(uuid4(), "gumruk_genel_tebligi.md", "secondary")
    quoted = CorpusSource(
        uuid4(),
        "bkk_2009-15481_gumruk_kanununun_bazi_maddelerinin_uygulanmasi_hakkinda_karar.md",
        "quoted",
    )
    law_chunks = {
        article: canonical_law_chunk(law, article) for article in ("142", "143")
    }
    circular_chunk = CorpusChunk(
        "circular-reference",
        circular.id,
        "Gümrük Kanununun 142 nci maddesi uygulanır.\n4458 sayılı Gümrük Kanununun 143 üncü maddesi uygulanır.",
        7,
        7,
        ("GÜMRÜK GENEL TEBLİĞİ", "MADDE 3"),
        {"document_type": "kanun"},
        None,
        None,
        "active",
    )
    circular_opening = CorpusChunk(
        "circular-opening",
        circular.id,
        "GÜMRÜK GENEL TEBLİĞİ\nResmi belge",
        0,
        0,
        ("GÜMRÜK GENEL TEBLİĞİ",),
        {},
        None,
        None,
        "active",
    )
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    reference = evidence_for_chunk(circular, circular_chunk)
    reference.question_ids = ["clock"]
    ledger.add([reference], context)
    calls: list[tuple[str, dict]] = []

    def named(args, _context):
        calls.append(("read_named_provision", dict(args)))
        candidates = [quoted] if discover_late else [law, quoted]
        return ToolOutcome(
            status=OutcomeStatus.AMBIGUOUS,
            summary="Same law appears in secondary source titles",
            data={
                "sources": [
                    {"source_id": str(source.id), "name": source.name}
                    for source in candidates
                ]
            },
        )

    def opening(args, _context):
        calls.append(("read_source_range", dict(args)))
        assert args["source_id"] != str(quoted.id)
        if args["source_id"] == str(circular.id):
            source, chunks = circular, [circular_opening]
        else:
            source, chunks = (
                law,
                [
                    canonical_law_chunk(law),
                    canonical_law_chunk(law, "1"),
                    canonical_law_chunk(law, "2"),
                ],
            )
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL,
            summary="Canonical opening only",
            data={"has_more": True, "next_position": 3},
            evidence=[evidence_for_chunk(source, chunk) for chunk in chunks],
        )

    def provision(args, _context):
        calls.append(("read_provision", dict(args)))
        assert args["source_id"] == str(law.id)
        original = evidence_for_chunk(law, law_chunks[args["article"]])
        original.metadata["article_closure_complete"] = True
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Own governing original",
            evidence=[original],
        )

    def search(args, _context):
        calls.append(("search_corpus", dict(args)))
        original = evidence_for_chunk(law, law_chunks["142"])
        original.metadata["article_closure_complete"] = True
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical article 142 found after ambiguous resolution",
            evidence=[original],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                handler=handler,
            )
            for name, handler in (
                ("read_named_provision", named),
                ("read_source_range", opening),
                ("read_provision", provision),
                ("search_corpus", search),
            )
        ]
    )
    broker = Mock()
    broker.chunk.side_effect = lambda _source_id, chunk_id, _context: (
        law,
        next(chunk for chunk in law_chunks.values() if chunk.id == chunk_id),
    )
    broker.related_catalog_sources.return_value = [], False
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    edges = expander.expand(plan(), frontier={1})
    assert {edge.article for edge in edges} == {"142", "143"}
    for edge in edges:
        assert edge.governing_citations
        for number in edge.governing_citations:
            original = ledger.get(number)
            assert original is not None and original.source_id == str(law.id)
        assert not edge.discovery_gaps
        if edge.article == "142":
            assert not edge.candidate_citations
        else:
            assert edge.candidate_citations
            assert all(
                expander._candidate_article(number) == "142"
                for number in edge.candidate_citations
            )
    assert broker.related_catalog_sources.call_count == 2
    law_openings = [
        args
        for name, args in calls
        if name == "read_source_range" and args["source_id"] == str(law.id)
    ]
    assert law_openings == [{"source_id": str(law.id), "start": 0, "limit": 3}]
    direct_reads = [args["article"] for name, args in calls if name == "read_provision"]
    assert sorted(direct_reads) == (["143"] if discover_late else ["142", "143"])
    names = [name for name, _ in calls]
    if not discover_late:
        assert max(
            index for index, name in enumerate(names) if name == "read_provision"
        ) < names.index("search_corpus")


@pytest.mark.parametrize(
    "unresolved", ["multiple_primaries", "wrong_number", "secondary_root"]
)
def test_primary_recovery_keeps_conflicting_original_identity_unresolved(
    unresolved: str,
) -> None:
    sources = [CorpusSource(uuid4(), "gumruk_kanunu_.md", "law")]
    if unresolved == "multiple_primaries":
        sources.append(CorpusSource(uuid4(), "gumruk_kanunu_.md", "another-law"))
    context = RunContext()
    ledger = EvidenceLedger()

    def chunk(source):
        original = canonical_law_chunk(source)
        if unresolved == "secondary_root":
            return CorpusChunk(
                original.id,
                source.id,
                original.text,
                0,
                0,
                (
                    "4458 SAYILI GÜMRÜK KANUNUNUN BAZI MADDELERİNİN UYGULANMASI HAKKINDA KARAR",
                ),
                original.metadata,
                None,
                None,
                "active",
            )
        return original

    ledger.add(
        [evidence_for_chunk(source, chunk(source)) for source in sources], context
    )

    def opening(args, _context):
        source = next(
            source for source in sources if str(source.id) == args["source_id"]
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical first original",
            evidence=[evidence_for_chunk(source, chunk(source))],
        )

    # No provision capability: accidental promotion would fail this test immediately.
    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_source_range",
                description="opening",
                parameters={"type": "object"},
                handler=opening,
            )
        ]
    )
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=Mock(),
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    edge = AuthorityDependency(
        edge_id="unresolved",
        need_ids=["clock"],
        instrument_name="Gümrük Kanunu",
        instrument_number="8917" if unresolved == "wrong_number" else "4458",
        article="143",
        qualifier=None,
        origins=[],
    )
    expander.edges[edge.edge_id] = edge
    expander._identify_sources(plan(), frontier=set(ledger.citation_numbers()))
    expander._read_own_governing([edge], plan())
    assert not edge.governing_citations
    assert all(receipt["tool"] == "read_source_range" for receipt in expander.receipts)


@pytest.mark.parametrize("partial_continuation", [False, True])
@pytest.mark.parametrize("same_law", [False, True])
def test_other_articles_of_governing_laws_remain_limiting_candidates(
    partial_continuation: bool,
    same_law: bool,
) -> None:
    first = CorpusSource(uuid4(), "8917 sayılı Faaliyet Kanunu.md", "first-law")
    second = CorpusSource(uuid4(), "5521 sayılı İzin Kanunu.md", "second-law")
    roots = {
        first.id: "8917 sayılı Faaliyet Kanunu",
        second.id: "5521 sayılı İzin Kanunu",
    }

    def chunk(source, article):
        return CorpusChunk(
            str(source.id) + "-" + (article or "opening"),
            source.id,
            "Kabul Tarihi: 01.01.2020"
            if article is None
            else "MADDE " + article + " — Özgün hüküm.",
            int(article) if article else 0,
            int(article) if article else 0,
            (roots[source.id], "MADDE " + article) if article else (roots[source.id],),
            {"document_type": "kanun"},
            None,
            None,
            "active",
        )

    first_original, second_original = chunk(first, "1"), chunk(second, "2")
    candidate_source = first if same_law else second
    candidate = chunk(candidate_source, "3")
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    governing = [
        evidence_for_chunk(first, first_original),
        evidence_for_chunk(second, second_original),
    ]
    for item in governing:
        item.question_ids = ["clock"]
        item.metadata["article_closure_complete"] = True
    ledger.add(governing, context)
    provision_calls = []

    def opening(args, _context):
        assert args.get("start", 0) == 0 and args["limit"] == 3
        source = first if args["source_id"] == str(first.id) else second
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical source opening",
            evidence=[evidence_for_chunk(source, chunk(source, None))],
        )

    def search(args, _context):
        is_first = args["query"].startswith("8917 ")
        original = (
            evidence_for_chunk(candidate_source, candidate)
            if is_first
            else evidence_for_chunk(second, second_original)
        )
        original.metadata["article_closure_complete"] = not is_first
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Different article of a law governing another need",
            evidence=[original],
        )

    def provision(args, _context):
        provision_calls.append(dict(args))
        assert args == {"source_id": str(candidate_source.id), "article": "3"}
        original = evidence_for_chunk(candidate_source, candidate)
        original.metadata["article_closure_complete"] = not partial_continuation
        return ToolOutcome(
            status=OutcomeStatus.PARTIAL
            if partial_continuation
            else OutcomeStatus.FOUND,
            summary="Required candidate article continuation",
            data={"evidence_truncated": True} if partial_continuation else {},
            evidence=[original],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                handler=handler,
            )
            for name, handler in (
                ("read_source_range", opening),
                ("search_corpus", search),
                ("read_provision", provision),
            )
        ]
    )
    broker = Mock()
    broker.chunk.side_effect = lambda source_id, _chunk_id, _context: (
        (first, first_original)
        if source_id == str(first.id)
        else (second, second_original)
    )
    broker.related_catalog_sources.return_value = [], False
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    edges = expander.expand(plan(), frontier={1, 2})
    first_edge = next(edge for edge in edges if edge.instrument_number == "8917")
    second_edge = next(edge for edge in edges if edge.instrument_number == "5521")
    assert first_edge.governing_citations == [1]
    assert second_edge.governing_citations == [2]
    assert provision_calls == [{"source_id": str(candidate_source.id), "article": "3"}]
    assert first_edge.candidate_citations
    for number in first_edge.candidate_citations:
        original = ledger.get(number)
        assert original is not None and original.source_id == str(candidate_source.id)
    assert not second_edge.candidate_citations
    assert (
        str(candidate_source.id) in first_edge.incomplete_source_ids
    ) is partial_continuation


@pytest.mark.parametrize(
    "outcome_status",
    [
        OutcomeStatus.FOUND,
        OutcomeStatus.PARTIAL,
        OutcomeStatus.ERROR,
        OutcomeStatus.DENIED,
        OutcomeStatus.NOT_FOUND,
        OutcomeStatus.UNAVAILABLE,
    ],
)
@pytest.mark.parametrize("metadata_missing", [False, True])
def test_bound_incomplete_governing_article_is_read_without_search_repeating_it(
    outcome_status: OutcomeStatus,
    metadata_missing: bool,
) -> None:
    law = CorpusSource(uuid4(), "gumruk_kanunu_.md", "law")
    body = canonical_law_chunk(law, "27")
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    original = evidence_for_chunk(law, body)
    original.question_ids = ["clock"]
    if metadata_missing:
        canonical = original.metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        canonical.pop("article_closure_complete", None)
    else:
        original.metadata["article_closure_complete"] = False
    ledger.add([original], context)
    calls = []

    def opening(_args, _context):
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Canonical law opening",
            evidence=[evidence_for_chunk(law, canonical_law_chunk(law))],
        )

    def provision(args, _context):
        calls.append(dict(args))
        assert args == {"source_id": str(law.id), "article": "27"}
        return ToolOutcome(
            status=outcome_status,
            summary="Governing provision continuation",
            data={"evidence_truncated": True}
            if outcome_status == OutcomeStatus.PARTIAL
            else {},
            evidence=[evidence_for_chunk(law, body)]
            if outcome_status in {OutcomeStatus.FOUND, OutcomeStatus.PARTIAL}
            else [],
        )

    def search(_args, _context):
        return ToolOutcome(
            status=OutcomeStatus.NOT_FOUND,
            summary="No additional limiting original found",
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                handler=handler,
            )
            for name, handler in (
                ("read_source_range", opening),
                ("read_provision", provision),
                ("search_corpus", search),
            )
        ]
    )
    broker = Mock()
    broker.chunk.return_value = law, body
    broker.related_catalog_sources.return_value = [], False
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    edges = expander.expand(plan(), frontier={1})
    assert len(edges) == 1 and edges[0].governing_citations == [1]
    assert calls == [{"source_id": str(law.id), "article": "27"}]
    assert bool(edges[0].discovery_gaps) is (outcome_status != OutcomeStatus.FOUND)
    assert ((str(law.id), "27") in expander._closed_provisions) is (
        outcome_status == OutcomeStatus.FOUND
    )


@pytest.mark.parametrize(
    "read_shape",
    [
        "initial_complete",
        "valid_chain",
        "tail_only",
        "midstream_partial",
        "clipped_terminal",
    ],
)
def test_provision_closure_requires_the_initial_prefix_and_advancing_chain(
    read_shape: str,
) -> None:
    source = CorpusSource(uuid4(), "gumruk_kanunu_.md", "law")
    original = evidence_for_chunk(source, canonical_law_chunk(source, "27"))
    context = RunContext()
    ledger = EvidenceLedger()
    ledger.add([original], context)
    continued = []

    def continuation(args, _context):
        continued.append(dict(args))
        assert args == {"source_id": str(source.id), "article": "27", "start": 10}
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original structural boundary reached",
            evidence=[original],
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name="read_provision",
                description="provision",
                parameters={"type": "object"},
                handler=continuation,
            )
        ]
    )
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=Mock(),
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    arguments: dict[str, JsonValue] = {"source_id": str(source.id), "article": "27"}
    data = {}
    status = "found"
    if read_shape == "valid_chain":
        status, data = (
            "partial",
            {"evidence_truncated": True, "evidence_next_position": 10},
        )
    elif read_shape == "tail_only":
        arguments["start"] = 10
    elif read_shape == "midstream_partial":
        arguments["start"] = 10
        status, data = "partial", {"next_position": 20}
    elif read_shape == "clipped_terminal":
        data = {"scan_truncated": True}
    expander._continue_provisions(
        [
            {
                "tool": "read_provision",
                "host_arguments": arguments,
                "status": status,
                "need_ids": ["clock"],
                "citations": [1],
                "data": data,
            }
        ],
        plan(),
    )
    complete = read_shape in {"initial_complete", "valid_chain"}
    assert ((str(source.id), "27") in expander._closed_provisions) is complete
    assert (str(source.id) in expander._incomplete_reads) is not complete
    assert bool(continued) is (read_shape == "valid_chain")


@pytest.mark.parametrize(
    "invalid_proof",
    [
        "label_only",
        "foreign_chunk",
        "projected_heading",
        "quoted_other_source",
        "conflicting_roots",
        "conflicting_number",
        "late_chunk",
        "untrusted",
    ],
)
def test_canonical_root_identity_rejects_unverified_or_conflicting_proof(
    invalid_proof: str,
) -> None:
    source = CorpusSource(uuid4(), "gumruk_kanunu_.md", "canonical-law")
    item = evidence_for_chunk(source, canonical_law_chunk(source))
    openings = [item]
    if invalid_proof == "label_only":
        item.metadata["heading_path"] = []
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_heading_path"] = []
    elif invalid_proof == "foreign_chunk":
        item.source_id = str(uuid4())
    elif invalid_proof == "projected_heading":
        assert item.search_doc is not None
        item.search_doc.metadata["regulatory_heading_path"] = [
            "4458 SAYILI GÜMRÜK KANUNU",
            "different projection",
        ]
    elif invalid_proof == "quoted_other_source":
        source = CorpusSource(source.id, "gumruk_yonetmeligi_.md", source.file_id)
    elif invalid_proof == "conflicting_roots":
        other = evidence_for_chunk(source, canonical_law_chunk(source, "1"))
        other.metadata["heading_path"] = ["8917 SAYILI FAALİYET KANUNU", "MADDE 1"]
        assert other.search_doc is not None
        other.search_doc.metadata["regulatory_heading_path"] = other.metadata[
            "heading_path"
        ]
        openings.append(other)
    elif invalid_proof == "conflicting_number":
        source = CorpusSource(source.id, "8917 sayılı Gümrük Kanunu.md", source.file_id)
    elif invalid_proof == "late_chunk":
        item.metadata["position"] = 100
    else:
        item.metadata["untrusted"] = True
    assert _canonical_opening_kind(source, openings) == SourceKind.UNKNOWN


def test_direct_statute_seeds_own_anchor_and_candidates_never_crawl_full_regulations() -> (
    None
):
    law = CorpusSource(uuid4(), "8917 sayılı Faaliyet Kanunu.md", "law")
    regulation = CorpusSource(uuid4(), "Faaliyet Yönetmeliği.md", "regulation")
    law_body = CorpusChunk(
        "law-body",
        law.id,
        "MADDE 27 — Yetkili idareye başvuru gerekir.",
        40,
        40,
        ("8917 sayılı Faaliyet Kanunu", "MADDE 27"),
        {"document_type": "kanun", "title": "8917 sayılı Faaliyet Kanunu"},
        None,
        None,
        "active",
    )
    law_opening = CorpusChunk(
        "law-opening",
        law.id,
        "FAALİYET KANUNU\nKanun Numarası: 8917",
        0,
        0,
        ("8917 sayılı Faaliyet Kanunu",),
        {},
        None,
        None,
        "active",
    )
    regulation_body = CorpusChunk(
        "reg-body",
        regulation.id,
        "Faaliyet Kanununun 27 nci maddesinin uygulanmasında başvuru belgesi gerekir.",
        30,
        30,
        ("Faaliyet Yönetmeliği", "MADDE 5"),
        {"document_type": "yönetmelik"},
        None,
        None,
        "active",
    )
    regulation_opening = CorpusChunk(
        "reg-opening",
        regulation.id,
        "FAALİYET YÖNETMELİĞİ\nResmi belge",
        0,
        0,
        ("Faaliyet Yönetmeliği",),
        {},
        None,
        None,
        "active",
    )
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    governing = evidence_for_chunk(law, law_body)
    governing.question_ids = ["clock"]
    governing.metadata["article_closure_complete"] = True
    ledger.add([governing], context)
    calls: list[tuple[str, dict]] = []
    broker = Mock()
    broker.chunk.return_value = law, law_body
    broker.related_catalog_sources.return_value = [], False

    def opening(args, _context):
        calls.append(("read_source_range", dict(args)))
        assert args.get("start", 0) == 0 and args["limit"] == 3
        source, chunk = (
            (law, law_opening)
            if args["source_id"] == str(law.id)
            else (regulation, regulation_opening)
        )
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original identity",
            evidence=[evidence_for_chunk(source, chunk)],
        )

    def search(args, _context):
        calls.append(("search_corpus", dict(args)))
        item = evidence_for_chunk(regulation, regulation_body)
        item.metadata["article_closure_complete"] = True
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Complete matching article",
            evidence=[item],
        )

    specs = [
        ToolSpec(
            name=name, description=name, parameters={"type": "object"}, handler=handler
        )
        for name, handler in (("read_source_range", opening), ("search_corpus", search))
    ]
    registry = CapabilityRegistry(specs)
    acquirer = SupersearchAcquirer(
        registry,
        context,
        ledger,
        WorkflowPolicy(
            timeout_seconds=float("inf"),
            max_tools=sys.maxsize,
            max_search_calls=sys.maxsize,
        ),
    )
    expander = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    edges = expander.expand(plan(), frontier={1})
    assert len(edges) == 1 and edges[0].article == "27"
    assert edges[0].governing_citations == [1]
    assert edges[0].candidate_citations and not edges[0].incomplete_source_ids
    assert len([call for call in calls if call[0] == "search_corpus"]) == 1
    assert broker.related_catalog_sources.call_count == 1
    assert all(
        args["limit"] == 3 for name, args in calls if name == "read_source_range"
    )
    # A candidate regulation's incidental reference does not create another edge.
    assert len(expander.expand(plan(), frontier={1})) == 1


def test_partial_governing_and_focused_reads_without_cursor_remain_open() -> None:
    context = RunContext()
    ledger = EvidenceLedger()
    registry = CapabilityRegistry([])
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=Mock(),
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )
    expander._continue_provisions(
        [
            {
                "tool": "read_named_provision",
                "status": "partial",
                "need_ids": ["clock"],
                "host_arguments": {"source_name": "Faaliyet Kanunu", "article": "27"},
                "data": {"source_id": "source", "evidence_truncated": True},
            }
        ],
        plan(),
    )
    assert (
        "source" in expander._incomplete_reads and "clock" in expander._incomplete_needs
    )
    expander._continue_focused(
        [
            {
                "tool": "search_source_text",
                "status": "partial",
                "need_ids": ["clock"],
                "host_arguments": {"source_id": "candidate", "pattern": "8917"},
                "data": {"scan_truncated": True},
            }
        ],
        plan(),
    )
    assert "candidate" in expander._incomplete_reads


def test_ambiguous_authenticated_primary_keeps_bound_partial_article_open() -> None:
    first = CorpusSource(uuid4(), "gumruk_kanunu_.md", "first-law")
    second = CorpusSource(uuid4(), "gumruk_kanunu_.md", "second-law")
    body = canonical_law_chunk(first, "27")
    openings = {
        str(source.id): (source, canonical_law_chunk(source))
        for source in (first, second)
    }
    context = RunContext(
        timeout_seconds=float("inf"), budget=SharedBudget(unlimited_execution=True)
    )
    ledger = EvidenceLedger()
    governing = evidence_for_chunk(first, body)
    governing.question_ids = ["clock"]
    governing.metadata["article_closure_complete"] = False
    ledger.add(
        [governing, evidence_for_chunk(second, openings[str(second.id)][1])],
        context,
    )

    def opening(arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        source_id = arguments["source_id"]
        assert isinstance(source_id, str)
        assert arguments == {"source_id": source_id, "start": 0, "limit": 3}
        source, chunk = openings[source_id]
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Authenticated canonical law opening",
            evidence=[evidence_for_chunk(source, chunk)],
        )

    def search(_arguments: dict[str, JsonValue], _context: RunContext) -> ToolOutcome:
        return ToolOutcome(
            status=OutcomeStatus.NOT_FOUND,
            summary="No additional limiting original found",
        )

    registry = CapabilityRegistry(
        [
            ToolSpec(
                name=name,
                description=name,
                parameters={"type": "object"},
                handler=handler,
            )
            for name, handler in (
                ("read_source_range", opening),
                ("search_corpus", search),
            )
        ]
    )
    canonical_chunks = {
        (source_id, chunk.id): (source, chunk)
        for source_id, (source, chunk) in openings.items()
    }
    canonical_chunks[str(first.id), body.id] = first, body

    def canonical_chunk(
        source_id: str, chunk_id: str, _context: RunContext
    ) -> tuple[CorpusSource, CorpusChunk]:
        return canonical_chunks[source_id, chunk_id]

    broker = Mock()
    broker.chunk.side_effect = canonical_chunk
    broker.related_catalog_sources.return_value = [], False
    acquirer = SupersearchAcquirer(registry, context, ledger, WorkflowPolicy())
    expander = SupersearchDependencyExpander(
        broker=broker,
        acquirer=acquirer,
        ledger=ledger,
        context=context,
        source_kinds={},
    )

    edges = expander.expand(plan(), frontier={1})

    assert len(edges) == 1 and edges[0].governing_citations == [1]
    assert expander.verified_kinds == {
        str(first.id): SourceKind.STATUTE,
        str(second.id): SourceKind.STATUTE,
    }
    assert not expander._closed_provisions
    assert not expander._incomplete_reads and not expander._incomplete_needs
    assert edges[0].discovery_gaps == [
        "A required original continuation did not complete; its unread legal interaction remains a source gap."
    ]
    assert all(receipt["tool"] != "read_provision" for receipt in expander.receipts)
