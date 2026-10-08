"""Material-frontier closure never turns unrelated regulations into full-file reads."""

import sys
from unittest.mock import Mock
from uuid import uuid4

import pytest

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
from onyx.legal_composite.models import WorkflowPolicy
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
    assert all(
        not edge.discovery_gaps and not edge.candidate_citations for edge in edges
    )
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
