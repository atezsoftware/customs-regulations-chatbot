"""Material-frontier closure never turns unrelated regulations into full-file reads."""

import sys
from unittest.mock import Mock
from uuid import uuid4

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
from onyx.legal_composite.models import WorkflowPolicy
from onyx.supersearch.acquisition import SupersearchAcquirer
from onyx.supersearch.dependencies import SupersearchDependencyExpander
from tests.unit.onyx.supersearch.test_engine import plan


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
