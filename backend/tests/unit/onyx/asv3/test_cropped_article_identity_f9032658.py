"""Captured canonical headings and ordered siblings from the f9032658 incident."""

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID

import pytest

from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.corpus_tools import build_corpus_specs, evidence_for_chunk
from onyx.asv3.cropped_article_bindings import (
    CroppedArticleBindings,
    capture_cropped_dispatched_read,
    cropped_composed_read_capture,
    cropped_identity_ledger,
    cropped_named_authority_gap,
    ordered_source_read_data,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.harness import Harness
from onyx.asv3.models import (
    CapabilityCall,
    Decision,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolReceipt,
)
from onyx.asv3.registry import CapabilityRegistry
from onyx.asv3.sandbox import build_sandbox_specs
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker

_FIXTURE = (
    Path(__file__).with_name("fixtures") / "cropped_article_identity_f9032658.json"
)
_OPTIONS = {
    "strict_reference_boundaries": True,
    "syntactic_reference_binding": True,
    "strict_numeric_paragraphs": True,
}


def _items(*, inserted: bool) -> list[EvidenceItem]:
    rows = json.loads(_FIXTURE.read_text())["rows"]
    result = []
    for row in rows[3:] if inserted else rows[:3]:
        assert hashlib.sha256(row["text"].encode()).hexdigest() == row["text_hash"]
        result.append(
            EvidenceItem(
                source_id=row["source_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                text_hash=row["text_hash"],
                metadata={
                    "position": row["position"],
                    "canonical_metadata": row["metadata"],
                },
                search_doc=SearchDoc(
                    document_id=row["source_id"],
                    chunk_ind=row["position"],
                    semantic_identifier=row["metadata"]["title"],
                    blurb=row["text"],
                    source_type=DocumentSource.FILE,
                    boost=0,
                    hidden=False,
                    metadata={"regulatory_chunk_id": row["chunk_id"]},
                    match_highlights=[],
                ),
            )
        )
    return result


def _setup(
    items: list[EvidenceItem], *, follow: bool = False
) -> tuple[RunContext, EvidenceLedger, CroppedArticleBindings, ToolReceipt]:
    context = RunContext(
        run_id="captured-f9032658",
        scope={"owner": "captured-owner", "corpus": 15},
        services={"asv3_exp_cropped": True},
    )
    state = CroppedArticleBindings(context, "Captured incident question")
    context.services["cropped_article_bindings"] = state
    ledger = EvidenceLedger()
    numbers = ledger.add(items, context)
    call = CapabilityCall(
        name="follow_reference" if follow else "read_provision",
        arguments=(
            {
                "source_id": "6af208e4-f1ee-4af0-afc1-9f87f17b1c79",
                "chunk_id": "rc_21c0100ad3ba6f63b843acbc57065695561750de",
                "target_source_id": items[0].source_id,
                "article": "179",
            }
            if follow
            else {"source_id": items[0].source_id, "article": "191"}
        ),
    )
    receipt = ToolReceipt(
        call=call,
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Captured canonical ordered provision read",
            evidence=items,
            data=ordered_source_read_data(items, context),
        ),
        evidence_ids=numbers,
        elapsed_seconds=0,
    )
    return context, ledger, state, receipt


def _record(
    current: RunContext,
    ledger: EvidenceLedger,
    state: CroppedArticleBindings,
    receipt: ToolReceipt,
    *,
    composed: bool,
) -> int:
    if not composed:
        return state.record_receipt(receipt, current, ledger)
    parent = CapabilityCall(name="compose_tool_calls", arguments={"steps": []})
    with cropped_composed_read_capture(parent, current):
        capture_cropped_dispatched_read(receipt.call, receipt.outcome, current)
    assert state.export()["reads"] == []
    committed = ToolReceipt(
        call=parent,
        outcome=ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Executed and committed composition",
            evidence=receipt.outcome.evidence,
        ),
        evidence_ids=receipt.evidence_ids,
        elapsed_seconds=0,
    )
    return state.record_receipt(committed, current, ledger)


@pytest.mark.parametrize("composed", [False, True])
def test_captured_follow_reference_retains_179_paragraph_identity(
    composed: bool,
) -> None:
    current, ledger, state, receipt = _setup(_items(inserted=False), follow=True)
    before = copy.deepcopy(ledger.export())
    answers = [
        "5237 sayılı Türk Ceza Kanunu m.179/2 uygulanır [2].",
        "5237 sayılı Türk Ceza Kanunu m.179/3 uygulanır [3].",
    ]
    for answer in answers:
        assert cropped_named_authority_gap(answer, ledger, current, **_OPTIONS)
    assert _record(current, ledger, state, receipt, composed=composed) == 2
    for answer in answers:
        assert cropped_named_authority_gap(answer, ledger, current, **_OPTIONS) is None
        assert canonical_named_authority_gap(answer, ledger, **_OPTIONS)
    assert cropped_named_authority_gap(
        answers[0].replace("179/2", "179/3"), ledger, current, **_OPTIONS
    )
    assert cropped_named_authority_gap(
        answers[0].replace("5237", "4458"), ledger, current, **_OPTIONS
    )
    assert ledger.export() == before
    restored = CroppedArticleBindings(current, "Captured incident question")
    restored.restore(state.export(), current, ledger)
    assert restored.identity_bindings(current, ledger) == state.identity_bindings(
        current, ledger
    )


@pytest.mark.parametrize("composed", [False, True])
def test_captured_inserted_heading_and_clauses_bind_without_relabelling_base(
    composed: bool,
) -> None:
    current, ledger, state, receipt = _setup(_items(inserted=True))
    before = copy.deepcopy(ledger.export())
    answer = "4458 sayılı Gümrük Kanunu m.191/A koşulları aranır [1][2][3]."
    assert cropped_named_authority_gap(answer, ledger, current, **_OPTIONS)
    assert _record(current, ledger, state, receipt, composed=composed) == 3
    assert cropped_named_authority_gap(answer, ledger, current, **_OPTIONS) is None
    rows = cropped_identity_ledger(ledger, current).authority_metadata()
    assert [row["article_no"] for row in rows] == ["191/A"] * 3
    assert [row["clause_label"] for row in rows] == [None, "a", "b"]
    assert canonical_named_authority_gap(answer, ledger, **_OPTIONS)
    assert cropped_named_authority_gap(
        answer.replace("191/A", "191"), ledger, current, **_OPTIONS
    )
    assert cropped_named_authority_gap(
        answer.replace("4458", "5237"), ledger, current, **_OPTIONS
    )
    assert ledger.export() == before


@pytest.mark.parametrize("variant", ["gapped", "next_article", "wrong_clause"])
def test_inserted_identity_does_not_cross_a_missing_or_conflicting_continuation(
    variant: str,
) -> None:
    items = _items(inserted=True)
    if variant == "gapped":
        items[1].metadata["position"] = 909
    elif variant == "next_article":
        items[1].text = "**MADDE 192-** Yeni hüküm."
        items[1].text_hash = hashlib.sha256(items[1].text.encode()).hexdigest()
    else:
        canonical = items[1].metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        canonical["clause_label"] = "b"
    current, ledger, state, receipt = _setup(items)
    assert state.record_receipt(receipt, current, ledger) == 1
    assert set(state.identity_bindings(current, ledger)) == {1}
    assert cropped_named_authority_gap(
        "4458 sayılı Gümrük Kanunu m.191/A koşulları aranır [2][3].",
        ledger,
        current,
        **_OPTIONS,
    )


@pytest.mark.parametrize("field", ["target_article_no", "original_heading_path"])
def test_inserted_anchor_rejects_foreign_or_amendment_metadata(field: str) -> None:
    items = _items(inserted=True)
    canonical = items[0].metadata["canonical_metadata"]
    assert isinstance(canonical, dict)
    canonical[field] = "191/A" if field == "target_article_no" else ["MADDE 192"]
    current, ledger, state, receipt = _setup(items)
    assert state.record_receipt(receipt, current, ledger) == 0
    assert state.identity_bindings(current, ledger) == {}


@pytest.mark.parametrize("tamper", ["text", "metadata", "scope", "target_source"])
def test_captured_receipt_keeps_hash_scope_and_follow_target_fences(
    tamper: str,
) -> None:
    current, ledger, state, receipt = _setup(_items(inserted=False), follow=True)
    item = ledger._items[2]
    if tamper == "text":
        item.text += " Değişmiş metin."
    elif tamper == "metadata":
        canonical = item.metadata["canonical_metadata"]
        assert isinstance(canonical, dict)
        canonical["paragraph_no"] = "3"
    elif tamper == "scope":
        current.scope = {"corpus": "foreign"}
    else:
        receipt.call.arguments["target_source_id"] = receipt.call.arguments["source_id"]
    with pytest.raises(ValueError):
        state.record_receipt(receipt, current, ledger)
    assert state.export()["reads"] == []


def test_actual_composed_follow_reference_commits_target_paragraphs() -> None:
    items = _items(inserted=False)
    chunks = []
    for item in items:
        metadata = item.metadata["canonical_metadata"]
        assert isinstance(metadata, dict)
        headings = metadata["heading_path"]
        assert isinstance(headings, list)
        position = item.metadata["position"]
        assert type(position) is int and item.chunk_id is not None
        chunks.append(
            CorpusChunk(
                id=item.chunk_id,
                source_id=UUID(item.source_id),
                text=item.text,
                position=position,
                projection_ordinal=0,
                heading_path=tuple(str(value) for value in headings),
                metadata=metadata,
                validity_start=None,
                validity_end=None,
                status="active",
            )
        )
    origin = CorpusSource(
        UUID("6af208e4-f1ee-4af0-afc1-9f87f17b1c79"), "Yönetmelik", "origin"
    )
    origin_chunk = CorpusChunk(
        id="rc_21c0100ad3ba6f63b843acbc57065695561750de",
        source_id=origin.id,
        text="5237 sayılı Türk Ceza Kanunu madde 179 uyarınca işlem yapılır.",
        position=1,
        projection_ordinal=0,
        heading_path=("Yönetmelik", "MADDE 97"),
        metadata={"document_type": "yonetmelik"},
        validity_start=None,
        validity_end=None,
        status="active",
    )

    class Broker(MemoryBroker):
        def chunk(
            self, source_id: str, chunk_id: str, context: RunContext
        ) -> tuple[CorpusSource, CorpusChunk | None]:
            if source_id == str(origin.id) and chunk_id == origin_chunk.id:
                context.check_active()
                return origin, origin_chunk
            return super().chunk(source_id, chunk_id, context)

    broker = Broker(chunks)
    broker.item = CorpusSource(
        chunks[0].source_id, "5237 Sayılı TÜRK CEZA KANUNU", "target"
    )
    context = RunContext(
        run_id="actual-follow-reference",
        scope={"corpus": 15},
        services={"asv3_exp_cropped": True},
    )
    state = CroppedArticleBindings(context, "Captured incident question")
    context.services["cropped_article_bindings"] = state
    ledger = EvidenceLedger()
    ledger.add([evidence_for_chunk(broker.item, chunk) for chunk in chunks], context)
    registry = CapabilityRegistry(build_corpus_specs(broker))
    registry.register(
        next(
            spec
            for spec in build_sandbox_specs(broker)
            if spec.name == "compose_tool_calls"
        )
    )
    harness = Harness(
        registry=registry,
        context=context,
        request="Captured incident question",
        decide=lambda _view: Decision(),
        evidence=ledger,
    )
    call = CapabilityCall(
        name="compose_tool_calls",
        arguments={
            "steps": [
                {
                    "id": "tckref",
                    "tool": "follow_reference",
                    "arguments": {
                        "source_id": str(origin.id),
                        "chunk_id": origin_chunk.id,
                        "target_source_id": str(broker.item.id),
                        "article": "179",
                    },
                }
            ]
        },
    )
    outcome = registry.dispatch(call, context)
    assert outcome.status == OutcomeStatus.FOUND
    assert state.export()["reads"] == []
    harness._commit_receipt(ToolReceipt(call=call, outcome=outcome, elapsed_seconds=0))
    assert {
        number: binding.paragraph_no
        for number, binding in state.identity_bindings(context, ledger).items()
    } == {2: "2", 3: "3"}
    assert (
        cropped_named_authority_gap(
            "5237 sayılı Türk Ceza Kanunu m.179/2 uygulanır [2].",
            ledger,
            context,
            **_OPTIONS,
        )
        is None
    )
    reads = state.export()["reads"]
    assert isinstance(reads, list) and len(reads) == 1
    saved = reads[0]
    assert isinstance(saved, dict) and saved["method"] == "follow_reference"
