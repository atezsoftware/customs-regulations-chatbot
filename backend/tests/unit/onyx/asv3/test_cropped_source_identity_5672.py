"""A captured decision quoting 4458 cannot impersonate that statute's original."""

import json
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from onyx.asv3.authority import known_statute_source_ids
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import EvidenceItem, OutcomeStatus, RunContext
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker

_CAPTURE = json.loads(
    (
        Path(__file__).parent / "fixtures" / "cropped_source_identity_5672.json"
    ).read_text()
)
_LAW = "12328001-2f21-407e-b230-7d50a8dbf28f"
_DECISION = "240f4072-02b0-4f7d-a2f4-9295d8aa4279"


def originals() -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(row["item"]) for row in _CAPTURE["records"]]


@pytest.mark.parametrize("query", ["4458 sayılı Kanun", "4458 sayılı Gümrük Kanunu"])
def test_captured_short_and_formal_law_queries_resolve_only_the_actual_statute(
    query: str,
) -> None:
    ledger, context = EvidenceLedger(), RunContext()
    ledger.add(originals(), context)
    before = ledger.export()
    assert known_statute_source_ids(query, ledger) == [_LAW]
    assert ledger.export() == before


def test_legacy_law_type_does_not_promote_the_decisions_quoted_statute() -> None:
    ledger, context = EvidenceLedger(), RunContext()
    decision = next(item for item in originals() if item.source_id == _DECISION)
    canonical = decision.metadata["canonical_metadata"]
    assert isinstance(canonical, dict) and canonical["document_type"] == "kanun"
    ledger.add([decision], context)
    assert known_statute_source_ids("4458 sayılı Kanun", ledger) == []
    assert (
        canonical_named_authority_gap(
            "4458 sayılı Kanun m.33 uygulanır [1].",
            ledger,
            strict_reference_boundaries=True,
            syntactic_reference_binding=True,
        )
        is not None
    )


def test_captured_decision_remains_readable_by_its_own_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = next(item for item in originals() if item.source_id == _DECISION)
    canonical = decision.metadata["canonical_metadata"]
    assert isinstance(canonical, dict) and decision.chunk_id is not None
    chunk = CorpusChunk(
        decision.chunk_id,
        UUID(_DECISION),
        decision.text,
        0,
        int(canonical["chunk_index"]),
        tuple(canonical["heading_path"]),
        canonical,
        None,
        None,
        "active",
    )
    broker = MemoryBroker([chunk])
    broker.item = CorpusSource(
        UUID(_DECISION), str(canonical["title"]), "captured-original"
    )
    title = str(canonical["heading_path"][0])
    queries: list[str] = []

    def sources(
        query: str, _context: RunContext, **_kwargs: Any
    ) -> tuple[list[CorpusSource], bool]:
        queries.append(query)
        return [broker.item], False

    monkeypatch.setattr(broker, "sources", sources)
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_args: None)
    ledger, context = EvidenceLedger(), RunContext()
    ledger.add([decision], context)
    context.services["evidence"] = ledger
    tool = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=True)
        if spec.name == "read_named_provision"
    )
    result = tool.handler({"source_name": title, "article": "33"}, context)
    assert queries == [title]
    assert result.status == OutcomeStatus.FOUND
    assert len(result.evidence) == 1
    assert result.evidence[0].source_id == _DECISION
    assert result.evidence[0].text_hash == decision.text_hash
