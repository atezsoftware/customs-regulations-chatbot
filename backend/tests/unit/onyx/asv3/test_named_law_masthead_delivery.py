"""Natural law names require a full own masthead, never metadata alone."""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue
from sqlalchemy.dialects import postgresql

from onyx.asv3 import corpus_tools
from onyx.asv3.authority import known_statute_source_ids
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.cropped_article_bindings import (
    CroppedArticleBindings,
    capture_cropped_dispatched_read,
    cropped_composed_read_capture,
    cropped_named_authority_gap,
)
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    CapabilityCall,
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolReceipt,
    ToolSpec,
)
from onyx.asv3.shared_reads import SharedReads
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from onyx.db.models import User
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.asv3.test_numbered_law_identity_213 import chunk
from tests.unit.onyx.asv3.test_numbered_source_resolution import catalogue, filters

CAPTURES = json.loads(
    (Path(__file__).parent / "fixtures/numbered_intake_display_names.json").read_text()
)["laws"]


def captured(capture: dict[str, Any]) -> tuple[CorpusSource, list[CorpusChunk]]:
    identifier = UUID(capture["source_id"])
    name = re.sub(r"^\d+\s+sayılı\s+", "", capture["display_name"])
    return CorpusSource(identifier, name, "retained-owned-original"), [
        chunk(row, identifier) for row in capture["candidate_heads"]
    ]


def deliver(
    ledger: EvidenceLedger,
    context: RunContext,
    items: list[EvidenceItem],
    *,
    partial_head: bool = False,
) -> list[int]:
    numbers = ledger.add(items, context)
    ledger.record_delivery(
        "originals",
        "tool",
        [
            {
                "citation": number,
                "text": item.text[:20] if partial_head and index == 0 else item.text,
            }
            for index, (number, item) in enumerate(zip(numbers, items, strict=True))
        ],
    )
    return numbers


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda row: row["number"])
@pytest.mark.parametrize("formal", [False, True])
def test_actual_natural_names_resolve_bare_and_full_numbered_provision_reads(
    monkeypatch: pytest.MonkeyPatch, capture: dict[str, Any], formal: bool
) -> None:
    source, heads = captured(capture)
    assert not asv3_corpus._source_has_own_law_number(source, heads, capture["number"])
    assert asv3_corpus._source_has_unnumbered_own_law_identity(
        source, heads, capture["number"]
    )
    query = capture["display_name"] if formal else capture["number"] + " sayılı Kanun"
    session = catalogue(monkeypatch, [[], [source]] if formal else [[source]])
    scope = filters()
    monkeypatch.setattr(
        asv3_corpus, "_read_identity_heads_batch", lambda **_: {source.id: heads}
    )
    broker = MemoryBroker([chunk(row, source.id) for row in capture["provision"]])
    broker.item = source
    broker.filters = scope

    def lookup(
        name: str, context: RunContext, **kwargs: Any
    ) -> tuple[list[CorpusSource], bool]:
        return asv3_corpus.find_sources(
            session,
            user=cast(User, object()),
            filters=scope,
            query=name,
            limit=kwargs["limit"],
            allow_canonical_filename_fallback=True,
            check_active=context.check_active,
        )

    monkeypatch.setattr(broker, "sources", lookup)
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_: None)
    spec = next(
        item
        for item in build_corpus_specs(broker, named_provision_reads=True)
        if item.name == "read_named_provision"
    )
    ledger = EvidenceLedger()
    context = RunContext(services={"asv3_exp_cropped": True, "evidence": ledger})
    bindings = CroppedArticleBindings(context, query)
    context.services["cropped_article_bindings"] = bindings
    args = {"source_name": query, "article": capture["article"]}
    result = spec.handler(args, context)
    assert result.status == OutcomeStatus.FOUND
    article = result.evidence[: len(capture["provision"])]
    assert [item.text for item in article] == [
        row["text"] for row in capture["provision"]
    ]
    assert [item.chunk_id for item in article] == [
        row["id"] for row in capture["provision"]
    ]
    assert len(result.evidence) == len(article) + 1
    assert (
        result.data["next_position"]
        == max(row["position"] for row in capture["provision"]) + 1
    )
    assert result.evidence[-1].text == capture["candidate_heads"][0]["text"]
    ordered = result.data["cropped_ordered_source_read"]
    assert isinstance(ordered, dict)
    assert [row["position"] for row in ordered["members"]] == [
        row["position"] for row in capture["provision"]
    ]
    numbers = ledger.add(result.evidence, context)
    ledger.record_delivery(
        "named",
        "tool",
        [
            {"citation": n, "text": item.text}
            for n, item in zip(numbers, result.evidence, strict=True)
        ],
    )
    bindings.record_receipt(
        ToolReceipt(
            call=CapabilityCall(name=spec.name, arguments=args),
            outcome=result,
            evidence_ids=numbers,
            elapsed_seconds=0,
        ),
        context,
        ledger,
    )
    assert known_statute_source_ids(capture["number"] + " sayılı Kanun", ledger) == [
        str(source.id)
    ]
    for title in (capture["number"] + " sayılı Kanun", capture["display_name"]):
        assert (
            cropped_named_authority_gap(
                f"{title} m.{capture['article']} uygulanır [1].",
                ledger,
                context,
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            )
            is None
        )
        assert (
            canonical_named_authority_gap(
                f"{title} m.{capture['article']} uygulanır [1].",
                ledger,
                strict_reference_boundaries=True,
                syntactic_reference_binding=True,
            )
            is None
        )
    sql = str(
        session.execute.call_args_list[0]
        .args[0]
        .compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )
    assert ("document_number" in sql) is (not formal)
    assert "document_set__user_file" in sql and "project__user_file" in sql


@pytest.mark.parametrize("capture", CAPTURES, ids=lambda row: row["number"])
def test_resolve_retains_real_head_then_explicit_id_article_completes_authority(
    monkeypatch: pytest.MonkeyPatch, capture: dict[str, Any]
) -> None:
    source, heads = captured(capture)
    session = catalogue(monkeypatch, [[source]])
    scope = filters()
    monkeypatch.setattr(
        asv3_corpus, "_read_identity_heads_batch", lambda **_: {source.id: heads}
    )
    broker = MemoryBroker([chunk(row, source.id) for row in capture["provision"]])
    broker.item, broker.filters = source, scope
    monkeypatch.setattr(
        broker,
        "sources",
        lambda query, context, **_kwargs: asv3_corpus.find_sources(
            session,
            user=cast(User, object()),
            filters=scope,
            query=query,
            allow_canonical_filename_fallback=True,
            check_active=context.check_active,
        ),
    )
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_: None)
    ledger = EvidenceLedger()
    context = RunContext(services={"asv3_exp_cropped": True, "evidence": ledger})
    specs = {
        spec.name: spec
        for spec in build_corpus_specs(broker, named_provision_reads=True)
    }
    query = capture["number"] + " sayılı Kanun"
    resolved = specs["resolve_source"].handler({"query": query}, context)
    assert resolved.status == OutcomeStatus.FOUND
    assert len(resolved.evidence) == 1
    deliver(ledger, context, resolved.evidence)
    assert known_statute_source_ids(query, ledger) == [str(source.id)]
    assert (
        canonical_named_authority_gap(
            f"{query} m.{capture['article']} uygulanır [1].", ledger
        )
        is not None
    )
    # A refetched source has no ephemeral head transport. Its real article is still required.
    assert broker.source(str(source.id), context).identity_originals == ()
    article = specs["read_provision"].handler(
        {"source_id": str(source.id), "article": capture["article"]}, context
    )
    assert article.status == OutcomeStatus.FOUND
    assert len(article.evidence) == len(capture["provision"])
    numbers = deliver(ledger, context, article.evidence)
    assert (
        canonical_named_authority_gap(
            f"{query} m.{capture['article']} uygulanır [{numbers[0]}].", ledger
        )
        is None
    )


def named_setup(
    monkeypatch: pytest.MonkeyPatch, *, partial: bool = False
) -> tuple[
    ToolSpec, MemoryBroker, RunContext, EvidenceLedger, CorpusSource, list[CorpusChunk]
]:
    capture = next(row for row in CAPTURES if row["number"] == "6001")
    source, heads = captured(capture)
    witnessed = replace(source, identity_originals=(heads[0],))
    rows = capture["provision"][:3] if partial else capture["provision"]
    broker = MemoryBroker([chunk(row, source.id) for row in rows], partial=partial)
    broker.item = source
    monkeypatch.setattr(
        broker, "sources", lambda *_args, **_kwargs: ([witnessed], False)
    )
    monkeypatch.setattr(broker, "related_sources_for_provision", lambda *_args: None)
    ledger = EvidenceLedger()
    context = RunContext(services={"asv3_exp_cropped": True, "evidence": ledger})
    context.services["cropped_article_bindings"] = CroppedArticleBindings(
        context, "6001 m.30"
    )
    spec = next(
        spec
        for spec in build_corpus_specs(broker, named_provision_reads=True)
        if spec.name == "read_named_provision"
    )
    return spec, broker, context, ledger, source, heads


def test_partial_article_keeps_exact_ordered_interval_and_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, context, ledger, _, _ = named_setup(monkeypatch, partial=True)
    args: dict[str, JsonValue] = {"source_name": "6001 sayılı Kanun", "article": "30"}
    result = spec.handler(args, context)
    assert result.status == OutcomeStatus.PARTIAL
    assert result.data["scan_truncated"] is True
    assert result.data["next_position"] == 106
    assert result.data["evidence_next_position"] is None
    raw = cast(dict[str, Any], result.data["cropped_ordered_source_read"])
    assert [member["position"] for member in raw["members"]] == [103, 104, 105]
    assert result.evidence[-1].metadata["position"] == 0
    numbers = deliver(ledger, context, result.evidence)
    state = cast(CroppedArticleBindings, context.services["cropped_article_bindings"])
    state.record_receipt(
        ToolReceipt(
            call=CapabilityCall(name=spec.name, arguments=args),
            outcome=result,
            evidence_ids=numbers,
            elapsed_seconds=0,
        ),
        context,
        ledger,
    )
    assert all(
        binding.chunk_id != result.evidence[-1].chunk_id
        for binding in state.identity_bindings(context, ledger).values()
    )


def test_clipped_head_never_promotes_article_and_retains_article_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, broker, context, ledger, _, _ = named_setup(monkeypatch)
    monkeypatch.setattr(
        corpus_tools, "MAX_RESPONSE_CHARS", sum(len(row.text) for row in broker.items)
    )
    result = spec.handler(
        {"source_name": "6001 sayılı Kanun", "article": "30"}, context
    )
    assert result.status == OutcomeStatus.PARTIAL
    assert result.data["identity_evidence_truncated"] is True
    assert [item.text for item in result.evidence] == [row.text for row in broker.items]
    # A tool-provided metadata flag cannot forge the ledger-owned delivery marker.
    for item in result.evidence:
        item.metadata["original_completely_delivered"] = True
    deliver(ledger, context, result.evidence)
    assert known_statute_source_ids("6001 sayılı Kanun", ledger) == []


def test_shared_article_cache_does_not_capture_callers_identity_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, broker, context, _, source, _ = named_setup(monkeypatch)
    context.services["shared_reads"] = SharedReads(
        fence=lambda *_: "same-publication", producer_context=lambda caller: caller
    )
    first = spec.handler({"source_name": "6001 sayılı Kanun", "article": "30"}, context)
    monkeypatch.setattr(broker, "sources", lambda *_args, **_kwargs: ([source], False))
    second = spec.handler(
        {"source_name": "6001 sayılı Kanun", "article": "30"}, context
    )
    assert len(first.evidence) == len(broker.items) + 1
    assert len(second.evidence) == len(broker.items)
    assert second.data["shared_read_reuse"] == "completed"
    assert [item.identity for item in first.evidence[:-1]] == [
        item.identity for item in second.evidence
    ]


def test_composed_named_read_commits_only_its_article_members(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, _, context, ledger, _, _ = named_setup(monkeypatch)
    call = CapabilityCall(
        name=spec.name, arguments={"source_name": "6001 sayılı Kanun", "article": "30"}
    )
    parent = CapabilityCall(name="compose_tool_calls")
    with cropped_composed_read_capture(parent, context):
        result = spec.handler(call.arguments, context)
        capture_cropped_dispatched_read(call, result, context)
    numbers = deliver(ledger, context, result.evidence)
    state = cast(CroppedArticleBindings, context.services["cropped_article_bindings"])
    state.record_receipt(
        ToolReceipt(
            call=parent, outcome=result, evidence_ids=numbers, elapsed_seconds=0
        ),
        context,
        ledger,
    )
    stored = cast(list[dict[str, Any]], state.export()["reads"])
    assert len(stored) == 1
    assert [member["position"] for member in stored[0]["read"]["members"]] == list(
        range(103, 117)
    )
    assert len(stored[0]["citations"]) == 14
    assert (
        cropped_named_authority_gap(
            "6001 sayılı Kanun m.30 uygulanır [1].", ledger, context
        )
        is None
    )


@pytest.mark.parametrize("kind", ["Kanun", "LAW", "court_decision"])
def test_identity_head_kind_uses_the_same_closed_casefolded_law_family(
    monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    spec, broker, context, ledger, source, heads = named_setup(monkeypatch)
    head = replace(heads[0], metadata={**heads[0].metadata, "document_type": kind})
    witnessed = replace(source, identity_originals=(head,))
    monkeypatch.setattr(
        broker, "sources", lambda *_args, **_kwargs: ([witnessed], False)
    )
    args: dict[str, JsonValue] = {"source_name": "6001 sayılı Kanun", "article": "30"}
    result = spec.handler(args, context)
    numbers = deliver(ledger, context, result.evidence)
    state = cast(CroppedArticleBindings, context.services["cropped_article_bindings"])
    receipt = ToolReceipt(
        call=CapabilityCall(name=spec.name, arguments=args),
        outcome=result,
        evidence_ids=numbers,
        elapsed_seconds=0,
    )
    if kind == "court_decision":
        with pytest.raises(ValueError, match="unrelated identity original"):
            state.record_receipt(receipt, context, ledger)
    else:
        state.record_receipt(receipt, context, ledger)
        assert (
            cropped_named_authority_gap(
                "6001 sayılı Kanun m.30 uygulanır [1].", ledger, context
            )
            is None
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unrelated",
        "wrong_source",
        "changed_hash",
        "missing_article",
        "conflicting_root",
    ],
)
def test_ordered_adapter_rejects_non_identity_extras_and_changed_members(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    spec, _, context, ledger, _, _ = named_setup(monkeypatch)
    args: dict[str, JsonValue] = {"source_name": "6001 sayılı Kanun", "article": "30"}
    result = spec.handler(args, context)
    if mutation == "unrelated":
        result.evidence[-1] = result.evidence[-1].model_copy(
            update={"text": "Madde 99- unrelated provision", "text_hash": ""}
        )
    elif mutation == "wrong_source":
        result.evidence[-1] = result.evidence[-1].model_copy(
            update={"source_id": str(uuid4())}
        )
    elif mutation == "changed_hash":
        result.evidence[-1] = result.evidence[-1].model_copy(
            update={"text_hash": "0" * 64}
        )
    elif mutation == "missing_article":
        result.evidence.pop(0)
    else:
        result.evidence[0].metadata["heading_path"] = ["213 sayılı Maden Kanunu"]
    # Exercise adapter directly; normal ledger.add independently rejects changed hashes.
    receipt = ToolReceipt(
        call=CapabilityCall(name=spec.name, arguments=args),
        outcome=result,
        evidence_ids=list(range(1, len(result.evidence) + 1)),
        elapsed_seconds=0,
    )
    with pytest.raises(ValueError):
        record = CroppedArticleBindings._receipt_read(receipt)
        assert record is not None
        state = cast(
            CroppedArticleBindings, context.services["cropped_article_bindings"]
        )
        state._validate_members(record, list(result.evidence))
