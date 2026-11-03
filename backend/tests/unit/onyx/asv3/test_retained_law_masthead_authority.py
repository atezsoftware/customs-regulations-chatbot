"""Natural law names require a full own masthead, never metadata alone."""

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.authority import known_statute_source_ids
from onyx.asv3.cheap_authority import canonical_named_authority_gap
from onyx.asv3.corpus_tools import evidence_for_chunk
from onyx.asv3.evidence import EvidenceLedger
from onyx.asv3.models import (
    EvidenceItem,
    RunContext,
)
from onyx.db.asv3_corpus import CorpusChunk, CorpusSource
from tests.unit.onyx.asv3.test_numbered_law_identity_213 import chunk

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


@pytest.mark.parametrize(
    "mutation",
    [
        "absent",
        "undelivered",
        "partly_delivered",
        "wrong_number",
        "conflicting_head",
        "conflicting_article",
        "conflicting_article_title",
        "wrong_source",
        "wrong_title",
        "truncated",
        "quoted",
        "code",
        "foreign_type",
        "doc_binding",
    ],
)
def test_article_authority_requires_delivered_consistent_same_source_own_head(
    mutation: str,
) -> None:
    capture = next(row for row in CAPTURES if row["number"] == "5393")
    source, heads = captured(capture)
    head = evidence_for_chunk(source, heads[0])
    article = [
        evidence_for_chunk(source, chunk(row, source.id))
        for row in capture["provision"]
    ]
    if mutation == "wrong_number":
        head = head.model_copy(
            update={"text": head.text.replace("5393", "3213"), "text_hash": ""}
        )
    elif mutation == "conflicting_article":
        article[0].metadata["heading_path"] = ["213 sayılı Belediye Kanunu", "MADDE 14"]
    elif mutation == "conflicting_article_title":
        cast(dict[str, JsonValue], article[0].metadata["canonical_metadata"])[
            "title"
        ] = "213 sayılı Belediye Kanunu"
    elif mutation == "wrong_source":
        foreign = replace(source, id=uuid4())
        head = evidence_for_chunk(foreign, replace(heads[0], source_id=foreign.id))
    elif mutation == "wrong_title":
        cast(dict[str, JsonValue], head.metadata["canonical_metadata"])["title"] = (
            "Maden Kanunu"
        )
    elif mutation == "truncated":
        head.metadata["truncated"] = True
    elif mutation in {"quoted", "code"}:
        prefix = "> " if mutation == "quoted" else "```text\n"
        head = head.model_copy(update={"text": prefix + head.text, "text_hash": ""})
    elif mutation == "foreign_type":
        cast(dict[str, JsonValue], head.metadata["canonical_metadata"])[
            "document_type"
        ] = "court_decision"
    elif mutation == "doc_binding":
        assert head.search_doc is not None
        head.search_doc.document_id = str(uuid4())
    ledger, context = EvidenceLedger(), RunContext()
    if mutation == "undelivered":
        ledger.add([head], context)
    elif mutation != "absent":
        deliver(ledger, context, [head], partial_head=mutation == "partly_delivered")
    if mutation == "conflicting_head":
        conflict = replace(
            heads[0], id="other-head", text=heads[0].text.replace("5393", "3213")
        )
        deliver(ledger, context, [evidence_for_chunk(source, conflict)])
    numbers = deliver(ledger, context, article)
    if mutation not in {"undelivered", "partly_delivered"}:
        assert str(source.id) not in known_statute_source_ids(
            "5393 sayılı Kanun", ledger
        )
    assert (
        canonical_named_authority_gap(
            f"5393 sayılı Kanun m.14 uygulanır [{numbers[0]}].", ledger
        )
        is not None
    )
    if mutation in {"conflicting_article", "conflicting_article_title"}:
        assert known_statute_source_ids("213 sayılı Kanun", ledger) == []
        assert (
            canonical_named_authority_gap(
                f"213 sayılı Kanun m.14 uygulanır [{numbers[0]}].", ledger
            )
            is not None
        )


@pytest.mark.parametrize("repeated_head", [False, True])
def test_real_head_delivery_survives_bounded_log_eviction_and_restore(
    repeated_head: bool,
) -> None:
    capture = next(row for row in CAPTURES if row["number"] == "5393")
    source, heads = captured(capture)
    items = [
        evidence_for_chunk(source, heads[0]),
        evidence_for_chunk(source, chunk(capture["provision"][0], source.id)),
    ]
    ledger, context = EvidenceLedger(), RunContext()
    numbers = ledger.add(items, context)
    ledger.record_delivery(
        "head", "tool", [{"citation": numbers[0], "text": items[0].text}]
    )
    for index in range(205):
        records: list[dict[str, JsonValue]] = [
            {"citation": numbers[1], "text": items[1].text}
        ]
        if repeated_head:
            records.append({"citation": numbers[0], "text": items[0].text})
        ledger.record_delivery(
            f"article-{index}",
            "tool",
            records,
        )
    exported = ledger.export()
    assert isinstance(exported["pinned_delivery_calls"], list)
    assert isinstance(exported["deliveries"], list)
    assert "head" in exported["pinned_delivery_calls"]
    assert len(exported["pinned_delivery_calls"]) == 1
    assert len(exported["deliveries"]) == 201
    restored = EvidenceLedger()
    restored.restore(exported, context)
    assert restored.export() == exported
    for retained in (ledger, restored):
        assert known_statute_source_ids("5393 sayılı Kanun", retained) == [
            str(source.id)
        ]
        assert (
            canonical_named_authority_gap(
                f"5393 sayılı Kanun m.14 uygulanır [{numbers[1]}].", retained
            )
            is None
        )


@pytest.mark.parametrize(
    "mutation", ["partial_flag", "offset", "passage_hash", "source_id", "chunk_id"]
)
def test_restored_complete_delivery_requires_the_full_original_receipt(
    mutation: str,
) -> None:
    capture = next(row for row in CAPTURES if row["number"] == "5393")
    source, heads = captured(capture)
    ledger, context = EvidenceLedger(), RunContext()
    deliver(
        ledger,
        context,
        [evidence_for_chunk(source, heads[0])],
        partial_head=mutation == "partial_flag",
    )
    exported = ledger.export()
    deliveries = cast(list[dict[str, Any]], exported["deliveries"])
    record = deliveries[0]["records"][0]
    if mutation == "partial_flag":
        assert record["complete"] is False
        record["complete"] = True
    elif mutation == "offset":
        record["end_char"] -= 1
    elif mutation == "passage_hash":
        record["passage_hash"] = "0" * 64
    else:
        record[mutation] = str(uuid4())
    with pytest.raises(ValueError, match="Invalid complete evidence delivery"):
        EvidenceLedger().restore(exported, context)


@pytest.mark.parametrize(
    "missing",
    [
        "source_id",
        "chunk_id",
        "start_char",
        "end_char",
        "passage_hash",
        "legacy_minimal",
    ],
)
def test_legacy_delivery_receipts_restore_exactly_without_new_article_authority(
    missing: str,
) -> None:
    capture = next(row for row in CAPTURES if row["number"] == "5393")
    source, heads = captured(capture)
    items = [
        evidence_for_chunk(source, heads[0]),
        evidence_for_chunk(source, chunk(capture["provision"][0], source.id)),
    ]
    ledger, context = EvidenceLedger(), RunContext()
    numbers = deliver(ledger, context, items)
    exported = ledger.export()
    record = cast(list[dict[str, Any]], exported["deliveries"])[0]["records"][0]
    fields = (
        ("source_id", "chunk_id", "start_char", "end_char", "passage_hash")
        if missing == "legacy_minimal"
        else (missing,)
    )
    for field in fields:
        record.pop(field)
    restored = EvidenceLedger()
    restored.restore(exported, context)
    assert restored.export() == exported
    head_row = restored.authority_metadata()[0]
    assert "original_completely_delivered" not in head_row
    assert (
        canonical_named_authority_gap(
            f"5393 sayılı Kanun m.14 uygulanır [{numbers[1]}].", restored
        )
        is not None
    )
    restored.record_delivery(
        "fresh-head", "tool", [{"citation": numbers[0], "text": items[0].text}]
    )
    for index in range(205):
        restored.record_delivery(
            f"later-article-{index}",
            "tool",
            [{"citation": numbers[1], "text": items[1].text}],
        )
    fresh = restored.export()
    assert "fresh-head" in cast(list[str], fresh["pinned_delivery_calls"])
    resumed = EvidenceLedger()
    resumed.restore(fresh, context)
    assert resumed.export() == fresh
    assert (
        canonical_named_authority_gap(
            f"5393 sayılı Kanun m.14 uygulanır [{numbers[1]}].", resumed
        )
        is None
    )
