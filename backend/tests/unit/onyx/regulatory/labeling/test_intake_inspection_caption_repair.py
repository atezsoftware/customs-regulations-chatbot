"""Actual26-row criterion context stays strict without changing model semantics."""

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from onyx.regulatory.amendments.annexes.context_dependencies import context_hash
from onyx.regulatory.labeling.intake_dependent_clause_repair import (
    DependentClauseRepairSource,
)
from onyx.regulatory.labeling.intake_inspection_caption_repair import (
    legacy_inspection_caption_repair_candidate,
    legacy_inspection_caption_shape,
)


def _case() -> tuple[str, list[DependentClauseRepairSource]]:
    data = json.loads(
        (
            Path(__file__).parent / "fixtures/legacy_inspection_criteria_context.json"
        ).read_text()
    )
    return data["target_chunk_id"], [
        DependentClauseRepairSource(**row) for row in data["sources"]
    ]


def _candidate(chunk_id: str, sources: list[DependentClauseRepairSource]) -> bool:
    return legacy_inspection_caption_repair_candidate(
        chunk_id=chunk_id, sources=sources
    )


def test_actual_26_current_members_and_honest_missing_first_table_title() -> None:
    chunk_id, sources = _case()
    first = next(source for source in sources if source.snapshot["position"] == 15)
    assert "section_title" not in first.snapshot["chunk_metadata"]
    assert (
        "section_title"
        not in first.snapshot["chunk_metadata"]["intake_legal_metadata"]["fields"]
    )
    before = deepcopy(sources)
    assert _candidate(chunk_id, sources)
    assert sources == before
    assert not _candidate(sources[0].chunk_id, sources)


@pytest.mark.parametrize("index", range(26))
@pytest.mark.parametrize("kind", ["text", "metadata"])
def test_every_delivered_current_member_is_bound(index: int, kind: str) -> None:
    chunk_id, sources = _case()
    if kind == "text":
        sources[index] = replace(
            sources[index], current_text=sources[index].current_text + "changed"
        )
    else:
        sources[index].current_snapshot["chunk_metadata"]["section_title"] = "changed"
    assert not _candidate(chunk_id, sources)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-member",
        "duplicate-member",
        "wrong-file",
        "inactive",
        "missing-boundary",
        "wrong-heading",
        "title-heading-not-body",
        "fabricated-first-table-title",
        "null-first-table-title",
        "stale-target",
        "stale-table",
        "wrong-next-title",
        "wrong-table-index",
        "missing-end-ordinal",
        "wrong-context-set",
    ],
)
def test_closed_criterion_scope_and_literal_provenance_are_required(
    mutation: str,
) -> None:
    chunk_id, sources = _case()
    by_pos = {source.snapshot["position"]: source for source in sources}
    target, table, boundary = by_pos[14], by_pos[15], by_pos[22]

    def meta(source: DependentClauseRepairSource) -> dict[str, Any]:
        return source.snapshot["chunk_metadata"]

    def sync(source: DependentClauseRepairSource) -> None:
        source.current_snapshot.update(
            deepcopy({key: source.snapshot[key] for key in source.current_snapshot})
        )

    if mutation == "missing-member":
        sources.pop(0)
    elif mutation == "duplicate-member":
        sources[-1] = sources[0]
    elif mutation == "wrong-file":
        sources[0] = replace(sources[0], file_id="different")
    elif mutation == "inactive":
        sources[0] = replace(sources[0], active=False)
    elif mutation == "missing-boundary":
        sources.remove(boundary)
    elif mutation == "wrong-heading":
        target.snapshot["heading_path"] = ["EK 3"]
        meta(target)["heading_path"] = ["EK 3"]
        sync(target)
    elif mutation == "title-heading-not-body":
        meta(target)["intake_legal_metadata"]["fields"]["section_title"][
            "evidence_scope"
        ] = "heading"
        sync(target)
    elif mutation == "null-first-table-title":
        meta(table)["section_title"] = None
        sync(table)
    elif mutation == "fabricated-first-table-title":
        meta(table)["section_title"] = target.text
        sync(table)
    elif mutation == "stale-target":
        meta(target)["intake_legal_metadata"]["input_stale"] = True
        sync(target)
    elif mutation == "stale-table":
        meta(table)["intake_legal_metadata"]["input_stale"] = True
        sync(table)
    elif mutation == "wrong-next-title":
        updated = replace(
            boundary,
            text="D.ANOTHER SECTION",
            current_text="D.ANOTHER SECTION",
            canonical_text_sha256=context_hash("D.ANOTHER SECTION"),
        )
        sources[sources.index(boundary)] = updated
    elif mutation == "wrong-table-index":
        meta(by_pos[19])["table_index"] = 4
        sync(by_pos[19])
    elif mutation == "missing-end-ordinal":
        old = by_pos[21]
        text = old.text.replace("| 20 |", "| 21 |")
        assert text != old.text
        sources[sources.index(old)] = replace(
            old, text=text, current_text=text, canonical_text_sha256=context_hash(text)
        )
    elif mutation == "wrong-context-set":
        target.snapshot["context_member_ids"] = [chunk_id]
    assert not _candidate(chunk_id, sources)


def test_runtime_candidate_does_not_allowlist_database_identities() -> None:
    chunk_id, sources = _case()
    mapping = {
        source.chunk_id: f"fresh-{index}" for index, source in enumerate(sources)
    }

    def rename(value: Any) -> Any:
        if isinstance(value, str):
            return mapping.get(value, value)
        if isinstance(value, list):
            return [rename(item) for item in value]
        if isinstance(value, dict):
            return {key: rename(item) for key, item in value.items()}
        return value

    moved = [
        replace(
            source,
            chunk_id=mapping[source.chunk_id],
            file_id="another-owned-pdf",
            snapshot=rename(source.snapshot),
            current_snapshot=rename(source.current_snapshot),
        )
        for source in sources
    ]
    assert _candidate(mapping[chunk_id], moved)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "B. MUAYENE ÜNİTESİ KRİTERLERİ",
        "B.MUAYENE ÜNİTESİ KRİTERLERİ\nYeni zorunluluk",
        "C.ALTYAPI VE PERSONEL KRİTERLERİ",
    ],
)
def test_routing_never_admits_other_or_operative_text(text: str) -> None:
    assert not legacy_inspection_caption_shape(text)
