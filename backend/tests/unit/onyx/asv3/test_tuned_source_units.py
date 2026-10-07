"""Exact retrieval lineage and explicit local structure preserve conditions without article expansion."""

from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from onyx.asv3.models import RunContext
from onyx.asv3.workflow_variant import ASV3_TUNED_VARIANT
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import SearchDoc
from onyx.db import asv3_corpus
from tests.unit.onyx.asv3.test_search_hydration import (
    Rows,
    boundary,
    read,
    row,
    validation_broker,
)


@pytest.mark.parametrize("tuned", [False, True])
@pytest.mark.parametrize("ordinal", [88, 87])
def test_broker_hydrates_aggregate_children_only_in_tuned_with_verified_ordinal(
    monkeypatch: pytest.MonkeyPatch,
    tuned: bool,
    ordinal: int,
) -> None:
    source = uuid4()
    items = [
        row(source, i, ["Law", "MADDE 8", "(1) Conditions", f"{i}"]) for i in range(3)
    ]
    session, arguments = boundary(monkeypatch, items)
    outline = session.execute.side_effect

    def execute(statement: Any) -> Rows:
        if len(statement.selected_columns) == 4:
            return Rows(
                [
                    SimpleNamespace(
                        id="aggregate",
                        projection_ordinal=88,
                        chunk_type="hierarchical_aggregate",
                        chunk_metadata={
                            "source_regulatory_chunk_ids": [item.id for item in items]
                        },
                    )
                ]
            )
        return cast(Rows, outline(statement))

    session.execute.side_effect = execute
    broker = validation_broker(monkeypatch, session, arguments)
    doc = SearchDoc(
        document_id=str(source),
        chunk_ind=ordinal,
        semantic_identifier="Law",
        link=None,
        blurb="Untrusted derived text",
        source_type=DocumentSource.USER_FILE,
        boost=0,
        hidden=False,
        match_highlights=[],
        metadata={
            "regulatory_chunk_id": "aggregate",
            "source_regulatory_chunk_ids": ["foreign"],
        },
    )
    context = RunContext()
    if tuned:
        context.services["asv3_workflow_variant"] = ASV3_TUNED_VARIANT
    hydrated = broker.hydrate_search_centers([doc], context)[(str(source), ordinal)]
    assert [item.chunk_id for item in hydrated] == (
        [item.id for item in items] if tuned and ordinal == 88 else []
    )
    assert all(item.text != doc.blurb for item in hydrated)


def test_selected_clause_completes_only_its_own_condition_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [
        row(source, 0, ["Law", "MADDE 8", "(1) First conditions"]),
        row(source, 1, ["Law", "MADDE 8", "(1) First conditions", "a) First"]),
        row(source, 2, ["Law", "MADDE 8", "(1) First conditions", "b) Second"]),
        row(source, 3, ["Law", "MADDE 8", "(2) Separate rule"]),
        row(source, 4, ["Law", "MADDE 9", "(1) First conditions"]),
    ]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[1], items[2]], local_units_only=True)
    assert {chunk.id for chunk in result.chunks} == {item.id for item in items[:3]}
    assert result.members[items[1].id] == result.members[items[2].id]
    assert not any(result.complete.values())  # A fıkra is not the complete article.


def test_example_facts_and_conclusion_join_without_neighboring_examples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    items = [
        row(source, 0, ["Guidance", "Örnek 2", "Facts"]),
        row(source, 1, ["Guidance", "Örnek 2", "Conclusion"]),
        row(source, 2, ["Guidance", "Örnek 3", "Facts"]),
    ]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[0]], local_units_only=True)
    assert result.members[items[0].id] == (items[0].id, items[1].id)


def test_unstructured_hit_keeps_exact_center(monkeypatch: pytest.MonkeyPatch) -> None:
    source = uuid4()
    items = [row(source, i, ["Law", "MADDE 8"]) for i in range(3)]
    _, arguments = boundary(monkeypatch, items)
    result = read(arguments, [items[1]], local_units_only=True)
    assert result.members[items[1].id] == (items[1].id,)


@pytest.mark.parametrize("qualified", [False, True])
@pytest.mark.parametrize("defect", [None, "ordinal", "missing", "duplicate", "self"])
def test_aggregate_lineage_uses_verified_center_not_search_payload(
    monkeypatch: pytest.MonkeyPatch,
    qualified: bool,
    defect: str | None,
) -> None:
    source = uuid4()
    item = row(source, 1, ["Law", "MADDE 8", "(1) First conditions"])
    session, arguments = boundary(monkeypatch, [item])
    members = ["atomic-a", "atomic-b"]
    if defect == "missing":
        members = []
    elif defect == "duplicate":
        members = ["atomic-a", "atomic-a"]
    elif defect == "self":
        members = ["aggregate"]
    streams: list[Rows] = []
    if qualified:
        monkeypatch.setattr(
            asv3_corpus, "qualified_file_ids", lambda *_args: frozenset({source})
        )
        binding = SimpleNamespace(
            projection=SimpleNamespace(
                ordinal=18, source_json='{"regulatory_chunk_id":"aggregate"}'
            ),
            derived_role="hierarchical_aggregate",
            representation_metadata={"source_regulatory_chunk_ids": members},
        )

        def bindings(*_args: Any, **kwargs: Any) -> Any:
            assert kwargs["canonical_chunk_ids"] == ("aggregate",)
            yield binding

        monkeypatch.setattr(asv3_corpus, "iter_public_temporal_bindings", bindings)
        arguments["index"] = MagicMock()
    else:

        def execute(statement: Any) -> Rows:
            assert "text" not in [column.key for column in statement.selected_columns]
            assert "user_file_id" in str(statement)
            values = Rows(
                [
                    SimpleNamespace(
                        id="aggregate",
                        projection_ordinal=18,
                        chunk_type="hierarchical_aggregate",
                        chunk_metadata={"source_regulatory_chunk_ids": members},
                    )
                ]
            )
            streams.append(values)
            return values

        session.execute.side_effect = execute
    result = asv3_corpus.resolve_search_center_members(
        **{key: value for key, value in arguments.items() if not key.startswith("_")},
        center_ordinals={"aggregate": 19 if defect == "ordinal" else 18},
    )
    assert result == {"aggregate": ("atomic-a", "atomic-b") if defect is None else ()}
    assert all(stream.closed for stream in streams)


def test_local_units_reject_overlapping_visible_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    first = row(source, 0, ["Law", "MADDE 8", "(1) Conditions"])
    second = row(source, 0, ["Law", "MADDE 8", "(1) Conditions"])
    second.id += "-alternative"
    _, arguments = boundary(monkeypatch, [first, second], as_of=date(2026, 1, 1))
    with pytest.raises(asv3_corpus.CorpusScopeUnavailable, match="overlapping"):
        read(arguments, [first], local_units_only=True)
