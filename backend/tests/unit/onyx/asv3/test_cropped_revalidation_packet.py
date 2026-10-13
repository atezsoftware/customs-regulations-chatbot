"""Packet revalidation keeps date-separated originals and fail-closed delivery."""

from dataclasses import replace
from datetime import date
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from onyx.asv3 import corpus_tools
from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import RunContext
from onyx.context.search.models import IndexFilters
from onyx.db.asv3_corpus import (
    CorpusChunk,
    CorpusScopeUnavailable,
    CorpusSearchCenterRead,
    CorpusSource,
)
from onyx.db.models import User


def setup_packet(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    first, second = uuid4(), uuid4()
    sources = {key: CorpusSource(key, str(key), str(key)) for key in (first, second)}
    days = [date(2025, 1, 1), date(2026, 1, 1)]
    originals = {
        (key, day): CorpusChunk(
            str(key),
            key,
            f"exact legal rule {day}",
            3,
            14,
            ("MADDE 3",),
            {"read_as_of_date": day.isoformat()},
            day,
            None,
            "active",
        )
        for key, day in ((first, days[0]), (second, days[0]), (first, days[1]))
    }
    items = [
        evidence_for_chunk(sources[key], chunk) for (key, _), chunk in originals.items()
    ]
    # Every retained occurrence must be verified, including repeats of the same original.
    items.append(items[0].model_copy(deep=True))
    broker = CorpusBroker(cast(User, object()), IndexFilters(access_control_list=[]))
    calls: list[tuple[date | None, dict[UUID, tuple[str, ...]]]] = []

    def read(
        *,
        filters: IndexFilters,
        source_chunk_ids: dict[UUID, tuple[str, ...]],
        **_: Any,
    ) -> dict[UUID, CorpusSearchCenterRead]:
        calls.append((filters.as_of_date, source_chunk_ids))
        assert filters.as_of_date is not None
        return {
            key: CorpusSearchCenterRead(
                sources[key], (originals[key, filters.as_of_date],), None
            )
            for key in source_chunk_ids
        }

    batch = MagicMock(side_effect=read)
    monkeypatch.setattr(corpus_tools, "read_search_centers_batch", batch)
    return dict(
        broker=broker,
        items=items,
        originals=originals,
        batch=batch,
        calls=calls,
        days=days,
    )


def test_all_retained_occurrences_are_verified_in_exact_date_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = setup_packet(monkeypatch)
    before = [item.model_dump(mode="json") for item in state["items"]]
    state["broker"].revalidate_evidence(
        state["items"], RunContext(services={"asv3_exp_cropped": True})
    )
    assert [day for day, _ in state["calls"]] == state["days"]
    assert [len(requests) for _, requests in state["calls"]] == [2, 1]
    assert all(
        len(ids) == 1 for _, requests in state["calls"] for ids in requests.values()
    )
    assert before == [item.model_dump(mode="json") for item in state["items"]]


@pytest.mark.parametrize(
    "defect",
    [
        "text",
        "hash",
        "ordinal",
        "duplicate_retained_hash",
        "missing",
        "ambiguous",
        "wrong_date",
        "cancel",
        "publication",
    ],
)
def test_changed_evidence_or_failed_packet_never_validates(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    state = setup_packet(monkeypatch)
    item = state["items"][0]
    key = UUID(item.source_id), date.fromisoformat(item.metadata["read_as_of_date"])
    original = state["originals"][key]
    if defect == "text":
        state["originals"][key] = replace(original, text="new text")
    elif defect == "hash":
        item.text_hash = "0" * 64
    elif defect == "duplicate_retained_hash":
        state["items"][-1].text_hash = "0" * 64
    elif defect == "ordinal":
        state["originals"][key] = replace(original, projection_ordinal=15)
    elif defect == "wrong_date":
        state["broker"].filters.as_of_date = date(2027, 1, 1)
    elif defect == "publication":
        state["batch"].side_effect = PermissionError("revoked at final fence")
    elif defect in ("missing", "ambiguous"):
        read = state["batch"].side_effect

        def changed(**kwargs: Any) -> Any:
            result = read(**kwargs)
            source = next(iter(result))
            current = result[source]
            result[source] = replace(
                current, chunks=() if defect == "missing" else current.chunks * 2
            )
            return result

        state["batch"].side_effect = changed
    context = RunContext(services={"asv3_exp_cropped": True})
    if defect == "cancel":

        def active() -> None:
            if state["batch"].call_count:
                raise RuntimeError("cancelled")

        monkeypatch.setattr(context, "check_active", active)
    with pytest.raises((CorpusScopeUnavailable, PermissionError, RuntimeError)):
        state["broker"].revalidate_evidence(state["items"], context)


@pytest.mark.parametrize("cropped", [None, False, "true"])
def test_non_cropped_revalidation_never_uses_packet(
    monkeypatch: pytest.MonkeyPatch, cropped: object
) -> None:
    state = setup_packet(monkeypatch)
    legacy = MagicMock()
    monkeypatch.setattr(corpus_tools, "run_functions_tuples_in_parallel", legacy)
    state["broker"].revalidate_evidence(
        state["items"], RunContext(services={"asv3_exp_cropped": cropped})
    )
    state["batch"].assert_not_called()
    assert len(legacy.call_args.args[0]) == 3
