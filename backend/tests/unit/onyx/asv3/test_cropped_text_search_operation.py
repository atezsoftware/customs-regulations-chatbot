"""Paged literal and regex scans retain their complete result and final fences."""

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from pydantic import JsonValue

from onyx.asv3 import corpus_tools
from onyx.asv3.models import OutcomeStatus, RunContext, RunStopped
from onyx.db import asv3_corpus as corpus
from tests.unit.onyx.asv3.test_cropped_verified_source_operation import setup_operation


def setup_scan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    count: int = 2001,
    after_page: Callable[[int], None] | None = None,
) -> tuple[dict[str, Any], corpus_tools.CorpusBroker]:
    state = setup_operation(monkeypatch)
    source = state["source"]
    broker = corpus_tools.CorpusBroker(
        state["kwargs"]["user"],
        state["kwargs"]["filters"],
        query_indexes={source.id: state["index"]},
    )
    rows = [
        corpus.CorpusChunk(
            f"chunk-{n}",
            source.id,
            "İSTANBUL araştırma usulü — istanbul" if n % 200 == 0 else f"Original {n}",
            n,
            n,
            (f"MADDE {n}",),
            {"version": "retained"},
            None,
            None,
            "active",
        )
        for n in range(count)
    ]
    state["pages"] = []

    def read_page(
        _session: Any, **kwargs: Any
    ) -> tuple[list[corpus.CorpusChunk], bool]:
        start, limit = kwargs["start"], kwargs["limit"]
        state["pages"].append((start, limit))
        if after_page:
            after_page(len(state["pages"]))
        return rows[start : start + limit], start + limit < len(rows)

    monkeypatch.setattr(corpus, "_read_source_chunk_page", read_page)
    monkeypatch.setattr(
        corpus, "filter_publication_read", lambda _observation, rows, _key: rows
    )
    return state, broker


@pytest.mark.parametrize(
    "pattern,mode",
    [("istanbul", "literal"), ("[Aa]raştırma", "regex"), ("Absent", "literal")],
)
@pytest.mark.parametrize("count,start", [(2001, 0), (2001, 2000), (37, 0)])
def test_full_scan_result_unchanged_with_two_fresh_authorizations(
    monkeypatch: pytest.MonkeyPatch, pattern: str, mode: str, count: int, start: int
) -> None:
    state, broker = setup_scan(monkeypatch, count=count)
    tool = next(
        spec
        for spec in corpus_tools.build_corpus_specs(broker)
        if spec.name == "search_source_text"
    )
    args: dict[str, JsonValue] = {
        "source_id": str(state["source"].id),
        "pattern": pattern,
        "mode": mode,
        "start": start,
    }
    before = tool.handler(args, RunContext())
    old_pages = state["pages"].copy()
    old_authorizations = state["find"].call_count
    assert old_authorizations == 1 + 2 * len(old_pages)
    state["find"].reset_mock()
    state["pages"].clear()
    after = tool.handler(args, RunContext(services={"asv3_exp_cropped": True}))
    assert after.model_dump(mode="json") == before.model_dump(mode="json")
    assert state["pages"] == old_pages
    assert state["find"].call_count == 2
    assert (
        state["find"].call_args_list[0].args[0]
        is not state["find"].call_args_list[1].args[0]
    )
    if count == 2001 and start == 0:
        assert len(old_pages) == 20
        assert old_authorizations == 41
        assert after.status == OutcomeStatus.PARTIAL
        assert after.data["next_position"] == 2000
        assert after.data["scan_truncated"] is True


@pytest.mark.parametrize(
    "defect", ["acl", "identity", "publication", "cancel", "index"]
)
def test_actual_text_search_returns_no_evidence_when_final_fence_fails(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    context = RunContext(services={"asv3_exp_cropped": True})

    def mutate(page: int) -> None:
        if page != 20:
            return
        if defect == "acl":
            state["find"].return_value = ([], False)
        elif defect == "identity":
            state["find"].return_value = (
                [replace(state["source"], file_id="changed")],
                False,
            )
        elif defect == "publication":
            state["publication"].side_effect = PermissionError("publishing")
        elif defect == "cancel":
            context.cancel()
        else:
            broker.query_indexes[state["source"].id] = state["index"].model_copy(
                update={"index_uuid": "changed"}
            )

    state, broker = setup_scan(monkeypatch, after_page=mutate)
    tool = next(
        spec
        for spec in corpus_tools.build_corpus_specs(broker)
        if spec.name == "search_source_text"
    )
    if defect == "cancel":
        with pytest.raises(RunStopped, match="cancelled"):
            tool.handler(
                {"source_id": str(state["source"].id), "pattern": "istanbul"}, context
            )
        assert corpus._VERIFIED_SOURCE_OPERATION.get() is None
        return
    result = tool.handler(
        {"source_id": str(state["source"].id), "pattern": "istanbul"}, context
    )
    assert result.status in {
        OutcomeStatus.DENIED,
        OutcomeStatus.UNAVAILABLE,
        OutcomeStatus.CANCELLED,
    }
    assert result.evidence == []
    assert result.original_reads == []
    assert corpus._VERIFIED_SOURCE_OPERATION.get() is None


@pytest.mark.parametrize(
    "services",
    [
        {},
        {"asv3_exp_cropped": False},
        {"asv3_exp_cropped": "true"},
        {"asv3_exp_cropped": True, "asv3_exp_cheap": True},
    ],
)
def test_only_literal_cropped_uses_scan_operation(
    monkeypatch: pytest.MonkeyPatch, services: dict[str, object]
) -> None:
    state, broker = setup_scan(monkeypatch)
    tool = next(
        spec
        for spec in corpus_tools.build_corpus_specs(broker)
        if spec.name == "search_source_text"
    )
    result = tool.handler(
        {"source_id": str(state["source"].id), "pattern": "istanbul"},
        RunContext(services=services),
    )
    assert result.status == OutcomeStatus.PARTIAL
    assert state["find"].call_count == 41


def test_scan_byte_limit_and_evidence_limit_preserve_exact_continuation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus_tools, "MAX_SCAN_BYTES", 200)
    monkeypatch.setattr(corpus_tools, "MAX_RESPONSE_CHARS", 10)
    state, broker = setup_scan(monkeypatch)
    tool = next(
        spec
        for spec in corpus_tools.build_corpus_specs(broker)
        if spec.name == "search_source_text"
    )
    args: dict[str, JsonValue] = {
        "source_id": str(state["source"].id),
        "pattern": "istanbul",
    }
    old = tool.handler(args, RunContext())
    new = tool.handler(args, RunContext(services={"asv3_exp_cropped": True}))
    assert new.model_dump(mode="json") == old.model_dump(mode="json")
    assert new.status == OutcomeStatus.PARTIAL
    assert new.data["evidence_truncated"] is True
    assert new.data["evidence_next_position"] == 0
    position = new.data["next_position"]
    assert isinstance(position, int) and 0 < position < 100
