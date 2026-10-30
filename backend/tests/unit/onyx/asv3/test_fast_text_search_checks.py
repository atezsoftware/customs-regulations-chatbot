"""Fast text scans reuse the producer's fresh check, with a final fresh fence."""

from collections.abc import Iterator
from unittest.mock import MagicMock

import pytest
from pydantic import JsonValue

from onyx.asv3 import corpus_tools
from onyx.asv3.models import RunContext, RunStopped
from onyx.db.asv3_corpus import CorpusChunk
from tests.unit.onyx.asv3.test_cropped_text_search_operation import setup_scan


@pytest.mark.parametrize("fast", [None, False, True, 1, "true"])
@pytest.mark.parametrize("cropped", [False, True])
@pytest.mark.parametrize(
    "mode,pattern", [("literal", "istanbul"), ("regex", "[Aa]raştırma")]
)
def test_same_chunks_results_and_pages_with_one_fresh_check_per_yield(
    monkeypatch: pytest.MonkeyPatch,
    fast: object,
    cropped: bool,
    mode: str,
    pattern: str,
) -> None:
    state, broker = setup_scan(monkeypatch)
    spec = next(
        s
        for s in corpus_tools.build_corpus_specs(broker)
        if s.name == "search_source_text"
    )
    args: dict[str, JsonValue] = {
        "source_id": str(state["source"].id),
        "pattern": pattern,
        "mode": mode,
    }
    before_check = MagicMock(return_value=False)
    before = spec.handler(
        args, RunContext(services={"asv3_exp_cropped": cropped}, cancelled=before_check)
    )
    pages = list(state["pages"])
    state["pages"].clear()
    after_check = MagicMock(return_value=False)
    after = spec.handler(
        args,
        RunContext(
            services={"asv3_exp_cropped": cropped, "asv3_fast": fast},
            cancelled=after_check,
        ),
    )
    assert after.model_dump(mode="json") == before.model_dump(mode="json")
    assert state["pages"] == pages
    assert before_check.call_count - after_check.call_count == (
        1999 if fast is True else 0
    )


@pytest.mark.parametrize("stop", ["parent", "remote", "error", "deadline"])
def test_every_page_still_rechecks_parent_remote_stop_errors_and_deadline(
    monkeypatch: pytest.MonkeyPatch, stop: str
) -> None:
    stopped = False

    def remote_check() -> bool:
        if stopped and stop == "error":
            raise RuntimeError("stop cache unavailable")
        return stopped and stop == "remote"

    parent = RunContext(cancelled=remote_check, services={"asv3_fast": True})
    child = parent.child()

    def stop_at_page(page: int) -> None:
        nonlocal stopped
        if page != 2:
            return
        stopped = True
        if stop == "parent":
            parent.cancel()
        if stop == "deadline":
            child.deadline = 0

    state, broker = setup_scan(monkeypatch, after_page=stop_at_page)
    spec = next(
        s
        for s in corpus_tools.build_corpus_specs(broker)
        if s.name == "search_source_text"
    )
    error = RuntimeError if stop == "error" else RunStopped
    with pytest.raises(error):
        spec.handler(
            {"source_id": str(state["source"].id), "pattern": "istanbul"}, child
        )
    assert len(state["pages"]) == 2


def test_stream_bound_to_another_context_cannot_supply_the_handlers_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, broker = setup_scan(monkeypatch, count=1)
    other = RunContext()
    stream = corpus_tools.ChunkScan(broker, str(state["source"].id), other)
    monkeypatch.setattr(broker, "iter_chunks", lambda *_args, **_kwargs: stream)
    spec = next(
        s
        for s in corpus_tools.build_corpus_specs(broker)
        if s.name == "search_source_text"
    )
    with pytest.raises(RunStopped):
        spec.handler(
            {"source_id": str(state["source"].id), "pattern": "istanbul"},
            RunContext(services={"asv3_fast": True}, cancelled=lambda: True),
        )


def test_cancel_after_last_chunk_is_fenced_before_returning_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, broker = setup_scan(monkeypatch, count=1)
    context = RunContext(services={"asv3_fast": True})
    add = corpus_tools.EvidenceSelection.add

    def cancel_after_match(
        selection: corpus_tools.EvidenceSelection, chunk: CorpusChunk
    ) -> None:
        add(selection, chunk)
        context.cancel()

    monkeypatch.setattr(corpus_tools.EvidenceSelection, "add", cancel_after_match)
    spec = next(
        s
        for s in corpus_tools.build_corpus_specs(broker)
        if s.name == "search_source_text"
    )
    with pytest.raises(RunStopped):
        spec.handler(
            {"source_id": str(state["source"].id), "pattern": "istanbul"}, context
        )


def test_overridden_stream_cannot_bypass_check_before_processing_the_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, broker = setup_scan(monkeypatch, count=1)
    identifier = str(state["source"].id)
    row = next(iter(corpus_tools.ChunkScan(broker, identifier, RunContext())))
    context = RunContext(services={"asv3_fast": True})

    class UncheckedScan(corpus_tools.ChunkScan):
        def __iter__(self) -> Iterator[CorpusChunk]:
            context.cancel()
            yield row

    stream = UncheckedScan(broker, identifier, context)
    monkeypatch.setattr(broker, "iter_chunks", lambda *_args, **_kwargs: stream)
    add = MagicMock()
    monkeypatch.setattr(corpus_tools.EvidenceSelection, "add", add)
    spec = next(
        s
        for s in corpus_tools.build_corpus_specs(broker)
        if s.name == "search_source_text"
    )
    with pytest.raises(RunStopped):
        spec.handler({"source_id": identifier, "pattern": "istanbul"}, context)
    add.assert_not_called()
