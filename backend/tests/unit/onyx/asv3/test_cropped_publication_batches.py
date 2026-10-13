"""Cropped exact-original batches retain every live authorization fence."""

from collections.abc import Generator
from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.asv3.models import RunContext
from onyx.db import asv3_corpus
from onyx.db.asv3_corpus import CorpusChunk, CorpusScopeUnavailable
from onyx.document_index.publication_models import PublicationIndexSnapshot
from onyx.regulatory.amendments.annexes.models import AnnexTemporalProjection
from tests.unit.onyx.asv3.test_search_hydration import (
    boundary,
    retained,
    row,
    validation_broker,
)


def read_batch(arguments: dict[str, Any], ids: tuple[str, ...]) -> list[CorpusChunk]:
    return list(
        asv3_corpus.iter_source_chunks_by_ids(
            **{
                key: value
                for key, value in arguments.items()
                if not key.startswith("_")
            },
            chunk_ids=ids,
            batch_publication_checks=True,
        )
    )


@pytest.mark.parametrize("cropped", [False, True])
def test_batch_optimization_is_cropped_only_and_keeps_all_exact_originals(
    monkeypatch: pytest.MonkeyPatch, cropped: bool
) -> None:
    source = uuid4()
    originals = [
        row(source, n, [f"MADDE {n + 1}"], f"Original condition {n}") for n in range(64)
    ]
    session, arguments = boundary(monkeypatch, originals)
    broker = validation_broker(monkeypatch, session, arguments)
    check = MagicMock(side_effect=lambda _observation, values, *_args: list(values))
    monkeypatch.setattr(asv3_corpus, "filter_publication_read", check)
    broker.revalidate_evidence(
        [retained(item) for item in originals],
        RunContext(services={"asv3_exp_cropped": cropped}),
    )
    # Cropped uses the shared packet entry/final publication fences, not per-row filters.
    assert check.call_count == (0 if cropped else len(originals))
    assert session.scalars.call_count == 1
    assert all(stream.closed for stream in arguments["_streams"])


@pytest.mark.parametrize(
    "defect", ["entry_acl", "exit_acl", "publication", "cancel", "read"]
)
def test_no_original_escapes_when_any_batch_fence_fails(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    source = uuid4()
    originals = [row(source, n, ["MADDE 3"]) for n in range(3)]
    session, arguments = boundary(monkeypatch, originals)
    source_check = asv3_corpus.require_source
    calls = 0

    def require(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if (defect == "entry_acl" and calls == 1) or (
            defect == "exit_acl" and calls == 2
        ):
            raise PermissionError("Source revoked")
        return source_check(*args, **kwargs)

    monkeypatch.setattr(asv3_corpus, "require_source", require)
    if defect == "publication":
        monkeypatch.setattr(
            asv3_corpus,
            "require_publication_files",
            MagicMock(side_effect=CorpusScopeUnavailable("Publication changed")),
        )
    elif defect == "cancel":
        active = 0

        def check_active() -> None:
            nonlocal active
            active += 1
            if active == len(originals) + 2:
                raise RuntimeError("Cancelled before delivery")

        arguments["check_active"] = check_active
    elif defect == "read":

        def fail_stream() -> Generator[object, None, None]:
            yield originals[0]
            raise RuntimeError("Interrupted original read")

        session.scalars.side_effect = lambda *_args, **_kwargs: fail_stream()
    received: list[CorpusChunk] = []
    with pytest.raises((PermissionError, CorpusScopeUnavailable, RuntimeError)):
        for chunk in asv3_corpus.iter_source_chunks_by_ids(
            **{
                key: value
                for key, value in arguments.items()
                if not key.startswith("_")
            },
            chunk_ids=tuple(item.id for item in originals),
            batch_publication_checks=True,
        ):
            received.append(chunk)
    assert received == []
    if defect == "entry_acl":
        session.scalars.assert_not_called()


def test_same_source_publication_filter_hides_entire_changed_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = uuid4()
    originals = [row(source, n, ["MADDE 3"]) for n in range(3)]
    _, arguments = boundary(monkeypatch, originals)
    check = MagicMock(return_value=[])
    final = MagicMock()
    monkeypatch.setattr(asv3_corpus, "filter_publication_read", check)
    monkeypatch.setattr(asv3_corpus, "require_publication_files", final)
    assert read_batch(arguments, tuple(item.id for item in originals)) == []
    assert [item.id for item in check.call_args.args[1]] == [
        item.id for item in originals
    ]
    final.assert_called_once()


@pytest.mark.parametrize("defect", ["text", "hash", "ordinal", "missing"])
def test_cropped_revalidation_still_rejects_changed_originals(
    monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    source = uuid4()
    original = row(source, 0, ["MADDE 10/A"])
    session, arguments = boundary(monkeypatch, [original])
    evidence = retained(original)
    broker = validation_broker(monkeypatch, session, arguments)
    if defect == "text":
        original.text += " changed"
    elif defect == "hash":
        evidence.text_hash = "0" * 64
    elif defect == "ordinal":
        original.projection_ordinal += 1
    else:
        original.status = "superseded"
    with pytest.raises(CorpusScopeUnavailable):
        broker.revalidate_evidence(
            [evidence], RunContext(services={"asv3_exp_cropped": True})
        )


def test_qualified_batch_uses_exact_snapshot_and_date_without_timeless_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import json

    source = uuid4()
    originals = [row(source, n, ["MADDE 3"]) for n in range(3)]
    session, arguments = boundary(monkeypatch, originals, as_of=date(2025, 12, 31))
    snapshot = cast(PublicationIndexSnapshot, object())
    arguments["index"] = snapshot
    monkeypatch.setattr(
        asv3_corpus, "qualified_file_ids", lambda *_args: frozenset([source])
    )
    calls: list[dict[str, Any]] = []

    def bindings(
        _session: Session, identifier: object, **kwargs: Any
    ) -> Generator[AnnexTemporalProjection, None, None]:
        calls.append(kwargs)
        assert identifier == source
        for original in originals:
            yield cast(
                AnnexTemporalProjection,
                SimpleNamespace(
                    derived_role="canonical",
                    semantic_position=original.position,
                    projection=SimpleNamespace(
                        ordinal=original.projection_ordinal,
                        source_json=json.dumps(
                            {
                                "regulatory_chunk_id": original.id,
                                "heading_path": original.heading_path,
                            }
                        ),
                    ),
                    representation_text=original.text,
                    representation_metadata={},
                    effective_start=None,
                    effective_end=None,
                ),
            )

    monkeypatch.setattr(asv3_corpus, "iter_public_temporal_bindings", bindings)
    chunks = read_batch(arguments, tuple(item.id for item in originals))
    assert [chunk.text for chunk in chunks] == [item.text for item in originals]
    assert len(calls) == 1
    assert calls[0]["index"] is snapshot
    assert calls[0]["as_of_date"] == date(2025, 12, 31)
    assert calls[0]["canonical_chunk_ids"] == tuple(item.id for item in originals)
    assert all(chunk.metadata["read_as_of_date"] == "2025-12-31" for chunk in chunks)
    session.scalars.assert_not_called()
    arguments["index"] = None
    with pytest.raises(CorpusScopeUnavailable, match="verified query index"):
        read_batch(arguments, tuple(item.id for item in originals))
