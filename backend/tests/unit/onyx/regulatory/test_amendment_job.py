from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from onyx.regulatory.amendments import job


def test_resume_reuses_segmentation_and_skips_completed_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=9,
        raw_text="original",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[
            {"instruction_text": "MADDE 1"},
            {"instruction_text": "MADDE 2"},
        ],
        processed_instruction_count=1,
    )
    db_session = MagicMock()

    @contextmanager
    def _session():
        yield db_session

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    segmentation = MagicMock()
    monkeypatch.setattr(job, "segment_amendment_text", segmentation)
    find_candidates = MagicMock(return_value=[])
    monkeypatch.setattr(job, "find_candidates", find_candidates)
    persist_unmatched = MagicMock(return_value=True)
    monkeypatch.setattr(job, "persist_unmatched_checkpoint", persist_unmatched)
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=9, lease_generation=3)

    segmentation.assert_not_called()
    assert [
        item.kwargs["instruction"].instruction_text
        for item in find_candidates.call_args_list
    ] == ["MADDE 2"]
    persist_unmatched.assert_called_once()


def test_first_run_persists_segmentation_before_instruction_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=10,
        raw_text="MADDE 1",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[],
        processed_instruction_count=0,
    )
    db_session = MagicMock()

    @contextmanager
    def _session():
        yield db_session

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    segment_result = SimpleNamespace(
        reference_date="2026-08-26",
        instructions=[
            SimpleNamespace(model_dump=lambda: {"instruction_text": "MADDE 1"})
        ],
    )
    monkeypatch.setattr(
        job, "segment_amendment_text", MagicMock(return_value=segment_result)
    )

    def _persist_segmentation(*_args: object, **kwargs: object) -> bool:
        batch.segmented_instructions = list(kwargs["instructions"])
        batch.reference_date = "2026-08-26"
        return True

    persist_segmentation = MagicMock(side_effect=_persist_segmentation)
    monkeypatch.setattr(job, "persist_segmentation_checkpoint", persist_segmentation)
    monkeypatch.setattr(job, "find_candidates", MagicMock(return_value=[]))
    monkeypatch.setattr(
        job, "persist_unmatched_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=10, lease_generation=1)

    persist_segmentation.assert_called_once()


def test_segmentation_runs_without_an_open_database_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=11,
        raw_text="MADDE 1",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[],
        processed_instruction_count=0,
    )
    session_depth = 0

    @contextmanager
    def _session():
        nonlocal session_depth
        session_depth += 1
        try:
            yield MagicMock()
        finally:
            session_depth -= 1

    def segment(*_args: object) -> SimpleNamespace:
        assert session_depth == 0
        return SimpleNamespace(
            reference_date=None,
            instructions=[
                SimpleNamespace(model_dump=lambda: {"instruction_text": "MADDE 1"})
            ],
        )

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    monkeypatch.setattr(job, "segment_amendment_text", segment)
    monkeypatch.setattr(
        job, "persist_segmentation_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "find_candidates", MagicMock(return_value=[]))
    monkeypatch.setattr(
        job, "persist_unmatched_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=11, lease_generation=1)


def test_empty_segmentation_fails_instead_of_checkpointing_ambiguous_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=12,
        raw_text="belirsiz metin",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[],
        processed_instruction_count=0,
    )

    @contextmanager
    def _session():
        yield MagicMock()

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    monkeypatch.setattr(
        job,
        "segment_amendment_text",
        MagicMock(return_value=SimpleNamespace(reference_date=None, instructions=[])),
    )
    persist = MagicMock()
    monkeypatch.setattr(job, "persist_segmentation_checkpoint", persist)
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    with pytest.raises(RuntimeError, match="no instructions"):
        job.run_amendment_batch(batch_id=12, lease_generation=1)

    persist.assert_not_called()


def test_match_and_draft_llm_calls_run_outside_database_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=13,
        raw_text="MADDE 1",
        user_file_ids=["00000000-0000-0000-0000-000000000123"],
        reference_date=None,
        segmented_instructions=[{"instruction_text": "MADDE 1"}],
        processed_instruction_count=0,
    )
    session_depth = 0

    @contextmanager
    def _session():
        nonlocal session_depth
        session_depth += 1
        try:
            yield MagicMock()
        finally:
            session_depth -= 1

    candidate = SimpleNamespace(chunk_id="chunk-1")
    match = SimpleNamespace(old_chunk_id="chunk-1")
    context = SimpleNamespace()
    proposal = SimpleNamespace(instruction_index=0)

    def confirm(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert session_depth == 0
        return match

    def load_context(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert session_depth == 1
        return context

    def draft(*_args: object, **_kwargs: object) -> SimpleNamespace:
        assert session_depth == 0
        return proposal

    monkeypatch.setattr(job, "_session", _session)
    monkeypatch.setattr(job, "get_batch", lambda *_args: batch)
    monkeypatch.setattr(job, "find_candidates", MagicMock(return_value=[candidate]))
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)
    monkeypatch.setattr(job, "load_instruction_draft_context", load_context)
    monkeypatch.setattr(job, "draft_instruction_proposal", draft)
    monkeypatch.setattr(
        job, "persist_proposal_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=13, lease_generation=1)
