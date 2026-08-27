from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID

import pytest

from onyx.regulatory.amendments import job
from onyx.regulatory.amendments.models import AmendmentInstruction, MatchResult
from onyx.regulatory.amendments.ranker import CandidateChunk

_CREATOR_ID = UUID("00000000-0000-0000-0000-000000000321")


def _patch_empty_retriever(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    retriever = MagicMock()
    retriever.search.return_value = []
    monkeypatch.setattr(
        job,
        "build_amendment_search_retriever",
        MagicMock(return_value=retriever),
    )
    return retriever


def test_unmatched_initial_search_gets_one_recovery_before_giving_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text="Carnets can now cover up to 8 customs offices.",
        search_query=(
            "TIR karnesi kapsamında işlem yapılabilecek gümrük idaresi sayısı kaçtır?"
        ),
        recovery_query="TIR karnesi gümrük idaresi azami sayı",
    )
    candidate = CandidateChunk(
        chunk_id="tir-chunk-6",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="Bir TIR taşıması en fazla sekiz gümrük idaresini kapsayabilir.",
    )
    retriever = MagicMock()
    retriever.search.side_effect = [[], [candidate]]
    expected_match = MatchResult(
        old_chunk_id="tir-chunk-6",
        confidence=0.93,
        rationale="The recovered chunk governs the customs-office limit.",
    )
    confirm = MagicMock(return_value=expected_match)
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)

    candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=instruction,
    )

    assert candidates == [candidate]
    assert match == expected_match
    assert retriever.search.call_args_list[0].kwargs == {
        "instruction": instruction,
        "recovery": False,
    }
    assert retriever.search.call_args_list[1].kwargs == {
        "instruction": instruction,
        "recovery": True,
    }
    confirm.assert_called_once()


def test_match_rejection_gets_only_one_recovery_and_rechecks_merged_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instruction = AmendmentInstruction(
        instruction_text="Risk-inspection criteria were updated.",
        search_query="TIR işlemlerinde risk kriterlerine göre muayene nasıl yapılır?",
        recovery_query="TIR risk kriterleri muayene kontrol",
    )
    first = CandidateChunk(
        chunk_id="wrong",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="Yetkilendirilmiş yükümlü şartları.",
    )
    recovered = CandidateChunk(
        chunk_id="risk",
        user_file_id="00000000-0000-0000-0000-000000000123",
        text="Risk kriterlerine göre fiziki muayeneye sevk edilir.",
    )
    retriever = MagicMock()
    retriever.search.side_effect = [[first], [recovered]]
    expected_match = MatchResult(
        old_chunk_id="risk", confidence=0.91, rationale="Recovered target"
    )
    confirm = MagicMock(side_effect=[None, expected_match])
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)

    candidates, match = job.retrieve_and_confirm_instruction(
        retriever=retriever,
        llm=MagicMock(),
        instruction=instruction,
    )

    assert candidates == [first, recovered]
    assert match == expected_match
    assert retriever.search.call_count == 2
    assert confirm.call_count == 2


def test_resume_reuses_segmentation_and_skips_completed_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=9,
        document_set_id=7,
        created_by=_CREATOR_ID,
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
    retriever = _patch_empty_retriever(monkeypatch)
    persist_unmatched = MagicMock(return_value=True)
    monkeypatch.setattr(job, "persist_unmatched_checkpoint", persist_unmatched)
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=9, lease_generation=3)

    segmentation.assert_not_called()
    assert [
        item.kwargs["instruction"].instruction_text
        for item in retriever.search.call_args_list
        if item.kwargs["recovery"] is False
    ] == ["MADDE 2"]
    persist_unmatched.assert_called_once()


def test_first_run_persists_segmentation_before_instruction_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = SimpleNamespace(
        id=10,
        document_set_id=7,
        created_by=_CREATOR_ID,
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

    def _persist_segmentation(*_args: object, **kwargs: Any) -> bool:
        batch.segmented_instructions = list(kwargs["instructions"])
        batch.reference_date = "2026-08-26"
        return True

    persist_segmentation = MagicMock(side_effect=_persist_segmentation)
    monkeypatch.setattr(job, "persist_segmentation_checkpoint", persist_segmentation)
    _patch_empty_retriever(monkeypatch)
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
        document_set_id=7,
        created_by=_CREATOR_ID,
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
    _patch_empty_retriever(monkeypatch)
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
        document_set_id=7,
        created_by=_CREATOR_ID,
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
    _patch_empty_retriever(monkeypatch)
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
        document_set_id=7,
        created_by=_CREATOR_ID,
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
    retriever = MagicMock()
    retriever.search.return_value = [candidate]
    monkeypatch.setattr(
        job,
        "build_amendment_search_retriever",
        MagicMock(return_value=retriever),
    )
    monkeypatch.setattr(job, "confirm_instruction_match", confirm)
    monkeypatch.setattr(job, "load_instruction_draft_context", load_context)
    monkeypatch.setattr(job, "draft_instruction_proposal", draft)
    monkeypatch.setattr(
        job, "persist_proposal_checkpoint", MagicMock(return_value=True)
    )
    monkeypatch.setattr(job, "mark_batch_analyzed", MagicMock(return_value=True))
    monkeypatch.setattr(job, "get_default_llm", MagicMock(return_value=MagicMock()))

    job.run_amendment_batch(batch_id=13, lease_generation=1)
