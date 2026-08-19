from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import RETURN_SEPARATOR
from onyx.db.enums import RegulatoryIndexingItemStatus
from onyx.db.models import (
    RegulatoryChunk,
    RegulatoryIndexingItem,
    RegulatoryIndexingJob,
)
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.prompts.contextual_retrieval import (
    CONTEXTUAL_RAG_PROMPT1,
    CONTEXTUAL_RAG_PROMPT2,
)
from onyx.regulatory.indexing_jobs import contextual
from onyx.regulatory.indexing_jobs.contextual import (
    ContextApplySummary,
    apply_contextual_results,
    build_contextual_requests,
)
from onyx.regulatory.indexing_jobs.vertex_batch import (
    VertexBatchRequest,
    VertexBatchResult,
    VertexBatchResultError,
)


class _CharacterTokenizer(BaseTokenizer):
    def encode(self, string: str) -> list[int]:
        return [ord(character) for character in string]

    def tokenize(self, string: str) -> list[str]:
        return list(string)

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


def _job() -> RegulatoryIndexingJob:
    return cast(
        RegulatoryIndexingJob,
        SimpleNamespace(
            id=uuid4(),
            user_file_id=uuid4(),
            lease_generation=7,
            config_snapshot={
                "vertex": {"model_name": "gemini-3.1-flash-lite"},
            },
        ),
    )


def _row(
    job: RegulatoryIndexingJob,
    *,
    row_id: str,
    position: int,
    text: str,
    heading_path: list[str],
) -> RegulatoryChunk:
    return cast(
        RegulatoryChunk,
        SimpleNamespace(
            id=row_id,
            user_file_id=job.user_file_id,
            position=position,
            text=text,
            heading_path=heading_path,
        ),
    )


def _document_block(rows: list[RegulatoryChunk]) -> str:
    return "\n\n".join(
        f"[Canonical position: {row.position}]\n"
        f"Heading path: {' > '.join(row.heading_path)}\n{row.text}"
        for row in rows
    )


def _chunk_block(row: RegulatoryChunk) -> str:
    return (
        f"[Canonical position: {row.position}]\n"
        f"Heading path: {' > '.join(row.heading_path)}\n{row.text}"
    )


def test_contextual_requests_use_ordered_canonical_rows_and_preserve_legal_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    first = _row(
        job,
        row_id="rc_z_first",
        position=0,
        text="MADDE 12 - (1) Transit rejiminde teminat aranır.",
        heading_path=["BİRİNCİ BÖLÜM", "MADDE 12"],
    )
    second = _row(
        job,
        row_id="rc_a_second",
        position=1,
        text="(2) 4458 sayılı Kanunun 84 üncü maddesi uygulanır.",
        heading_path=["BİRİNCİ BÖLÜM", "MADDE 12", "Fıkra 2"],
    )
    ordered_rows = [first, second]
    expected_first_prompt = CONTEXTUAL_RAG_PROMPT1.format(
        document=_document_block(ordered_rows)
    ) + CONTEXTUAL_RAG_PROMPT2.format(chunk=_chunk_block(first))
    expected_first_request = VertexBatchRequest(prompt=expected_first_prompt)
    expected_second_prompt = CONTEXTUAL_RAG_PROMPT1.format(
        document=_document_block(ordered_rows)
    ) + CONTEXTUAL_RAG_PROMPT2.format(chunk=_chunk_block(second))
    expected_second_request = VertexBatchRequest(prompt=expected_second_prompt)
    first_item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=first.id,
            request_hash=expected_first_request.request_hash,
            status=RegulatoryIndexingItemStatus.PENDING.value,
            context=None,
        ),
    )
    second_item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=second.id,
            request_hash=expected_second_request.request_hash,
            status=RegulatoryIndexingItemStatus.PENDING.value,
            context=None,
        ),
    )
    monkeypatch.setattr(
        contextual,
        "get_max_input_tokens",
        lambda *_args, **_kwargs: 100_000,
    )

    requests = build_contextual_requests(
        job,
        [second, first],
        [second_item, first_item],
        _CharacterTokenizer(),
    )

    assert requests == [expected_first_request, expected_second_request]
    assert "BİRİNCİ BÖLÜM > MADDE 12 > Fıkra 2" in requests[1].prompt
    assert "4458 sayılı Kanunun 84 üncü maddesi" in requests[1].prompt
    assert requests[1].prompt.index("MADDE 12 - (1)") < requests[1].prompt.index(
        "4458 sayılı"
    )


def test_contextual_request_fits_reconstructed_document_to_model_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    first = _row(
        job,
        row_id="rc_first",
        position=0,
        text="MADDE 1 - " + "belgenin başı " * 40,
        heading_path=["BİRİNCİ BÖLÜM", "MADDE 1"],
    )
    middle = _row(
        job,
        row_id="rc_middle",
        position=1,
        text="MADDE 2 - Hedef hüküm.",
        heading_path=["BİRİNCİ BÖLÜM", "MADDE 2"],
    )
    last = _row(
        job,
        row_id="rc_last",
        position=2,
        text="MADDE 3 - " + "belgenin sonu " * 40,
        heading_path=["İKİNCİ BÖLÜM", "MADDE 3"],
    )
    monkeypatch.setattr(
        contextual,
        "get_max_input_tokens",
        lambda *_args, **_kwargs: 900,
    )

    request = contextual.contextual_request_for_row(
        job,
        [last, middle, first],
        middle,
        _CharacterTokenizer(),
    )

    assert len(request.prompt) <= 855
    assert "MADDE 1" in request.prompt
    assert "MADDE 3" in request.prompt
    assert "MADDE 2 - Hedef hüküm." in request.prompt


def test_apply_results_persists_only_fitted_generated_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    row = _row(
        job,
        row_id="rc_first",
        position=0,
        text="MADDE 1 - Transit rejiminde teminat aranır.",
        heading_path=["MADDE 1"],
    )
    sibling = _row(
        job,
        row_id="rc_second",
        position=1,
        text="MADDE 2 - Teminat idarece belirlenir.",
        heading_path=["MADDE 2"],
    )
    request = contextual.contextual_request_for_row(
        job,
        [row, sibling],
        row,
        _CharacterTokenizer(),
    )
    item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=row.id,
            request_hash=request.request_hash,
            status=RegulatoryIndexingItemStatus.PENDING.value,
            context=None,
        ),
    )
    persisted: list[dict[str, object]] = []

    def persist_context(*_args: object, **kwargs: object) -> bool:
        persisted.append(cast(dict[str, object], kwargs["context"]))
        return True

    monkeypatch.setattr(
        contextual, "persist_regulatory_indexing_item_context", persist_context
    )

    summary = apply_contextual_results(
        job,
        [row, sibling],
        [item],
        {
            request.request_hash: VertexBatchResult(
                request_hash=request.request_hash,
                context="Bu parça, transit teminatının aranmasını düzenler.",
            )
        },
        _CharacterTokenizer(),
        cast(Session, SimpleNamespace()),
    )

    assert persisted == [
        {
            "contextual_text": (
                "Bu parça, transit teminatının aranmasını düzenler." + RETURN_SEPARATOR
            )
        }
    ]
    assert summary == ContextApplySummary(
        context_ready_count=1,
        failed_count=0,
        pending_count=0,
        skipped_count=0,
    )
    assert "MADDE 1" not in str(persisted)


def test_apply_results_records_typed_failure_without_replacing_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    rows = [
        _row(
            job,
            row_id=f"rc_{position}",
            position=position,
            text=f"MADDE {position + 1} - Transit hükmü.",
            heading_path=[f"MADDE {position + 1}"],
        )
        for position in range(2)
    ]
    failed_request = contextual.contextual_request_for_row(
        job,
        rows,
        rows[0],
        _CharacterTokenizer(),
    )
    failed_item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=rows[0].id,
            request_hash=failed_request.request_hash,
            status=RegulatoryIndexingItemStatus.PENDING.value,
            context=None,
        ),
    )
    successful_item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=rows[1].id,
            request_hash="b" * 64,
            status=RegulatoryIndexingItemStatus.CONTEXT_READY.value,
            context={"contextual_text": "Korunacak başarılı bağlam."},
        ),
    )
    failures: list[tuple[str, str]] = []
    contexts: list[dict[str, object]] = []

    def persist_failure(*_args: object, **kwargs: object) -> bool:
        failures.append(
            (cast(str, kwargs["error_code"]), cast(str, kwargs["error_message"]))
        )
        return True

    monkeypatch.setattr(
        contextual, "persist_regulatory_indexing_item_failure", persist_failure
    )
    monkeypatch.setattr(
        contextual,
        "persist_regulatory_indexing_item_context",
        lambda *_args, **kwargs: contexts.append(kwargs["context"]) or True,
    )

    summary = apply_contextual_results(
        job,
        rows,
        [failed_item, successful_item],
        {
            failed_request.request_hash: VertexBatchResult(
                request_hash=failed_request.request_hash,
                error=VertexBatchResultError.SAFETY,
            )
        },
        _CharacterTokenizer(),
        cast(Session, SimpleNamespace()),
    )

    assert failures == [("safety_blocked", "safety_blocked")]
    assert contexts == []
    assert successful_item.context == {"contextual_text": "Korunacak başarılı bağlam."}
    assert summary == ContextApplySummary(
        context_ready_count=1,
        failed_count=1,
        pending_count=0,
        skipped_count=0,
    )
