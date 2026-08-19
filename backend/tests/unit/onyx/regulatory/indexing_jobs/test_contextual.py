import datetime
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
from onyx.llm.constants import LlmProviderNames
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.prompts.contextual_retrieval import (
    CONTEXTUAL_RAG_PROMPT1,
    CONTEXTUAL_RAG_PROMPT2,
)
from onyx.regulatory import contextual as regulatory_contextual
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
            validity_start_date=None,
            validity_end_date=None,
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
        embedding_tokenizer=_CharacterTokenizer(),
        contextual_tokenizer=_CharacterTokenizer(),
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
        contextual_tokenizer=_CharacterTokenizer(),
    )

    assert len(request.prompt) <= 855
    assert "MADDE 1" in request.prompt
    assert "MADDE 3" in request.prompt
    assert "MADDE 2 - Hedef hüküm." in request.prompt


def test_contextual_snapshot_selects_target_version_and_excludes_ambiguous_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job()
    old = _row(
        job,
        row_id="rc_old",
        position=0,
        text="MADDE 1 - Eski hüküm.",
        heading_path=["MADDE 1"],
    )
    old.validity_start_date = datetime.date(2020, 1, 1)
    old.validity_end_date = datetime.date(2024, 1, 1)
    replacement = _row(
        job,
        row_id="rc_replacement",
        position=0,
        text="MADDE 1 - Yeni hüküm.",
        heading_path=["MADDE 1"],
    )
    replacement.validity_start_date = datetime.date(2024, 1, 1)
    replacement.validity_end_date = None
    stable = _row(
        job,
        row_id="rc_stable",
        position=1,
        text="MADDE 2 - Ortak hüküm.",
        heading_path=["MADDE 2"],
    )
    stable.validity_start_date = None
    stable.validity_end_date = None
    overlap_a = _row(
        job,
        row_id="rc_overlap_a",
        position=2,
        text="MADDE 3 - Çelişen A.",
        heading_path=["MADDE 3"],
    )
    overlap_b = _row(
        job,
        row_id="rc_overlap_b",
        position=2,
        text="MADDE 3 - Çelişen B.",
        heading_path=["MADDE 3"],
    )
    for row in (overlap_a, overlap_b):
        row.validity_start_date = None
        row.validity_end_date = None
    rows = [replacement, overlap_b, stable, old, overlap_a]

    old_snapshot = regulatory_contextual.visible_regulatory_snapshot_for_target(
        rows, old, today=datetime.date(2026, 8, 19)
    )
    replacement_snapshot = regulatory_contextual.visible_regulatory_snapshot_for_target(
        rows, replacement, today=datetime.date(2026, 8, 19)
    )

    assert [row.id for row in old_snapshot] == [old.id, stable.id]
    assert [row.id for row in replacement_snapshot] == [replacement.id, stable.id]

    monkeypatch.setattr(
        contextual, "get_max_input_tokens", lambda *_args, **_kwargs: 100_000
    )
    request = contextual.contextual_request_for_row(
        job,
        rows,
        old,
        contextual_tokenizer=_CharacterTokenizer(),
    )
    assert "Eski hüküm" in request.prompt
    assert "Yeni hüküm" not in request.prompt
    assert "Ortak hüküm" in request.prompt
    assert "Çelişen A" not in request.prompt
    assert "Çelişen B" not in request.prompt


def test_contextual_budget_and_embedding_reserve_use_distinct_tokenizers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _OneTokenTokenizer(_CharacterTokenizer):
        def encode(self, string: str) -> list[int]:
            return [1] if string else []

        def decode(self, tokens: list[int]) -> str:
            return "representative" if tokens else ""

    job = _job()
    first = _row(
        job,
        row_id="rc_first",
        position=0,
        text="MADDE 1 - " + "uzun " * 120,
        heading_path=["MADDE 1"],
    )
    second = _row(
        job,
        row_id="rc_second",
        position=1,
        text="MADDE 2 - Kısa hüküm.",
        heading_path=["MADDE 2"],
    )
    for row in (first, second):
        row.validity_start_date = None
        row.validity_end_date = None
    monkeypatch.setattr(
        contextual, "get_max_input_tokens", lambda *_args, **_kwargs: 900
    )

    reserve = contextual.contextual_reserve_for_row(
        [first, second],
        first,
        embedding_tokenizer=_CharacterTokenizer(),
    )
    request = contextual.contextual_request_for_row(
        job,
        [first, second],
        second,
        contextual_tokenizer=_OneTokenTokenizer(),
    )

    assert reserve == 0
    assert "MADDE 1" in request.prompt


def test_vertex_contextual_factory_conservatively_fits_multibyte_turkish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = contextual.get_contextual_token_budget_tokenizer(
        model_provider=LlmProviderNames.VERTEX_AI,
        model_name="gemini-3.1-flash-lite",
    )
    assert not isinstance(tokenizer, _CharacterTokenizer)
    turkish_text = "Gümrük yükümlülüğü ölçülü yürütülür."
    encoded = tokenizer.encode(turkish_text)

    assert len(encoded) == len(turkish_text.encode("utf-8"))
    assert len(encoded) > len(turkish_text)
    assert tokenizer.decode(encoded) == turkish_text
    boundary_fragment = tokenizer.decode(tokenizer.encode("ölçü")[1:-1])
    assert "�" not in boundary_fragment
    boundary_fragment.encode("utf-8")

    job = _job()
    first = _row(
        job,
        row_id="rc_turkish_context",
        position=0,
        text="MADDE 1 - " + turkish_text * 80,
        heading_path=["BİRİNCİ BÖLÜM", "MADDE 1"],
    )
    target = _row(
        job,
        row_id="rc_target",
        position=1,
        text="MADDE 2 - Ölçülü işlem yapılır.",
        heading_path=["BİRİNCİ BÖLÜM", "MADDE 2"],
    )
    contextual_input_limit = 1_200
    monkeypatch.setattr(
        contextual,
        "get_max_input_tokens",
        lambda *_args, **_kwargs: contextual_input_limit,
    )

    request = contextual.contextual_request_for_row(
        job,
        [first, target],
        target,
        contextual_tokenizer=tokenizer,
    )
    safe_byte_token_limit = int(
        contextual_input_limit * (1 - contextual.GEN_AI_INPUT_TOKEN_SAFETY_MARGIN)
    )

    assert len(request.prompt.encode("utf-8")) <= safe_byte_token_limit
    assert "�" not in request.prompt
    assert "MADDE 1" in request.prompt
    assert "MADDE 2 - Ölçülü işlem yapılır." in request.prompt


def test_contextual_token_budget_factory_rejects_non_vertex_contract() -> None:
    with pytest.raises(ValueError, match="Vertex AI"):
        contextual.get_contextual_token_budget_tokenizer(
            model_provider=LlmProviderNames.OPENAI,
            model_name="gemini-3.1-flash-lite",
        )


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
        contextual_tokenizer=_CharacterTokenizer(),
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
        embedding_tokenizer=_CharacterTokenizer(),
        db_session=cast(Session, SimpleNamespace()),
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
        contextual_tokenizer=_CharacterTokenizer(),
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
        embedding_tokenizer=_CharacterTokenizer(),
        db_session=cast(Session, SimpleNamespace()),
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


def test_apply_results_rejects_canonical_row_owned_by_another_file() -> None:
    job = _job()
    row = _row(
        job,
        row_id="rc_foreign",
        position=0,
        text="MADDE 1 - Yabancı belge hükmü.",
        heading_path=["MADDE 1"],
    )
    row.user_file_id = uuid4()
    item = cast(
        RegulatoryIndexingItem,
        SimpleNamespace(
            id=uuid4(),
            job_id=job.id,
            regulatory_chunk_id=row.id,
            request_hash="f" * 64,
            status=RegulatoryIndexingItemStatus.PENDING.value,
            context=None,
        ),
    )

    with pytest.raises(
        contextual.ContextualMappingError,
        match="canonical rows do not belong to the indexing job",
    ):
        apply_contextual_results(
            job,
            [row],
            [item],
            {},
            embedding_tokenizer=_CharacterTokenizer(),
            db_session=cast(Session, SimpleNamespace()),
        )
