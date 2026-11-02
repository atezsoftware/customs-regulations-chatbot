"""Whole-table originals must survive legal selection and explicit canonical reads."""

import hashlib
import json
from uuid import uuid4

import pytest
from pydantic import JsonValue

from onyx.asv3.corpus_tools import build_corpus_specs
from onyx.asv3.models import CapabilityCall, OutcomeStatus, RunContext
from onyx.asv3.registry import CapabilityRegistry
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import BaseFilters, InferenceChunk
from onyx.context.search.utils import inference_section_from_single_chunk
from onyx.db.asv3_corpus import CorpusChunk
from onyx.natural_language_processing.utils import TiktokenTokenizer
from onyx.reranking.models import RerankOutcome, RerankResult
from onyx.tools.models import ToolResponse
from onyx.tools.tool_implementations.search.search_tool import _trim_sections_by_tokens
from shared_configs.configs import DOC_EMBEDDING_CONTEXT_SIZE
from tests.unit.onyx.asv3.test_corpus_source_sandbox import MemoryBroker
from tests.unit.onyx.tools.tool_implementations.search.test_search_tool_run import (
    _make_tool,
    _ordinary_chunk,
    _run,
)

# Exact largest table from the verified 20251220-3-1.pdf transcript.
_REAL_TABLE = r"""| (C) 6.0.2023<br>(R) 1.1.2025<br>(C) 23.5.2025 | | (C) 24.5.2023<br>(R) 1.1.2025<br>(C) 23.5.2025 | (C) 26.3.2020 | |
|---|---|---|---|---|
| | | | | (C) 26.3.2020 |
| (C) 1.4.2014<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | | | (C) 24.5.2023<br>(R) 1.1.2025<br>(C) 23.5.2025 |
| (C) 1.4.2014<br>(R) 1.1.2025 | | (C) 1.4.2014<br>(R) 1.1.2025 | | |
| | (C) 1.4.2014<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | | (C) 6.7.2023<br>(R) 1.1.2025<br>(C) 23.5.2025 |
| (C) 1.4.2014<br>(R) 1.1.2025<br>(C) 30.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025<br>(C) 30.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025<br>(C) 30.1.2025 | | |
| (C) 1.4.2014<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | | |
| (C) 1.2.2015<br>(R) 1.1.2025 | (C) 1.2.2015<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | | |
| (C) 1.4.2014<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | (C) 1.4.2014<br>(R) 1.1.2025 | | |
| (C) 1.8.2018 | (C) 1.6.2019 | (C) 1.10.2017 | (C) 29.4.2021 | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| | | | | |
| 1.5.2015<br>(C) 1.4.2022<br>(R) 1.1.2025 | (C) 1.5.2015<br>(R) 1.1.2025 | (C) 1.11.2024<br>(R) 1.1.2025 | (C) 1.9.2017<br>(R) 1.1.2025 | 1.6.2012 |
| 1.5.2015<br>(C) 1.4.2022<br>(R) 1.1.2025 | (C) 1.5.2015<br>(R) 1.1.2025 | (C) 1.9.2024<br>(R) 1.1.2025 | (C) 1.9.2017<br>(R) 1.1.2025 | 1.6.2012 |
| 1.2.2016<br>(C) 1.4.2022<br>(R) 1.1.2025 | (C) 1.5.2015<br>(R) 1.1.2025 | (C) 1.4.2025<br>(R) 1.4.2025 | (C) 1.5.2018<br>(R) 1.1.2025 | 1.6.2012 |
| (C) 1.5.2015<br>(R) 1.1.2025 | (C) 1.2.2015<br>(R) 1.1.2025 | (C) 1.12.2016<br>(R) 1.1.2025 | (C) 1.6.2018<br>(R) 1.1.2025 | (C) 1.1.2019<br>(R) 1.1.2025 |
| MK | RS | MD | GE | UA |"""


def _tokenizer() -> TiktokenTokenizer:
    return TiktokenTokenizer("text-embedding-3-large")


def _synthetic_table() -> str:
    lines = ["| Kod | Menşe koşulu |", "|---|---|"]
    while len(_tokenizer().encode("\n".join(lines))) < 1750:
        lines.append(
            f"| {len(lines)} | Tarife kodu, ürün açıklaması ve ortak menşe koşulu birlikte korunur. |"
        )
    lines.append("| SON_SATIR_KORUNDU | Ortak koşul bütün ürünlere uygulanır. |")
    result = "\n".join(lines)
    assert 1750 < len(_tokenizer().encode(result)) < 1848
    return result


def _chunk(text: str) -> InferenceChunk:
    return _ordinary_chunk(str(uuid4()), 0, text).model_copy(
        update={
            "regulatory_chunk_id": "whole-table",
            "heading_path": ["MADDE 1", "Tablo"],
            "metadata": {},
        }
    )


@pytest.mark.parametrize("synthetic", [False, True])
def test_usual_selection_budget_keeps_complete_table_and_final_row(
    synthetic: bool,
) -> None:
    table = _synthetic_table() if synthetic else _REAL_TABLE
    if not synthetic:
        assert len(_tokenizer().encode(table)) == 1143
    section = inference_section_from_single_chunk(_chunk(table))
    selected = _trim_sections_by_tokens(
        [section],
        max_tokens=25 * DOC_EMBEDDING_CONTEXT_SIZE,
        token_counter=lambda text: len(_tokenizer().encode(text)),
        max_chunks_per_section=1,
    )
    assert len(selected) == 1
    assert selected[0].combined_content == table
    assert selected[0].combined_content.endswith(table.splitlines()[-1])


@pytest.mark.parametrize("external_rerank,max_llm_chunks", [(False, 25), (True, 1)])
def test_actual_search_serialization_keeps_largest_source_table(
    external_rerank: bool, max_llm_chunks: int
) -> None:
    chunk = _chunk(_REAL_TABLE)
    rerank = RerankResult(
        ordered_chunks=[chunk],
        scores_by_chunk={(chunk.document_id, 0): 0.99},
        submitted_count=1,
        result_count=1,
        outcome=RerankOutcome.SUCCESS if external_rerank else RerankOutcome.DISABLED,
        fallback_used=not external_rerank,
    )
    responses: list[ToolResponse] = []
    _run(
        _make_tool(BaseFilters(regulatory_chunks_only=True), auto_detect_filters=False),
        connected_sources=[DocumentSource.USER_FILE],
        fused_chunks=[chunk],
        rerank_result=rerank,
        max_llm_chunks=max_llm_chunks,
        response_sink=responses,
        query="tablo menşe koşulları",
    )
    results = json.loads(responses[0].llm_facing_response)["results"]
    assert len(results) == 1 and results[0]["content"] == _REAL_TABLE
    assert results[0]["content"].endswith("| MK | RS | MD | GE | UA |")


def test_small_fallback_budget_omits_whole_table_without_clipping_or_rewriting_it() -> (
    None
):
    chunk = _chunk(_REAL_TABLE)
    section = inference_section_from_single_chunk(chunk)
    assert len(_tokenizer().encode(_REAL_TABLE)) + 75 > 2 * DOC_EMBEDDING_CONTEXT_SIZE
    assert (
        _trim_sections_by_tokens(
            [section],
            max_tokens=2 * DOC_EMBEDDING_CONTEXT_SIZE,
            token_counter=lambda text: len(_tokenizer().encode(text)),
        )
        == []
    )
    responses: list[ToolResponse] = []
    _run(
        _make_tool(BaseFilters(regulatory_chunks_only=True), auto_detect_filters=False),
        connected_sources=[DocumentSource.USER_FILE],
        fused_chunks=[chunk],
        max_llm_chunks=2,
        response_sink=responses,
        query="tablo menşe koşulları",
    )
    assert json.loads(responses[0].llm_facing_response)["results"] == []
    assert chunk.content == _REAL_TABLE


@pytest.mark.parametrize(
    "method", ["read_chunk", "read_source_range", "read_provision"]
)
def test_explicit_canonical_read_returns_full_1800_token_table(method: str) -> None:
    table = _synthetic_table()
    source_id = uuid4()
    row = CorpusChunk(
        id="whole-table",
        source_id=source_id,
        text=table,
        position=0,
        projection_ordinal=0,
        heading_path=("MADDE 1", "Tablo"),
        metadata={
            "article_no": "1",
            "intake_chunking_version": "4",
            "chunk_type": "table",
        },
        validity_start=None,
        validity_end=None,
        status="active",
    )
    broker = MemoryBroker([row])
    arguments: dict[str, JsonValue] = {"source_id": str(source_id)}
    if method == "read_chunk":
        arguments["chunk_id"] = row.id
    elif method == "read_provision":
        arguments["article"] = "1"
    outcome = CapabilityRegistry(build_corpus_specs(broker)).dispatch(
        CapabilityCall(name=method, arguments=arguments),
        RunContext(),
    )
    assert outcome.status == OutcomeStatus.FOUND
    assert len(outcome.evidence) == 1
    original = outcome.evidence[0]
    assert original.text == table and original.text.endswith(table.splitlines()[-1])
    assert original.text_hash == hashlib.sha256(table.encode()).hexdigest()
    assert original.chunk_id == row.id
    assert not outcome.data.get("evidence_truncated", False)
