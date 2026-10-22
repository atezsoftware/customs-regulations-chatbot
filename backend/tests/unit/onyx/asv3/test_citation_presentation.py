"""Published citations retain one exact original while offering a file-level label."""

import copy
from dataclasses import replace
from typing import Any, cast

import pytest

from onyx.asv3 import runtime
from onyx.asv3.citation_presentation import (
    citation_presentation_enabled,
    evidence_citation_presentation,
)
from onyx.asv3.corpus_tools import CorpusBroker, evidence_for_chunk
from onyx.asv3.models import (
    EvidenceItem,
    OutcomeStatus,
    RunContext,
    ToolOutcome,
    ToolSpec,
)
from onyx.asv3.source_tools import original_evidence
from onyx.asv3.workflow_variant import (
    ASV3_EXP_CHEAP_VARIANT,
    ASV3_EXP_CROPPED_VARIANT,
    ASV3_GUARDED_EXPERIMENTAL_VARIANT,
    ASV3_GUARDRAILS_V2_VARIANT,
    ASV3_GUARDRAILS_V3_VARIANT,
    ASV3_STANDARD_VARIANT,
    ASV3_TUNED_VARIANT,
)
from onyx.llm.model_response import ModelResponse
from onyx.regulatory.legistation_metadata import LegistationMetadata
from onyx.server.query_and_chat.streaming_models import AgentResponseStart, CitationInfo
from tests.unit.onyx.asv3.test_cropped_focused_repair_runtime import (
    priced_scripted_provider as priced_scripted_provider,
)
from tests.unit.onyx.asv3.test_runtime import packets, response, setup_run

pytestmark = pytest.mark.usefixtures(
    "empty_source_inventory", "priced_scripted_provider"
)


@pytest.mark.parametrize(
    "variant,profile,parallel,enabled",
    [
        (ASV3_STANDARD_VARIANT, "normal", False, True),
        (ASV3_EXP_CROPPED_VARIANT, "normal", False, True),
        (ASV3_STANDARD_VARIANT, "deep", False, False),
        (ASV3_STANDARD_VARIANT, "experimental", True, False),
        (ASV3_TUNED_VARIANT, "normal", False, False),
        (ASV3_EXP_CHEAP_VARIANT, "normal", False, False),
        (ASV3_GUARDED_EXPERIMENTAL_VARIANT, "normal", False, False),
        (ASV3_GUARDRAILS_V2_VARIANT, "normal", False, False),
        (ASV3_GUARDRAILS_V3_VARIANT, "normal", False, False),
    ],
)
def test_enrichment_only_applies_to_requested_workflows(
    variant: str, profile: str, parallel: bool, enabled: bool
) -> None:
    assert citation_presentation_enabled(variant, profile, parallel) is enabled


@pytest.mark.parametrize(
    "title",
    [None, "", "   ", "2024/9366 Sayılı Karar / Standart Depo Muafiyeti"],
)
def test_optional_title_preserves_original_identity_and_exact_passage(
    monkeypatch: pytest.MonkeyPatch, title: str | None
) -> None:
    _, broker, _, _, _ = setup_run(monkeypatch)
    source = replace(
        broker.sources[0],
        legistation_metadata=LegistationMetadata(official_title=title),
    )
    original = evidence_for_chunk(source, broker.chunks[str(source.id)])
    assert original.search_doc is not None
    before = original.model_dump(mode="json")
    result = evidence_citation_presentation(original)
    assert result.display_title == (
        title if title and title.strip() else original.search_doc.semantic_identifier
    )
    assert result.text == original.text and result.chunk_id == original.chunk_id
    assert original.model_dump(mode="json") == before


def test_native_original_keeps_its_locator_and_has_no_canonical_chunk_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, broker, _, _, _ = setup_run(monkeypatch)
    source = broker.sources[0]
    original = evidence_for_chunk(source, broker.chunks[str(source.id)])
    assert original.search_doc is not None
    original.chunk_id = None
    original.search_doc.chunk_ind = -1
    original.search_doc.metadata = {
        "asv3_native_locator": '{"page":2}',
        "asv3_citation_preview_url": "/api/asv3/citation/2/1",
    }
    before = original.model_dump(mode="json")
    result = evidence_citation_presentation(original)
    assert result.chunk_id is None
    assert result.text == original.text
    assert result.display_title == original.search_doc.semantic_identifier
    assert original.model_dump(mode="json") == before


@pytest.mark.parametrize(
    "variant,profile,enriched",
    [
        (ASV3_STANDARD_VARIANT, "normal", True),
        (ASV3_EXP_CROPPED_VARIANT, "normal", True),
        (ASV3_TUNED_VARIANT, "normal", False),
        (ASV3_STANDARD_VARIANT, "deep", False),
    ],
)
def test_actual_publication_keeps_two_chunks_of_one_file_separate(
    monkeypatch: pytest.MonkeyPatch, variant: str, profile: str, enriched: bool
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile=profile, workflow_variant=variant)
    title = "2024/9366 Sayılı Karar / Tam Resmî Başlık"
    source = replace(
        broker.sources[0],
        legistation_metadata=LegistationMetadata(official_title=title),
    )
    original = broker.chunks[str(source.id)]
    originals = [
        original,
        replace(
            original,
            id="chunk-1",
            position=1,
            projection_ordinal=1,
            text="İkinci özgün hüküm.\n\n  Boşluklar ve son satır korunur.\n",
        ),
    ]
    by_id = {chunk.id: chunk for chunk in originals}

    def page(
        source_id: str, context: RunContext, **_options: Any
    ) -> tuple[Any, list[Any], bool]:
        context.check_active()
        assert source_id == str(source.id)
        return source, originals, False

    def revalidate(items: list[EvidenceItem], context: RunContext) -> None:
        context.check_active()
        for item in items:
            assert item.chunk_id is not None
            assert item.text == by_id[item.chunk_id].text
        broker.revalidated.extend(items)

    monkeypatch.setattr(broker, "page", page)
    monkeypatch.setattr(broker, "revalidate_evidence", revalidate)
    decisions = 0

    def scripted(**_arguments: Any) -> ModelResponse:
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            return response(
                calls=[("read_source_range", {"source_id": str(source.id)})]
            )
        assert decisions == 2
        return response("İlk koşul [1]; ikinci koşul [2].")

    llm.invoke.side_effect = scripted
    if profile == "deep":
        from tests.unit.onyx.asv3.test_runtime import run_independent

        run_independent(**kwargs)
    else:
        runtime.run_asv3_loop(**kwargs)
    assert decisions == 2
    emitted = packets(queue)
    citations = [
        packet.obj for packet in emitted if isinstance(packet.obj, CitationInfo)
    ]
    assert [item.citation_number for item in citations] == [1, 2]
    assert len({item.document_id for item in citations}) == 1
    assert [item.chunk_ind for item in citations] == [0, 1]
    for number, (citation, original) in enumerate(
        zip(citations, originals, strict=True), 1
    ):
        assert citation.preview_url == f"/api/asv3/citation/2/{number}"
        assert citation.chunk_id == (original.id if enriched else None)
        assert citation.text == (original.text if enriched else None)
        assert citation.display_title == (title if enriched else None)
        assert citation.semantic_identifier is not None
        assert citation.semantic_identifier != title
    published = next(
        packet.obj for packet in emitted if isinstance(packet.obj, AgentResponseStart)
    )
    assert published.final_documents is not None
    assert len(published.final_documents) == 2
    assert [
        doc.metadata.get("asv3_source_display_name")
        for doc in published.final_documents
    ] == ([title, title] if enriched else [None, None])
    saved = copy.deepcopy(checkpoints[-1]["evidence"]["records"])
    assert [row["item"]["text"] for row in saved] == [item.text for item in originals]
    assert all(
        "asv3_source_display_name" not in row["item"]["search_doc"]["metadata"]
        for row in saved
    )
    assert broker.revalidated
    assert checkpoints[-1]["publication_status"] == "found"
    assert all(
        title not in message.model_dump_json()
        for call in llm.invoke.call_args_list
        for message in call.kwargs["prompt"]
    )


@pytest.mark.parametrize(
    "variant,enriched",
    [
        (ASV3_STANDARD_VARIANT, True),
        (ASV3_EXP_CROPPED_VARIANT, True),
        (ASV3_TUNED_VARIANT, False),
    ],
)
def test_native_title_uses_existing_authorized_source_read_only_for_presentation(
    monkeypatch: pytest.MonkeyPatch, variant: str, enriched: bool
) -> None:
    kwargs, broker, llm, checkpoints, queue = setup_run(monkeypatch)
    kwargs.pop("test_language")
    kwargs.update(research_profile="normal", workflow_variant=variant)
    title = "2024/9366 Sayılı Karar / Özgün Belgenin Resmî Başlığı"
    source = replace(
        broker.sources[0],
        legistation_metadata=LegistationMetadata(official_title=title),
    )
    original = original_evidence(
        str(source.id),
        "Yerel dosyadan çıkarılan tam hüküm.\n  Son satır.\n",
        "a" * 64,
        {"page": 2},
    )
    before = original.model_dump(mode="json")
    source_reads = 0

    def authorized_source(source_id: str, context: RunContext) -> Any:
        nonlocal source_reads
        context.check_active()
        assert source_id == str(source.id)
        source_reads += 1
        return source

    def attach(
        item: EvidenceItem, number: int, message_id: int, context: RunContext
    ) -> EvidenceItem:
        return CorpusBroker.attach_native_citation(
            cast(CorpusBroker, broker), item, number, message_id, context
        )

    def read_native(_args: dict[str, Any], context: RunContext) -> ToolOutcome:
        context.check_active()
        return ToolOutcome(
            status=OutcomeStatus.FOUND,
            summary="Original page read",
            evidence=[original],
        )

    def revalidate(items: list[EvidenceItem], context: RunContext) -> None:
        context.check_active()
        for item in items:
            assert item.text == original.text and item.text_hash == original.text_hash
        broker.revalidated.extend(items)

    monkeypatch.setattr(broker, "source", authorized_source, raising=False)
    monkeypatch.setattr(broker, "attach_native_citation", attach, raising=False)
    monkeypatch.setattr(broker, "revalidate_evidence", revalidate)
    monkeypatch.setattr(
        runtime,
        "build_corpus_specs",
        lambda _broker, **_options: [
            ToolSpec(
                name="read_native",
                description="Read a native page",
                parameters={"type": "object", "properties": {}},
                handler=read_native,
            )
        ],
    )
    decisions = 0

    def scripted(**_arguments: Any) -> ModelResponse:
        nonlocal decisions
        decisions += 1
        if decisions == 1:
            return response(calls=[("read_native", {})])
        assert decisions == 2
        return response("Özgün hükmün sonucu uygulanır [1].")

    llm.invoke.side_effect = scripted
    runtime.run_asv3_loop(**kwargs)
    assert decisions == 2 and source_reads == 1
    output = packets(queue)
    citation = next(
        packet.obj for packet in output if isinstance(packet.obj, CitationInfo)
    )
    assert citation.chunk_ind == -1 and citation.chunk_id is None
    assert citation.preview_url == "/api/asv3/citation/2/1"
    assert citation.text == (original.text if enriched else None)
    assert citation.display_title == (title if enriched else None)
    assert (
        citation.semantic_identifier is not None
        and title not in citation.semantic_identifier
    )
    publication = next(
        packet.obj for packet in output if isinstance(packet.obj, AgentResponseStart)
    )
    assert publication.final_documents is not None
    assert publication.final_documents[0].metadata.get("asv3_source_display_name") == (
        title if enriched else None
    )
    assert original.model_dump(mode="json") == before
    saved_item = checkpoints[-1]["evidence"]["records"][0]["item"]
    assert saved_item["text"] == original.text
    assert "legistation_metadata" not in saved_item["metadata"]
    assert title not in str(saved_item)
    assert all(
        title not in message.model_dump_json()
        for call in llm.invoke.call_args_list
        for message in call.kwargs["prompt"]
    )
