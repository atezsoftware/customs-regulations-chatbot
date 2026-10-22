"""Public stream and JSON adapters preserve every exact cited passage target."""

import json
from types import SimpleNamespace
from typing import cast

import pytest

from onyx.chat.chat_state import ChatStateContainer
from onyx.chat.chat_utils import reorganize_citations
from onyx.chat.citation_processor import DynamicCitationProcessor
from onyx.chat.models import AnswerStream
from onyx.chat.process_message import gather_stream, gather_stream_full
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    AgentResponseDelta,
    CitationInfo,
    Packet,
)
from tests.unit.onyx.chat.test_citation_processor import create_test_search_doc


def test_live_citations_expose_distinct_article_locations_for_the_same_file() -> None:
    processor = DynamicCitationProcessor()
    processor.update_citation_mapping(
        {
            97: create_test_search_doc(
                document_id="law",
                chunk_ind=1100,
                semantic_identifier="Gümrük Kanunu",
                metadata={"regulatory_heading_path": ["Gümrük Kanunu", "MADDE 241"]},
            ),
            250: create_test_search_doc(
                document_id="law",
                chunk_ind=1135,
                semantic_identifier="Gümrük Kanunu",
                metadata={"regulatory_heading_path": "Gümrük Kanunu > MADDE 242"},
            ),
        }
    )
    output = [
        *processor.process_token("Usulsüzlük [97]; itiraz [250]."),
        *processor.process_token(None),
    ]
    citations = [part for part in output if isinstance(part, CitationInfo)]
    assert [part.citation_number for part in citations] == [97, 250]
    assert [part.chunk_ind for part in citations] == [1100, 1135]
    assert [part.reference for part in citations] == [
        "Gümrük Kanunu > MADDE 241",
        "Gümrük Kanunu > MADDE 242",
    ]
    assert all(part.text is None and part.display_title is None for part in citations)


@pytest.mark.parametrize(
    ("heading", "reference"),
    [
        ("MADDE 241", "MADDE 241"),
        (["Gümrük Kanunu", "MADDE 241"], "Gümrük Kanunu > MADDE 241"),
        (None, None),
        (["MADDE 241", None], None),
        (241, None),
    ],
)
def test_recorded_location_ignores_malformed_metadata(
    heading: object, reference: str | None
) -> None:
    citation = CitationInfo(citation_number=97, document_id="law").with_source_metadata(
        {"regulatory_heading_path": heading}, link="https://example.test/law"
    )
    assert citation.reference == reference
    assert citation.link == "https://example.test/law"


@pytest.mark.parametrize("text", [None, "", " \n\t ", "Exact original\n"])
@pytest.mark.parametrize("full", [False, True])
def test_basic_and_full_json_preserve_all_stream_citation_fields(
    full: bool, text: str | None
) -> None:
    citations = [
        CitationInfo(
            citation_number=97,
            document_id="law",
            chunk_ind=1100,
            semantic_identifier="stored.md",
            display_title="Gümrük Kanunu",
            chunk_id="rc241",
            text=text,
            reference="MADDE 241",
            link="https://example.test/law",
            preview_url="/api/asv3/citation/6202/97",
            resolution_status="retained",
        ),
        CitationInfo(
            citation_number=250,
            document_id="law",
            chunk_ind=1135,
            semantic_identifier="stored.md",
            reference="MADDE 242",
        ),
        CitationInfo(
            citation_number=251,
            document_id="law",
            chunk_ind=-251,
            semantic_identifier="native page 2",
            text=text,
            preview_url="/api/asv3/citation/6202/251",
        ),
    ]
    stream = cast(
        AnswerStream,
        iter(
            [
                SimpleNamespace(reserved_assistant_message_id=6202),
                *[
                    Packet(placement=Placement(turn_index=0), obj=citation)
                    for citation in citations
                ],
                Packet(
                    placement=Placement(turn_index=0),
                    obj=AgentResponseDelta(
                        content="Rule [97]; appeal [250]; original [251]."
                    ),
                ),
            ]
        ),
    )
    result = (
        gather_stream_full(stream, ChatStateContainer())
        if full
        else gather_stream(stream)
    )
    payload = json.loads(result.model_dump_json())
    assert payload["citation_info"] == [
        json.loads(citation.model_dump_json()) for citation in citations
    ]
    assert payload["citation_info"][0]["text"] == text
    assert payload["citation_info"][2]["chunk_ind"] == -251
    assert payload["citation_info"][2]["chunk_id"] is None


def test_optional_renumbering_retains_complete_citation_presentation() -> None:
    original = CitationInfo(
        citation_number=97,
        document_id="law",
        chunk_ind=1100,
        semantic_identifier="stored.md",
        display_title="Official title",
        chunk_id="rc241",
        text="",
        reference="MADDE 241",
        link="https://example.test/law",
        preview_url="/api/asv3/citation/6202/97",
        resolution_status="retained",
    )
    answer, citations = reorganize_citations("Rule [[97]]()", [original])
    assert answer == "Rule [[1]]()"
    assert citations == [original.model_copy(update={"citation_number": 1})]
    assert original.citation_number == 97
