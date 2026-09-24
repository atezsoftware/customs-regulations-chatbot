"""Optional model-selected structural regulatory lookup."""

import json
from collections.abc import Callable
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session

from onyx.chat.emitter import Emitter
from onyx.context.search.models import IndexFilters, SearchDocsResponse
from onyx.context.search.utils import (
    convert_inference_sections_to_search_docs,
    inference_section_from_single_chunk,
)
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.regulatory_chunks import get_visible_regulatory_chunk_ids
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.regulatory.provision_lookup import (
    ProvisionLookupResult,
    ProvisionRequest,
    lookup_provision,
)
from onyx.server.query_and_chat.placement import Placement
from onyx.server.query_and_chat.streaming_models import (
    Packet,
    SearchToolDocumentsDelta,
    SearchToolQueriesDelta,
    SearchToolStart,
)
from onyx.tools.interface import Tool
from onyx.tools.models import ToolResponse
from onyx.tools.tool_implementations.utils import (
    convert_inference_sections_to_llm_string,
)
from onyx.utils.logger import setup_logger

logger = setup_logger()


class ProvisionToolOverrideKwargs(BaseModel):
    starting_citation_num: int = 1


class RegulatoryProvisionTool(Tool[ProvisionToolOverrideKwargs]):
    NAME = "get_regulatory_provision"
    DISPLAY_NAME = "Mevzuat Maddesi Bul"
    DESCRIPTION = (
        "Optional structural lookup of an identified legal instrument and article, "
        "with optional paragraph, clause and effective date. Useful when the source "
        "and provision are known; returns canonical citations and coverage status. "
        "Choose this or internal_search according to the evidence you need, in any order. "
        "Ambiguous, partial or missing results are not proof of absence; you may "
        "clarify the reference or use other internal_search modes. "
        "Do not invent source identifiers or interpret partial context as a verified subunit."
    )

    def __init__(
        self,
        *,
        tool_id: int,
        emitter: Emitter,
        document_index: DocumentIndex,
        filters_provider: Callable[[], IndexFilters],
    ) -> None:
        super().__init__(emitter)
        self._id = tool_id
        self.document_index = document_index
        self.filters_provider = filters_provider

    @property
    def id(self) -> int:
        return self._id

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESCRIPTION

    @property
    def display_name(self) -> str:
        return self.DISPLAY_NAME

    @classmethod
    def is_available(cls, db_session: Session) -> bool:
        from onyx.tools.tool_implementations.search.search_tool import SearchTool

        return SearchTool.is_available(db_session)

    def tool_definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": ProvisionRequest.model_json_schema(),
            },
        }

    def emit_start(self, placement: Placement) -> None:
        self.emitter.emit(
            Packet(
                placement=placement, obj=SearchToolStart(display_name=self.DISPLAY_NAME)
            )
        )

    def run(
        self,
        placement: Placement,
        override_kwargs: ProvisionToolOverrideKwargs,
        **llm_kwargs: Any,
    ) -> ToolResponse:
        request = None
        try:
            request = ProvisionRequest.model_validate(llm_kwargs)
        except ValidationError:
            result = ProvisionLookupResult(
                "invalid_reference",
                "Supply a source and one article number, with separate optional kind, paragraph, clause and date fields.",
            )
        else:
            self.emitter.emit(
                Packet(
                    placement=placement,
                    obj=SearchToolQueriesDelta(queries=[request.label]),
                )
            )
            try:
                filters = self.filters_provider()
                result = lookup_provision(
                    request, document_index=self.document_index, filters=filters
                )
                if result.chunks:
                    indexes = {
                        UUID(row.document_id): row.publication_index
                        for row in result.chunks
                        if row.publication_index is not None
                    }
                    with get_session_with_current_tenant() as session:
                        visible = get_visible_regulatory_chunk_ids(
                            session,
                            [
                                row.regulatory_chunk_id
                                for row in result.chunks
                                if row.regulatory_chunk_id is not None
                            ],
                            as_of_date=result.as_of_date,
                            query_indexes=indexes,
                        )
                    retained = [
                        row
                        for row in result.chunks
                        if row.regulatory_chunk_id in visible
                    ]
                    if len(retained) != len(result.chunks):
                        result.status = "partial" if retained else "unavailable"
                        result.detail = "Publication visibility changed or could not be verified; coverage is incomplete."
                    result.chunks = retained
            except Exception:
                logger.exception("Regulatory provision lookup failed")
                result = ProvisionLookupResult(
                    "unavailable",
                    "Structural lookup is temporarily unavailable. Other internal search modes may still be available.",
                )
        sections = [inference_section_from_single_chunk(row) for row in result.chunks]
        docs = convert_inference_sections_to_search_docs(sections, is_internet=False)
        body, mapping, chunk_mapping = convert_inference_sections_to_llm_string(
            sections,
            citation_start=override_kwargs.starting_citation_num,
            include_link=True,
        )
        payload = json.loads(body)
        payload.update(
            status=result.status,
            detail=result.detail,
            sources=result.sources,
            as_of_date=result.as_of_date.isoformat() if result.as_of_date else None,
            reference=request.model_dump(mode="json") if request else None,
        )
        self.emitter.emit(
            Packet(placement=placement, obj=SearchToolDocumentsDelta(documents=docs))
        )
        return ToolResponse(
            rich_response=SearchDocsResponse(
                search_docs=docs,
                displayed_docs=docs,
                citation_mapping=mapping,
                citation_chunk_mapping=chunk_mapping,
            ),
            llm_facing_response=json.dumps(payload, ensure_ascii=False),
        )
