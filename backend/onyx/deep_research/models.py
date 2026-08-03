from pydantic import BaseModel, Field

from onyx.chat.citation_processor import CitationMapping
from onyx.regulatory.candidate_answer_review import CandidateAnswerEvidenceChunk
from onyx.tools.models import ToolCallKickoff


class SpecialToolCalls(BaseModel):
    think_tool_call: ToolCallKickoff | None = None
    generate_report_tool_call: ToolCallKickoff | None = None


class ResearchAgentCallResult(BaseModel):
    intermediate_report: str
    # Citations that actually appeared in the intermediate report.
    citation_mapping: CitationMapping
    # Exact retrieved chunks that may be exposed only to a bounded correction pass.
    # Numbers are local to this research-agent call.
    evidence_citation_mapping: CitationMapping = Field(default_factory=dict)
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk] = Field(
        default_factory=list
    )


class CombinedResearchAgentCallResult(BaseModel):
    # The None is needed here to keep the mappings consistent
    # we later skip the failed research results but we need to know
    # which ones failed
    intermediate_reports: list[str | None]
    # Citations that actually appeared in the combined intermediate reports.
    citation_mapping: CitationMapping
    # Exact retrieved chunks in the accumulated global citation namespace.
    evidence_citation_mapping: CitationMapping = Field(default_factory=dict)
    exact_evidence_chunks: list[CandidateAnswerEvidenceChunk] = Field(
        default_factory=list
    )
