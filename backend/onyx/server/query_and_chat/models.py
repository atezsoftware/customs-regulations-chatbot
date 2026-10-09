from datetime import datetime
from enum import Enum
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator

from onyx.configs.constants import DocumentSource, MessageType, SessionType
from onyx.context.search.models import BaseFilters, SavedSearchDoc, SearchDoc, Tag
from onyx.db.enums import ChatSessionSharedStatus
from onyx.db.models import ChatSession
from onyx.file_store.models import FileDescriptor
from onyx.llm.override_models import LLMOverride
from onyx.llm.usage_cost import ResponseUsage
from onyx.server.query_and_chat.streaming_models import Packet

AUTO_PLACE_AFTER_LATEST_MESSAGE = -1


class MessageOrigin(str, Enum):
    """Origin of a chat message for telemetry tracking."""

    WEBAPP = "webapp"
    CHROME_EXTENSION = "chrome_extension"
    API = "api"
    SLACKBOT = "slackbot"
    WIDGET = "widget"
    DISCORDBOT = "discordbot"
    MOBILE = "mobile"
    BENCHMARK = "benchmark"
    UNKNOWN = "unknown"
    UNSET = "unset"


class MessageResponseIDInfo(BaseModel):
    user_message_id: int | None
    reserved_assistant_message_id: int


class ModelResponseSlot(BaseModel):
    """Pairs a reserved assistant message ID with its model display name."""

    message_id: int
    model_name: str


class MultiModelMessageResponseIDInfo(BaseModel):
    """Sent at the start of a multi-model streaming response.
    Contains the user message ID and one slot per model being run in parallel."""

    user_message_id: int | None
    responses: list[ModelResponseSlot]


class SourceTag(Tag):
    source: DocumentSource


class TagResponse(BaseModel):
    tags: list[SourceTag]


class UpdateChatSessionThreadRequest(BaseModel):
    # If not specified, use Onyx default persona
    chat_session_id: UUID
    new_alternate_model: str


class UpdateChatSessionTemperatureRequest(BaseModel):
    chat_session_id: UUID
    temperature_override: float


class UpdateChatSessionReasoningRequest(BaseModel):
    chat_session_id: UUID
    reasoning_effort_override: str | None = None


class ChatSessionCreationRequest(BaseModel):
    # If not specified, use Onyx default persona
    persona_id: int = 0
    description: str | None = None
    project_id: int | None = None


class ChatFeedbackRequest(BaseModel):
    chat_message_id: int
    is_positive: bool | None = None
    feedback_text: str | None = None
    predefined_feedback: str | None = None

    @model_validator(mode="after")
    def check_is_positive_or_feedback_text(self) -> "ChatFeedbackRequest":
        if self.is_positive is None and self.feedback_text is None:
            raise ValueError("Empty feedback received.")
        return self


# NOTE: This model is used for the core flow of the Onyx application, any changes to it should be reviewed and approved by an
# experienced team member. It is very important to 1. avoid bloat and 2. that this remains backwards compatible across versions.
class SendMessageRequest(BaseModel):
    message: str

    llm_override: LLMOverride | None = None
    # For multi-model mode: up to 3 LLM overrides to run in parallel.
    # When provided with >1 entry, triggers multi-model streaming.
    llm_overrides: list[LLMOverride] | None = None
    # Test-only override for deterministic LiteLLM mock responses.
    mock_llm_response: str | None = None

    allowed_tool_ids: list[int] | None = None
    forced_tool_id: int | None = None

    file_descriptors: list[FileDescriptor] = []

    internal_search_filters: BaseFilters | None = None

    deep_research: bool = False
    # Opts the default assistant into the structure-aware Atez regulatory
    # workflow. Normal chat remains on the standard Onyx agent loop.
    atez_search: bool = False
    # Independent, lower-latency profile. It shares the same indexed corpus,
    # authorization, citations, coverage plan, and validation contracts.
    atez_search_v2: bool = False
    # Explicit experiment opt-in. The deployed default remains the baseline path.
    atez_search_v2_labels: bool = False
    atez_search_v2_label_run_ids: list[UUID] = Field(
        default_factory=list, max_length=32
    )
    # ASv3 owns an independent adaptive research harness.
    atez_search_v3: bool = False
    # Independent bounded research with evidence review before publication.
    legal_composite: bool = False
    # Independent, selected-model research confined to the PC Külliyatı corpus.
    supersearch: bool = False
    # Independent Gemini Flash discovery with evidence and answer JEV checks.
    legal_review: bool = False
    asv3_research_profile: Literal["normal", "deep", "experimental"] = "deep"
    asv3_parallel_research: bool = False
    asv3_guarded_experimental: bool = False
    asv3_guardrails_v2: bool = False
    asv3_allow_external: bool = False
    asv3_resume_message_id: int | None = None

    # Headers to forward to MCP tool calls (e.g., user JWT token, user ID)
    # Example: {"Authorization": "Bearer <user_jwt>", "X-User-ID": "user123"}
    mcp_headers: dict[str, str] | None = None

    # Origin of the message for telemetry tracking
    origin: MessageOrigin = MessageOrigin.UNSET

    # Placement information for the message in the conversation tree:
    # - -1: auto-place after latest message in chain
    # - null: regeneration from root (first message)
    # - positive int: place after that specific parent message
    # NOTE: for regeneration, this is the only case currently where there is branching on the user message.
    # If the message of parent_message_id is a user message, the message will be ignored and it will use the
    # original user message for regeneration.
    parent_message_id: int | None = AUTO_PLACE_AFTER_LATEST_MESSAGE
    chat_session_id: UUID | None = None
    chat_session_info: ChatSessionCreationRequest | None = None

    # When True (default), returns StreamingResponse with SSE
    # When False, returns ChatFullResponse with complete data
    stream: bool = True

    # When False, disables citation generation:
    # - Citation markers like [1], [2] are removed from response text
    # - No CitationInfo packets are emitted during streaming
    include_citations: bool = True

    # Additional context injected into the LLM call but NOT stored in the DB
    # (not shown in chat history). Used e.g. by the Chrome extension to pass
    # the current tab URL when "Read this tab" is enabled.
    additional_context: str | None = None

    @model_validator(mode="after")
    def check_chat_session_id_or_info(self) -> "SendMessageRequest":
        if self.legal_review and any(
            (
                self.atez_search,
                self.atez_search_v2,
                self.atez_search_v3,
                self.legal_composite,
                self.supersearch,
                self.deep_research,
                self.atez_search_v2_labels,
                bool(self.atez_search_v2_label_run_ids),
                self.asv3_allow_external,
                self.asv3_parallel_research,
                self.asv3_guarded_experimental,
                self.asv3_guardrails_v2,
                self.asv3_resume_message_id is not None,
                self.asv3_research_profile != "deep",
            )
        ):
            raise ValueError(
                "Legal Review is mutually exclusive with other workflows and research options"
            )
        if self.legal_review and self.llm_overrides and len(self.llm_overrides) > 1:
            raise ValueError("Legal Review uses one Gemini 3.8 Flash model")
        if self.legal_review and (
            self.file_descriptors
            or self.additional_context
            or self.forced_tool_id is not None
        ):
            raise ValueError(
                "Legal Review accepts indexed corpus sources only; attachments, additional context and forced tools are unavailable"
            )
        if self.supersearch and any(
            (
                self.atez_search,
                self.atez_search_v2,
                self.atez_search_v3,
                self.legal_composite,
                self.deep_research,
                self.atez_search_v2_labels,
                bool(self.atez_search_v2_label_run_ids),
                self.asv3_allow_external,
                self.asv3_parallel_research,
                self.asv3_guarded_experimental,
                self.asv3_resume_message_id is not None,
                self.asv3_research_profile != "deep",
            )
        ):
            raise ValueError(
                "Supersearch is PC-only and mutually exclusive with other workflows "
                "and external research options"
            )
        if self.supersearch and self.llm_overrides and len(self.llm_overrides) > 1:
            raise ValueError("Supersearch uses one selected model per request")
        if self.supersearch and self.llm_override is not None and self.llm_overrides:
            raise ValueError("Supersearch requires one model override format")
        if self.supersearch and (
            self.file_descriptors
            or self.additional_context
            or self.forced_tool_id is not None
        ):
            raise ValueError(
                "Supersearch accepts sources only from the PC Külliyatı document "
                "set; attachments, additional context and forced tools are unavailable"
            )
        if self.legal_composite and any(
            (
                self.atez_search,
                self.atez_search_v2,
                self.atez_search_v3,
                self.deep_research,
            )
        ):
            raise ValueError(
                "Legal Composite and other workflows are mutually exclusive"
            )
        if self.legal_composite and self.llm_overrides and len(self.llm_overrides) > 1:
            raise ValueError("Legal Composite uses one answering model per request")
        if self.atez_search and self.atez_search_v2:
            raise ValueError("atez_search and atez_search_v2 are mutually exclusive")
        if self.atez_search_v2_labels and not self.atez_search_v2:
            raise ValueError("atez_search_v2_labels requires atez_search_v2")
        if self.atez_search_v2_label_run_ids and not self.atez_search_v2_labels:
            raise ValueError("atez_search_v2_label_run_ids requires labels")
        if sum((self.atez_search, self.atez_search_v2, self.atez_search_v3)) > 1:
            raise ValueError("ATEZ Search workflows are mutually exclusive")
        if self.atez_search_v3 and self.deep_research:
            raise ValueError("ASv3 and Deep Research are mutually exclusive")
        if (
            (self.atez_search_v3 or self.deep_research)
            and self.llm_overrides
            and len(self.llm_overrides) > 1
        ):
            raise ValueError("ASv3 uses one coordinator model per request")
        if self.asv3_research_profile == "normal" and not self.atez_search_v3:
            raise ValueError("The coordinator profile requires the ASv3 workflow")
        if self.asv3_research_profile == "experimental" and not self.atez_search_v3:
            raise ValueError(
                "The experimental profile requires explicit ASv3 selection"
            )
        if self.asv3_parallel_research and not (
            self.atez_search_v3 and self.asv3_research_profile == "experimental"
        ):
            raise ValueError(
                "Parallel experimental research requires the experimental ASv3 profile"
            )
        if self.asv3_guarded_experimental and not (
            self.atez_search_v3
            and self.asv3_research_profile == "normal"
            and not self.asv3_parallel_research
        ):
            raise ValueError(
                "Experimental Guardrails requires the normal non-parallel ASv3 profile"
            )
        if self.asv3_guarded_experimental and self.asv3_guardrails_v2:
            raise ValueError("ASv3 guardrails variants are mutually exclusive")
        if self.asv3_guardrails_v2 and not (
            self.atez_search_v3
            and self.asv3_research_profile == "normal"
            and not self.asv3_parallel_research
        ):
            raise ValueError(
                "Experimental Guardrails v2 requires the normal non-parallel ASv3 profile"
            )
        if self.asv3_resume_message_id is not None and not (
            self.atez_search_v3 or self.deep_research
        ):
            raise ValueError("An ASv3 checkpoint requires the ASv3 workflow")
        if self.asv3_allow_external and not (self.atez_search_v3 or self.deep_research):
            raise ValueError(
                "ASv3 external source permission requires the ASv3 workflow"
            )
        # If neither is provided, default to creating a new chat session using the
        # default ChatSessionCreationRequest values.
        if self.chat_session_id is None and self.chat_session_info is None:
            return self.model_copy(
                update={"chat_session_info": ChatSessionCreationRequest()}
            )
        if self.chat_session_id is not None and self.chat_session_info is not None:
            raise ValueError(
                "Only one of chat_session_id or chat_session_info should be provided, not both."
            )
        return self


class ChatMessageIdentifier(BaseModel):
    message_id: int


class ChatRenameRequest(BaseModel):
    chat_session_id: UUID
    name: str | None = None


class ChatSessionUpdateRequest(BaseModel):
    sharing_status: ChatSessionSharedStatus


class DeleteAllSessionsRequest(BaseModel):
    session_type: SessionType


class RenameChatSessionResponse(BaseModel):
    new_name: str  # This is only really useful if the name is generated


class ChatSessionDetails(BaseModel):
    id: UUID
    name: str | None
    persona_id: int | None = None
    time_created: str
    time_updated: str
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None = None
    current_temperature_override: float | None = None
    current_reasoning_effort_override: str | None = None

    @classmethod
    def from_model(cls, model: ChatSession) -> "ChatSessionDetails":
        return cls(
            id=model.id,
            name=model.description,
            persona_id=model.persona_id,
            time_created=model.time_created.isoformat(),
            time_updated=model.time_updated.isoformat(),
            shared_status=model.shared_status,
            current_alternate_model=model.current_alternate_model,
            current_temperature_override=model.temperature_override,
            current_reasoning_effort_override=model.reasoning_effort_override,
        )


class ChatSessionsResponse(BaseModel):
    sessions: list[ChatSessionDetails]
    has_more: bool = False


class ChatMessageDetail(BaseModel):
    chat_session_id: UUID | None = None
    message_id: int
    parent_message: int | None = None
    latest_child_message: int | None = None
    message: str
    reasoning_tokens: str | None = None
    message_type: MessageType
    context_docs: list[SavedSearchDoc] | None = None
    # Dict mapping citation number to document_id
    citations: dict[int, str] | None = None
    time_sent: datetime
    files: list[FileDescriptor]
    error: str | None = None
    current_feedback: str | None = None  # "like" | "dislike" | null
    processing_duration_seconds: float | None = None
    usage: ResponseUsage | None = None
    preferred_response_id: int | None = None
    model_display_name: str | None = None

    def model_dump(  # ty: ignore[invalid-method-override]
        self, *args: list, **kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        initial_dict = super().model_dump(
            mode="json",
            *args,
            **kwargs,  # ty: ignore[invalid-argument-type]
        )
        initial_dict["time_sent"] = self.time_sent.isoformat()
        return initial_dict


class SetPreferredResponseRequest(BaseModel):
    user_message_id: int
    preferred_response_id: int


class CurrentRunInfo(BaseModel):
    """In-flight run whose stream buffer can be replayed/tailed."""

    run_id: int


class ChatSessionDetailResponse(BaseModel):
    chat_session_id: UUID
    description: str | None
    persona_id: int | None = None
    persona_name: str | None
    personal_icon_name: str | None
    messages: list[ChatMessageDetail]
    time_created: datetime
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None
    current_temperature_override: float | None
    current_reasoning_effort_override: str | None
    deleted: bool = False
    owner_name: str | None = None
    packets: list[list[Packet]]
    # Set while a run is in flight and resumable: cursor-0 replay+tail is
    # available at /chat-session/{id}/resume-stream.
    current_run: CurrentRunInfo | None = None


class AdminSearchRequest(BaseModel):
    query: str
    filters: BaseFilters


class AdminSearchResponse(BaseModel):
    documents: list[SearchDoc]


class ChatSessionSummary(BaseModel):
    id: UUID
    name: str | None = None
    persona_id: int | None = None
    time_created: datetime
    shared_status: ChatSessionSharedStatus
    current_alternate_model: str | None = None
    current_temperature_override: float | None = None
    current_reasoning_effort_override: str | None = None


class ChatSessionGroup(BaseModel):
    title: str
    chats: list[ChatSessionSummary]


class ChatSearchResponse(BaseModel):
    groups: list[ChatSessionGroup]
    has_more: bool
    next_page: int | None = None
