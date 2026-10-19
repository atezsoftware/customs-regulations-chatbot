"""Contracts for live chat links rendered by an external application's frontend."""

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import AnyHttpUrl, BaseModel, Field, field_validator


class ExternalShareCreateRequest(BaseModel):
    share_url_base: AnyHttpUrl = Field(
        default=AnyHttpUrl("https://tariff.singlewindow.io/chat/shared"),
        description="External frontend route to which the opaque token is appended.",
    )

    @field_validator("share_url_base")
    @classmethod
    def validate_share_url_base(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        if (
            value.username is not None
            or value.password is not None
            or value.query is not None
            or value.fragment is not None
        ):
            raise ValueError(
                "Share URL cannot contain credentials, a query or a fragment"
            )
        localhost = value.host in {"localhost", "127.0.0.1", "[::1]", "::1"}
        if value.scheme != "https" and not (value.scheme == "http" and localhost):
            raise ValueError(
                "Share URL must use HTTPS except for localhost development"
            )
        return value


class ExternalShareCreateResponse(BaseModel):
    shareToken: str = Field(
        min_length=36,
        max_length=36,
        pattern=r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
        description="A new UUIDv4 share token, distinct from the chat session ID.",
        examples=["eb711ac5-7078-47bb-b8ea-2b3e3067b9ed"],
    )
    shareUrl: str


class ExternalSharedChatCitation(BaseModel):
    number: int
    title: str
    url: str | None = None


class ExternalSharedChatMessage(BaseModel):
    id: int
    sessionId: UUID
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    citations: list[ExternalSharedChatCitation] = Field(default_factory=list)


class ExternalSharedChatSession(BaseModel):
    id: UUID
    title: str | None
    createdAt: datetime
    updatedAt: datetime
    messageCount: int
    isShared: bool = True


class ExternalSharedChatMessagesResponse(BaseModel):
    session: ExternalSharedChatSession
    messages: list[ExternalSharedChatMessage]
