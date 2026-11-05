from pydantic import BaseModel, Field

from onyx.file_store.models import ChatFileType


class ChatAttachmentSnapshot(BaseModel):
    id: str
    type: ChatFileType
    name: str
    token_count: int
    size: int
    is_session_attachment: bool = True


class RejectedChatAttachment(BaseModel):
    name: str
    reason: str


class ChatAttachmentUploadResponse(BaseModel):
    files: list[ChatAttachmentSnapshot] = Field(default_factory=list)
    rejected_files: list[RejectedChatAttachment] = Field(default_factory=list)
