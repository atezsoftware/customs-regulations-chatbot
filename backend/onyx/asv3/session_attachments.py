"""Native user PDFs travel with normal research requests, never a pre-read call."""

from pydantic import BaseModel, Field

from onyx.llm.models import (
    ChatCompletionMessage,
    FileContentPart,
    SystemMessage,
    TextContentPart,
    UserMessage,
)
from onyx.prompts.asv3.attachments import SESSION_ATTACHMENT_CONTEXT

NATIVE_SESSION_ATTACHMENTS = "native_session_attachments"
# Reserve visual/context capacity per page, without tokenizing base64 transport.
PDF_PAGE_INPUT_TOKEN_RESERVE = 4096


class NativeSessionAttachment(BaseModel):
    file_id: str
    filename: str
    sha256: str
    page_count: int = Field(ge=1, le=500)
    part: FileContentPart = Field(repr=False)

    @property
    def input_token_reserve(self) -> int:
        return self.page_count * PDF_PAGE_INPUT_TOKEN_RESERVE


def session_attachment_prompt(
    prompt: list[ChatCompletionMessage], attachments: list[NativeSessionAttachment]
) -> list[ChatCompletionMessage]:
    if not attachments:
        return prompt
    prompt = [
        message.model_copy(
            update={"content": message.content + "\n\n" + SESSION_ATTACHMENT_CONTEXT}
        )
        if isinstance(message, SystemMessage)
        and SESSION_ATTACHMENT_CONTEXT not in message.content
        else message
        for message in prompt
    ]
    present = {
        part.file.file_data
        for message in prompt
        if isinstance(message.content, list)
        for part in message.content
        if isinstance(part, FileContentPart)
    }
    missing = [
        item.part for item in attachments if item.part.file.file_data not in present
    ]
    if not missing:
        return prompt
    # Keep the final JSON payload and native catalogue at the same message index.
    for index in range(len(prompt) - 1, -1, -1):
        message = prompt[index]
        if isinstance(message, UserMessage):
            content = message.content
            parts = (
                [TextContentPart(text=content)]
                if isinstance(content, str)
                else list(content)
            )
            return [
                *prompt[:index],
                message.model_copy(update={"content": [*parts, *missing]}),
                *prompt[index + 1 :],
            ]
    return [
        *prompt,
        UserMessage(
            content=[TextContentPart(text=SESSION_ATTACHMENT_CONTEXT), *missing]
        ),
    ]
