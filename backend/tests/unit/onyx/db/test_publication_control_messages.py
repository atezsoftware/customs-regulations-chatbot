"""Unavailable originals redact attachments without hiding a literal host stop notice."""

from typing import cast
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from onyx.asv3.progress import localized_notifications
from onyx.db import regulatory_chat_reads
from onyx.db.chat import (
    saved_message_citation_info,
    translate_db_message_to_chat_message_detail,
)
from onyx.file_store.models import ChatFileType, FileDescriptor
from onyx.regulatory.publication_reads import PublicationReadChanged
from onyx.server.query_and_chat.session_loading import (
    translate_assistant_message_to_packets,
)
from tests.unit.onyx.db.test_asv3_citation_details import fixture_message


@pytest.mark.parametrize("language", ["tr", "en"])
def test_unavailable_cancelled_control_preserves_status_with_all_sources_redacted(
    monkeypatch: pytest.MonkeyPatch, language: str
) -> None:
    message, _checkpoint = fixture_message("asv3_tuned")
    notice = localized_notifications(language)["cancelled"][1]
    message.message = notice
    message.error = None
    message.reasoning_tokens = "Never expose retained private reasoning."
    files: list[FileDescriptor] = [
        {"id": "unavailable-file", "type": ChatFileType.PLAIN_TEXT, "name": "source"}
    ]
    message.files = files
    assert message.search_docs and message.citations
    monkeypatch.setattr(
        regulatory_chat_reads, "message_publication_available", lambda _message: False
    )

    detail = translate_db_message_to_chat_message_detail(message)

    assert detail.message == notice and detail.error == notice
    assert detail.context_docs == [] and detail.citations is None
    assert detail.citation_info == [] and detail.reasoning_tokens is None
    assert detail.files == []
    assert saved_message_citation_info(message) == []
    assert (
        translate_assistant_message_to_packets(message, cast(Session, MagicMock()))
        == []
    )


@pytest.mark.parametrize(
    "text",
    [
        "Kanunun 211. maddesi uyarınca kaldırma başvurusu yapılabilir.[1]",
        "Çalışma durduruldu; geç sonuçlar kabul edilmeyecek. Kanun ceza öngörmez.[1]",
        "Çalışma durduruldu; geç sonuçlar kabul edilmeyecek.[1]",
        "…",
    ],
)
def test_unavailable_substantive_or_unproven_controls_keep_publication_mask(
    monkeypatch: pytest.MonkeyPatch, text: str
) -> None:
    message, _checkpoint = fixture_message("asv3_tuned")
    message.message = text
    monkeypatch.setattr(
        regulatory_chat_reads, "message_publication_available", lambda _message: False
    )

    detail = translate_db_message_to_chat_message_detail(message)

    assert detail.message == str(PublicationReadChanged())
    assert detail.error == str(PublicationReadChanged())
    assert detail.context_docs == [] and detail.citations is None
    assert detail.citation_info == [] and detail.reasoning_tokens is None
    assert detail.files == []
