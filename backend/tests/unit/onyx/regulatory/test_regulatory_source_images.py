"""Exact source-image validation and the authenticated storage-only boundary."""

import hashlib
import io
from collections.abc import Iterator
from typing import cast
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy.orm import Session

from onyx.auth.schemas import UserRole
from onyx.auth.users import current_user
from onyx.configs.constants import FileOrigin
from onyx.db.engine.sql_engine import get_session
from onyx.db.enums import Permission, UserFileStatus
from onyx.db.models import User, UserFile
from onyx.error_handling.exceptions import register_onyx_exception_handlers
from onyx.regulatory import source_image_evidence as images
from onyx.server.features.regulatory import api


def image_bytes(format: str = "PNG", size: tuple[int, int] = (8, 12)) -> bytes:
    content = io.BytesIO()
    Image.new("RGB", size, "white").save(content, format=format)
    return content.getvalue()


@pytest.mark.parametrize("format,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_descriptor_preserves_exact_image(format: str, mime: str) -> None:
    content = image_bytes(format)
    result = images.validate_source_image(content, mime)
    assert result.model_dump() == {
        "sha256": hashlib.sha256(content).hexdigest(),
        "byte_count": len(content),
        "mime_type": mime,
        "width": 8,
        "height": 12,
    }


@pytest.mark.parametrize(
    "content,mime",
    [
        (b"", "image/png"),
        (b"<svg/>", "image/png"),
        (b"%PDF-1.7", "image/png"),
        (image_bytes("GIF"), "image/gif"),
        (image_bytes(), "image/jpeg"),
        (image_bytes("JPEG"), "image/png"),
        (image_bytes(), None),
        (image_bytes()[:-4], "image/png"),
        (image_bytes("JPEG")[:-2], "image/jpeg"),
        (image_bytes() + b"<script/>", "image/png"),
        (b"\x89PNG\r\n\x1a\n" + b"\0\0\0\0IEND\xaeB`\x82", "image/png"),
        (image_bytes("JPEG")[:100] + b"\xff\xd9", "image/jpeg"),
    ],
)
def test_rejects_spoofed_truncated_or_unsupported_image(
    content: bytes, mime: str | None
) -> None:
    with pytest.raises(ValueError):
        images.validate_source_image(content, mime)


def test_size_pixel_and_animation_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    content = image_bytes()
    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_BYTES", len(content))
    images.validate_source_image(content, "image/png")
    with pytest.raises(ValueError):
        images.validate_source_image(content + b"x", "image/png")
    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_PIXELS", 95)
    with pytest.raises(ValueError, match="16 megapixels"):
        images.validate_source_image(content, "image/png")
    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_BYTES", 10000)
    monkeypatch.setattr(images, "MAX_SOURCE_IMAGE_PIXELS", 10000)
    animated = io.BytesIO()
    Image.new("RGB", (8, 12), "white").save(
        animated,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (8, 12), "black")],
    )
    with pytest.raises(ValueError, match="one image"):
        images.validate_source_image(animated.getvalue(), "image/png")


def test_dimension_bound() -> None:
    with pytest.raises(ValueError, match="8192 pixels"):
        images.validate_source_image(image_bytes(size=(8193, 1)), "image/png")


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, MagicMock, User, UserFile]]:
    user = User(
        id=uuid4(),
        role=UserRole.ADMIN,
        effective_permissions=[Permission.FULL_ADMIN_PANEL_ACCESS.value],
    )
    parent = UserFile(id=uuid4(), user_id=user.id, status=UserFileStatus.COMPLETED)
    session = MagicMock(spec=Session)
    session.get.return_value = parent
    store = MagicMock()
    store.save_file.return_value = str(uuid4())
    monkeypatch.setattr(api, "get_default_file_store", lambda: store)
    monkeypatch.setattr(api, "pending_intake_file_ids", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(api, "has_regulatory_chunks_for_file", lambda *_args: True)
    app = FastAPI()
    register_onyx_exception_handlers(app)
    app.include_router(api.router)
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[current_user] = lambda: user
    with TestClient(app) as http:
        yield http, store, user, parent


def test_http_upload_stores_only_exact_blob_and_parent_descriptor(
    client: tuple[TestClient, MagicMock, User, UserFile],
) -> None:
    http, store, user, parent = client
    content = image_bytes()
    response = http.post(
        f"/regulatory/files/{parent.id}/source-images",
        files={"file": ("../../page.png", content, "image/png")},
    )
    assert response.status_code == 200, response.text
    receipt = response.json()
    assert receipt["file_id"] == store.save_file.return_value
    assert receipt["user_file_id"] == str(parent.id)
    assert receipt["sha256"] == hashlib.sha256(content).hexdigest()
    args, kwargs = store.save_file.call_args
    assert args[0].read() == content
    assert kwargs["display_name"] == "page.png"
    assert kwargs["file_origin"] == FileOrigin.OTHER
    assert kwargs["file_type"] == "image/png"
    assert kwargs["file_metadata"]["purpose"] == "regulatory_source_image"
    assert kwargs["file_metadata"]["regulatory_user_file_id"] == str(parent.id)
    assert kwargs["file_metadata"]["uploaded_by"] == str(user.id)
    assert "file_id" not in kwargs
    assert parent.status == UserFileStatus.COMPLETED


def test_upload_reads_only_the_bounded_prefix(
    client: tuple[TestClient, MagicMock, User, UserFile],
) -> None:
    _http, store, user, parent = client
    reader = MagicMock(spec=io.BytesIO)
    reader.read.return_value = b"x" * (images.MAX_SOURCE_IMAGE_BYTES + 1)
    with pytest.raises(api.OnyxError):
        api.upload_source_image(
            parent.id,
            api.UploadFile(reader, filename="oversize.png"),
            user,
            MagicMock(get=MagicMock(return_value=parent)),
        )
    reader.read.assert_called_once_with(images.MAX_SOURCE_IMAGE_BYTES + 1)
    store.save_file.assert_not_called()


@pytest.mark.parametrize("permission", [False, True])
def test_missing_auth_or_permission_cannot_store(
    client: tuple[TestClient, MagicMock, User, UserFile], permission: bool
) -> None:
    http, store, user, parent = client
    if permission:
        user.effective_permissions = []
    else:

        def unauthenticated() -> User:
            raise HTTPException(status_code=401)

        cast(FastAPI, http.app).dependency_overrides[current_user] = unauthenticated
    response = http.post(
        f"/regulatory/files/{parent.id}/source-images",
        files={"file": ("page.png", image_bytes(), "image/png")},
    )
    assert response.status_code == (403 if permission else 401)
    store.save_file.assert_not_called()


@pytest.mark.parametrize(
    "boundary",
    [
        "missing",
        "foreign",
        "processing",
        "intake",
        "nonregulatory",
        "invalid",
        "oversize",
    ],
)
def test_parent_and_content_guards_precede_storage(
    client: tuple[TestClient, MagicMock, User, UserFile],
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    http, store, user, parent = client
    expected = 409
    if boundary == "missing":
        cast(FastAPI, http.app).dependency_overrides[get_session] = lambda: MagicMock(
            get=MagicMock(return_value=None)
        )
        expected = 404
    elif boundary == "foreign":
        user.role = UserRole.BASIC
        parent.user_id = uuid4()
        expected = 403
    elif boundary == "processing":
        parent.status = UserFileStatus.PROCESSING
    elif boundary == "intake":
        monkeypatch.setattr(
            api, "pending_intake_file_ids", lambda *_args, **_kwargs: [parent.id]
        )
    elif boundary == "nonregulatory":
        monkeypatch.setattr(api, "has_regulatory_chunks_for_file", lambda *_args: False)
    elif boundary == "invalid":
        expected = 400
    elif boundary == "oversize":
        monkeypatch.setattr(api, "MAX_SOURCE_IMAGE_BYTES", 10)
        expected = 413
    response = http.post(
        f"/regulatory/files/{parent.id}/source-images",
        files={
            "file": (
                "page.png",
                b"fake" if boundary == "invalid" else image_bytes(),
                "image/png",
            )
        },
    )
    assert response.status_code == expected, response.text
    store.save_file.assert_not_called()
