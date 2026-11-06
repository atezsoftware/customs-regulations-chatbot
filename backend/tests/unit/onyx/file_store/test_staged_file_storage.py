"""Fresh staged objects cannot outlive a failed FileRecord transaction."""

from collections.abc import Generator
from contextlib import contextmanager
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.file_store import file_store


def test_new_s3_object_is_deleted_if_metadata_save_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = file_store.S3BackedFileStore(bucket_name="test-bucket")
    client = MagicMock()
    monkeypatch.setattr(store, "_get_s3_client", lambda: client)

    @contextmanager
    def broken_session(_session: Session | None) -> Generator[Session, None, None]:
        raise RuntimeError("metadata unavailable")
        yield

    monkeypatch.setattr(
        file_store, "get_session_with_current_tenant_if_none", broken_session
    )
    with pytest.raises(RuntimeError, match="metadata unavailable"):
        store.save_file(
            BytesIO(b"# staged MD"),
            "stage.md",
            FileOrigin.OTHER,
            "text/markdown",
            cleanup_on_metadata_failure=True,
        )
    client.put_object.assert_called_once()
    client.delete_object.assert_called_once_with(
        Bucket="test-bucket", Key=client.put_object.call_args.kwargs["Key"]
    )


def test_cleanup_mode_never_accepts_an_existing_object_identifier() -> None:
    store = file_store.S3BackedFileStore(bucket_name="test-bucket")
    with pytest.raises(ValueError, match="new file"):
        store.save_file(
            BytesIO(b"# staged MD"),
            "stage.md",
            FileOrigin.OTHER,
            "text/markdown",
            file_id="potential-existing-file",
            cleanup_on_metadata_failure=True,
        )


def test_create_only_s3_failure_never_deletes_existing_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from botocore.exceptions import ClientError

    store = file_store.S3BackedFileStore(bucket_name="test-bucket")
    client = MagicMock()
    client.put_object.side_effect = ClientError(
        {"Error": {"Code": "PreconditionFailed"}}, "PutObject"
    )
    monkeypatch.setattr(store, "_get_s3_client", lambda: client)
    with pytest.raises(ClientError):
        store.save_file(
            BytesIO(b"# new"),
            "stage.md",
            FileOrigin.OTHER,
            "text/markdown",
            cleanup_on_metadata_failure=True,
        )
    assert client.put_object.call_args.kwargs["IfNoneMatch"] == "*"
    client.delete_object.assert_not_called()
