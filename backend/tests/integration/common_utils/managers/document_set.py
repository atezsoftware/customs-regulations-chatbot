import time
from typing import Any
from uuid import UUID, uuid4

from httpx import Response

from onyx.db.enums import UserFileStatus
from onyx.server.features.projects.models import (
    CategorizedFilesSnapshot,
    UserFileSnapshot,
)
from tests.integration.common_utils.constants import API_SERVER_URL, MAX_DELAY
from tests.integration.common_utils.http_client import client
from tests.integration.common_utils.test_models import DATestDocumentSet, DATestUser


class DocumentSetManager:
    @staticmethod
    def create(
        user_performing_action: DATestUser,
        name: str | None = None,
        description: str | None = None,
        cc_pair_ids: list[int] | None = None,
        is_public: bool = True,
        users: list[str] | None = None,
        groups: list[int] | None = None,
        federated_connectors: list[dict[str, Any]] | None = None,
    ) -> DATestDocumentSet:
        if name is None:
            name = f"test_doc_set_{str(uuid4())}"

        doc_set_creation_request = {
            "name": name,
            "description": description or name,
            "cc_pair_ids": cc_pair_ids or [],
            "is_public": is_public,
            "users": [str(UUID(user_id)) for user_id in (users or [])],
            "groups": groups or [],
            "federated_connectors": federated_connectors or [],
        }

        response = client.post(
            f"{API_SERVER_URL}/manage/admin/document-set",
            json=doc_set_creation_request,
            headers=user_performing_action.headers,
        )
        response.raise_for_status()

        return DATestDocumentSet(
            id=int(response.json()),
            name=name,
            description=description or name,
            cc_pair_ids=cc_pair_ids or [],
            is_public=is_public,
            is_up_to_date=True,
            users=users or [],
            groups=groups or [],
            federated_connectors=federated_connectors or [],
        )

    @staticmethod
    def edit(
        document_set: DATestDocumentSet,
        user_performing_action: DATestUser,
    ) -> bool:
        doc_set_update_request = {
            "id": document_set.id,
            "name": document_set.name,
            "description": document_set.description,
            "cc_pair_ids": document_set.cc_pair_ids,
            "is_public": document_set.is_public,
            "users": [str(UUID(user_id)) for user_id in document_set.users],
            "groups": document_set.groups,
            "federated_connectors": document_set.federated_connectors,
        }
        response = client.patch(
            f"{API_SERVER_URL}/manage/admin/document-set",
            json=doc_set_update_request,
            headers=user_performing_action.headers,
        )
        response.raise_for_status()
        return True

    @staticmethod
    def delete(
        document_set: DATestDocumentSet,
        user_performing_action: DATestUser,
    ) -> bool:
        response = client.delete(
            f"{API_SERVER_URL}/manage/admin/document-set/{document_set.id}",
            headers=user_performing_action.headers,
        )
        response.raise_for_status()
        return True

    @staticmethod
    def get_all(
        user_performing_action: DATestUser,
    ) -> list[DATestDocumentSet]:
        response = client.get(
            f"{API_SERVER_URL}/manage/document-set",
            headers=user_performing_action.headers,
        )
        response.raise_for_status()
        return [
            DATestDocumentSet(
                id=doc_set["id"],
                name=doc_set["name"],
                description=doc_set["description"],
                cc_pair_ids=[cc_pair["id"] for cc_pair in doc_set["cc_pair_summaries"]],
                is_public=doc_set["is_public"],
                is_up_to_date=doc_set["is_up_to_date"],
                users=[str(user_id) for user_id in doc_set["users"]],
                groups=doc_set["groups"],
                federated_connectors=doc_set["federated_connector_summaries"],
            )
            for doc_set in response.json()
        ]

    @staticmethod
    def upload_files(
        document_set_id: int,
        files: list[tuple[str, bytes, str]],  # (filename, content, content_type)
        user_performing_action: DATestUser,
    ) -> CategorizedFilesSnapshot:
        """Upload files to a document set via the admin API."""
        response = DocumentSetManager.upload_files_raw(
            document_set_id, files, user_performing_action
        )
        response.raise_for_status()
        return CategorizedFilesSnapshot.model_validate(response.json())

    @staticmethod
    def upload_files_raw(
        document_set_id: int,
        files: list[tuple[str, bytes, str]],  # (filename, content, content_type)
        user_performing_action: DATestUser,
    ) -> Response:
        """Upload without raising, for tests asserting on a rejection response."""
        headers = dict(user_performing_action.headers or {})
        headers.pop("Content-Type", None)

        return client.post(
            f"{API_SERVER_URL}/manage/admin/document-set/{document_set_id}/file/upload",
            files=[
                ("files", (filename, content, content_type))
                for filename, content, content_type in files
            ],
            headers=headers,
        )

    @staticmethod
    def get_files(
        document_set_id: int,
        user_performing_action: DATestUser,
    ) -> list[UserFileSnapshot]:
        response = client.get(
            f"{API_SERVER_URL}/manage/admin/document-set/{document_set_id}/files",
            headers=user_performing_action.headers,
        )
        response.raise_for_status()
        return [UserFileSnapshot.model_validate(item) for item in response.json()]

    @staticmethod
    def wait_for_files_indexed(
        document_set_id: int,
        expected_count: int,
        user_performing_action: DATestUser,
        timeout: float = MAX_DELAY,
    ) -> list[UserFileSnapshot]:
        """Poll until every uploaded file reaches a terminal status."""
        start = time.monotonic()
        while True:
            files = DocumentSetManager.get_files(
                document_set_id, user_performing_action
            )
            settled = [
                file
                for file in files
                if file.status in (UserFileStatus.COMPLETED, UserFileStatus.FAILED)
            ]
            if len(settled) >= expected_count:
                return files
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Document set {document_set_id} files did not settle within "
                    f"{timeout}s (settled={len(settled)}, expected={expected_count})"
                )
            time.sleep(2)

    @staticmethod
    def wait_for_files_chunked(
        document_set_id: int,
        expected_count: int,
        user_performing_action: DATestUser,
        timeout: float = MAX_DELAY,
    ) -> list[UserFileSnapshot]:
        """Poll until the production chunk phase has settled every file."""

        start = time.monotonic()
        while True:
            files = DocumentSetManager.get_files(
                document_set_id, user_performing_action
            )
            if len(files) >= expected_count and all(
                file.status is UserFileStatus.CHUNKED for file in files
            ):
                return files
            failed = [file for file in files if file.status is UserFileStatus.FAILED]
            if failed:
                raise RuntimeError(
                    f"Document set {document_set_id} failed during chunking: "
                    f"{[file.name for file in failed]}"
                )
            if time.monotonic() - start > timeout:
                raise TimeoutError(
                    f"Document set {document_set_id} files did not reach CHUNKED "
                    f"within {timeout}s"
                )
            time.sleep(2)

    @staticmethod
    def index_chunked_files(
        document_set_id: int,
        user_performing_action: DATestUser,
    ) -> int:
        response = client.post(
            f"{API_SERVER_URL}/manage/admin/document-set/"
            f"{document_set_id}/index-chunked",
            headers=user_performing_action.headers,
        )
        response.raise_for_status()
        return int(response.json()["queued"])

    @staticmethod
    def wait_for_sync(
        user_performing_action: DATestUser,
        document_sets_to_check: list[DATestDocumentSet] | None = None,
    ) -> None:
        # wait for document sets to be synced
        start = time.time()
        while True:
            doc_sets = DocumentSetManager.get_all(user_performing_action)
            if document_sets_to_check:
                check_ids = {doc_set.id for doc_set in document_sets_to_check}
                doc_set_ids = {doc_set.id for doc_set in doc_sets}
                if not check_ids.issubset(doc_set_ids):
                    raise RuntimeError("Document set not found")
                doc_sets = [doc_set for doc_set in doc_sets if doc_set.id in check_ids]
            all_up_to_date = all(doc_set.is_up_to_date for doc_set in doc_sets)

            if all_up_to_date:
                print("Document sets synced successfully.")
                break

            if time.time() - start > MAX_DELAY:
                not_synced_doc_sets = [
                    doc_set for doc_set in doc_sets if not doc_set.is_up_to_date
                ]
                raise TimeoutError(
                    f"Document sets were not synced within the {MAX_DELAY} seconds. "
                    f"Remaining unsynced document sets: {len(not_synced_doc_sets)}. "
                    f"IDs: {[doc_set.id for doc_set in not_synced_doc_sets]}"
                )
            else:
                not_synced_doc_sets = [
                    doc_set for doc_set in doc_sets if not doc_set.is_up_to_date
                ]
                print(
                    f"Document sets were not synced yet, waiting... "
                    f"{len(not_synced_doc_sets)}/{len(doc_sets)} document sets still syncing. "
                    f"IDs: {[doc_set.id for doc_set in not_synced_doc_sets]}"
                )

            time.sleep(2)

    @staticmethod
    def verify(
        document_set: DATestDocumentSet,
        user_performing_action: DATestUser,
        verify_deleted: bool = False,
    ) -> None:
        doc_sets = DocumentSetManager.get_all(user_performing_action)
        for doc_set in doc_sets:
            if doc_set.id == document_set.id:
                if verify_deleted:
                    raise ValueError(
                        f"Document set {document_set.id} found but should have been deleted"
                    )
                if (
                    doc_set.name == document_set.name
                    and set(doc_set.cc_pair_ids) == set(document_set.cc_pair_ids)
                    and doc_set.is_public == document_set.is_public
                    and set(doc_set.users) == set(document_set.users)
                    and set(doc_set.groups) == set(document_set.groups)
                    and doc_set.federated_connectors
                    == document_set.federated_connectors
                ):
                    return
        if not verify_deleted:
            raise ValueError(f"Document set {document_set.id} not found")
