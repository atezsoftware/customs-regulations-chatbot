from io import BytesIO
from zipfile import ZipFile

from openpyxl import Workbook

from tests.unit.onyx.server.test_chat_attachments import (
    AttachmentEnvironment,
    environment,
)

__all__ = ["environment"]


def test_malformed_xlsx_rejects_only_that_file_and_keeps_other_batch_uploads(
    environment: AttachmentEnvironment,
) -> None:
    workbook = Workbook()
    original_bytes = BytesIO()
    workbook.save(original_bytes)
    workbook.close()
    malformed = BytesIO()
    with ZipFile(original_bytes) as original, ZipFile(malformed, "w") as changed:
        for member in original.infolist():
            content = original.read(member.filename)
            if member.filename == "xl/worksheets/sheet1.xml":
                assert b'workbookViewId="0"' in content
                content = content.replace(
                    b'workbookViewId="0"', b'workbookViewId="bad"'
                )
            changed.writestr(member, content)

    response = environment.client.post(
        "/chat/files/upload",
        files=[
            ("files", ("before.txt", b"Before malformed file")),
            ("files", ("malformed.xlsx", malformed.getvalue())),
            ("files", ("after.csv", b"item,amount\ncustoms,120\n")),
        ],
        data={"chat_session_id": str(environment.chat.id)},
    )

    assert response.status_code == 200
    result = response.json()
    assert [file["name"] for file in result["files"]] == ["before.txt", "after.csv"]
    assert [file["name"] for file in result["rejected_files"]] == ["malformed.xlsx"]
    assert "malformed" in result["rejected_files"][0]["reason"]
    listed = environment.client.get(f"/chat/sessions/{environment.chat.id}/files")
    assert {file["name"] for file in listed.json()} == {"before.txt", "after.csv"}
