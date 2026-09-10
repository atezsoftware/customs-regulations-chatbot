"""Validate protected attachment bytes without granting ACL or read-gate access."""

from hashlib import sha256

from onyx.db.regulatory_original_ingestion import current_original_receipts
from onyx.error_handling.error_codes import OnyxErrorCode
from onyx.error_handling.exceptions import OnyxError


def require_original_attachment_bytes(
    file_id: str, content: bytes, *, plaintext: bool = False
) -> None:
    digest = sha256(content).hexdigest()
    for receipt in current_original_receipts(file_id):
        if receipt is None or digest != (
            receipt.plaintext_sha256 if plaintext else receipt.raw_sha256
        ):
            raise OnyxError(
                OnyxErrorCode.SERVICE_UNAVAILABLE,
                "The uploaded original no longer proves the current source; use dated search.",
            )
