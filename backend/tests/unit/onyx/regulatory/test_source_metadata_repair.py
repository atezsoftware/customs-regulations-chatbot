import base64
import hashlib
from uuid import uuid4

from onyx.regulatory.amendments.annexes.publication_representations import _snapshot
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    canonical_row,
)


def test_image_recovery_uses_asset_and_source_position_between_unique_neighbors() -> (
    None
):
    from onyx.regulatory.source_metadata_repair import recover_markdown_image_parents

    file_id = uuid4()
    rows = [
        _snapshot(canonical_row(file_id, n, body)).model_copy(
            update={"heading_path": ["Same heading"]}
        )
        for n, body in enumerate(
            [
                "First section",
                "Repeated instructions",
                "Middle section",
                "Repeated instructions",
                "Last section",
            ]
        )
    ]
    image = rows[1].model_copy(
        update={
            "id": "image",
            "position": 5,
            "chunk_type": "image",
            "metadata": {
                "bound_to_regulatory_chunk_id": "lost",
                "image_file_id": "asset",
                "image_order": 1,
            },
        }
    )
    raw = (
        "First section\n\n![](data:image/png;base64,"
        + base64.b64encode(b"image bytes").decode()
        + ")\n\nRepeated instructions\n\nMiddle section\n\nRepeated instructions\n\nLast section"
    )
    proofs = recover_markdown_image_parents(
        [*rows, image],
        raw.encode(),
        {"asset": hashlib.sha256(b"image bytes").hexdigest()},
    )
    assert proofs[image.id].parent_id == rows[1].id
    assert proofs[image.id].parent_id != rows[3].id
    assert proofs[image.id].source_sha256 == hashlib.sha256(raw.encode()).hexdigest()


def test_image_recovery_rejects_wrong_asset_and_unrelated_source_text() -> None:
    from onyx.regulatory.source_metadata_repair import recover_markdown_image_parents

    parent = _snapshot(canonical_row(uuid4(), 0, "Other section"))
    image = parent.model_copy(
        update={
            "id": "image",
            "position": 1,
            "metadata": {
                "bound_to_regulatory_chunk_id": "lost",
                "image_file_id": "asset",
                "image_order": 1,
            },
        }
    )
    raw = b"![](data:image/png;base64,aW1hZ2U=)\n\nUnrelated source\n\nOther section"
    assert (
        recover_markdown_image_parents([parent, image], raw, {"asset": "wrong"}) == {}
    )
    assert (
        recover_markdown_image_parents(
            [parent, image], raw, {"asset": hashlib.sha256(b"image").hexdigest()}
        )
        == {}
    )
