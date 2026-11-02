"""Native source boundaries must not be inferred from presentation alone."""

import pytest

from onyx.regulatory.document_intake.structural_readers import (
    build_native_chapter_reader_proofs,
)
from tests.unit.onyx.regulatory.test_intake_native_chapter_readers import (
    ASSET,
    FILE,
    HTML,
    build,
    scope,
)


def test_no_children_incomplete_articles_and_nonstandalone_text_are_not_readers() -> (
    None
):
    assert build(rows=scope()[:1]) == {}
    assert build(rows=scope()[:2]) == {}
    assert (
        build(
            rows=[
                scope()[0].model_copy(
                    update={"text": "Belgelerin sunulması\nNormatif hüküm"}
                ),
                *scope()[1:],
            ]
        )
        == {}
    )
    raw = HTML.encode()
    with pytest.raises(ValueError, match="raw source changed"):
        build_native_chapter_reader_proofs(
            raw_html=raw,
            expected_sha256="0" * 64,
            user_file_id=FILE,
            source_asset_id=ASSET,
            rows=scope(),
        )


@pytest.mark.parametrize(
    "html",
    [
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<p id="caption" class="caption" style="text-align:left;font-weight:normal">Belgelerin sunulması</p>',
        )
        + "<style>#caption#caption {text-align:center;font-weight:bold}</style>",
        HTML.replace(
            "text-align:center;font-weight:bold",
            "text-align:center;font-weight:normal!important;font-weight:bold",
        ),
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<p class="caption" hidden>Belgelerin sunulması</p>',
        ),
        "<template>" + HTML + "</template>",
        HTML.replace(
            '<p class="caption">Belgelerin sunulması</p>',
            '<p class="caption" style="all:initial">Belgelerin sunulması</p>',
        ),
        HTML + '<link rel="stylesheet" href="https://example.test/unverified.css">',
        HTML + "<style>@media print {p {font-weight:normal}}</style>",
    ],
)
def test_css_cascade_and_nonrendered_source_cannot_falsely_prove_caption(
    html: str,
) -> None:
    assert build(html) == {}
