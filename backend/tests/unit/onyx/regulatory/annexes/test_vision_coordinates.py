"""The actual structured generator validates coordinates before persisting evidence."""

import hashlib
import json
from unittest.mock import MagicMock

import pytest
from jsonschema import Draft202012Validator

from onyx.llm.model_response import Choice, Message, ModelResponse
from onyx.regulatory.amendments.annexes import extraction
from onyx.regulatory.amendments.annexes.models import AcquiredAsset, AnnexRenderedPage
from onyx.regulatory.amendments.pdf_vision import prepare_pdf_source


def response(box: object) -> ModelResponse:
    return ModelResponse(
        id="fixture",
        created="2026-09-12",
        choice=Choice(
            message=Message(
                content=json.dumps(
                    {
                        "elements": [
                            {
                                "kind": "table_cell",
                                "text": "Unclear rate",
                                "table_role": "unknown",
                                "box": box,
                                "status": "uncertain",
                                "issues": ["uncertain_value"],
                            }
                        ]
                    }
                )
            )
        ),
    )


@pytest.fixture
def llm(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    monkeypatch.setattr(
        extraction,
        "run_in_isolated_process",
        lambda *_args, **_kwargs: (
            [],
            [AnnexRenderedPage(page=1, width=200, height=100, png=b"fixture pixels")],
        ),
    )
    model = MagicMock()
    model.config.model_name = "fixture"
    model.config.model_provider = "fixture"
    model.invoke.return_value = response(
        {"left": 0.1, "top": 0.2, "right": 0.8, "bottom": 0.9}
    )
    return model


def test_provider_requires_four_named_coordinates_and_preserves_durable_tuples(
    llm: MagicMock,
) -> None:
    result = extraction.extract_annex_structure(
        b"original pixels", "image/png", vision_llm=llm
    )
    schema = llm.invoke.call_args.kwargs["structured_response_format"]["json_schema"][
        "schema"
    ]
    element_schema = schema["$defs"][
        schema["properties"]["elements"]["items"]["$ref"].split("/")[-1]
    ]
    box = schema["$defs"][element_schema["properties"]["box"]["$ref"].split("/")[-1]]
    assert box["type"] == "object" and box["additionalProperties"] is False
    assert (
        set(box["properties"])
        == set(box["required"])
        == {"left", "top", "right", "bottom"}
    )
    assert all(item["type"] == "number" for item in box["properties"].values())
    validator = Draft202012Validator(schema)
    for invalid in (
        [0, 0, 1, 1, 0.5],
        {"left": 0, "top": 0, "right": 1},
        {"left": 0, "top": 0, "right": 1, "bottom": 1, "extra": 0},
    ):
        assert not validator.is_valid(
            json.loads(response(invalid).choice.message.content or "")
        )
    element = result.elements[0]
    assert element.locator.normalized_box == (0.1, 0.2, 0.8, 0.9)
    assert element.locator.original_box == (20, 20, 160, 90)
    assert element.text == "Unclear rate" and element.status == "uncertain"
    assert element.issues == ["uncertain_value"] and element.table_role == "unknown"
    assert result.source_sha256 == hashlib.sha256(b"original pixels").hexdigest()
    assert result.schema_version == 1
    assert result.model_dump(mode="json")["elements"][0]["locator"][
        "normalized_box"
    ] == [0.1, 0.2, 0.8, 0.9]


@pytest.mark.parametrize(
    "invalid",
    [
        [0, 0, 1, 1],
        [0, 0, 1, 1, 0.5],
        {"left": 0, "top": 0, "right": 1},
        {"left": 0, "top": 0, "right": 1, "bottom": 1, "extra": 0},
        *[
            {"left": value, "top": 0, "right": 1, "bottom": 1}
            for value in (
                -0.1,
                1,
                2,
                float("nan"),
                float("inf"),
                float("-inf"),
                True,
                "0",
                None,
            )
        ],
        {"left": 0, "top": 0.9, "right": 1, "bottom": 0.1},
        {"left": 0, "top": 0, "right": 1, "bottom": 1.1},
    ],
)
def test_invalid_coordinate_answer_is_corrected_under_original_deadline(
    monkeypatch: pytest.MonkeyPatch, llm: MagicMock, invalid: object
) -> None:
    now = [100.0]
    monkeypatch.setattr(extraction.time, "monotonic", lambda: now[0])
    corrected = llm.invoke.return_value

    def invoke(*_args: object, **_kwargs: object) -> ModelResponse:
        if llm.invoke.call_count == 1:
            now[0] = 104
            return response(invalid)
        return corrected

    llm.invoke.side_effect = invoke
    result = extraction.extract_annex_structure(
        b"original pixels", "image/png", vision_llm=llm, vision_deadline=110
    )
    assert result.elements[0].locator.normalized_box == (0.1, 0.2, 0.8, 0.9)
    first, second = llm.invoke.call_args_list
    assert first.kwargs["timeout_override"] == 10
    assert second.kwargs["timeout_override"] == 6
    assert first.args[0][1] == second.args[0][1]
    assert "box" in second.args[0][-1].content
    assert result.elements[0].issues == ["uncertain_value"]


@pytest.mark.parametrize("deadline_exhausted", [False, True])
def test_invalid_coordinate_exhaustion_never_persists_partial_output(
    monkeypatch: pytest.MonkeyPatch, llm: MagicMock, deadline_exhausted: bool
) -> None:
    now = [100.0]
    monkeypatch.setattr(extraction.time, "monotonic", lambda: now[0])

    def invalid(*_args: object, **_kwargs: object) -> ModelResponse:
        if deadline_exhausted:
            now[0] = 110
        return response({"left": 1, "top": 0, "right": 0, "bottom": 1})

    llm.invoke.side_effect = invalid
    store = MagicMock()
    with pytest.raises(
        TimeoutError if deadline_exhausted else ValueError,
        match="deadline" if deadline_exhausted else "after 2 attempts",
    ):
        prepare_pdf_source(
            AcquiredAsset(
                sha256="a" * 64,
                content=b"pdf",
                mime_type="application/pdf",
                display_name="fixture.pdf",
            ),
            store=store,
            llm=llm,
            deadline=110,
        )
    assert llm.invoke.call_count == (1 if deadline_exhausted else 2)
    store.save_file.assert_not_called()
