import hashlib
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from PIL import Image

from onyx.regulatory.amendments.annexes import extraction, job
from onyx.regulatory.amendments.annexes.models import AnnexVisionWireResult


@pytest.mark.parametrize(
    "mode", ["document", "empty", "scene", "cached", "cached_empty"]
)
def test_image_source_preparation_requires_visible_text_and_reuses_verified_text(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    from onyx.regulatory.amendments import analysis_llm

    output = BytesIO()
    Image.new("RGB", (120, 160), "white").save(output, format="PNG")
    content = output.getvalue()
    text = "MADDE 1- Yeni metin."
    blobs = {"upload": content, "frozen-text": text.encode()}
    store = MagicMock()
    store.read_file.side_effect = lambda name: BytesIO(blobs[name])

    def save(stream: BytesIO, **_kwargs: object) -> str:
        name = f"saved-{len(blobs)}"
        blobs[name] = stream.read()
        return name

    store.save_file.side_effect = save
    previous = SimpleNamespace(
        sha256=hashlib.sha256(content).hexdigest(),
        mime_type="image/png",
        original_url=None,
        final_url=None,
        file_id="upload",
        byte_count=len(content),
        text_file_id=None if mode == "cached_empty" else "frozen-text",
        text_sha256=None
        if mode == "cached_empty"
        else hashlib.sha256(text.encode()).hexdigest(),
    )
    package = SimpleNamespace(
        input_spec={"mime_type": "image/png", "display_name": "source.png"},
        input_file_id="upload",
        manifest_file_id=None,
    )
    monkeypatch.setattr(
        job, "claim_source_package", lambda *_args, **_kw: (package, uuid4())
    )
    monkeypatch.setattr(
        job,
        "list_source_assets",
        lambda *_args: [previous] if mode.startswith("cached") else [],
    )
    monkeypatch.setattr(job, "get_session_with_current_tenant", MagicMock())
    monkeypatch.setattr(job, "get_default_file_store", lambda: store)
    monkeypatch.setattr(job, "extend_source_package_lease", lambda *_args, **_kw: True)
    finish, failed = MagicMock(return_value=True), MagicMock()
    monkeypatch.setattr(job, "finish_source_package", finish)
    monkeypatch.setattr(job, "mark_source_package_failed", failed)
    model = MagicMock()
    model.config.model_provider = "fixture"
    model.config.model_name = "vision"
    model_factory = MagicMock(return_value=model)
    monkeypatch.setattr(
        analysis_llm,
        "get_amendment_analysis_llm",
        lambda **kwargs: model_factory(**kwargs),
    )

    def visual_response(*_args: object, **kwargs: object) -> AnnexVisionWireResult:
        assert "Never describe the scene" in str(kwargs["system_prompt"])
        return AnnexVisionWireResult.model_validate(
            {
                "elements": []
                if mode == "empty"
                else [
                    {
                        "kind": "image_region" if mode == "scene" else "text",
                        "text": "A landscape photo" if mode == "scene" else text,
                        "box": {"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.2},
                        "status": "readable",
                        "issues": [],
                    }
                ]
            }
        )

    generate = MagicMock(side_effect=visual_response)
    monkeypatch.setattr(extraction, "generate_structured", generate)
    if mode in ("empty", "scene", "cached_empty"):
        with pytest.raises(ValueError):
            job.run_source_package(package_id=uuid4(), environment="local-test")
        finish.assert_not_called()
        failed.assert_called_once()
        if mode == "cached_empty":
            assert (
                str(failed.call_args.kwargs["failure"])
                == "image_source_requires_new_preparation"
            )
            generate.assert_not_called()
            store.save_file.assert_not_called()
        return
    job.run_source_package(package_id=uuid4(), environment="local-test")
    result = finish.call_args.kwargs["result"]
    assert result.status == "ready"
    assert result.assets[0].content == content
    assert result.assets[0].text == text
    assert result.assets[0].page_count == 1
    if mode == "cached":
        model_factory.assert_not_called()
        generate.assert_not_called()
    else:
        asset = finish.call_args.kwargs["assets"][0]
        assert blobs[asset.text_file_id] == text.encode()
        assert asset.text_sha256 == hashlib.sha256(text.encode()).hexdigest()
