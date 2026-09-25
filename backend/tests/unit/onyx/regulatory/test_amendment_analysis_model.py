from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import copy_context
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from onyx.regulatory.amendments import analysis_llm
from onyx.regulatory.amendments.model_choice import AmendmentAnalysisModel


def test_selected_vertex_model_follows_parallel_calls_and_resets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def session():
        yield MagicMock()

    monkeypatch.setattr(analysis_llm, "get_session_with_current_tenant", session)
    monkeypatch.setattr(
        analysis_llm,
        "fetch_vertex_model_configuration",
        lambda _session, name: SimpleNamespace(name=name, llm_provider=object()),
    )
    monkeypatch.setattr(
        analysis_llm.LLMProviderView, "from_model", lambda _: MagicMock()
    )
    factory = MagicMock(side_effect=lambda **kwargs: kwargs["model_name"])
    monkeypatch.setattr(analysis_llm, "llm_from_provider", factory)
    with analysis_llm.use_analysis_model("gemini-3.5-flash-lite"):
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = [
                pool.submit(copy_context().run, analysis_llm.get_amendment_analysis_llm)
                for _ in range(4)
            ]
            assert [r.result() for r in results] == ["gemini-3.5-flash-lite"] * 4
    assert analysis_llm.get_amendment_analysis_llm() == "gemini-3.8-flash"
    assert (
        analysis_llm.get_amendment_analysis_llm(model=AmendmentAnalysisModel.FLASH_LITE)
        == "gemini-3.5-flash-lite"
    )
    assert analysis_llm.get_amendment_analysis_llm() == "gemini-3.8-flash"


def test_missing_selected_vertex_model_never_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def session():
        yield MagicMock()

    monkeypatch.setattr(analysis_llm, "get_session_with_current_tenant", session)
    monkeypatch.setattr(
        analysis_llm, "fetch_vertex_model_configuration", lambda *_: None
    )
    with analysis_llm.use_analysis_model("gemini-3.5-flash-lite"):
        with pytest.raises(RuntimeError, match="gemini-3.5-flash-lite"):
            analysis_llm.get_amendment_analysis_llm()


def test_lite_annex_validation_restores_saved_model_in_a_later_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from uuid import uuid4

    from onyx.configs import app_configs
    from onyx.db import (
        amendment_analysis_settings,
        amendment_sources,
        regulatory_annex_publication,
    )
    from onyx.regulatory.amendments.annexes import analysis
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft

    selected = AmendmentAnalysisModel.FLASH_LITE
    model_config = {"model_name": selected.value}
    model_hash = analysis.context_hash(model_config)
    draft = AnnexChangeDraft(
        batch_id=17,
        instruction_indices=[0],
        instruction_texts=["EK-1 değiştirilmiştir"],
        annex_label="EK-1",
        user_file_id=uuid4(),
        source_package_id=uuid4(),
        source_manifest_sha256="manifest",
        source_graph=[],
        source_graph_sha256=analysis.context_hash([]),
        original_source_text_sha256="original",
        preparation_configuration={
            "analysis_model": model_hash,
            "vision_model": model_hash,
            "context_model": "none",
            "transport": "normal",
        },
    )

    @contextmanager
    def session():
        yield MagicMock()

    monkeypatch.setattr(analysis, "get_session_with_current_tenant", session)
    monkeypatch.setattr(app_configs, "REGULATORY_BATCH_INDEXING_ENABLED", False)
    monkeypatch.setattr(
        analysis, "capture_preparation_configuration", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        analysis, "get_batch", lambda *_args: SimpleNamespace(id=17, document_set_id=1)
    )
    saved_model = MagicMock(return_value=selected)
    monkeypatch.setattr(amendment_analysis_settings, "load_analysis_model", saved_model)
    monkeypatch.setattr(
        regulatory_annex_publication, "load_annex_context_settings", lambda *_args: None
    )
    monkeypatch.setattr(amendment_sources, "list_source_assets", lambda *_args: [])
    monkeypatch.setattr(
        amendment_sources,
        "require_ready_source_package",
        lambda *_args, **_kwargs: SimpleNamespace(
            manifest_file_id="file", manifest_sha256="manifest"
        ),
    )
    monkeypatch.setattr(analysis, "get_default_file_store", lambda: MagicMock())
    monkeypatch.setattr(analysis, "read_source_graph", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        analysis,
        "read_original_source_text",
        lambda *_args, **_kwargs: ("text", "original"),
    )
    monkeypatch.setattr(analysis, "resolve_review_context_llm", lambda *_args: None)

    def resolve(**kwargs):
        resolved = kwargs.get("model", analysis_llm._ANALYSIS_MODEL.get())
        result = MagicMock()
        result.config.model_dump.return_value = {"model_name": resolved.value}
        return result

    monkeypatch.setattr(analysis_llm, "get_amendment_analysis_llm", resolve)
    # The analysis process's ContextVar has gone; the next request starts at Flash.
    assert analysis_llm._ANALYSIS_MODEL.get() == AmendmentAnalysisModel.FLASH
    analysis.validate_live_review_runtime(draft)
    assert saved_model.call_args.args[1] == 17


@pytest.mark.parametrize("resume", [False, True])
def test_annex_preparation_restores_saved_model_outside_analysis_child(
    monkeypatch: pytest.MonkeyPatch,
    resume: bool,
) -> None:
    from contextlib import nullcontext
    from uuid import uuid4

    from onyx.background.celery.tasks.regulatory_amendments import (
        annex_preparation as worker,
    )
    from onyx.db import amendment_analysis_settings, regulatory_annex_changes
    from onyx.regulatory.amendments.annexes import analysis, corrections
    from onyx.regulatory.amendments.annexes.models import AnnexChangeDraft
    from shared_configs import contextvars

    draft = AnnexChangeDraft(
        batch_id=17,
        instruction_indices=[0],
        instruction_texts=["EK-1"],
        annex_label="EK-1",
    )
    payload = draft.model_dump(mode="json")
    job = SimpleNamespace(
        generation=1,
        expected_review_sha256="review",
        checkpoint=payload if resume else None,
        corrections=[],
        corrected_by=uuid4(),
    )
    monkeypatch.setattr(
        worker, "get_session_with_current_tenant", lambda: nullcontext(MagicMock())
    )
    monkeypatch.setattr(
        worker, "claim_review_preparation", lambda *_args, **_kwargs: job
    )
    monkeypatch.setattr(worker, "_heartbeat", lambda *_args: nullcontext())
    monkeypatch.setattr(worker, "touch_review_preparation", lambda **_kwargs: None)
    monkeypatch.setattr(contextvars, "get_current_tenant_id", lambda: "public")
    monkeypatch.setattr(
        regulatory_annex_changes,
        "require_current_annex_review",
        lambda *_args, **_kwargs: SimpleNamespace(review_payload=payload),
    )
    revised = MagicMock()
    monkeypatch.setattr(regulatory_annex_changes, "revise_annex_review", revised)
    monkeypatch.setattr(
        amendment_analysis_settings,
        "load_analysis_model",
        lambda *_args: AmendmentAnalysisModel.FLASH_LITE,
    )
    llm = MagicMock()
    factory = MagicMock(return_value=llm)
    monkeypatch.setattr(analysis_llm, "get_amendment_analysis_llm", factory)
    revalidate = MagicMock(return_value=draft)
    monkeypatch.setattr(corrections, "revalidate_annex_review", revalidate)
    validate_runtime = MagicMock()
    monkeypatch.setattr(analysis, "validate_live_review_runtime", validate_runtime)
    monkeypatch.setattr(analysis, "prepare_review_context", lambda value: value)
    worker.prepare_annex_review.run(
        review_id=str(uuid4()),
        tenant_id="public",
        environment=worker.config.REGULATORY_ANNEX_ENVIRONMENT,
        database_identity=worker.config.ANNEX_DATABASE_IDENTITY,
    )
    revised.assert_called_once()
    if resume:
        validate_runtime.assert_called_once()
        revalidate.assert_not_called()
    else:
        factory.assert_called_once_with(model=AmendmentAnalysisModel.FLASH_LITE)
        assert revalidate.call_args.kwargs["llm"] is llm
        assert revalidate.call_args.kwargs["vision_llm"] is llm
