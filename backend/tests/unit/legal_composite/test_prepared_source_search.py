from datetime import date
from types import SimpleNamespace
from typing import Any, cast
from uuid import UUID, uuid4

import pytest

from onyx.chat.emitter import NullEmitter
from onyx.configs.constants import DocumentSource
from onyx.context.search.models import IndexFilters, InferenceChunk, PersonaSearchInfo
from onyx.db.asv3_corpus import CorpusSource
from onyx.db.legal_composite_sources import (
    SourceClassification,
    SourceKind,
    classify_source,
)
from onyx.db.models import User
from onyx.document_index.interfaces_new import DocumentIndex
from onyx.legal_composite import search
from onyx.legal_composite.search import CompositeSearchTool
from onyx.llm.interfaces import LLM
from onyx.natural_language_processing.search_nlp_models import EmbeddingModel
from onyx.regulatory.heading_path import RegulatoryProvisionReference
from onyx.tools.tool_implementations.search.search_tool import SearchTool


def tool() -> CompositeSearchTool:
    return CompositeSearchTool.from_fork(
        SearchTool(
            tool_id=1,
            emitter=NullEmitter(),
            user=cast(User, SimpleNamespace(id=uuid4(), is_anonymous=False)),
            persona_search_info=PersonaSearchInfo(
                document_set_names=["PC Külliyatı"],
                search_start_date=None,
                attached_document_ids=[],
                hierarchy_node_ids=[],
            ),
            llm=cast(LLM, object()),
            document_index=cast(DocumentIndex, object()),
            user_selected_filters=IndexFilters(
                access_control_list=["user-acl"],
                source_type=[DocumentSource.USER_FILE],
                document_set=["PC Külliyatı"],
                forced_document_set=["PC Külliyatı"],
                asv3_document_set_id=15,
                regulatory_chunks_only=True,
                as_of_date=date(2022, 1, 1),
            ),
            auto_detect_filters=False,
            project_id_filter=None,
        )
    )


def test_shared_search_keeps_type_steps_and_isolates_changed_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tool()
    owned.enable_shared_prepared_work()
    source_id = str(uuid4())
    assert owned.user_selected_filters is not None
    owned.user_selected_filters = owned.user_selected_filters.model_copy(
        update={"attached_document_ids": [source_id]}
    )
    model = EmbeddingModel.__new__(EmbeddingModel)
    monkeypatch.setattr(search, "embedding_model_key", lambda _: "fixed-test-encoder")
    calls = 0

    def pipeline(**_kwargs: Any) -> list[InferenceChunk]:
        nonlocal calls
        calls += 1
        return [chunk(source_id)]

    monkeypatch.setattr(search, "search_pipeline", pipeline)
    receipts: list[dict[str, Any]] = []

    def retrieve(lane: CompositeSearchTool) -> list[InferenceChunk]:
        return lane._run_search_for_query(
            "same question",
            0.5,
            False,
            100,
            ["fresh-acl"],
            model,
            [],
            lane.user_selected_filters,
        )

    first = owned.fork_for_independent_context()
    first.configure_prepared_source_lane(SourceKind.STATUTE, receipts.append)
    second = owned.fork_for_independent_context()
    second.configure_prepared_source_lane(SourceKind.COMMUNIQUE, receipts.append)
    assert retrieve(first)[0].document_id == source_id
    assert retrieve(second)[0].document_id == source_id
    assert calls == 1
    assert [r["source_kind"] for r in receipts] == ["statute", "communique"]
    assert [r["shared_exact_retrieval"] for r in receipts] == [False, True]
    assert second.user_selected_filters is not None
    second.user_selected_filters = second.user_selected_filters.model_copy(
        update={"as_of_date": date(2020, 1, 1)}
    )
    assert retrieve(second)[0].document_id == source_id
    assert calls == 2


def chunk(source_id: str, position: int = 0) -> InferenceChunk:
    return InferenceChunk(
        chunk_id=position,
        document_id=source_id,
        source_type=DocumentSource.USER_FILE,
        semantic_identifier=source_id,
        title="Wrong metadata heading",
        blurb="original",
        content="original",
        source_links={},
        image_file_id=None,
        section_continuation=False,
        boost=0,
        score=1.0,
        hidden=False,
        metadata={"document_type": "statute"},
        match_highlights=[],
        doc_summary="",
        chunk_context="",
        updated_at=None,
    )


def record(source_id: str, opening: str) -> SourceClassification:
    return classify_source(
        CorpusSource(UUID(source_id), "mahkeme.md", "file"),
        ("statute",),
        opening_texts=(opening,),
    )


def run(
    owned: CompositeSearchTool,
    *,
    num_hits: int = 1,
    query: str = "royalty vergi uygulaması",
    provision_reference: RegulatoryProvisionReference | None = None,
) -> list[InferenceChunk]:
    return owned._run_search_for_query(
        query,
        0.5,
        False,
        num_hits,
        ["fresh-acl"],
        cast(
            EmbeddingModel,
            SimpleNamespace(
                search_settings_id=1,
                model_name="embedding",
                provider_type=None,
                query_prefix=None,
                normalize=True,
                api_url=None,
                api_version=None,
                deployment_name=None,
                reduced_dimension=None,
            ),
        ),
        [],
        owned.user_selected_filters,
        provision_reference,
    )


def test_prepared_ids_filter_before_retrieval_without_opening_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tool()
    assert isinstance(owned.user_selected_filters, IndexFilters)
    wanted, forbidden = str(uuid4()), str(uuid4())
    owned.user_selected_filters.attached_document_ids = [wanted]
    receipts = []
    owned.configure_prepared_source_lane(
        SourceKind.PRESIDENTIAL_DECREE, receipts.append
    )
    calls = []

    def retrieve(**kwargs: Any) -> list[InferenceChunk]:
        request = kwargs["chunk_search_request"]
        assert request.user_selected_filters.attached_document_ids == [wanted]
        assert kwargs["acl_filters"] == ["fresh-acl"]
        calls.append(request)
        return [chunk(forbidden), chunk(wanted)]

    monkeypatch.setattr(search, "search_pipeline", retrieve)
    assert [row.document_id for row in run(owned, num_hits=5)] == [wanted]
    assert len(calls) == 1
    assert receipts[0]["runtime_opening_reads"] == 0
    assert receipts[0]["runtime_classifications"] == 0
    assert receipts[0]["unavailable_source_ids"] == [forbidden]
    assert receipts[0]["incomplete"] is True


@pytest.mark.parametrize("kind", list(SourceKind))
def test_borrowed_provision_cannot_filter_own_heading_in_any_lane(
    monkeypatch: pytest.MonkeyPatch, kind: SourceKind
) -> None:
    owned = tool()
    assert isinstance(owned.user_selected_filters, IndexFilters)
    source_id = str(uuid4())
    owned.user_selected_filters.attached_document_ids = [source_id]
    owned.configure_prepared_source_lane(kind)
    result = chunk(source_id)
    result.metadata["heading_path"] = ["Different document", "MADDE 9"]
    monkeypatch.setattr(search, "search_pipeline", lambda **_kwargs: [result])
    assert run(
        owned,
        query="Faaliyet Kanunu madde 27",
        provision_reference=RegulatoryProvisionReference("27", None),
    ) == [result]


@pytest.mark.parametrize("bad_scope", ["empty", "unbound", "acl_bypass"])
def test_unscoped_prepared_lane_never_falls_back_to_broad_search(
    monkeypatch: pytest.MonkeyPatch, bad_scope: str
) -> None:
    owned = tool()
    assert isinstance(owned.user_selected_filters, IndexFilters)
    owned.configure_prepared_source_lane(SourceKind.UNKNOWN)
    owned.user_selected_filters.attached_document_ids = [str(uuid4())]
    if bad_scope == "empty":
        owned.user_selected_filters.attached_document_ids = []
    elif bad_scope == "unbound":
        owned.user_selected_filters.asv3_document_set_id = None
    else:
        owned.bypass_acl = True

    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("Retrieval must not run")

    monkeypatch.setattr(search, "search_pipeline", forbidden)
    with pytest.raises(PermissionError):
        run(owned)


def test_host_related_search_does_not_apply_borrowed_article_heading_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tool()
    assert isinstance(owned.user_selected_filters, IndexFilters)
    observed = []

    def inherited(_self: Any, *args: Any) -> list[InferenceChunk]:
        observed.append(args[-1])
        return []

    monkeypatch.setattr(SearchTool, "_run_search_for_query", inherited)
    run(owned, provision_reference=RegulatoryProvisionReference("27", None))
    assert observed == [None]


def test_saturation_is_visible_without_claiming_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owned = tool()
    assert isinstance(owned.user_selected_filters, IndexFilters)
    source_id = str(uuid4())
    owned.user_selected_filters.attached_document_ids = [source_id]
    receipts = []
    owned.configure_prepared_source_lane(SourceKind.UNKNOWN, receipts.append)
    monkeypatch.setattr(search, "search_pipeline", lambda **_kwargs: [chunk(source_id)])
    assert len(run(owned)) == 1
    assert receipts[0]["candidate_window_saturated"] is True
    assert receipts[0]["corpus_absence_verified"] is False


def test_independent_fork_retains_prepared_lane_and_scope() -> None:
    owned = tool()
    assert isinstance(owned.user_selected_filters, IndexFilters)
    source_id = str(uuid4())
    owned.user_selected_filters.attached_document_ids = [source_id]
    receipts = []
    owned.configure_prepared_source_lane(SourceKind.COMMUNIQUE, receipts.append)
    fork = owned.fork_for_independent_context()
    assert fork._prepared_source_kind == SourceKind.COMMUNIQUE
    assert fork._record_prepared_search == receipts.append
    assert isinstance(fork.user_selected_filters, IndexFilters)
    assert fork.user_selected_filters.attached_document_ids is not None
    fork.user_selected_filters.attached_document_ids.append(str(uuid4()))
    assert owned.user_selected_filters.attached_document_ids == [source_id]
