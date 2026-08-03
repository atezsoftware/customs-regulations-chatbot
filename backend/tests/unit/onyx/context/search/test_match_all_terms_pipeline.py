from unittest.mock import MagicMock

from pytest import MonkeyPatch

from onyx.context.search import pipeline
from onyx.context.search.models import ChunkSearchRequest


def test_search_pipeline_propagates_high_term_coverage(
    monkeypatch: MonkeyPatch,
) -> None:
    captured_request = None

    def fake_search_chunks(**kwargs: object) -> list[object]:
        nonlocal captured_request
        captured_request = kwargs["query_request"]
        return []

    monkeypatch.setattr(pipeline, "search_chunks", fake_search_chunks)
    monkeypatch.setattr(
        pipeline,
        "fetch_ee_implementation_or_noop",
        lambda *_args: lambda *, chunks, **_kwargs: chunks,
    )

    pipeline.search_pipeline(
        chunk_search_request=ChunkSearchRequest(
            query="exact legal phrase",
            high_term_coverage=True,
            bypass_acl=True,
        ),
        document_index=MagicMock(),
        user=MagicMock(id=None, is_anonymous=True),
        persona_search_info=None,
    )

    assert captured_request is not None
    assert captured_request.high_term_coverage is True
