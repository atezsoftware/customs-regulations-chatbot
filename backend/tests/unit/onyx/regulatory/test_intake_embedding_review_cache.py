"""A file's encoder guards reuse native evidence while rechecking authority."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from threading import Event
from typing import cast
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from onyx.db import document_intake_embedding as policy
from onyx.db.enums import UserFileStatus
from onyx.document_index.publication_models import FileOwnership
from onyx.regulatory import writer_publication as writer
from onyx.regulatory.document_intake.embedding import (
    FrozenEmbeddingGroup,
    encode_intake_embedding_groups,
)
from onyx.regulatory.labeling import chapter_caption_cache as cache
from tests.unit.onyx.regulatory.indexing_jobs.owned_publication_test_helpers import (
    OwnedAuthority,
    canonical_row,
    writer_inputs,
)
from tests.unit.onyx.regulatory.labeling.test_chapter_caption import FILE, RAW, sources
from tests.unit.onyx.regulatory.labeling.test_chapter_caption_cache import identity


@pytest.mark.parametrize("failure", [None, "authority", "encoder"])
def test_actual_writer_scopes_native_reuse_across_all_encoder_guards_and_cleanup(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    authority = OwnedAuthority(FILE)
    inputs = writer_inputs(FILE, [canonical_row(FILE, 0, "Exact source")])
    inputs.file.status = UserFileStatus.CHUNKED
    attempt = uuid4()
    read = Mock(return_value=RAW)
    parse = Mock(wraps=cache.build_chapter_caption_source_proof)
    monkeypatch.setattr(cache, "build_chapter_caption_source_proof", parse)
    checks = 0
    current = True

    def proof() -> None:
        cache.chapter_caption_native_proof(
            identity=identity(), chunk_id="caption", sources=sources(), read_raw=read
        )

    def check(owner: FileOwnership, token: object, **_kwargs: object) -> bool:
        nonlocal checks
        assert owner == authority.owner and token == attempt
        checks += 1
        # Each real review nests its own operation and reloads current authority.
        with cache.native_caption_proof_operation():
            proof()
        return current

    def encode(texts: list[str]) -> list[list[float]]:
        # Provider threads inherit only the caller's bounded evidence context.
        proof()
        if failure == "encoder":
            raise RuntimeError("encoder failed")
        return [[0.1, 0.2, 0.3] for _ in texts]

    def prepare(*_args: object, **kwargs: object) -> MagicMock:
        nonlocal current
        callback = cast(Callable[[], None], kwargs["before_embedding_request"])
        assert callable(callback)

        def guard() -> None:
            nonlocal current
            if failure == "authority" and checks >= 4:
                current = False
            callback()

        vectors = encode_intake_embedding_groups(
            [
                FrozenEmbeddingGroup(str(index), (f"source {index}",))
                for index in range(8)
            ],
            encode=encode,
            expected_dimension=3,
            check_owner=guard,
        )
        assert list(vectors) == [str(index) for index in range(8)]
        manifest = MagicMock()
        manifest.model_copy.return_value = manifest
        return manifest

    @contextmanager
    def heartbeat(_owner: FileOwnership) -> Iterator[Event]:
        yield Event()

    monkeypatch.setattr(
        "onyx.db.document_intake_review.require_publication_file_for_tenant", Mock()
    )
    monkeypatch.setattr("onyx.db.user_file.validate_user_file_index_request", Mock())
    monkeypatch.setattr(writer, "PublicationStore", authority.for_scope)
    monkeypatch.setattr(writer, "recover_owned_writer_before_next", lambda owner: owner)
    monkeypatch.setattr(writer, "publication_heartbeat", heartbeat)
    monkeypatch.setattr(
        "onyx.db.regulatory_writer_publication.load_owned_writer_inputs",
        lambda _owner: inputs,
    )
    monkeypatch.setattr(
        "onyx.regulatory.writer_projection.prepare_owned_correction", prepare
    )
    monkeypatch.setattr(
        "onyx.document_index.elasticsearch.client.ElasticsearchClient", MagicMock()
    )
    monkeypatch.setattr(policy, "current_intake_embedding_scope", check)
    publish = Mock()
    monkeypatch.setattr(writer, "execute_writer_publication", publish)
    for operation in (1, 2):
        current, checks = True, 0
        if failure is None:
            assert (
                writer.republish_user_file(
                    FILE,
                    "tenant-a",
                    include_chunked=True,
                    index_request_attempt_id=attempt,
                )
                == 1
            )
            assert checks >= 3
        else:
            with pytest.raises(
                RuntimeError if failure == "encoder" else ValueError,
                match="encoder failed" if failure == "encoder" else "authority changed",
            ):
                writer.republish_user_file(
                    FILE,
                    "tenant-a",
                    include_chunked=True,
                    index_request_attempt_id=attempt,
                )
        assert read.call_count == parse.call_count == operation
        assert len(authority.released) == operation
    assert publish.call_count == (2 if failure is None else 0)
    proof()
    assert read.call_count == parse.call_count == 3
