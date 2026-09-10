"""Stream-buffer behavior on the real cache backends: the chunk write/read
roundtrip, done-marking, and gap signaling must hold identically on Redis and
the Postgres (lite) backend."""

from uuid import uuid4

from onyx.cache.interface import CacheBackend
from onyx.chat.stream_buffer import StreamBufferWriter, _chunk_key, read_stream_chunks


def test_roundtrip_and_done(cache: CacheBackend) -> None:
    session_id = uuid4()
    writer = StreamBufferWriter(cache=cache, chat_session_id=session_id, run_id=7)
    writer.append_line('{"a": 1}\n')
    writer.flush()
    writer.append_line('{"b": 2}\n')
    writer.mark_done()

    read = read_stream_chunks(cache, session_id, 7, cursor=0)
    assert read is not None
    assert "".join(read.blocks) == '{"a": 1}\n{"b": 2}\n'
    assert read.done
    assert not read.gap

    tail = read_stream_chunks(cache, session_id, 7, cursor=read.next_cursor)
    assert tail is not None
    assert tail.blocks == []
    assert tail.done


def test_missing_chunk_is_gap(cache: CacheBackend) -> None:
    session_id = uuid4()
    writer = StreamBufferWriter(cache=cache, chat_session_id=session_id, run_id=8)
    writer.append_line('{"a": 1}\n')
    writer.flush()
    writer.append_line('{"b": 2}\n')
    writer.flush()

    # Simulate eviction of the first chunk.
    cache.delete(_chunk_key(session_id, 8, 0))

    read = read_stream_chunks(cache, session_id, 8, cursor=0)
    assert read is not None
    assert read.gap
    assert read.blocks == []


def test_missing_run_returns_none(cache: CacheBackend) -> None:
    assert read_stream_chunks(cache, uuid4(), 999, cursor=0) is None


def test_publication_change_invalidates_buffer_before_replay(
    cache: CacheBackend,
) -> None:
    from datetime import timedelta

    from onyx.regulatory.publication_reads import (
        PublicationReadEvidence,
        public_read_store,
    )
    from tests.external_dependency_unit.regulatory.test_publication_primitives import (
        create_owned_file,
    )

    with create_owned_file() as file_id:
        authority = public_read_store()
        owner = authority.acquire(file_id, owner_id=uuid4(), ttl=timedelta(seconds=30))
        evidence = PublicationReadEvidence(
            observation=authority.observe(), user_file_ids=(file_id,)
        )
        session_id = uuid4()
        writer = StreamBufferWriter(cache=cache, chat_session_id=session_id, run_id=10)
        writer.append_line('{"answer":"old"}\n', evidence=evidence)
        writer.flush()
        before = read_stream_chunks(cache, session_id, 10, cursor=0)
        assert before is not None and before.blocks
        authority.close_gate(owner)
        after = read_stream_chunks(cache, session_id, 10, cursor=0)
        assert after is not None and after.gap and not after.blocks
        writer.append_line('{"answer":"late old"}\n', evidence=evidence)
        writer.mark_done()
        final = read_stream_chunks(cache, session_id, 10, cursor=0)
        assert final is not None and final.gap and not final.blocks
